// encode-score: score video encodes against their source on the GPU, every frame, in one pass over all the files.
//
// Every frame pair gets VMAF (libvmaf's CUDA backend), PSNR per plane, CAMBI banding of both files, mean color, and
// detail measured in 16 by 16 windows: the amount of fine and mid detail against the source's, the share of the source's
// detail the encode reproduces, and the share of the source's lasting texture it reproduces (texture that stays put over
// the 6 frames either side, so grain cannot count). The run is summarized per segment and as a whole, with a 0 to 100
// score from four parts (see stats.hpp).
#include "cache.hpp"
#include "cambi.hpp"
#include "decode.hpp"
#include "gpu.hpp"
#include "stats.hpp"
extern "C" {
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_cuda.h>
#include <libvmaf/libvmaf.h>
#include <libvmaf/libvmaf_cuda.h>
#include <libavutil/pixdesc.h>
// libvmaf's internal accessor for the stream a picture's work starts on (cuda/picture_cuda.h)
CUstream vmaf_cuda_picture_get_stream(VmafPicture *pic);
}
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <vector>

static const char *kUsage =
    "usage: encode-score SOURCE ENCODE [ENCODE ...] [options]\n"
    "  --start SECONDS    where to begin in the encodes (default 0)\n"
    "  --offset SECONDS   added to --start for the source, when the encodes start later in it; `auto` finds it by\n"
    "                     matching thumbnails within 10 seconds either side of 0, `auto=SECONDS` either side of a\n"
    "                     guess (default auto)\n"
    "  --seconds N        how much to score (default: to the end of the shortest file)\n"
    "  --segment SECONDS  length of the summary segments (default 60)\n"
    "  --model NAME       VMAF model (default: vmaf_4k_v0.6.1 for 4K pictures, vmaf_v0.6.1 below)\n"
    "  --json FILE        write every per-frame and per-segment figure\n";

struct Options {
    std::string source, json, model = "auto";
    std::vector<std::string> encodes;
    double start = 0, offset = 0, seconds = -1, segment = 60;
    bool find_offset = true;
};

static Options parse(int argc, char **argv)
{
    Options o;
    std::vector<std::string> positional;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto value = [&] { if (i + 1 >= argc) throw std::runtime_error(a + " needs a value"); return std::string(argv[++i]); };
        if (a == "--start") o.start = std::stod(value());
        else if (a == "--offset") {
            std::string v = value();
            o.find_offset = v.rfind("auto", 0) == 0;
            o.offset = o.find_offset ? (v.size() > 5 ? std::stod(v.substr(5)) : 0.0) : std::stod(v);
        }
        else if (a == "--seconds") o.seconds = std::stod(value());
        else if (a == "--segment") o.segment = std::stod(value());
        else if (a == "--json") o.json = value();
        else if (a == "--model") o.model = value();
        else if (a == "-h" || a == "--help") { std::fputs(kUsage, stdout); std::exit(0); }
        else if (a.rfind("--", 0) == 0) throw std::runtime_error("unknown option " + a + "\n" + kUsage);
        else positional.push_back(a);
    }
    if (positional.size() < 2) throw std::runtime_error(std::string("expected SOURCE and at least one ENCODE\n") + kUsage);
    o.source = positional[0];
    o.encodes.assign(positional.begin() + 1, positional.end());
    return o;
}

static void vmaf_check(int err, const char *what)
{
    if (err) throw std::runtime_error(std::string(what) + " failed (libvmaf error " + std::to_string(err) + ")");
}

// A device buffer that frees itself.
struct Buffer {
    CUdeviceptr ptr = 0;
    explicit Buffer(size_t bytes) { cu_check(cuMemAlloc(&ptr, bytes), "allocating GPU memory"); }
    ~Buffer() { if (ptr) cuMemFree(ptr); }
    Buffer(const Buffer &) = delete;
};

// The planes of one frame of one file at the pipeline's bit depth.
struct Planes {
    Buffer y16, u, v;
    Planes(size_t n) : y16(n * 2), u(n / 2), v(n / 2) {}
};

// A decoded frame as the kernels read it (kernels.cu's Frame).
struct FrameView {
    const void *y, *u, *v;
    int ypitch, cpitch, semi, bytes, shift, depth;
};

// Page-locked host memory, which the GPU copies to and from without the host waiting.
struct Pinned {
    void *ptr = nullptr;
    explicit Pinned(size_t bytes) { cu_check(cuMemAllocHost(&ptr, bytes), "allocating host memory"); }
    ~Pinned() { if (ptr) cuMemFreeHost(ptr); }
    Pinned(const Pinned &) = delete;
};

// GPU memory for a frame decoded on the processor.
struct Upload {
    CUdeviceptr ptr = 0;
    size_t size = 0;
    ~Upload() { if (ptr) cuMemFree(ptr); }
    CUdeviceptr reserve(size_t bytes)
    {
        if (bytes > size) {
            if (ptr) cuMemFree(ptr);
            cu_check(cuMemAlloc(&ptr, bytes), "allocating upload memory");
            size = bytes;
        }
        return ptr;
    }
};

// How the kernels should read a frame: GPU frames as NVDEC left them (P010 or NV12), processor frames after copying
// their planes to the GPU on `stream`. Only 4:2:0 at 8 or 10 bits is supported.
static FrameView view_of(const AVFrame *f, Upload &up, CUstream stream)
{
    if (f->format == AV_PIX_FMT_CUDA) {
        const AVPixelFormat sw = ((AVHWFramesContext *)f->hw_frames_ctx->data)->sw_format;
        if (sw == AV_PIX_FMT_P010LE) return {f->data[0], f->data[1], nullptr, f->linesize[0], f->linesize[1], 1, 2, 6, 10};
        if (sw == AV_PIX_FMT_NV12) return {f->data[0], f->data[1], nullptr, f->linesize[0], f->linesize[1], 1, 1, 0, 8};
        throw std::runtime_error(std::string("unsupported GPU frame format ") + av_get_pix_fmt_name(sw) + " (4:2:0 at 8 or 10 bits only)");
    }
    const AVPixelFormat fmt = (AVPixelFormat)f->format;
    int semi, bytes, shift, depth;
    if (fmt == AV_PIX_FMT_YUV420P) { semi = 0; bytes = 1; shift = 0; depth = 8; }
    else if (fmt == AV_PIX_FMT_YUV420P10LE) { semi = 0; bytes = 2; shift = 0; depth = 10; }
    else if (fmt == AV_PIX_FMT_NV12) { semi = 1; bytes = 1; shift = 0; depth = 8; }
    else if (fmt == AV_PIX_FMT_P010LE) { semi = 1; bytes = 2; shift = 6; depth = 10; }
    else throw std::runtime_error(std::string("unsupported pixel format ") + av_get_pix_fmt_name(fmt) + " (4:2:0 at 8 or 10 bits only)");
    const size_t ysize = (size_t)f->linesize[0] * f->height, csize = (size_t)f->linesize[1] * ((f->height + 1) / 2);
    const CUdeviceptr base = up.reserve(ysize + 2 * csize);
    // from the page-locked memory the decoder gives a processor frame, so the frame is kept until the copy is done
    cu_check(cuMemcpyHtoDAsync(base, f->data[0], ysize, stream), "uploading a frame");
    cu_check(cuMemcpyHtoDAsync(base + ysize, f->data[1], csize, stream), "uploading a frame");
    if (!semi) cu_check(cuMemcpyHtoDAsync(base + ysize + csize, f->data[2], csize, stream), "uploading a frame");
    return {(const void *)base, (const void *)(base + ysize), semi ? nullptr : (const void *)(base + ysize + csize),
            f->linesize[0], f->linesize[1], semi, bytes, shift, depth};
}

// What a file's first frame and header say: whether it decodes on the GPU's decoder, its frames' bit depth and size, its
// frame rate and its length.
struct Probe {
    bool gpu;
    int depth, w, h;
    double fps, seconds;
};

static Probe probe(const std::string &path, AVBufferRef *device, CUstream stream)
{
    for (bool gpu : {true, false}) {
        try {
            Decoder d(path, device, 0, gpu);
            AVFrame *f = d.next();
            if (!f) throw std::runtime_error(path + " has no frames");
            Upload up;
            const int depth = view_of(f, up, stream).depth;
            cu_check(cuStreamSynchronize(stream), "probing");
            av_frame_free(&f);
            return {gpu, depth, d.width(), d.height(), d.frame_rate(), d.duration()};
        } catch (const std::exception &) {
            if (!gpu) throw;
        }
    }
    return {};
}

// Frames per second at which `path` decodes from `start` seconds on the GPU's decoder or the processor, measured over
// 96 frames after the first 24 (fewer give too high a figure on the processor, whose threads have frames in hand). The
// measurement stops early, with a lower figure, once it is known to be slower than `floor`.
static double decode_rate(const std::string &path, AVBufferRef *device, bool gpu, double start, double floor)
{
    constexpr int kSkip = 24, kFrames = 96;
    const auto t0 = std::chrono::steady_clock::now();
    auto since = [](std::chrono::steady_clock::time_point t) { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count(); };
    Decoder d(path, device, start, gpu);
    std::chrono::steady_clock::time_point t1;
    int n = 0;
    double elapsed = 0;
    for (AVFrame *f; n < kSkip + kFrames && (f = d.next()); n++) {
        av_frame_free(&f);
        if (n == kSkip) t1 = std::chrono::steady_clock::now();
        if (n > kSkip) elapsed = since(t1);
        if (floor > 0 && since(t0) * floor > kSkip + kFrames) break;
    }
    return n > kSkip + 1 ? (n - kSkip - 1) / elapsed : 0;
}

// Which files decode on the GPU's decoder (true) and which on the processor, from each file's rate on each (`gpu_rate`
// 0 where the GPU cannot decode it). All files decode at once, and the GPU's decoder and the processor each take as
// long as all the files given to them together, so the files are split to make the slower of the two as fast as
// possible. A file leaves the GPU's decoder only for a gain of more than 10%, which measurement noise does not give.
static std::vector<bool> place(const std::vector<double> &gpu_rate, const std::vector<double> &cpu_rate)
{
    const size_t n = gpu_rate.size();
    auto rate = [&](const std::vector<bool> &on_gpu) {
        double g = 0, c = 0;  // seconds per frame of each
        for (size_t i = 0; i < n; i++) (on_gpu[i] ? g : c) += 1 / (on_gpu[i] ? gpu_rate[i] : cpu_rate[i]);
        return 1 / std::max(g, c);
    };
    std::vector<bool> start(n);
    for (size_t i = 0; i < n; i++) start[i] = gpu_rate[i] > 0;
    std::vector<bool> best = start;
    double best_rate = rate(start) * 1.1;
    for (unsigned mask = 1; mask < (1u << n); mask++) {
        std::vector<bool> on_gpu = start;
        bool possible = true;
        for (size_t i = 0; i < n; i++)
            if (mask >> i & 1) { possible &= on_gpu[i] && cpu_rate[i] > 0; on_gpu[i] = false; }
        if (possible && rate(on_gpu) > best_rate) { best = on_gpu; best_rate = rate(on_gpu); }
    }
    return best;
}

struct Kernel {
    CUfunction frame_planes, to_picture, window_stats, persist_stats, plane_sse, color_sums;
    explicit Kernel(const Kernels &k)
        : frame_planes(k.get("frame_planes")), to_picture(k.get("to_picture")), window_stats(k.get("window_stats")), persist_stats(k.get("persist_stats")), plane_sse(k.get("plane_sse")),
          color_sums(k.get("color_sums")) {}
};

struct Frame {
    double vmaf = 0, cambi = 0, cambi_source = 0, psnr[3] = {};
};

static constexpr int kRing = 13, kRadius = 6, kStats = 8, kPersist = 5;

// The figures of one frame of one encode in host memory: PSNR's squared errors, mean color, and per window the detail
// stats and the lasting texture stats of the frame kRadius back.
struct Results {
    Pinned mem;
    unsigned long long *sse;
    double *color;
    float *stats, *persist;
    explicit Results(size_t windows)
        : mem(48 + windows * (kStats + kPersist) * 4), sse((unsigned long long *)mem.ptr), color((double *)mem.ptr + 3),
          stats((float *)((char *)mem.ptr + 48)), persist(stats + windows * kStats) {}
};

// The GPU works on one frame while the host reads the figures of the frame before it, so whatever a frame's figures
// are read from comes in two, used by frame % 2.
struct Candidate {
    std::string path;
    int x0 = 0, y0 = 0;
    std::unique_ptr<Planes> planes;
    std::vector<std::unique_ptr<Buffer>> mid;  // the encode's mid detail planes, by frame % (kRadius + 1)
    std::unique_ptr<Buffer> stats, persist, sse, color;
    std::unique_ptr<Cambi> cambi[2];
    std::unique_ptr<Results> results[2];
    Upload upload[2];
    VmafContext *vmaf = nullptr;
    VmafModel *model = nullptr;
    std::vector<Frame> frames;
    std::vector<Accum> segments;
    std::vector<std::vector<uint8_t>> mask_ring;  // the structured windows of the last kRadius + 1 frames
};

// PSNR as libvmaf computes it at bit depth `depth`: peak 2^depth - 1, capped at 6 * depth + 12 dB.
static double psnr(unsigned long long sse, size_t n, int depth)
{
    const double mse = (double)sse / (double)n, peak = (double)((1 << depth) - 1);
    return std::min(10.0 * std::log10(peak * peak / std::max(mse, 1e-16)), 6.0 * depth + 12);
}

// numpy.quantile with linear interpolation, as the Python version chose its structured windows.
static float quantile(std::vector<float> v, double q)
{
    const double pos = q * (double)(v.size() - 1);
    const size_t lo = (size_t)pos;
    std::nth_element(v.begin(), v.begin() + lo, v.end());
    float a = v[lo];
    if (lo + 1 >= v.size()) return a;
    float b = *std::min_element(v.begin() + lo + 1, v.end());
    return (float)(a + (b - a) * (pos - (double)lo));
}

// Thumbnails (64 by 36 luma means of the cropped picture) and times of `count` frames of a file from `start` seconds.
struct Thumbnails {
    std::string path;
    bool gpu;
    double start;
    int count;
    std::vector<std::vector<float>> thumbs;
    std::vector<double> times;
};

// Fills in the thumbnails of each file, decoding the files side by side.
static void thumbnails(std::vector<Thumbnails> &files, AVBufferRef *device, CUstream stream, CUfunction tiny, int w, int h)
{
    std::vector<std::unique_ptr<Decoder>> decoders;
    for (const Thumbnails &t : files) decoders.push_back(std::make_unique<Decoder>(t.path, device, std::max(t.start, 0.0), t.gpu));
    std::vector<bool> ended(files.size());
    Upload up;
    CUdeviceptr out;
    cu_check(cuMemAlloc(&out, 64 * 36 * 4), "thumbnails");
    for (bool more = true; more;) {
        more = false;
        for (size_t i = 0; i < files.size(); i++) {
            Thumbnails &t = files[i];
            Decoder &d = *decoders[i];
            if (ended[i] || (int)t.thumbs.size() >= t.count) continue;
            AVFrame *f = d.next();
            if (!(ended[i] = !f)) {
                FrameView v = view_of(f, up, stream);
                int xx = (d.width() - w) / 2 & ~1, yy = (d.height() - h) / 2 & ~1, ww = w, hh = h;
                void *a[] = {&v, &xx, &yy, &ww, &hh, &out};
                cu_check(cuLaunchKernel(tiny, 64, 36, 1, 32, 8, 1, 0, stream, a, nullptr), "thumbnails");
                t.thumbs.emplace_back(64 * 36);
                cu_check(cuMemcpyDtoHAsync(t.thumbs.back().data(), out, 64 * 36 * 4, stream), "thumbnails");
                cu_check(cuStreamSynchronize(stream), "thumbnails");
                t.times.push_back(d.seconds(f));
                av_frame_free(&f);
                more = true;
            }
        }
    }
    cuMemFree(out);
}

// The offset to add to the encode's times so its frames match the source's: the shift, to the frame, at which the
// encode's thumbnails best match the source's, which cover `search` seconds either side of `guess`.
static double find_offset(const Thumbnails &ref, const Thumbnails &probe, double guess, double search)
{
    if (probe.thumbs.size() < 24 || ref.thumbs.size() < probe.thumbs.size()) throw std::runtime_error("too few frames to find the offset");
    const size_t n = probe.thumbs.size();
    double best = 1e30;
    size_t lag = 0;
    for (size_t l = 0; l + n <= ref.thumbs.size(); l++) {
        double total = 0;
        for (size_t i = 0; i < n && total < best * n * 2304; i++)
            for (int p = 0; p < 64 * 36; p++) total += std::fabs(ref.thumbs[l + i][p] - probe.thumbs[i][p]);
        const double mean = total / (n * 2304.0);
        if (mean < best) { best = mean; lag = l; }
    }
    if (best > 16.0)  // mean thumbnail difference in 10-bit code values; matching pictures are well under 5
        throw std::runtime_error(probe.path + " does not match the source within " + std::to_string((int)search) +
                                 " s of offset " + std::to_string(guess) + " (closest mean difference " + std::to_string(best) + "); pass --offset");
    return ref.times[lag] - probe.times[0];
}

int main(int argc, char **argv) try {
    Options o = parse(argc, argv);
    av_log_set_level(AV_LOG_ERROR);
    AVBufferRef *device = nullptr;
    if (av_hwdevice_ctx_create(&device, AV_HWDEVICE_TYPE_CUDA, nullptr, nullptr, AV_CUDA_USE_PRIMARY_CONTEXT) < 0)
        throw std::runtime_error("no CUDA device");
    CUcontext ctx = ((AVCUDADeviceContext *)((AVHWDeviceContext *)device->data)->hwctx)->cuda_ctx;
    cu_check(cuCtxPushCurrent(ctx), "making the CUDA context current");
    Kernels kernels;
    kernels.load();
    Kernel k(kernels);
    // all GPU work, the decoders' copies of their frames included, goes on one stream, so it runs in the order it is
    // queued without the host waiting in between
    CUstream stream;
    cu_check(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING), "creating a stream");
    ((AVCUDADeviceContext *)((AVHWDeviceContext *)device->data)->hwctx)->stream = stream;

    // the files' bit depths and the picture area they share; the source is files[0]
    std::vector<std::string> files = {o.source};
    files.insert(files.end(), o.encodes.begin(), o.encodes.end());
    std::vector<Probe> probes;
    for (const std::string &f : files) probes.push_back(probe(f, device, stream));
    int w = probes[0].w, h = probes[0].h, pd = 8;
    const double fps = probes[0].fps;
    double length = 1e30;  // the seconds of video the run will cover, as far as the files say
    for (size_t i = 0; i < files.size(); i++) {
        const Probe &p = probes[i];
        w = std::min(w, p.w); h = std::min(h, p.h); pd = std::max(pd, p.depth);
        if (p.seconds > 0) length = std::min(length, p.seconds - o.start - (i ? 0 : o.offset));
    }
    if (o.seconds > 0) length = std::min(length, o.seconds);
    w &= ~1; h &= ~1;
    // where each file decodes: the GPU's decoder where it can, unless 5 minutes or more of video are to be scored, so
    // that measuring the rates costs little against the run, and giving some files to the processor makes decoding
    // faster
    std::vector<double> gpu_rate(files.size(), 0), cpu_rate(files.size(), 0);
    std::vector<bool> on_gpu(files.size());
    for (size_t i = 0; i < files.size(); i++) on_gpu[i] = probes[i].gpu;
    double gpu_time = 0;  // seconds per frame of decoding all the files the GPU can decode on it
    if (length >= 300) {
        for (size_t i = 0; i < files.size(); i++)
            if (probes[i].gpu) gpu_time += 1 / (gpu_rate[i] = decode_rate(files[i], device, true, o.start + (i ? 0 : o.offset), 0));
        // a file the GPU can decode is only worth moving if the processor alone decodes it faster than that
        for (size_t i = 0; i < files.size(); i++)
            cpu_rate[i] = decode_rate(files[i], device, false, o.start + (i ? 0 : o.offset), probes[i].gpu ? 1 / gpu_time : 0);
        on_gpu = place(gpu_rate, cpu_rate);
    }
    for (size_t i = 0; i < files.size(); i++) {
        if (!probes[i].gpu) std::fprintf(stderr, "%s decodes on the processor (no GPU decoder for its format)\n", files[i].c_str());
        else if (!on_gpu[i])
            std::fprintf(stderr, "%s decodes on the processor (%.0f fps there, %.0f on the GPU's decoder), which makes decoding faster\n",
                         files[i].c_str(), cpu_rate[i], gpu_rate[i]);
    }
    const bool source_gpu = on_gpu[0];
    std::vector<bool> encode_gpu(on_gpu.begin() + 1, on_gpu.end());
    // finding the offset decodes one file at a time, each where it decodes fastest alone
    auto alone_gpu = [&](size_t i) { return probes[i].gpu && gpu_rate[i] >= cpu_rate[i]; };
    if (o.find_offset) {
        // 8 seconds of each encode against the source within 10 seconds either side of the guess
        const double search = 10;
        std::vector<Thumbnails> thumbs = {{o.source, alone_gpu(0), o.start + o.offset - search, (int)((2 * search + 8) * fps) + 2}};
        for (size_t i = 0; i < o.encodes.size(); i++) thumbs.push_back({o.encodes[i], alone_gpu(i + 1), o.start, (int)(8 * fps)});
        thumbnails(thumbs, device, stream, kernels.get("tiny_luma"), w, h);
        std::vector<double> found;
        for (size_t i = 1; i < thumbs.size(); i++) found.push_back(find_offset(thumbs[0], thumbs[i], o.offset, search));
        for (double f : found)
            if (std::fabs(f - found[0]) > 0.5 / fps) throw std::runtime_error("the encodes start at different points of the source; score them separately");
        o.offset = found[0];
        std::printf("offset of the source against the encodes: %+.3f s\n", o.offset);
    }
    std::vector<std::unique_ptr<Decoder>> decoders;  // of files[i]
    decoders.push_back(std::make_unique<Decoder>(o.source, device, o.start + o.offset, source_gpu));
    std::vector<Candidate> cands(o.encodes.size());
    for (size_t c = 0; c < cands.size(); c++) {
        cands[c].path = o.encodes[c];
        decoders.push_back(std::make_unique<Decoder>(o.encodes[c], device, o.start, encode_gpu[c]));
    }
    const int sx = (probes[0].w - w) / 2 & ~1, sy = (probes[0].h - h) / 2 & ~1;
    const long limit = o.seconds > 0 ? (long)std::llround(o.seconds * fps) : -1;
    const long per_segment = std::max(1L, (long)std::llround(o.segment * fps));
    const std::string model_name = o.model != "auto" ? o.model : (w >= 3840 ? "vmaf_4k_v0.6.1" : "vmaf_v0.6.1");
    const size_t n = (size_t)w * h, plane = n;
    const int ww = w / 16, wh = h / 16, windows = ww * wh;

    Planes src(n);
    Upload source_upload[2];
    // processor frames are copied to the GPU on a stream of their own, so the copies overlap the work on the stream
    CUstream copies;
    cu_check(cuStreamCreate(&copies, CU_STREAM_NON_BLOCKING), "creating a stream");
    CUevent copied;
    cu_check(cuEventCreate(&copied, CU_EVENT_DISABLE_TIMING), "creating an event");
    int up10 = 10 - pd;
    Buffer src_color(3 * 8);
    Buffer ring(n * 4 * kRing);  // the source's mid detail planes, by frame % kRing
    std::unique_ptr<Cambi> source_cambi[2];
    for (auto &c : source_cambi) c = std::make_unique<Cambi>(kernels, w, h, pd);
    CambiCache cambi_cache(o.source, sx, sy, w, h, pd);
    long cambi_cached = 0;
    // per frame % 2: the source's mean color, its timestamp and cached CAMBI, and an event for when the frame's figures
    // are in host memory
    Pinned source_color(2 * 3 * 8);
    int64_t source_pts[2];
    std::optional<double> source_cached[2];
    CUevent done[2];
    for (CUevent &e : done) cu_check(cuEventCreate(&e, CU_EVENT_DISABLE_TIMING), "creating an event");
    std::vector<AVFrame *> held[2];
    CUevent pictures;
    cu_check(cuEventCreate(&pictures, CU_EVENT_DISABLE_TIMING), "creating an event");

    for (size_t ci = 0; ci < cands.size(); ci++) {
        Candidate &c = cands[ci];
        c.x0 = (probes[ci + 1].w - w) / 2 & ~1;
        c.y0 = (probes[ci + 1].h - h) / 2 & ~1;
        c.planes = std::make_unique<Planes>(n);
        for (int i = 0; i <= kRadius; i++) c.mid.push_back(std::make_unique<Buffer>(n * 4));
        c.stats = std::make_unique<Buffer>((size_t)windows * kStats * 4);
        c.persist = std::make_unique<Buffer>((size_t)windows * kPersist * 4);
        c.sse = std::make_unique<Buffer>(3 * 8);
        c.color = std::make_unique<Buffer>(3 * 8);
        for (int i = 0; i < 2; i++) {
            c.cambi[i] = std::make_unique<Cambi>(kernels, w, h, pd);
            c.results[i] = std::make_unique<Results>(windows);
        }
        c.mask_ring.assign(kRadius + 1, std::vector<uint8_t>(windows));
        VmafConfiguration cfg = {};
        cfg.log_level = VMAF_LOG_LEVEL_WARNING;
        vmaf_check(vmaf_init(&c.vmaf, cfg), "vmaf_init");
        VmafCudaState *cu_state = nullptr;
        VmafCudaConfiguration cu_cfg = {ctx};
        vmaf_check(vmaf_cuda_state_init(&cu_state, cu_cfg), "vmaf_cuda_state_init");
        vmaf_check(vmaf_cuda_import_state(c.vmaf, cu_state), "vmaf_cuda_import_state");
        VmafCudaPictureConfiguration pic_cfg = {};
        pic_cfg.pic_params = {(unsigned)w, (unsigned)h, (unsigned)pd, VMAF_PIX_FMT_YUV420P};
        pic_cfg.pic_prealloc_method = VMAF_CUDA_PICTURE_PREALLOCATION_METHOD_DEVICE;
        vmaf_check(vmaf_cuda_preallocate_pictures(c.vmaf, pic_cfg), "vmaf_cuda_preallocate_pictures");
        VmafModelConfig model_cfg = {};
        model_cfg.name = "vmaf";
        vmaf_check(vmaf_model_load(&c.model, &model_cfg, model_name.c_str()), ("loading model " + model_name).c_str());
        vmaf_check(vmaf_use_features_from_model(c.vmaf, c.model), "vmaf_use_features_from_model");
    }

    auto planes_of = [&](AVFrame *f, int x0, int y0, Planes &p, Upload &up) {
        FrameView view = view_of(f, up, copies);
        if (f->format != AV_PIX_FMT_CUDA) {
            cu_check(cuEventRecord(copied, copies), "event");
            cu_check(cuStreamWaitEvent(stream, copied, 0), "event");
        }
        int xx = x0, yy = y0, pw = w, ph = h, depth = pd;
        void *a[] = {&view, &xx, &yy, &depth, &p.y16.ptr, &p.u.ptr, &p.v.ptr, &pw, &ph};
        Kernels::launch(k.frame_planes, w, h, a, stream);
    };
    auto to_vmaf = [&](Candidate &c, CUdeviceptr y16, VmafPicture *pic) {
        vmaf_check(vmaf_cuda_fetch_preallocated_picture(c.vmaf, pic), "fetching a picture");
        CUdeviceptr dst = (CUdeviceptr)pic->data[0];
        int pw = w, ph = h, pitch = (int)pic->stride[0], bytes = pd > 8 ? 2 : 1;
        void *a[] = {&y16, &pw, &ph, &dst, &pitch, &bytes};
        Kernels::launch(k.to_picture, w, h, a, stream);
    };
    auto sums = [&](CUfunction f, void **args) { cu_check(cuLaunchKernel(f, 256, 1, 1, 256, 1, 1, 0, stream, args, nullptr), "reduction"); };
    auto copy_back = [&](void *dst, const Buffer &src, size_t bytes) { cu_check(cuMemcpyDtoHAsync(dst, src.ptr, bytes, stream), "reading figures"); };

    // Queues the GPU work of frame `f`: the source first, then each encode, with each CAMBI on its own stream and
    // libvmaf on its own streams, and the figures copied to host memory at the end.
    auto queue = [&](long f, AVFrame *sf, const std::vector<AVFrame *> &cf) {
        const int at = (int)(f % 2), slot = (int)(f % kRing), cslot = (int)(f % (kRadius + 1));
        planes_of(sf, sx, sy, src, source_upload[at]);
        source_pts[at] = sf->best_effort_timestamp;
        source_cached[at] = cambi_cache.find(sf->best_effort_timestamp);
        if (!source_cached[at]) source_cambi[at]->launch(src.y16.ptr, w, stream);
        cu_check(cuMemsetD32Async(src_color.ptr, 0, 6, stream), "clearing");
        { size_t cn = n / 4; void *a[] = {&src.u.ptr, &src.v.ptr, &cn, &up10, &src_color.ptr}; sums(k.color_sums, a); }
        copy_back((double *)source_color.ptr + 3 * at, src_color, 3 * 8);
        const bool texture = f >= 2 * kRadius;
        const long centre = f - kRadius;
        int pw = w, ph = h;
        for (size_t ci = 0; ci < cands.size(); ci++) {
            Candidate &c = cands[ci];
            Planes &p = *c.planes;
            Results &r = *c.results[at];
            planes_of(cf[ci], c.x0, c.y0, p, c.upload[at]);
            c.cambi[at]->launch(p.y16.ptr, w, stream);
            VmafPicture ref, dist;
            to_vmaf(c, src.y16.ptr, &ref);
            to_vmaf(c, p.y16.ptr, &dist);
            CUdeviceptr ring_slot = ring.ptr + (size_t)slot * plane * 4, mid_d = c.mid[cslot]->ptr;
            void *a[] = {&src.y16.ptr, &p.y16.ptr, &up10, &pw, &ph, &ring_slot, &mid_d, &c.stats->ptr};
            cu_check(cuLaunchKernel(k.window_stats, ww, wh, 1, 256, 1, 1, 0, stream, a, nullptr), "window stats");
            cu_check(cuMemsetD32Async(c.sse->ptr, 0, 6, stream), "clearing");
            cu_check(cuMemsetD32Async(c.color->ptr, 0, 6, stream), "clearing");
            CUdeviceptr null = 0, sse = c.sse->ptr, s1 = sse + 8, s2 = sse + 16;
            size_t cn = n / 4;
            void *y[] = {&src.y16.ptr, &p.y16.ptr, &null, &null, (void *)&n, &sse};  // at the pipeline's bit depth, as libvmaf
            sums(k.plane_sse, y);
            void *u[] = {&src.u.ptr, &p.u.ptr, &null, &null, &cn, &s1};
            sums(k.plane_sse, u);
            void *v[] = {&src.v.ptr, &p.v.ptr, &null, &null, &cn, &s2};
            sums(k.plane_sse, v);
            void *col[] = {&p.u.ptr, &p.v.ptr, &cn, &up10, &c.color->ptr};
            sums(k.color_sums, col);
            copy_back(r.sse, *c.sse, 3 * 8);
            copy_back(r.color, *c.color, 3 * 8);
            copy_back(r.stats, *c.stats, (size_t)windows * kStats * 4);
            if (texture) {
                // lasting texture for the frame kRadius back, now that the frames after it are in
                int cring = (int)(centre % kRing);
                CUdeviceptr mid_c = c.mid[centre % (kRadius + 1)]->ptr;
                size_t pl = plane;
                void *pa[] = {&ring.ptr, &pl, &cring, &mid_c, &pw, &c.persist->ptr};
                cu_check(cuLaunchKernel(k.persist_stats, ww, wh, 1, 256, 1, 1, 0, stream, pa, nullptr), "texture stats");
                copy_back(r.persist, *c.persist, (size_t)windows * kPersist * 4);
            }
            // libvmaf starts on the pictures' own streams, which wait for the pictures to be written; it returns once
            // its work on the frame before is done
            cu_check(cuEventRecord(pictures, stream), "event");
            for (VmafPicture *pic : {&ref, &dist}) cu_check(cuStreamWaitEvent(vmaf_cuda_picture_get_stream(pic), pictures, 0), "event");
            vmaf_check(vmaf_read_pictures(c.vmaf, &ref, &dist, (unsigned)f), "vmaf_read_pictures");
        }
        cu_check(cuEventRecord(done[at], stream), "event");
    };

    // Adds up the figures of frame `f` once they are in host memory.
    auto collect = [&](long f) {
        const int at = (int)(f % 2), cslot = (int)(f % (kRadius + 1));
        const bool texture = f >= 2 * kRadius;
        const long centre = f - kRadius;
        cu_check(cuEventSynchronize(done[at]), "waiting for a frame");
        for (AVFrame *&fr : held[at]) av_frame_free(&fr);
        const double *scol = (const double *)source_color.ptr + 3 * at;
        const double source_banding = source_cached[at] ? *source_cached[at] : source_cambi[at]->finish();
        if (source_cached[at]) cambi_cached++;
        else cambi_cache.add(source_pts[at], source_banding);
        for (size_t ci = 0; ci < cands.size(); ci++) {
            Candidate &c = cands[ci];
            const Results &r = *c.results[at];
            const double banding = c.cambi[at]->finish();
            const float *st = r.stats;
            Frame fr;
            fr.cambi = banding; fr.cambi_source = source_banding;
            fr.psnr[0] = psnr(r.sse[0], n, pd); fr.psnr[1] = psnr(r.sse[1], n / 4, pd); fr.psnr[2] = psnr(r.sse[2], n / 4, pd);
            c.frames.push_back(fr);
            const size_t seg = (size_t)(f / per_segment);
            if (c.segments.size() <= seg) c.segments.resize(seg + 1);
            Accum &acc = c.segments[seg];
            acc.frames++;
            acc.cambi += banding; acc.cambi_source += source_banding;
            for (int i = 0; i < 3; i++) {
                acc.psnr[i] += fr.psnr[i];
                acc.color_r[i] += scol[i] / (double)(n / 4);
                acc.color_d[i] += r.color[i] / (double)(n / 4);
            }
            std::vector<float> oriented(windows);
            for (int i = 0; i < windows; i++) oriented[i] = st[(size_t)i * kStats + 6];
            const float q = quantile(oriented, 0.8);
            std::vector<uint8_t> &mask = c.mask_ring[cslot];
            for (int i = 0; i < windows; i++) {
                const float *s = &st[(size_t)i * kStats];
                const bool m = s[6] >= q && s[6] > 0;
                mask[i] = m;
                const double er[3] = {s[0], s[1], s[6]}, ed[3] = {s[2], s[3], s[7]};
                for (int key = 0; key < 3; key++) {
                    acc.all_r[key] += er[key]; acc.all_d[key] += ed[key];
                    if (m) {
                        acc.st_r[key] += er[key]; acc.st_d[key] += ed[key];
                        acc.amount[key].add(100.0 * std::sqrt(ed[key] / std::max(er[key], 1e-9)));
                    }
                }
                for (int key = 0; key < 2; key++) {
                    acc.sh_all_r[key] += er[key]; acc.sh_all_x[key] += s[4 + key];
                    if (m) {
                        acc.sh_st_r[key] += er[key]; acc.sh_st_x[key] += s[4 + key];
                        acc.matching[key].add(100.0 * s[4 + key] / std::max(er[key], 1e-9));
                    }
                }
            }
            if (texture) {
                Accum &cacc = c.segments[(size_t)(centre / per_segment)];
                const std::vector<uint8_t> &cmask = c.mask_ring[centre % (kRadius + 1)];
                for (int i = 0; i < windows; i++) {
                    const float *s = &r.persist[(size_t)i * kPersist];
                    const double agree = s[2] / std::sqrt(std::max((double)s[3] * s[4], 1e-9));
                    if (cmask[i] && agree > 0.5 && s[0] > 0) {
                        cacc.own += s[0]; cacc.mine += s[1];
                        cacc.lasting.add(100.0 * s[1] / s[0]);
                    }
                }
            }
        }
    };

    // A file decoding on the processor goes back to the GPU's decoder, from its next frame, when over 5 seconds the loop
    // waits for it for more than a tenth of the time and runs slower than all files decoding on the GPU would, as when
    // other programs come to load the processor.
    auto now = [] { return std::chrono::steady_clock::now(); };
    auto since = [](std::chrono::steady_clock::time_point t) { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count(); };
    std::vector<double> waited(files.size(), 0), last(files.size(), 0);  // seconds waited for each, and its last frame's time
    auto window = now();
    long window_frames = 0;
    auto fall_back = [&] {
        const double el = since(window);
        for (size_t i = 0; i < files.size(); i++)
            if (!on_gpu[i] && probes[i].gpu && waited[i] > 0.1 * el && window_frames / el * gpu_time < 1) {
                decoders[i] = std::make_unique<Decoder>(files[i], device, last[i] + 1 / fps, true);
                on_gpu[i] = true;
                std::fprintf(stderr, "%s fell behind on the processor and decodes on the GPU's decoder from %.3f s\n", files[i].c_str(), last[i] + 1 / fps);
            }
        std::fill(waited.begin(), waited.end(), 0);
        window = now();
        window_frames = 0;
    };

    auto t0 = std::chrono::steady_clock::now();
    long frame = 0;
    while (limit < 0 || frame < limit) {
        std::vector<AVFrame *> got;
        bool ended = false;
        for (size_t i = 0; i < files.size(); i++) {
            const auto t = now();
            got.push_back(decoders[i]->next());
            waited[i] += since(t);
            if (got.back()) last[i] = decoders[i]->seconds(got.back());
            ended |= !got.back();
        }
        if (ended) { for (AVFrame *f : got) av_frame_free(&f); break; }
        AVFrame *sf = got[0];
        std::vector<AVFrame *> cf(got.begin() + 1, got.end());
        queue(frame, sf, cf);
        // the frames are kept until their figures are in, since the GPU copies processor frames from them meanwhile
        held[frame % 2] = got;
        if (frame > 0) collect(frame - 1);
        frame++;
        if (++window_frames % 16 == 0 && since(window) >= 5) fall_back();
        if (frame % 2400 == 0) {
            double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            std::fprintf(stderr, "%.1f min scored, %.0f fps\n", frame / fps / 60, frame / el);
        }
    }
    if (frame > 0) collect(frame - 1);
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (!cambi_cache.file().empty())
        std::fprintf(stderr, "source CAMBI of %ld of %ld frames read from %s\n", cambi_cached, frame, cambi_cache.file().c_str());

    FILE *json = o.json.empty() ? nullptr : std::fopen(o.json.c_str(), "w");
    if (json) std::fprintf(json, "{\"source\": \"%s\", \"start\": %.6f, \"offset\": %.6f, \"frames\": %ld, \"fps\": %.6f, \"width\": %d, \"height\": %d, \"model\": \"%s\", \"segment_seconds\": %g, \"encodes\": [", o.source.c_str(), o.start, o.offset, frame, fps, w, h, model_name.c_str(), o.segment);
    std::printf("%ld frames (%.1f min) of %dx%d in %.0f s (%.0f fps), VMAF model %s\n", frame, frame / fps / 60, w, h, seconds, frame / seconds, model_name.c_str());
    for (size_t ci = 0; ci < cands.size(); ci++) {
        Candidate &c = cands[ci];
        vmaf_check(vmaf_read_pictures(c.vmaf, nullptr, nullptr, 0), "flushing libvmaf");
        for (long i = 0; i < frame; i++) {
            vmaf_check(vmaf_score_at_index(c.vmaf, c.model, &c.frames[i].vmaf, (unsigned)i), "reading a score");
            Accum &acc = c.segments[(size_t)(i / per_segment)];
            acc.vmaf += c.frames[i].vmaf;
            acc.vmaf_min = std::min(acc.vmaf_min, c.frames[i].vmaf);
        }
        Accum total;
        for (const Accum &a : c.segments) total.merge(a);
        auto line = [&](const char *label, const Accum &a) {
            Accum::Parts p = a.score();
            std::printf("  %-14s score %5.1f  fidelity %5.1f  texture %5.1f  banding %5.1f  grain %5.1f | VMAF %6.2f (worst frame %5.1f)  CAMBI %.2f vs %.2f  lasting texture median %3.0f%% worst tenth %4.0f%%  fine detail %3.0f%%\n",
                        label, p.score, p.fidelity, p.texture, p.banding, p.grain, a.vmaf / a.frames, a.vmaf_min, a.cambi / a.frames,
                        a.cambi_source / a.frames, a.lasting.quantile(0.5), a.lasting.quantile(0.1), Accum::kept(a.all_d[0], a.all_r[0]));
        };
        std::printf("\n%s\n", c.path.c_str());
        line("whole run", total);
        std::vector<size_t> order(c.segments.size());
        for (size_t i = 0; i < order.size(); i++) order[i] = i;
        std::sort(order.begin(), order.end(), [&](size_t a, size_t b) { return c.segments[a].score().score < c.segments[b].score().score; });
        std::printf("  worst segments:\n");
        for (size_t i = 0; i < std::min<size_t>(5, order.size()); i++) {
            char label[32];
            double t = o.start + order[i] * o.segment;
            std::snprintf(label, sizeof label, "%02d:%02d:%02d", (int)(t / 3600), (int)(t / 60) % 60, (int)t % 60);
            line(label, c.segments[order[i]]);
        }
        if (json) {
            // a figure that could not be measured (no lasting texture in a short run, say) is written as null
            auto num = [](double v, int digits) {
                if (!std::isfinite(v)) return std::string("null");
                char buf[64];
                std::snprintf(buf, sizeof buf, "%.*f", digits, v);
                return std::string(buf);
            };
            auto list = [&](std::initializer_list<double> vs, int digits) {
                std::string out = "[";
                for (double v : vs) out += (out.size() > 1 ? ", " : "") + num(v, digits);
                return out + "]";
            };
            auto parts = [&](const Accum &a) {
                Accum::Parts p = a.score();
                const double f = (double)a.frames;
                std::string out = "{\"frames\": " + std::to_string(a.frames) + ", \"score\": " + num(p.score, 4) + ", \"fidelity\": " + num(p.fidelity, 4) +
                    ", \"texture\": " + num(p.texture, 4) + ", \"banding\": " + num(p.banding, 4) + ", \"grain\": " + num(p.grain, 4) +
                    ", \"vmaf\": " + num(a.vmaf / f, 6) + ", \"vmaf_min\": " + num(a.vmaf_min, 6) +
                    ", \"psnr_y\": " + num(a.psnr[0] / f, 4) + ", \"psnr_cb\": " + num(a.psnr[1] / f, 4) + ", \"psnr_cr\": " + num(a.psnr[2] / f, 4) +
                    ", \"cambi\": " + num(a.cambi / f, 6) + ", \"cambi_source\": " + num(a.cambi_source / f, 6) +
                    ", \"lasting\": {\"windows\": " + std::to_string(a.lasting.n) + ", \"median\": " + num(a.lasting.quantile(0.5), 3) +
                    ", \"worst_tenth\": " + num(a.lasting.quantile(0.1), 3) + ", \"total\": " + num(a.own > 0 ? 100.0 * a.mine / a.own : NAN, 3) + "}" +
                    ", \"amount\": {\"all\": " + list({Accum::kept(a.all_d[0], a.all_r[0]), Accum::kept(a.all_d[1], a.all_r[1]), Accum::kept(a.all_d[2], a.all_r[2])}, 3) +
                    ", \"structure\": " + list({Accum::kept(a.st_d[0], a.st_r[0]), Accum::kept(a.st_d[1], a.st_r[1]), Accum::kept(a.st_d[2], a.st_r[2])}, 3) +
                    ", \"median\": " + list({a.amount[0].quantile(0.5), a.amount[1].quantile(0.5), a.amount[2].quantile(0.5)}, 3) +
                    ", \"worst_tenth\": " + list({a.amount[0].quantile(0.1), a.amount[1].quantile(0.1), a.amount[2].quantile(0.1)}, 3) + "}" +
                    ", \"matching\": {\"all\": " + list({100.0 * a.sh_all_x[0] / a.sh_all_r[0], 100.0 * a.sh_all_x[1] / a.sh_all_r[1]}, 3) +
                    ", \"structure\": " + list({100.0 * a.sh_st_x[0] / a.sh_st_r[0], 100.0 * a.sh_st_x[1] / a.sh_st_r[1]}, 3) +
                    ", \"median\": " + list({a.matching[0].quantile(0.5), a.matching[1].quantile(0.5)}, 3) +
                    ", \"worst_tenth\": " + list({a.matching[0].quantile(0.1), a.matching[1].quantile(0.1)}, 3) + "}" +
                    ", \"color_source\": " + list({a.color_r[0] / f, a.color_r[1] / f, a.color_r[2] / f}, 4) +
                    ", \"color_encode\": " + list({a.color_d[0] / f, a.color_d[1] / f, a.color_d[2] / f}, 4) + "}";
                std::fputs(out.c_str(), json);
            };
            std::fprintf(json, "%s{\"path\": \"%s\", \"whole\": ", ci ? ", " : "", c.path.c_str());
            parts(total);
            std::fprintf(json, ", \"segments\": [");
            for (size_t i = 0; i < c.segments.size(); i++) { if (i) std::fprintf(json, ", "); parts(c.segments[i]); }
            std::fprintf(json, "], \"per_frame\": {\"vmaf\": [");
            for (long i = 0; i < frame; i++) std::fprintf(json, "%s%.6f", i ? "," : "", c.frames[i].vmaf);
            std::fprintf(json, "], \"cambi\": [");
            for (long i = 0; i < frame; i++) std::fprintf(json, "%s%.6f", i ? "," : "", c.frames[i].cambi);
            std::fprintf(json, "], \"cambi_source\": [");
            for (long i = 0; i < frame; i++) std::fprintf(json, "%s%.6f", i ? "," : "", c.frames[i].cambi_source);
            std::fprintf(json, "], \"psnr_y\": [");
            for (long i = 0; i < frame; i++) std::fprintf(json, "%s%.3f", i ? "," : "", c.frames[i].psnr[0]);
            std::fprintf(json, "]}}");
        }
        vmaf_model_destroy(c.model);
        vmaf_close(c.vmaf);
    }
    if (json) { std::fprintf(json, "]}\n"); std::fclose(json); }
    av_buffer_unref(&device);
    return 0;
} catch (const std::exception &e) {
    std::fprintf(stderr, "encode-score: %s\n", e.what());
    return 1;
}
