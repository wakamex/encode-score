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
}
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
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

// The pictures of one frame of one file: luma as float and as 10-bit integers, chroma, and the box-filtered luma.
struct Planes {
    Buffer y, y16, u, v, b3, b9;
    Planes(size_t n) : y(n * 4), y16(n * 2), u(n / 2), v(n / 2), b3(n * 4), b9(n * 4) {}
};

// A decoded frame as the kernels read it (kernels.cu's Frame).
struct FrameView {
    const void *y, *u, *v;
    int ypitch, cpitch, semi, bytes, shift, depth;
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
// their planes to the GPU. Only 4:2:0 at 8 or 10 bits is supported.
static FrameView view_of(const AVFrame *f, Upload &up)
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
    cu_check(cuMemcpyHtoD(base, f->data[0], ysize), "uploading a frame");
    cu_check(cuMemcpyHtoD(base + ysize, f->data[1], csize), "uploading a frame");
    if (!semi) cu_check(cuMemcpyHtoD(base + ysize + csize, f->data[2], csize), "uploading a frame");
    return {(const void *)base, (const void *)(base + ysize), semi ? nullptr : (const void *)(base + ysize + csize),
            f->linesize[0], f->linesize[1], semi, bytes, shift, depth};
}

// Whether `path` decodes on the GPU, and the bit depth of its frames.
static std::pair<bool, int> probe(const std::string &path, AVBufferRef *device)
{
    for (bool gpu : {true, false}) {
        try {
            Decoder d(path, device, 0, gpu);
            AVFrame *f = d.next();
            if (!f) throw std::runtime_error(path + " has no frames");
            Upload up;
            const int depth = view_of(f, up).depth;
            av_frame_free(&f);
            return {gpu, depth};
        } catch (const std::exception &) {
            if (!gpu) throw;
        }
    }
    return {false, 0};
}

struct Kernel {
    CUfunction frame_planes, to_picture, box_h, box_v, window_stats, persist_stats, plane_sse, color_sums;
    explicit Kernel(const Kernels &k)
        : frame_planes(k.get("frame_planes")), to_picture(k.get("to_picture")), box_h(k.get("box_h")), box_v(k.get("box_v")),
          window_stats(k.get("window_stats")), persist_stats(k.get("persist_stats")), plane_sse(k.get("plane_sse")),
          color_sums(k.get("color_sums")) {}
};

struct Frame {
    double vmaf = 0, cambi = 0, cambi_source = 0, psnr[3] = {};
};

static constexpr int kRing = 13, kRadius = 6, kStats = 8, kPersist = 5;

// Wall time per stage when ESCORE_PROFILE is set; each mark waits for the GPU so the time lands on the right stage.
struct Profile {
    bool on = getenv("ESCORE_PROFILE") != nullptr;
    std::vector<std::pair<std::string, double>> stages;
    std::chrono::steady_clock::time_point last = std::chrono::steady_clock::now();
    void mark(const char *stage)
    {
        if (!on) return;
        cuCtxSynchronize();
        auto now = std::chrono::steady_clock::now();
        double dt = std::chrono::duration<double>(now - last).count();
        last = now;
        for (auto &s : stages) if (s.first == stage) { s.second += dt; return; }
        stages.push_back({stage, dt});
    }
    void print() const
    {
        if (!on) return;
        std::fprintf(stderr, "stage seconds:");
        for (auto &s : stages) std::fprintf(stderr, " %s %.2f", s.first.c_str(), s.second);
        std::fprintf(stderr, "\n");
    }
};

struct Candidate {
    std::string path;
    std::unique_ptr<Decoder> decoder;
    int x0 = 0, y0 = 0;
    std::unique_ptr<Planes> planes;
    std::vector<std::unique_ptr<Buffer>> mid;  // the encode's mid detail planes, by frame % (kRadius + 1)
    std::unique_ptr<Buffer> stats, persist, sse, color;
    std::unique_ptr<Cambi> cambi;
    Upload upload;
    CUevent ready = nullptr;
    VmafContext *vmaf = nullptr;
    VmafModel *model = nullptr;
    std::vector<Frame> frames;
    std::vector<Accum> segments;
    std::vector<std::vector<float>> window_ring;  // window stats of the last kRadius + 1 frames, for the texture measure
    std::vector<std::vector<uint8_t>> mask_ring;
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

// Thumbnails (64 by 36 luma means of the cropped picture) and times of `count` frames of `path` from `start` seconds.
static std::vector<std::vector<float>> thumbnails(const std::string &path, bool gpu, AVBufferRef *device, CUfunction tiny,
                                                  double start, int count, int w, int h, std::vector<double> &times)
{
    Decoder d(path, device, std::max(start, 0.0), gpu);
    Upload up;
    const int x0 = (d.width() - w) / 2 & ~1, y0 = (d.height() - h) / 2 & ~1;
    CUdeviceptr out;
    cu_check(cuMemAlloc(&out, 64 * 36 * 4), "thumbnails");
    std::vector<std::vector<float>> result;
    while ((int)result.size() < count) {
        AVFrame *f = d.next();
        if (!f) break;
        FrameView v = view_of(f, up);
        int xx = x0, yy = y0, ww = w, hh = h;
        void *a[] = {&v, &xx, &yy, &ww, &hh, &out};
        cu_check(cuLaunchKernel(tiny, 64, 36, 1, 32, 8, 1, 0, nullptr, a, nullptr), "thumbnails");
        result.emplace_back(64 * 36);
        cu_check(cuMemcpyDtoH(result.back().data(), out, 64 * 36 * 4), "thumbnails");
        times.push_back(d.seconds(f));
        av_frame_free(&f);
    }
    cuMemFree(out);
    return result;
}

// The offset to add to the encode's times so its frames match the source's: the shift, to the frame, at which 8 seconds
// of the encode's thumbnails best match the source's within `search` seconds either side of `guess`.
static double find_offset(const std::string &source, bool source_gpu, const std::string &encode, bool encode_gpu,
                          AVBufferRef *device, CUfunction tiny, double start, double guess, double search, int w, int h, double fps)
{
    std::vector<double> ptimes, rtimes;
    const int probe_count = (int)(8 * fps);
    auto probe = thumbnails(encode, encode_gpu, device, tiny, start, probe_count, w, h, ptimes);
    const double ref_start = std::max(start + guess - search, 0.0);
    auto ref = thumbnails(source, source_gpu, device, tiny, ref_start, (int)((2 * search + 8) * fps) + 2, w, h, rtimes);
    if (probe.size() < 24 || ref.size() < probe.size()) throw std::runtime_error("too few frames to find the offset");
    double best = 1e30;
    size_t lag = 0;
    for (size_t l = 0; l + probe.size() <= ref.size(); l++) {
        double total = 0;
        for (size_t i = 0; i < probe.size() && total < best * probe.size() * 2304; i++)
            for (int p = 0; p < 64 * 36; p++) total += std::fabs(ref[l + i][p] - probe[i][p]);
        const double mean = total / (probe.size() * 2304.0);
        if (mean < best) { best = mean; lag = l; }
    }
    if (best > 16.0)  // mean thumbnail difference in 10-bit code values; matching pictures are well under 5
        throw std::runtime_error(encode + " does not match the source within " + std::to_string((int)search) +
                                 " s of offset " + std::to_string(guess) + " (closest mean difference " + std::to_string(best) + "); pass --offset");
    return rtimes[lag] - ptimes[0];
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

    // which files decode on the GPU, their bit depths, and the picture area they share
    int w, h, pd = 8;
    double fps;
    const auto [source_gpu, source_depth] = probe(o.source, device);
    std::vector<bool> encode_gpu;
    pd = std::max(pd, source_depth);
    {
        Decoder d(o.source, device, 0, source_gpu);
        w = d.width(); h = d.height(); fps = d.frame_rate();
        for (const std::string &e : o.encodes) {
            const auto [gpu, depth] = probe(e, device);
            encode_gpu.push_back(gpu);
            pd = std::max(pd, depth);
            Decoder c(e, device, 0, gpu);
            w = std::min(w, c.width());
            h = std::min(h, c.height());
        }
    }
    w &= ~1; h &= ~1;
    for (size_t i = 0; i <= o.encodes.size(); i++)
        if (!(i == 0 ? source_gpu : (bool)encode_gpu[i - 1]))
            std::fprintf(stderr, "%s decodes on the processor (no GPU decoder for its format)\n", (i == 0 ? o.source : o.encodes[i - 1]).c_str());
    if (o.find_offset) {
        const CUfunction tiny = kernels.get("tiny_luma");
        std::vector<double> found;
        for (size_t i = 0; i < o.encodes.size(); i++)
            found.push_back(find_offset(o.source, source_gpu, o.encodes[i], encode_gpu[i], device, tiny, o.start, o.offset, 10.0, w, h, fps));
        for (double f : found)
            if (std::fabs(f - found[0]) > 0.5 / fps) throw std::runtime_error("the encodes start at different points of the source; score them separately");
        o.offset = found[0];
        std::printf("offset of the source against the encodes: %+.3f s\n", o.offset);
    }
    Decoder source(o.source, device, o.start + o.offset, source_gpu);
    std::vector<Candidate> cands(o.encodes.size());
    for (size_t c = 0; c < cands.size(); c++) {
        cands[c].path = o.encodes[c];
        cands[c].decoder = std::make_unique<Decoder>(o.encodes[c], device, o.start, encode_gpu[c]);
    }
    const int sx = (source.width() - w) / 2 & ~1, sy = (source.height() - h) / 2 & ~1;
    const long limit = o.seconds > 0 ? (long)std::llround(o.seconds * fps) : -1;
    const long per_segment = std::max(1L, (long)std::llround(o.segment * fps));
    const std::string model_name = o.model != "auto" ? o.model : (w >= 3840 ? "vmaf_4k_v0.6.1" : "vmaf_v0.6.1");
    const size_t n = (size_t)w * h, plane = n;
    const int ww = w / 16, wh = h / 16, windows = ww * wh;

    Planes src(n);
    Upload source_upload;
    int up10 = 10 - pd;
    Buffer h3(n * 4), h9(n * 4), src_color(3 * 8);
    Buffer ring(n * 4 * kRing);  // the source's mid detail planes, by frame % kRing
    Cambi source_cambi(kernels, w, h, pd);
    CambiCache cambi_cache(o.source, sx, sy, w, h, pd);
    long cambi_cached = 0;
    CUevent source_ready;
    cu_check(cuEventCreate(&source_ready, CU_EVENT_DISABLE_TIMING), "creating an event");

    for (Candidate &c : cands) {
        c.x0 = (c.decoder->width() - w) / 2 & ~1;
        c.y0 = (c.decoder->height() - h) / 2 & ~1;
        c.planes = std::make_unique<Planes>(n);
        for (int i = 0; i <= kRadius; i++) c.mid.push_back(std::make_unique<Buffer>(n * 4));
        c.stats = std::make_unique<Buffer>((size_t)windows * kStats * 4);
        c.persist = std::make_unique<Buffer>((size_t)windows * kPersist * 4);
        c.sse = std::make_unique<Buffer>(3 * 8);
        c.color = std::make_unique<Buffer>(3 * 8);
        c.cambi = std::make_unique<Cambi>(kernels, w, h, pd);
        cu_check(cuEventCreate(&c.ready, CU_EVENT_DISABLE_TIMING), "creating an event");
        c.window_ring.assign(kRadius + 1, std::vector<float>((size_t)windows * kStats));
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
        FrameView view = view_of(f, up);
        int xx = x0, yy = y0, pw = w, ph = h, depth = pd;
        void *a[] = {&view, &xx, &yy, &depth, &p.y.ptr, &p.y16.ptr, &p.u.ptr, &p.v.ptr, &pw, &ph};
        Kernels::launch(k.frame_planes, w, h, a);
        void *c[] = {&p.y.ptr, &h3.ptr, &h9.ptr, &pw, &ph};
        Kernels::launch(k.box_h, w, h, c);
        void *d[] = {&h3.ptr, &h9.ptr, &p.b3.ptr, &p.b9.ptr, &pw, &ph};
        Kernels::launch(k.box_v, w, h, d);
    };
    auto to_vmaf = [&](Candidate &c, CUdeviceptr y16, VmafPicture *pic) {
        vmaf_check(vmaf_cuda_fetch_preallocated_picture(c.vmaf, pic), "fetching a picture");
        CUdeviceptr dst = (CUdeviceptr)pic->data[0];
        int pw = w, ph = h, pitch = (int)pic->stride[0], bytes = pd > 8 ? 2 : 1;
        void *a[] = {&y16, &pw, &ph, &dst, &pitch, &bytes};
        Kernels::launch(k.to_picture, w, h, a);
    };
    auto sums = [&](CUfunction f, void **args) { cu_check(cuLaunchKernel(f, 256, 1, 1, 256, 1, 1, 0, nullptr, args, nullptr), "reduction"); };

    std::vector<float> host_persist((size_t)windows * kPersist);
    Profile prof;
    auto t0 = std::chrono::steady_clock::now();
    long frame = 0;
    while (limit < 0 || frame < limit) {
        AVFrame *sf = source.next();
        std::vector<AVFrame *> cf;
        bool ended = !sf;
        for (Candidate &c : cands) { cf.push_back(c.decoder->next()); ended |= !cf.back(); }
        if (ended) { av_frame_free(&sf); for (AVFrame *f : cf) av_frame_free(&f); break; }
        prof.mark("decode");
        // queue the GPU work for this frame: the source first, then each encode, with each CAMBI on its own stream
        planes_of(sf, sx, sy, src, source_upload);
        const std::optional<double> cached = cambi_cache.find(sf->best_effort_timestamp);
        if (!cached) {
            cu_check(cuEventRecord(source_ready, nullptr), "event");
            source_cambi.launch(src.y16.ptr, w, source_ready);
        }
        cu_check(cuMemsetD32(src_color.ptr, 0, 6), "clearing");
        { size_t cn = n / 4; void *a[] = {&src.u.ptr, &src.v.ptr, &cn, &up10, &src_color.ptr}; sums(k.color_sums, a); }
        const int slot = (int)(frame % kRing), cslot = (int)(frame % (kRadius + 1));
        const bool texture = frame >= 2 * kRadius;
        const long centre = frame - kRadius;
        int pw = w, ph = h;
        for (size_t ci = 0; ci < cands.size(); ci++) {
            Candidate &c = cands[ci];
            Planes &p = *c.planes;
            planes_of(cf[ci], c.x0, c.y0, p, c.upload);
            cu_check(cuEventRecord(c.ready, nullptr), "event");
            c.cambi->launch(p.y16.ptr, w, c.ready);
            VmafPicture ref, dist;
            to_vmaf(c, src.y16.ptr, &ref);
            to_vmaf(c, p.y16.ptr, &dist);
            CUdeviceptr ring_slot = ring.ptr + (size_t)slot * plane * 4, mid_d = c.mid[cslot]->ptr;
            void *a[] = {&src.y.ptr, &src.b3.ptr, &src.b9.ptr, &p.y.ptr, &p.b3.ptr, &p.b9.ptr, &pw, &ph, &ring_slot, &mid_d, &c.stats->ptr};
            cu_check(cuLaunchKernel(k.window_stats, ww, wh, 1, 256, 1, 1, 0, nullptr, a, nullptr), "window stats");
            cu_check(cuMemsetD32(c.sse->ptr, 0, 6), "clearing");
            cu_check(cuMemsetD32(c.color->ptr, 0, 6), "clearing");
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
            if (texture) {
                // lasting texture for the frame kRadius back, now that the frames after it are in
                int cring = (int)(centre % kRing);
                CUdeviceptr mid_c = c.mid[centre % (kRadius + 1)]->ptr;
                size_t pl = plane;
                void *pa[] = {&ring.ptr, &pl, &cring, &mid_c, &pw, &c.persist->ptr};
                cu_check(cuLaunchKernel(k.persist_stats, ww, wh, 1, 256, 1, 1, 0, nullptr, pa, nullptr), "texture stats");
            }
            // libvmaf's CUDA path gives NaN scores for some frames when its work overlaps other GPU work in the same
            // context (seen on generated test video, and absent with blocking launches), so it gets the GPU to itself
            cu_check(cuCtxSynchronize(), "waiting before libvmaf");
            vmaf_check(vmaf_read_pictures(c.vmaf, &ref, &dist, (unsigned)frame), "vmaf_read_pictures");
            cu_check(cuCtxSynchronize(), "waiting for libvmaf");
        }
        prof.mark("queue");
        double scol[3];
        cu_check(cuMemcpyDtoH(scol, src_color.ptr, sizeof scol), "reading color");
        const double source_banding = cached ? *cached : source_cambi.finish();
        if (cached) cambi_cached++;
        else cambi_cache.add(sf->best_effort_timestamp, source_banding);
        prof.mark("gpu");

        // bookkeeping on the host
        for (size_t ci = 0; ci < cands.size(); ci++) {
            Candidate &c = cands[ci];
            const double banding = c.cambi->finish();
            std::vector<float> &st = c.window_ring[cslot];
            cu_check(cuMemcpyDtoH(st.data(), c.stats->ptr, st.size() * 4), "reading window stats");
            unsigned long long sse[3];
            double col[3];
            cu_check(cuMemcpyDtoH(sse, c.sse->ptr, sizeof sse), "reading PSNR");
            cu_check(cuMemcpyDtoH(col, c.color->ptr, sizeof col), "reading color");
            Frame fr;
            fr.cambi = banding; fr.cambi_source = source_banding;
            fr.psnr[0] = psnr(sse[0], n, pd); fr.psnr[1] = psnr(sse[1], n / 4, pd); fr.psnr[2] = psnr(sse[2], n / 4, pd);
            c.frames.push_back(fr);
            const size_t seg = (size_t)(frame / per_segment);
            if (c.segments.size() <= seg) c.segments.resize(seg + 1);
            Accum &acc = c.segments[seg];
            acc.frames++;
            acc.cambi += banding; acc.cambi_source += source_banding;
            for (int i = 0; i < 3; i++) {
                acc.psnr[i] += fr.psnr[i];
                acc.color_r[i] += scol[i] / (double)(n / 4);
                acc.color_d[i] += col[i] / (double)(n / 4);
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
                cu_check(cuMemcpyDtoH(host_persist.data(), c.persist->ptr, host_persist.size() * 4), "reading texture stats");
                Accum &cacc = c.segments[(size_t)(centre / per_segment)];
                const std::vector<uint8_t> &cmask = c.mask_ring[centre % (kRadius + 1)];
                for (int i = 0; i < windows; i++) {
                    const float *s = &host_persist[(size_t)i * kPersist];
                    const double agree = s[2] / std::sqrt(std::max((double)s[3] * s[4], 1e-9));
                    if (cmask[i] && agree > 0.5 && s[0] > 0) {
                        cacc.own += s[0]; cacc.mine += s[1];
                        cacc.lasting.add(100.0 * s[1] / s[0]);
                    }
                }
            }
        }
        prof.mark("host");
        av_frame_free(&sf);
        for (AVFrame *f : cf) av_frame_free(&f);
        frame++;
        if (frame % 2400 == 0) {
            double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            std::fprintf(stderr, "%.1f min scored, %.0f fps\n", frame / fps / 60, frame / el);
        }
    }
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    prof.print();
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
