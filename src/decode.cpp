#include "decode.hpp"
extern "C" {
#include <libavutil/hwcontext_cuda.h>
#include <libavutil/imgutils.h>
}
#include <stdexcept>

static AVPixelFormat pick_cuda(AVCodecContext *, const AVPixelFormat *formats)
{
    for (const AVPixelFormat *f = formats; *f != AV_PIX_FMT_NONE; f++)
        if (*f == AV_PIX_FMT_CUDA) return *f;
    return AV_PIX_FMT_NONE;  // no hardware path: fail loudly rather than decode on the processor
}

// The decoder for `id` that can decode into CUDA frames; software-only decoders such as libdav1d are skipped.
static const AVCodec *cuda_decoder(AVCodecID id)
{
    void *it = nullptr;
    while (const AVCodec *c = av_codec_iterate(&it)) {
        if (c->id != id || !av_codec_is_decoder(c)) continue;
        for (int i = 0; const AVCodecHWConfig *h = avcodec_get_hw_config(c, i); i++)
            if (h->device_type == AV_HWDEVICE_TYPE_CUDA && (h->methods & AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX)) return c;
    }
    return nullptr;
}

static void av_check(int err, const std::string &what)
{
    if (err >= 0) return;
    char msg[256];
    av_strerror(err, msg, sizeof msg);
    throw std::runtime_error(what + ": " + msg);
}

static void free_pinned(void *ctx, uint8_t *data)
{
    cuCtxPushCurrent((CUcontext)ctx);
    cuMemFreeHost(data);
    cuCtxPopCurrent(nullptr);
}

AVBufferRef *Decoder::alloc_pinned(void *opaque, size_t size)
{
    CUcontext ctx = ((Decoder *)opaque)->ctx_;
    void *p = nullptr;
    cuCtxPushCurrent(ctx);
    const CUresult r = cuMemHostAlloc(&p, size, CU_MEMHOSTALLOC_PORTABLE);
    cuCtxPopCurrent(nullptr);
    if (r != CUDA_SUCCESS) return nullptr;
    AVBufferRef *b = av_buffer_create((uint8_t *)p, size, free_pinned, ctx, 0);
    if (!b) free_pinned(ctx, (uint8_t *)p);
    return b;
}

// A frame's buffer from page-locked memory, laid out as FFmpeg's default get_buffer2 lays it out: dimensions and line
// sizes aligned as the decoder needs, and padding past the end. All planes share one buffer.
int Decoder::pinned_buffer(AVCodecContext *c, AVFrame *f, int flags)
{
    Decoder *d = (Decoder *)c->opaque;
    const AVPixelFormat fmt = (AVPixelFormat)f->format;
    int w = f->width, h = f->height, align[AV_NUM_DATA_POINTERS], linesize[4];
    avcodec_align_dimensions2(c, &w, &h, align);
    for (;;) {
        if (av_image_fill_linesizes(linesize, fmt, w) < 0) return avcodec_default_get_buffer2(c, f, flags);
        bool aligned = true;
        for (int i = 0; i < 4; i++) aligned &= linesize[i] % align[i] == 0;
        if (aligned) break;
        w += w & ~(w - 1);
    }
    size_t sizes[4];
    const ptrdiff_t ls[4] = {linesize[0], linesize[1], linesize[2], linesize[3]};
    if (av_image_fill_plane_sizes(sizes, fmt, h, ls) < 0) return avcodec_default_get_buffer2(c, f, flags);
    const size_t total = sizes[0] + sizes[1] + sizes[2] + sizes[3] + 16 + 64 - 1;
    {
        std::lock_guard<std::mutex> g(d->pool_lock_);
        if (total != d->pool_size_) {
            av_buffer_pool_uninit(&d->pool_);  // its buffers are freed as the frames using them are
            d->pool_ = av_buffer_pool_init2(total, d, alloc_pinned, nullptr);
            d->pool_size_ = total;
        }
        f->buf[0] = d->pool_ ? av_buffer_pool_get(d->pool_) : nullptr;
    }
    if (!f->buf[0]) return avcodec_default_get_buffer2(c, f, flags);
    uint8_t *p = f->buf[0]->data;
    for (int i = 0; i < 4 && sizes[i]; i++) {
        f->data[i] = p;
        f->linesize[i] = linesize[i];
        p += sizes[i];
    }
    f->extended_data = f->data;
    return 0;
}

Decoder::Decoder(const std::string &path, AVBufferRef *device, double start, bool gpu) : start_(start), gpu_(gpu)
{
    ctx_ = ((AVCUDADeviceContext *)((AVHWDeviceContext *)device->data)->hwctx)->cuda_ctx;
    av_check(avformat_open_input(&format_, path.c_str(), nullptr, nullptr), "opening " + path);
    av_check(avformat_find_stream_info(format_, nullptr), "reading " + path);
    stream_ = av_find_best_stream(format_, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
    av_check(stream_, "finding the video stream of " + path);
    const AVCodecID id = format_->streams[stream_]->codecpar->codec_id;
    const AVCodec *codec = gpu ? cuda_decoder(id) : avcodec_find_decoder(id);
    if (!codec) throw std::runtime_error(std::string("no ") + (gpu ? "GPU " : "") + "decoder for the video of " + path);
    codec_ = avcodec_alloc_context3(codec);
    av_check(avcodec_parameters_to_context(codec_, format_->streams[stream_]->codecpar), "decoder setup");
    if (gpu) {
        codec_->hw_device_ctx = av_buffer_ref(device);
        codec_->get_format = pick_cuda;
        codec_->extra_hw_frames = kQueue + kHeld + 1;
    } else {
        codec_->thread_count = 0;  // as many threads as the processor has
        codec_->opaque = this;
        codec_->get_buffer2 = pinned_buffer;
    }
    av_check(avcodec_open2(codec_, codec, nullptr), "opening the decoder for " + path);
    for (int i = 0; i < (int)format_->nb_streams; i++)
        if (i != stream_) format_->streams[i]->discard = AVDISCARD_ALL;
    if (start > 0) {
        AVStream *s = format_->streams[stream_];
        av_check(av_seek_frame(format_, stream_, (int64_t)(start / av_q2d(s->time_base)), AVSEEK_FLAG_BACKWARD), "seeking " + path);
    }
    thread_ = std::thread(&Decoder::run, this);
}

Decoder::~Decoder()
{
    { std::lock_guard<std::mutex> g(lock_); stop_ = true; }
    changed_.notify_all();
    if (thread_.joinable()) thread_.join();
    for (AVFrame *f : queue_) av_frame_free(&f);
    avcodec_free_context(&codec_);
    avformat_close_input(&format_);
    av_buffer_pool_uninit(&pool_);
}

void Decoder::run()
{
    AVPacket *packet = av_packet_alloc();
    AVFrame *frame = av_frame_alloc();
    const double tb = av_q2d(format_->streams[stream_]->time_base), half = 0.5 / frame_rate();
    try {
        bool flushing = false;
        while (true) {
            int err = avcodec_receive_frame(codec_, frame);
            if (err == AVERROR(EAGAIN)) {
                if (flushing) break;
                int r = av_read_frame(format_, packet);
                if (r == AVERROR_EOF) { avcodec_send_packet(codec_, nullptr); flushing = true; continue; }
                av_check(r, "reading a packet");
                if (packet->stream_index == stream_) av_check(avcodec_send_packet(codec_, packet), "decoding");
                av_packet_unref(packet);
                continue;
            }
            if (err == AVERROR_EOF) break;
            av_check(err, "decoding a frame");
            if (gpu_ && frame->format != AV_PIX_FMT_CUDA) throw std::runtime_error("the decoder returned a frame outside GPU memory");
            if (frame->best_effort_timestamp != AV_NOPTS_VALUE && frame->best_effort_timestamp * tb < start_ - half) {
                av_frame_unref(frame);
                continue;
            }
            std::unique_lock<std::mutex> g(lock_);
            changed_.wait(g, [&] { return queue_.size() < kQueue || stop_; });
            if (stop_) break;
            queue_.push_back(av_frame_clone(frame));
            av_frame_unref(frame);
            changed_.notify_all();
        }
    } catch (const std::exception &e) {
        std::lock_guard<std::mutex> g(lock_);
        error_ = e.what();
    }
    av_frame_free(&frame);
    av_packet_free(&packet);
    std::lock_guard<std::mutex> g(lock_);
    done_ = true;
    changed_.notify_all();
}

AVFrame *Decoder::next()
{
    std::unique_lock<std::mutex> g(lock_);
    changed_.wait(g, [&] { return !queue_.empty() || done_; });
    if (!error_.empty()) throw std::runtime_error(error_);
    if (queue_.empty()) return nullptr;
    AVFrame *f = queue_.front();
    queue_.pop_front();
    changed_.notify_all();
    return f;
}
