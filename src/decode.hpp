// Decoding of one video file on a thread of its own, on the GPU's video decoder into GPU memory or on the processor into
// page-locked host memory, which the GPU copies from without the host waiting.
#pragma once
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
}
#include <cuda.h>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <string>
#include <thread>

class Decoder {
public:
    // Opens `path` and decodes its first video stream, on the GPU with `device` or else on the processor, dropping
    // frames before `start` seconds.
    Decoder(const std::string &path, AVBufferRef *device, double start, bool gpu = true);
    ~Decoder();
    // The next frame in decode order of display, or nullptr after the last one. The caller frees it.
    AVFrame *next();
    int width() const { return codec_->width; }
    int height() const { return codec_->height; }
    double frame_rate() const { return av_q2d(format_->streams[stream_]->avg_frame_rate); }
    // The file's length in seconds, or 0 when it does not say.
    double duration() const { return format_->duration > 0 ? (double)format_->duration / AV_TIME_BASE : 0; }
    // The presentation time of a frame this decoder returned, in seconds.
    double seconds(const AVFrame *f) const { return f->best_effort_timestamp * av_q2d(format_->streams[stream_]->time_base); }

private:
    void run();
    static int pinned_buffer(AVCodecContext *c, AVFrame *f, int flags);
    static AVBufferRef *alloc_pinned(void *opaque, size_t size);
    AVFormatContext *format_ = nullptr;
    AVCodecContext *codec_ = nullptr;
    int stream_ = -1;
    double start_;
    bool gpu_;
    std::deque<AVFrame *> queue_;
    bool done_ = false, stop_ = false;
    std::string error_;
    std::mutex lock_;
    std::condition_variable changed_;
    std::thread thread_;
    CUcontext ctx_;
    std::mutex pool_lock_;
    AVBufferPool *pool_ = nullptr;  // page-locked buffers of pool_size_ bytes for processor frames
    size_t pool_size_ = 0;
    // each frame queued or held by the caller holds a decode surface, and AV1 allows at most 32; the caller holds two
    static constexpr size_t kQueue = 4, kHeld = 2;
};
