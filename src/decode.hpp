// Hardware decoding of one video file into frames that stay in GPU memory, on a thread of its own.
#pragma once
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
}
#include <condition_variable>
#include <deque>
#include <mutex>
#include <string>
#include <thread>

class Decoder {
public:
    // Opens `path` and decodes its first video stream, on the GPU with `device` or else on the processor, dropping
    // frames before `start` seconds. GPU frames stay in GPU memory; processor frames are in host memory.
    Decoder(const std::string &path, AVBufferRef *device, double start, bool gpu = true);
    ~Decoder();
    // The next frame in decode order of display, or nullptr after the last one. The caller frees it.
    AVFrame *next();
    int width() const { return codec_->width; }
    int height() const { return codec_->height; }
    double frame_rate() const { return av_q2d(format_->streams[stream_]->avg_frame_rate); }
    // The presentation time of a frame this decoder returned, in seconds.
    double seconds(const AVFrame *f) const { return f->best_effort_timestamp * av_q2d(format_->streams[stream_]->time_base); }

private:
    void run();
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
    static constexpr size_t kQueue = 4;  // each queued frame holds a decode surface, and AV1 allows at most 32
};
