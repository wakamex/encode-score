// CAMBI banding score of a 10-bit luma plane on the GPU, matching libvmaf's cambi feature with its default options.
// launch() queues the whole computation on the instance's own stream and finish() returns the score, so several
// instances (the source and each encode, and consecutive frames) run side by side. Each launch() is followed by its
// finish() before the instance's next launch().
#pragma once
#include "gpu.hpp"
#include <cstdint>

class Cambi {
public:
    // `depth` is the bit depth of the planes it will be given, 8 or 10.
    Cambi(const Kernels &k, int w, int h, int depth);
    ~Cambi();
    Cambi(const Cambi &) = delete;
    // Starts scoring the plane at `img` (10-bit values, `pitch` in pixels). The plane is copied on `stream`, after the
    // work already queued there, and the rest runs on the instance's own stream, so the plane can be overwritten by
    // work queued on `stream` afterwards.
    void launch(CUdeviceptr img, int pitch, CUstream stream);
    // The score, clamped to 1000 as libvmaf does.
    double finish();

private:
    static constexpr int kScales = 5;
    double topk_exact(int scale);
    CUfunction from8_, mask_kernel_, decimate16_, decimate8_, mode_, cvalues_, radix_, sum_above_;
    CUstream stream_ = nullptr;
    CUevent copied_ = nullptr;
    int w_, h_, depth_, window_, mask_index_, blocks_ = 0;
    unsigned shared_;
    int sw_[kScales], sh_[kScales], first_[kScales + 1];
    struct Params { int w, h, pad, band_base, band_size, vlt; int tvi[4]; int weight[4]; } params_;
    // the picture and mask of each scale, scratch for one picture, and the c-values of all scales one after another
    CUdeviceptr image_[kScales] = {}, mask_[kScales] = {}, scratch_ = 0, cvalues_all_ = 0, hist_ = 0, sums_ = 0;
    size_t cvalues_at_[kScales + 1];  // where each scale's c-values start, and their total
    struct Sums { double sum[kScales]; unsigned long long count[kScales]; } *host_ = nullptr;
};
