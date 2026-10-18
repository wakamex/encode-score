// CAMBI banding score of a 10-bit luma plane on the GPU, matching libvmaf's cambi feature with its default options.
// launch() queues the whole computation on the instance's own stream and finish() returns the score, so several
// instances (the source and each encode) run side by side.
#pragma once
#include "gpu.hpp"
#include <cstdint>

class Cambi {
public:
    // `depth` is the bit depth of the planes it will be given, 8 or 10.
    Cambi(const Kernels &k, int w, int h, int depth);
    ~Cambi();
    Cambi(const Cambi &) = delete;
    // Starts scoring the plane at `img` (10-bit values, `pitch` in pixels) once `ready` has fired.
    void launch(CUdeviceptr img, int pitch, CUevent ready);
    // The score, clamped to 1000 as libvmaf does.
    double finish();

private:
    static constexpr int kScales = 5;
    double topk_exact(int scale);
    CUfunction from8_, derivative_, mask_kernel_, decimate16_, decimate8_, mode_, rows_, rows_small_, packed_, radix_, sum_above_;
    CUstream stream_ = nullptr;
    int w_, h_, depth_, window_, mask_index_;
    int sw_[kScales], sh_[kScales];
    struct Params { int w, h, pad, band_base, band_size, vlt; int tvi[4]; int weight[4]; } params_;
    CUdeviceptr image_[2] = {}, mask_[2] = {}, deriv_ = 0, cvalues_[kScales] = {}, hist_ = 0, sums_ = 0;
    struct Sums { double sum[kScales]; unsigned long long count[kScales]; } *host_ = nullptr;
};
