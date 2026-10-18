#include "cambi.hpp"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>

// CAMBI's luminance tables come from libvmaf itself, so the thresholds are the same ones its CPU code uses.
extern "C" {
typedef struct { int foot, head; } VmafLumaRange;
typedef double (*VmafEOTF)(double);
int vmaf_luminance_init_luma_range(VmafLumaRange *range, int bitdepth, int pixel_range);
int vmaf_luminance_init_eotf(VmafEOTF *eotf, const char *name);
int get_tvi_for_diff(int diff, double tvi_threshold, int bitdepth, VmafLumaRange range, VmafEOTF eotf);
int get_vlt_luma(double threshold, VmafLumaRange range, VmafEOTF eotf);
}

static constexpr int kScaleWeights[5] = {16, 8, 4, 2, 1}, kTileW = 64, kTileH = 32;  // kernels.cu's CV_TX and CV_TY
static constexpr double kTopk = 0.6, kTvi = 0.019, kMax = 1000.0;

static int ceil_log2(uint32_t n)
{
    int s = 0;
    for (uint32_t t = n ? n - 1 : 0; t; t >>= 1) s++;
    return s;
}

// cambi.c's spatial_pooling keeps clip(topk * n, 1, n) values, with the product truncated to an int
static size_t topk_count(size_t n) { return std::clamp((size_t)(int)(kTopk * (double)n), (size_t)1, n); }

Cambi::Cambi(const Kernels &k, int w, int h, int depth) : w_(w), h_(h), depth_(depth)
{
    derivative_ = k.get("cambi_derivative"); mask_kernel_ = k.get("cambi_mask");
    from8_ = k.get("cambi_from8"); decimate16_ = k.get("cambi_decimate16"); decimate8_ = k.get("cambi_decimate8"); mode_ = k.get("cambi_mode");
    cvalues_ = k.get("cambi_cvalues");
    radix_ = k.get("radix_histogram"); sum_above_ = k.get("sum_above");
    cu_check(cuStreamCreate(&stream_, CU_STREAM_NON_BLOCKING), "creating a stream");
    cu_check(cuEventCreate(&copied_, CU_EVENT_DISABLE_TIMING), "creating an event");
    window_ = ((65 * (w + h) / 375) >> 4) | 1;  // cambi.c's adjust_window_size
    mask_index_ = (49 + 3 * (ceil_log2((uint32_t)((w >> 6) * (h >> 6))) - 11) - 1) >> 1;
    VmafLumaRange range;
    VmafEOTF eotf;
    vmaf_luminance_init_luma_range(&range, 10, 1 /* limited */);
    vmaf_luminance_init_eotf(&eotf, "bt1886");
    const int diffs = 4, weights[4] = {1, 2, 3, 4};
    for (int d = 0; d < diffs; d++) {
        params_.tvi[d] = get_tvi_for_diff(d + 1, kTvi, 10, range, eotf) + diffs;
        params_.weight[d] = weights[d];
    }
    params_.vlt = get_vlt_luma(0.0, range, eotf);
    params_.band_base = std::max(params_.vlt - 3 * diffs + 1, 0);
    params_.band_size = params_.tvi[diffs - 1] + 1 - params_.band_base;
    params_.pad = window_ >> 1;
    // column counts, the counts of 32 pixels and the numbers of the values, as cambi_cvalues lays them out
    const int columns = kTileW + 2 * params_.pad + 1;
    shared_ = (unsigned)(columns * 32 + (columns >> 4) * 8 + 8 + 32 * 33 * 2 + (params_.band_size + 2 * diffs) * 2);
    sw_[0] = w; sh_[0] = h;
    for (int s = 1; s < kScales; s++) { sw_[s] = (sw_[s - 1] + 1) >> 1; sh_[s] = (sh_[s - 1] + 1) >> 1; }
    size_t total = 0;
    for (int s = 0; s < kScales; s++) {
        const size_t m = (size_t)sw_[s] * sh_[s];
        cu_check(cuMemAlloc(&image_[s], m * 2), "cambi buffers");
        cu_check(cuMemAlloc(&mask_[s], m), "cambi buffers");
        cvalues_at_[s] = total;
        total += m;
        first_[s] = blocks_;
        blocks_ += (sw_[s] + kTileW - 1) / kTileW * ((sh_[s] + kTileH - 1) / kTileH);
    }
    first_[kScales] = blocks_;
    cvalues_at_[kScales] = total;
    const size_t n = (size_t)w * h;
    cu_check(cuMemAlloc(&scratch_, n * 2), "cambi buffers");
    cu_check(cuMemAlloc(&deriv_, n), "cambi buffers");
    cu_check(cuMemAlloc(&cvalues_all_, total * 4), "cambi buffers");
    cu_check(cuMemAlloc(&hist_, 256 * 4), "cambi buffers");
    cu_check(cuMemAlloc(&sums_, sizeof(Sums)), "cambi buffers");
    cu_check(cuMemAllocHost((void **)&host_, sizeof(Sums)), "cambi buffers");
}

Cambi::~Cambi()
{
    for (CUdeviceptr p : {scratch_, deriv_, cvalues_all_, hist_, sums_}) if (p) cuMemFree(p);
    for (int s = 0; s < kScales; s++) for (CUdeviceptr p : {image_[s], mask_[s]}) if (p) cuMemFree(p);
    if (host_) cuMemFreeHost(host_);
    if (copied_) cuEventDestroy(copied_);
    if (stream_) cuStreamDestroy(stream_);
}

void Cambi::launch(CUdeviceptr img, int pitch, CUstream stream)
{
    int w = w_, h = h_;
    // scale 0 works on a copy in scratch_; an 8-bit picture is brought to 10 bits and through the anti-dithering filter
    // on the way, as cambi.c's preprocessing does
    if (depth_ == 8) {
        void *a[] = {&img, &pitch, &w, &h, &scratch_};
        Kernels::launch(from8_, w, h, a, stream);
    } else {
        CUDA_MEMCPY2D m = {};
        m.srcMemoryType = m.dstMemoryType = CU_MEMORYTYPE_DEVICE;
        m.srcDevice = img; m.srcPitch = (size_t)pitch * 2; m.dstDevice = scratch_; m.dstPitch = (size_t)w * 2;
        m.WidthInBytes = (size_t)w * 2; m.Height = h;
        cu_check(cuMemcpy2DAsync(&m, stream), "cambi copy");
    }
    cu_check(cuEventRecord(copied_, stream), "cambi copy");
    cu_check(cuStreamWaitEvent(stream_, copied_, 0), "cambi copy");
    { void *a[] = {&scratch_, &w, &w, &h, &deriv_}; Kernels::launch(derivative_, w, h, a, stream_); }
    { int idx = mask_index_; void *a[] = {&deriv_, &w, &h, &idx, &mask_[0]}; Kernels::launch(mask_kernel_, w, h, a, stream_); }
    // each scale's picture is the mode filter of the one before it decimated, and its mask the mask before it decimated
    for (int s = 0; s < kScales; s++) {
        w = sw_[s]; h = sh_[s];
        if (s > 0) {
            int pw = sw_[s - 1];
            { void *a[] = {&image_[s - 1], &pw, &scratch_, &w, &h}; Kernels::launch(decimate16_, w, h, a, stream_); }
            { void *a[] = {&mask_[s - 1], &pw, &mask_[s], &w, &h}; Kernels::launch(decimate8_, w, h, a, stream_); }
        }
        void *a[] = {&scratch_, &w, &h, &image_[s]};
        Kernels::launch(mode_, w, h, a, stream_);
    }
    // the c-values of all scales in one launch (kernels.cu's CambiScales)
    struct { CUdeviceptr img[kScales], mask[kScales], out[kScales]; int w[kScales], h[kScales], first[kScales + 1]; } scales;
    for (int s = 0; s < kScales; s++) {
        scales.img[s] = image_[s]; scales.mask[s] = mask_[s]; scales.out[s] = cvalues_all_ + cvalues_at_[s] * 4;
        scales.w[s] = sw_[s]; scales.h[s] = sh_[s]; scales.first[s] = first_[s];
    }
    scales.first[kScales] = first_[kScales];
    cu_check(cuMemsetD32Async(cvalues_all_, 0, cvalues_at_[kScales], stream_), "clearing c-values");
    void *a[] = {&scales, &params_};
    cu_check(cuLaunchKernel(cvalues_, blocks_, 1, 1, 32, 1, 1, shared_, stream_, a, nullptr), "cambi c-values");
    // the sum and count of the positive c-values, which give the pooled score whenever no more than 60% are positive
    cu_check(cuMemsetD8Async(sums_, 0, sizeof(Sums), stream_), "clearing sums");
    for (int s = 0; s < kScales; s++) {
        size_t n = (size_t)sw_[s] * sh_[s];
        float zero = 0.0f;
        CUdeviceptr out = scales.out[s], sum = sums_ + s * 8, count = sums_ + offsetof(Sums, count) + s * 8;
        void *b[] = {&out, &n, &zero, &sum, &count};
        cu_check(cuLaunchKernel(sum_above_, 256, 1, 1, 256, 1, 1, 0, stream_, b, nullptr), "summing c-values");
    }
    cu_check(cuMemcpyDtoHAsync(host_, sums_, sizeof(Sums), stream_), "reading sums");
}

// Mean of the k largest c-values of a scale when more than k are positive: a radix select finds the k-th largest value,
// then the values above it are summed and the rest of the k are that value.
double Cambi::topk_exact(int s)
{
    const size_t n = (size_t)sw_[s] * sh_[s], k = topk_count(n);
    unsigned prefix = 0, high = 0, hist[256];
    size_t remaining = k;
    for (int shift = 24; shift >= 0; shift -= 8) {
        cu_check(cuMemsetD32Async(hist_, 0, 256, stream_), "radix");
        CUdeviceptr values = cvalues_all_ + cvalues_at_[s] * 4;
        void *args[] = {&values, (void *)&n, &prefix, &high, &shift, &hist_};
        cu_check(cuLaunchKernel(radix_, 256, 1, 1, 256, 1, 1, 0, stream_, args, nullptr), "radix");
        cu_check(cuMemcpyDtoHAsync(hist, hist_, sizeof hist, stream_), "radix");
        cu_check(cuStreamSynchronize(stream_), "radix");
        size_t above = 0;
        int digit = 255;
        for (; digit > 0 && above + hist[digit] < remaining; digit--) above += hist[digit];
        remaining -= above;
        prefix |= (unsigned)digit << shift;
        high |= 255u << shift;
    }
    float threshold;
    std::memcpy(&threshold, &prefix, 4);
    cu_check(cuMemsetD8Async(sums_, 0, 16, stream_), "sum");
    CUdeviceptr sum = sums_, count = sums_ + 8;
    CUdeviceptr values = cvalues_all_ + cvalues_at_[s] * 4;
    void *args[] = {&values, (void *)&n, &threshold, &sum, &count};
    cu_check(cuLaunchKernel(sum_above_, 256, 1, 1, 256, 1, 1, 0, stream_, args, nullptr), "sum");
    cu_check(cuMemcpyDtoHAsync(host_, sums_, 16, stream_), "sum");
    cu_check(cuStreamSynchronize(stream_), "sum");
    const double total = host_->sum[0];
    const unsigned long long above = ((const unsigned long long *)host_->sum)[1];
    return (total + (double)(k - above) * threshold) / (double)k;
}

double Cambi::finish()
{
    cu_check(cuStreamSynchronize(stream_), "cambi");
    const Sums got = *host_;
    double total = 0;
    for (int s = 0; s < kScales; s++) {
        const size_t k = topk_count((size_t)sw_[s] * sh_[s]);
        const double pooled = got.count[s] <= k ? got.sum[s] / (double)k : topk_exact(s);
        total += pooled * kScaleWeights[s];
    }
    const int odd = 2 * (window_ >> 1) + 1;
    return std::min(total / (odd * odd), kMax);
}
