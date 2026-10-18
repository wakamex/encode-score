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

static constexpr int kScaleWeights[5] = {16, 8, 4, 2, 1}, kTx = 32, kTy = 8, kSharedMax = 101376;
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
    rows_ = k.get("cambi_cvalues_rows"); rows_small_ = k.get("cambi_cvalues_rows_small"); packed_ = k.get("cambi_cvalues_packed");
    radix_ = k.get("radix_histogram"); sum_above_ = k.get("sum_above");
    cu_check(cuStreamCreate(&stream_, CU_STREAM_NON_BLOCKING), "creating a stream");
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
    sw_[0] = w; sh_[0] = h;
    for (int s = 1; s < kScales; s++) { sw_[s] = (sw_[s - 1] + 1) >> 1; sh_[s] = (sh_[s - 1] + 1) >> 1; }
    const size_t n = (size_t)w * h;
    for (int i = 0; i < 2; i++) {
        cu_check(cuMemAlloc(&image_[i], n * 2), "cambi buffers");
        cu_check(cuMemAlloc(&mask_[i], n), "cambi buffers");
    }
    cu_check(cuMemAlloc(&deriv_, n), "cambi buffers");
    for (int s = 0; s < kScales; s++) cu_check(cuMemAlloc(&cvalues_[s], (size_t)sw_[s] * sh_[s] * 4), "cambi buffers");
    cu_check(cuMemAlloc(&hist_, 256 * 4), "cambi buffers");
    cu_check(cuMemAlloc(&sums_, sizeof(Sums)), "cambi buffers");
    cu_check(cuMemAllocHost((void **)&host_, sizeof(Sums)), "cambi buffers");
}

Cambi::~Cambi()
{
    for (CUdeviceptr p : {image_[0], image_[1], mask_[0], mask_[1], deriv_, hist_, sums_}) if (p) cuMemFree(p);
    for (CUdeviceptr p : cvalues_) if (p) cuMemFree(p);
    if (host_) cuMemFreeHost(host_);
    if (stream_) cuStreamDestroy(stream_);
}

void Cambi::launch(CUdeviceptr img, int pitch, CUevent ready)
{
    cu_check(cuStreamWaitEvent(stream_, ready, 0), "waiting for the picture");
    int w = w_, h = h_;
    // scale 0 works on a copy, since the mode filter and decimation rewrite the picture; an 8-bit picture is brought to
    // 10 bits and through the anti-dithering filter on the way, as cambi.c's preprocessing does
    if (depth_ == 8) {
        void *a[] = {&img, &pitch, &w, &h, &image_[0]};
        Kernels::launch(from8_, w, h, a, stream_);
    } else {
        CUDA_MEMCPY2D m = {};
        m.srcMemoryType = m.dstMemoryType = CU_MEMORYTYPE_DEVICE;
        m.srcDevice = img; m.srcPitch = (size_t)pitch * 2; m.dstDevice = image_[0]; m.dstPitch = (size_t)w * 2;
        m.WidthInBytes = (size_t)w * 2; m.Height = h;
        cu_check(cuMemcpy2DAsync(&m, stream_), "cambi copy");
    }
    { void *a[] = {&image_[0], &w, &w, &h, &deriv_}; Kernels::launch(derivative_, w, h, a, stream_); }
    { int idx = mask_index_; void *a[] = {&deriv_, &w, &h, &idx, &mask_[0]}; Kernels::launch(mask_kernel_, w, h, a, stream_); }
    cu_check(cuMemsetD8Async(sums_, 0, sizeof(Sums), stream_), "clearing sums");
    // the picture of the current scale is image_[0] and its mask mask_[cur]; image_[1] is scratch
    int cur = 0;
    for (int s = 0; s < kScales; s++) {
        if (s > 0) {
            int pw = w;
            w = sw_[s]; h = sh_[s];
            { void *a[] = {&image_[0], &pw, &image_[1], &w, &h}; Kernels::launch(decimate16_, w, h, a, stream_); }
            std::swap(image_[0], image_[1]);
            { void *a[] = {&mask_[cur], &pw, &mask_[1 - cur], &w, &h}; Kernels::launch(decimate8_, w, h, a, stream_); }
            cur = 1 - cur;
        }
        { void *a[] = {&image_[0], &w, &h, &image_[1]}; Kernels::launch(mode_, w, h, a, stream_); }
        std::swap(image_[0], image_[1]);
        Params p = params_;
        p.w = w; p.h = h;
        void *a[] = {&image_[0], &mask_[cur], &p, &cvalues_[s]};
        // large tiles on the full-size scales and small tiles below, so every scale keeps the GPU busy
        const bool large = s < 2;
        const int tx = large ? 128 : 32, ty = large ? 64 : 32, warps = large ? 6 : 8;
        const unsigned shared = (unsigned)((tx + 2 * p.pad) * (ty + 2 * p.pad) * 2 + warps * (tx + 2 * p.pad) * 32 + warps * 32 * 33 * 2);
        cu_check(cuMemsetD32Async(cvalues_[s], 0, (size_t)w * h, stream_), "clearing c-values");
        if (shared <= kSharedMax) {
            CUfunction f = large ? rows_ : rows_small_;
            cu_check(cuFuncSetAttribute(f, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, (int)shared), "shared memory");
            cu_check(cuLaunchKernel(f, (w + tx - 1) / tx, (h + ty - 1) / ty, 1, warps * 32, 1, 1, shared, stream_, a, nullptr), "cambi c-values");
        } else {  // windows too large for shared memory (pictures far above 4K)
            const unsigned plain = (unsigned)((kTx + 2 * p.pad) * (kTy + 2 * p.pad) * 2);
            cu_check(cuFuncSetAttribute(packed_, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, (int)plain), "shared memory");
            cu_check(cuLaunchKernel(packed_, (w + kTx - 1) / kTx, (h + kTy - 1) / kTy, 1, kTx, kTy, 1, plain, stream_, a, nullptr), "cambi c-values");
        }
        // the sum and count of the positive c-values, which give the pooled score whenever no more than 60% are positive
        size_t n = (size_t)w * h;
        float zero = 0.0f;
        CUdeviceptr sum = sums_ + s * 8, count = sums_ + offsetof(Sums, count) + s * 8;
        void *b[] = {&cvalues_[s], &n, &zero, &sum, &count};
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
        void *args[] = {&cvalues_[s], (void *)&n, &prefix, &high, &shift, &hist_};
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
    void *args[] = {&cvalues_[s], (void *)&n, &threshold, &sum, &count};
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
