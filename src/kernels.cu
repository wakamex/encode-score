// GPU kernels for encode-score. Compiled to a fatbin and launched through the CUDA driver API.
#include <cstdint>

// ---- CAMBI (Contrast Aware Multiscale Banding Index), a port of libvmaf's cambi.c that gives the same scores ----
#include "cambi_lut.h"

#define MASK_ROWS 32

// The spatial mask: 1 where more than `index` pixels of the 7x7 square around a pixel (none outside the picture) are
// flat, that is equal their right and bottom neighbours (the last column and row count as equal), as cambi.c's
// get_derivative_data_for_row. Each warp moves down MASK_ROWS rows of a strip of 32 columns: per row, its lanes find
// which of the 64 pixels from 16 left of the strip are flat, count the 7 around each column, and keep the sum of the
// counts of the last 7 rows. Blocks are 32 x 4 threads, one strip per warp.
extern "C" __global__ void cambi_mask(const uint16_t *img, int w, int h, int index, uint8_t *mask)
{
    constexpr int R = 3;
    const int lane = threadIdx.x, x0 = (blockIdx.x * blockDim.y + threadIdx.y) * 32, y0 = blockIdx.y * MASK_ROWS;
    if (x0 >= w) return;
    const int xa = x0 - 16 + lane, xb = x0 + 16 + lane;  // the two pixels of the row this lane looks at
    auto at = [&](int x, int y) { return x >= 0 && x < w && y >= 0 && y < h ? (int)img[(size_t)y * w + x] : -1; };
    int a = at(xa, y0 - R), b = at(xb, y0 - R);
    int sum = 0, last[2 * R + 1] = {};  // the counts of the last 7 rows, oldest first
    for (int y = y0 - R; y < y0 + MASK_ROWS + R; y++) {
        const int na = at(xa, y + 1), nb = at(xb, y + 1);
        int ra = __shfl_down_sync(0xffffffff, a, 1);
        const int rb = __shfl_down_sync(0xffffffff, b, 1), b0 = __shfl_sync(0xffffffff, b, 0);
        if (lane == 31) ra = b0;
        const bool fa = a >= 0 && (xa == w - 1 || a == ra) && (y == h - 1 || a == na);
        const bool fb = b >= 0 && (xb == w - 1 || b == rb) && (y == h - 1 || b == nb);
        const unsigned long long flat = __ballot_sync(0xffffffff, fa) | (unsigned long long)__ballot_sync(0xffffffff, fb) << 32;
        const int count = __popcll(flat >> (16 + lane - R) & 0x7f);
        sum += count - last[0];
#pragma unroll
        for (int k = 0; k < 2 * R; k++) last[k] = last[k + 1];
        last[2 * R] = count;
        const int out = y - R;
        if (out >= y0 && out < h && x0 + lane < w) mask[(size_t)out * w + x0 + lane] = sum > index;
        a = na; b = nb;
    }
}

// Every second pixel of every second row: dst (w, h) from src with row pitch `spitch`.
extern "C" __global__ void cambi_decimate16(const uint16_t *src, int spitch, uint16_t *dst, int w, int h)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < w && y < h) dst[(size_t)y * w + x] = src[(size_t)(2 * y) * spitch + 2 * x];
}

extern "C" __global__ void cambi_decimate8(const uint8_t *src, int spitch, uint8_t *dst, int w, int h)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < w && y < h) dst[(size_t)y * w + x] = src[(size_t)(2 * y) * spitch + 2 * x];
}

__device__ __forceinline__ uint16_t mode3(uint16_t a, uint16_t b, uint16_t c)
{
    if (a == b || a == c) return a;
    if (b == c) return b;
    return min(a, min(b, c));
}

#define CAMBI_DIFFS 4
#define CAMBI_NONE 0xFFFF

struct CambiParams {
    int w, h, pad, band_base, band_size, vlt;
    int tvi[CAMBI_DIFFS];     // in cambi.c's adjusted space (value + number of diffs)
    int weight[CAMBI_DIFFS];
};

// One round of a radix select over non-negative floats (as their bit patterns): counts the elements whose bits above
// `shift + 8` equal `prefix`, by their next 8 bits.
extern "C" __global__ void radix_histogram(const unsigned *v, size_t n, unsigned prefix, unsigned high_mask, int shift, unsigned *hist)
{
    __shared__ unsigned local[256];
    for (int i = threadIdx.x; i < 256; i += blockDim.x) local[i] = 0;
    __syncthreads();
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        unsigned b = v[i];
        if ((b & high_mask) == prefix) atomicAdd(&local[(b >> shift) & 255], 1u);
    }
    __syncthreads();
    for (int i = threadIdx.x; i < 256; i += blockDim.x) if (local[i]) atomicAdd(&hist[i], local[i]);
}

// The sum and count of the elements strictly above `threshold`.
extern "C" __global__ void sum_above(const float *v, size_t n, float threshold, double *sum, unsigned long long *count)
{
    double s = 0;
    unsigned long long c = 0;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
        if (v[i] > threshold) { s += v[i]; c++; }
    for (int o = 16; o; o >>= 1) { s += __shfl_down_sync(0xffffffff, s, o); c += __shfl_down_sync(0xffffffff, c, o); }
    if ((threadIdx.x & 31) == 0) { atomicAdd(sum, s); atomicAdd(count, c); }
}

#define CAMBI_SCALES 5
#define CV_TX 64
#define CV_TY 32

// The pictures of all scales, whose c-values one launch of cambi_cvalues computes.
struct CambiScales {
    const uint16_t *img[CAMBI_SCALES];
    const uint8_t *mask[CAMBI_SCALES];
    float *out[CAMBI_SCALES];
    int w[CAMBI_SCALES], h[CAMBI_SCALES];
    int first[CAMBI_SCALES + 1];  // the first block of each scale
    double *sum;                  // per scale, the sum and count of the positive c-values
    unsigned long long *count;
};

// The value of a pixel less band_base, or CAMBI_NONE outside the picture, the mask or the band.
__device__ __forceinline__ int cv_value(const uint16_t *img, const uint8_t *mask, const CambiParams &p, int x, int y)
{
    if (x < 0 || x >= p.w || y < 0 || y >= p.h) return CAMBI_NONE;
    const size_t k = (size_t)y * p.w + x;
    const unsigned u = (unsigned)img[k] - (unsigned)p.band_base;
    return mask[k] && u < (unsigned)p.band_size ? (int)u : CAMBI_NONE;
}

// Where the bins of column c start; the padding puts the groups of lanes on different banks.
__device__ __forceinline__ int cv_col(int c, int bins) { return c * bins + (c >> 4) * 8; }

// The room for column counts of the tile, its margins and one column past them, in up to 32 bins. The shared memory
// of cambi_cvalues is this, then the counts of 32 pixels (32 x 33 uint16_t), then the numbers of the values
// (band_size + 2 * CAMBI_DIFFS uint16_t).
__device__ __forceinline__ int cv_cols_bytes(int pad) { return (CV_TX + 2 * pad + 1) * 32 + ((CV_TX + 2 * pad + 1) >> 4) * 8 + 8; }

// The passes of cambi_cvalues over a tile whose c-values need `relevant` numbered values, counted in B bins. The warp
// works as 32 / B groups of B lanes, each group sliding along its own part of the row.
template <int B>
__device__ void cv_passes(const uint16_t *img, const uint8_t *mask, const CambiParams &p, float *out, uint8_t *cols,
                          uint16_t *counts, const uint16_t *rank, int relevant, int x0, int y0, int rows, double &sum,
                          unsigned &positive)
{
    // groups of lanes, steps between flushes of the counts, pixels per group and row, and flushes per row
    constexpr int G = 32 / B, K = 32 / G, PART = CV_TX / G, F = CV_TX / 32;
    const int lane = threadIdx.x, g = lane / B, b = lane % B;
    const int span = 2 * p.pad + 1, cw = CV_TX + 2 * p.pad;
    // the pixels this lane computes, one per flush: K consecutive pixels of a group's part per flush
    const int mine = (lane / K) * PART + lane % K;
    for (int b0 = 0;;) {
        // this pass counts values b0 .. b0 + B - 1 and serves those whose neighbours within CAMBI_DIFFS it counts
        const int lo = b0 ? b0 + CAMBI_DIFFS : 0, hi = b0 + B >= relevant ? relevant : b0 + B - CAMBI_DIFFS;
        auto bin = [&](int c, int y) {
            const int u = cv_value(img, mask, p, x0 - p.pad + c, y);
            return u == CAMBI_NONE ? (unsigned)B : (unsigned)(rank[u + CAMBI_DIFFS] - b0);
        };
        for (int c = lane; c < cw; c += 32) {
            uint8_t *col = cols + cv_col(c, B);
            for (int i = 0; i < B; i++) col[i] = 0;
#pragma unroll 8
            for (int dy = -p.pad; dy <= p.pad; dy++) { const unsigned k = bin(c, y0 + dy); if (k < B) col[k]++; }
        }
        __syncwarp();
        for (int r = 0; r < rows; r++) {
            const int y = y0 + r;
            if (r > 0) {
                for (int c = lane; c < cw; c += 32) {
                    uint8_t *col = cols + cv_col(c, B);
                    const unsigned leave = bin(c, y - 1 - p.pad), enter = bin(c, y + p.pad);
                    if (leave < B) col[leave]--;
                    if (enter < B) col[enter]++;
                }
                __syncwarp();
            }
            int u[F], own[F];
            bool any = false;
#pragma unroll
            for (int f = 0; f < F; f++) {
                u[f] = cv_value(img, mask, p, x0 + mine + f * K, y);
                own[f] = u[f] == CAMBI_NONE ? CAMBI_NONE : rank[u[f] + CAMBI_DIFFS];
                any |= own[f] >= lo && own[f] < hi;
            }
            if (!__any_sync(0xffffffff, any)) continue;
            // lane b of group g holds the window count of value b0 + b at the group's current pixel
            const int xs = g * PART;
            unsigned n = 0;
            for (int c = 0; c < span; c++) n += cols[cv_col(xs + c, B) + b];
#pragma unroll
            for (int f = 0; f < F; f++) {
                for (int j = 0; j < K; j++) {
                    const int x = xs + f * K + j;
                    counts[(g * K + j) * (B + 1) + b] = (uint16_t)n;
                    n += cols[cv_col(x + span, B) + b] - cols[cv_col(x, B) + b];  // reads one column past the last at the end
                }
                __syncwarp();
                if (own[f] >= lo && own[f] < hi) {
                    const uint16_t *cnt = counts + lane * (B + 1) - b0;
                    // values within CAMBI_DIFFS of this one have numbers within CAMBI_DIFFS of its own, so this pass counts them
                    auto count = [&](int q) -> unsigned { const unsigned k = rank[q + CAMBI_DIFFS]; return k == CAMBI_NONE ? 0u : cnt[k]; };
                    const int value = u[f] + p.band_base + CAMBI_DIFFS;
                    const unsigned p0 = cnt[own[f]];
                    float c = 0.0f;
#pragma unroll
                    for (int d = 0; d < CAMBI_DIFFS; d++) {
                        if (value <= p.tvi[d] && value + d + 1 > p.vlt) {
                            const unsigned p1 = count(u[f] + d + 1), p2 = count(u[f] - d - 1);  // 0 below the band
                            const unsigned pm = p1 > p2 ? p1 : p2;
                            const float val = (float)(p.weight[d] * p0 * pm) * reciprocal_lut[pm + p0];
                            if (val > c) c = val;
                        }
                    }
                    out[(size_t)y * p.w + x0 + mine + f * K] = c;
                    if (c > 0) { sum += c; positive++; }
                }
                __syncwarp();
            }
        }
        if (hi == relevant) return;
        b0 = hi - CAMBI_DIFFS;
        __syncwarp();
    }
}

// The c-values of cambi.c's c_value_pixel for every scale, one warp per tile of CV_TX by CV_TY pixels, with `out`
// cleared by the caller, and the sum and count of the positive ones, which the caller clears too. The warp first numbers, in order, the values the tile's c-values can need: those of the masked
// pixels in the tile and its window margins that lie within CAMBI_DIFFS of a value of the tile's own pixels, usually a
// handful. It then counts window pixels by those numbers: for every column of the tile and its margins it keeps the
// counts in the vertical window around the current row, updated by the rows that enter and leave as it moves down, and
// slides a window of those along the row. With 8 or 16 numbers or fewer, groups of 8 or 16 lanes slide along different
// parts of the row at once; with more, 32 lanes count 32 numbers per pass. A tile's warp works through its rows alone,
// so the scales share one launch to keep the GPU busy while the small ones finish.
extern "C" __global__ void cambi_cvalues(CambiScales sc, CambiParams p)
{
    extern __shared__ uint16_t shared16[];
    uint8_t *shared = (uint8_t *)shared16;
    int s = 0;
    while ((int)blockIdx.x >= sc.first[s + 1]) s++;
    const int tile = blockIdx.x - sc.first[s], across = (sc.w[s] + CV_TX - 1) / CV_TX;
    p.w = sc.w[s]; p.h = sc.h[s];
    const uint16_t *img = sc.img[s];
    const uint8_t *mask = sc.mask[s];
    const int lane = threadIdx.x, cw = CV_TX + 2 * p.pad, nv = p.band_size + 2 * CAMBI_DIFFS;
    const int x0 = tile % across * CV_TX, y0 = tile / across * CV_TY, rows = min(CV_TY, p.h - y0);
    uint8_t *cols = shared;
    uint16_t *counts = (uint16_t *)(shared + cv_cols_bytes(p.pad)), *rank = counts + 32 * 33;  // rank by value + CAMBI_DIFFS
    // which values occur around the tile and in it, by value + CAMBI_DIFFS, in the room of the column counts
    uint8_t *occupied = cols, *served = cols + nv;
    for (int i = lane; i < 2 * nv; i += 32) cols[i] = 0;
    __syncwarp();
    // most tiles of most pictures have no pixel in the mask and the band, and so no c-value to compute
    bool any = false;
#pragma unroll 4
    for (int r = 0; r < rows; r++)
        for (int c = lane; c < CV_TX; c += 32) {
            const int u = cv_value(img, mask, p, x0 + c, y0 + r);
            if (u != CAMBI_NONE) { served[u + CAMBI_DIFFS] = 1; any = true; }
        }
    if (!__any_sync(0xffffffff, any)) return;
#pragma unroll 4
    for (int r = 0; r < rows + 2 * p.pad; r++)
        for (int c = lane; c < cw; c += 32) {
            const int u = cv_value(img, mask, p, x0 - p.pad + c, y0 - p.pad + r);
            if (u != CAMBI_NONE) occupied[u + CAMBI_DIFFS] = 1;
        }
    __syncwarp();
    int relevant = 0;
    for (int i0 = 0; i0 < nv; i0 += 32) {
        const int i = i0 + lane;
        bool need = false;
        if (i < nv && occupied[i])
            for (int k = -CAMBI_DIFFS; k <= CAMBI_DIFFS; k++) need |= i + k >= 0 && i + k < nv && served[i + k];
        const unsigned m = __ballot_sync(0xffffffff, need);
        if (i < nv) rank[i] = need ? relevant + __popc(m & ((1u << lane) - 1)) : CAMBI_NONE;
        relevant += __popc(m);
    }
    __syncwarp();
    if (relevant == 0) return;
    float *out = sc.out[s];
    double sum = 0;
    unsigned positive = 0;
    if (relevant <= 8) cv_passes<8>(img, mask, p, out, cols, counts, rank, relevant, x0, y0, rows, sum, positive);
    else if (relevant <= 16) cv_passes<16>(img, mask, p, out, cols, counts, rank, relevant, x0, y0, rows, sum, positive);
    else cv_passes<32>(img, mask, p, out, cols, counts, rank, relevant, x0, y0, rows, sum, positive);
    for (int o = 16; o; o >>= 1) { sum += __shfl_down_sync(0xffffffff, sum, o); positive += __shfl_down_sync(0xffffffff, positive, o); }
    if (lane == 0 && positive) { atomicAdd(&sc.sum[s], sum); atomicAdd(&sc.count[s], (unsigned long long)positive); }
}

// ---- Detail, texture, grain, color and PSNR measures ----

// A decoded frame as the kernels read it: planar or semi-planar (chroma interleaved in plane u) 4:2:0, with 8-bit or
// 16-bit samples whose value is the sample shifted right by `shift` (6 for P010) and has `depth` bits.
struct Frame {
    const void *y, *u, *v;
    int ypitch, cpitch, semi, bytes, shift, depth;
};

__device__ __forceinline__ unsigned sample(const void *plane, int pitch, int x, int y, const Frame &f)
{
    const char *row = (const char *)plane + (size_t)y * pitch;
    return (f.bytes == 1 ? ((const uint8_t *)row)[x] : ((const uint16_t *)row)[x]) >> f.shift;
}

// The frame cropped at (x0, y0) to w x h: luma and both chroma planes as integers at the pipeline's bit depth `pd` (8
// or 10). An 8-bit frame in a 10-bit pipeline is shifted up by 2, as libvmaf does.
extern "C" __global__ void frame_planes(Frame f, int x0, int y0, int pd, uint16_t *y16_out, uint16_t *u_out,
                                        uint16_t *v_out, int w, int h)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    const unsigned l = sample(f.y, f.ypitch, x + x0, y + y0, f);
    y16_out[(size_t)y * w + x] = l << (pd - f.depth);
    if ((x & 1) || (y & 1)) return;
    const int cx = (x + x0) / 2, cy = (y + y0) / 2, ci = (y / 2) * (w / 2) + x / 2;
    const unsigned u = f.semi ? sample(f.u, f.cpitch, 2 * cx, cy, f) : sample(f.u, f.cpitch, cx, cy, f);
    const unsigned v = f.semi ? sample(f.u, f.cpitch, 2 * cx + 1, cy, f) : sample(f.v, f.cpitch, cx, cy, f);
    u_out[ci] = u << (pd - f.depth);
    v_out[ci] = v << (pd - f.depth);
}

// The sums of N values over a block of 256 threads, into part[k][0]: each adds within warps, then the warps' sums in
// order.
template <int N>
__device__ __forceinline__ void block_sums(float *v, float (*part)[8])
{
    const int t = threadIdx.x;
    for (int k = 0; k < N; k++) {
        for (int o = 16; o; o >>= 1) v[k] += __shfl_down_sync(0xffffffff, v[k], o);
        if ((t & 31) == 0) part[k][t >> 5] = v[k];
    }
    __syncthreads();
    if (t < N) {
        float s = 0;
        for (int k = 0; k < 8; k++) s += part[t][k];
        part[t][0] = s;
    }
    __syncthreads();
}

#define WIN 16
#define WINDOW_STATS 8

// Per 16 by 16 window of one frame pair: fine and mid detail energy of the source and the encode, the energy each shares
// with the other, and the oriented detail of each (the difference of the eigenvalues of the summed structure tensor of
// the 3-pixel-smoothed picture). Fine detail is luma less its 3 by 3 box mean, mid detail the 3 by 3 mean less the 9 by 9
// mean, with the picture's edge pixels repeated beyond it; luma is on a 10-bit scale (the planes shifted up by `up`).
// Also writes the mid detail planes for the texture measure. One block of 256 threads per window; out[window *
// WINDOW_STATS + i] is: 0 source fine, 1 source mid, 2 encode fine, 3 encode mid, 4 shared fine, 5 shared mid, 6 source
// oriented, 7 encode oriented.
extern "C" __global__ void window_stats(const uint16_t *yr, const uint16_t *yd, int up, int w, int h, float *mid_r,
                                        float *mid_d, float *out)
{
    // per picture: luma of the window and 4 pixels around it, its horizontal sums of 3 (for the columns from one before
    // the window to one after) and of 9, and the 3 by 3 means from one pixel before the window to one after
    // (rows of Y and B3 are 48 apart, so the two rows a warp reads are on different banks)
    constexpr int R = 4, T = WIN + 2 * R, S = 48;
    __shared__ float Y[2][T][S], H3[2][T][WIN + 2], H9[2][T][WIN], B3[2][WIN + 2][S];
    __shared__ float part[12][8];
    const int x0 = blockIdx.x * WIN, y0 = blockIdx.y * WIN, t = threadIdx.x;
    for (int k = t; k < 2 * T * T; k += 256) {
        const int pic = k / (T * T), r = k / T % T, c = k % T;
        const int yy = min(max(y0 + r - R, 0), h - 1), xx = min(max(x0 + c - R, 0), w - 1);
        Y[pic][r][c] = (float)((pic ? yd : yr)[(size_t)yy * w + xx] << up);
    }
    __syncthreads();
    for (int k = t; k < 2 * T * (WIN + 2); k += 256) {
        const int pic = k / (T * (WIN + 2)), r = k / (WIN + 2) % T, j = k % (WIN + 2);
        H3[pic][r][j] = Y[pic][r][j + R - 2] + Y[pic][r][j + R - 1] + Y[pic][r][j + R];
    }
    // four sums of 9 per thread, each the one before plus the pixel entering and less the one leaving (exact, as the
    // values are integers)
    for (int k = t; k < 2 * T * WIN / 4; k += 256) {
        const int pic = k / (T * WIN / 4), r = k / (WIN / 4) % T, j = k % (WIN / 4) * 4;
        float s9 = 0;
        for (int d = 0; d < 9; d++) s9 += Y[pic][r][j + d];
        H9[pic][r][j] = s9;
        for (int q = 1; q < 4; q++) H9[pic][r][j + q] = s9 += Y[pic][r][j + q + 8] - Y[pic][r][j + q - 1];
    }
    __syncthreads();
    for (int k = t; k < 2 * (WIN + 2) * (WIN + 2); k += 256) {
        const int pic = k / ((WIN + 2) * (WIN + 2)), i = k / (WIN + 2) % (WIN + 2), j = k % (WIN + 2);
        B3[pic][i][j] = (H3[pic][i + R - 2][j] + H3[pic][i + R - 1][j] + H3[pic][i + R][j]) / 9.0f;
    }
    __syncthreads();
    const int tx = t & 15, ty = t >> 4, x = x0 + tx, y = y0 + ty;
    const size_t i = (size_t)y * w + x;
    float fine[2], mid[2], gx[2] = {0, 0}, gy[2] = {0, 0};
    for (int pic = 0; pic < 2; pic++) {
        float s9 = 0;
        for (int d = 0; d < 9; d++) s9 += H9[pic][ty + d][tx];
        const float b3 = B3[pic][ty + 1][tx + 1], b9 = s9 / 81.0f;
        fine[pic] = Y[pic][ty + R][tx + R] - b3;
        mid[pic] = b3 - b9;
        if (x > 0 && x < w - 1) gx[pic] = B3[pic][ty + 1][tx + 2] - B3[pic][ty + 1][tx];
        if (y > 0 && y < h - 1) gy[pic] = B3[pic][ty + 2][tx + 1] - B3[pic][ty][tx + 1];
    }
    mid_r[i] = mid[0];
    mid_d[i] = mid[1];
    float v[12] = {fine[0] * fine[0], mid[0] * mid[0], fine[1] * fine[1], mid[1] * mid[1], fine[0] * fine[1], mid[0] * mid[1],
                   gx[0] * gx[0], gy[0] * gy[0], gx[0] * gy[0], gx[1] * gx[1], gy[1] * gy[1], gx[1] * gy[1]};
    block_sums<12>(v, part);
    if (t == 0) {
        float *o = out + ((size_t)blockIdx.y * gridDim.x + blockIdx.x) * WINDOW_STATS;
        auto sum = [&](int k) { return part[k][0]; };
        for (int k = 0; k < 6; k++) o[k] = sum(k);
        o[6] = sqrtf((sum(6) - sum(7)) * (sum(6) - sum(7)) + 4 * sum(8) * sum(8));
        o[7] = sqrtf((sum(9) - sum(10)) * (sum(9) - sum(10)) + 4 * sum(11) * sum(11));
    }
}

#define PERSIST_STATS 5

// The texture measure for one centre frame, per window: from the source's mid detail planes of the 6 frames before and
// after (`ring` holds 13 planes, centre at slot `centre`), the means A of offsets +-2, 4, 6 and B of +-1, 3, 5, and
// T = (A + B) / 2. out: 0 sum(source mid * T), 1 sum(encode mid * T), 2 sum(A * B), 3 sum(A * A), 4 sum(B * B).
extern "C" __global__ void persist_stats(const float *ring, size_t plane, int centre, const float *mid_d, int w, float *out)
{
    __shared__ float part[5][8];
    const int wx = blockIdx.x, wy = blockIdx.y;
    const int x = wx * WIN + (threadIdx.x & 15), y = wy * WIN + (threadIdx.x >> 4);
    const size_t i = (size_t)y * w + x;
    auto at = [&](int off) { return ring[(size_t)((centre + off + 13) % 13) * plane + i]; };
    const float a = (at(-6) + at(-4) + at(-2) + at(2) + at(4) + at(6)) / 6.0f;
    const float b = (at(-5) + at(-3) + at(-1) + at(1) + at(3) + at(5)) / 6.0f;
    const float t = (a + b) / 2;
    float v[5] = {at(0) * t, mid_d[i] * t, a * b, a * a, b * b};
    block_sums<5>(v, part);
    if (threadIdx.x == 0) {
        float *o = out + ((size_t)wy * gridDim.x + wx) * PERSIST_STATS;
        for (int k = 0; k < 5; k++) o[k] = part[k][0];
    }
}

// Squared differences of a pair of 10-bit planes (for PSNR), and the sums of each plane, for one row per thread block.
extern "C" __global__ void plane_sse(const uint16_t *a, const uint16_t *b, const float *fa, const float *fb, size_t n,
                                     unsigned long long *sse)
{
    unsigned long long s = 0;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        long long d = a ? (long long)a[i] - b[i] : (long long)fa[i] - (long long)fb[i];
        s += (unsigned long long)(d * d);
    }
    for (int o = 16; o; o >>= 1) s += __shfl_down_sync(0xffffffff, s, o);
    if ((threadIdx.x & 31) == 0) atomicAdd(sse, s);
}

// Sums of the chroma values and of the color strength (distance from neutral 512) of a pair of chroma planes, on a
// 10-bit scale (`up` is the shift from the planes' bit depth to 10).
// The sums of values are exact integers; each thread adds its few strengths in single precision, which keeps the
// arithmetic off the GPU's slow double-precision units.
extern "C" __global__ void color_sums(const uint16_t *u, const uint16_t *v, size_t n, int up, double *sums)
{
    unsigned long long su = 0, sv = 0;
    float ss = 0;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        const int uu = u[i] << up, vv = v[i] << up, du = uu - 512, dv = vv - 512;
        su += uu; sv += vv; ss += sqrtf((float)(du * du + dv * dv));
    }
    double strength = ss;
    for (int o = 16; o; o >>= 1) {
        su += __shfl_down_sync(0xffffffff, su, o); sv += __shfl_down_sync(0xffffffff, sv, o);
        strength += __shfl_down_sync(0xffffffff, strength, o);
    }
    if ((threadIdx.x & 31) == 0) { atomicAdd(&sums[0], (double)su); atomicAdd(&sums[1], (double)sv); atomicAdd(&sums[2], strength); }
}

// cambi.c's filter_mode in one pass: the vertical mode of the horizontal modes of three neighbours (first and last
// column keep their value in the horizontal step), with the first and last row unchanged.
extern "C" __global__ void cambi_mode(const uint16_t *img, int w, int h, uint16_t *out)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    size_t i = (size_t)y * w + x;
    if (y == 0 || y == h - 1) { out[i] = img[i]; return; }
    uint16_t m[3];
    for (int k = 0; k < 3; k++) {
        const uint16_t *row = img + (size_t)(y - 1 + k) * w;
        m[k] = (x == 0 || x == w - 1) ? row[x] : mode3(row[x - 1], row[x], row[x + 1]);
    }
    out[i] = mode3(m[0], m[1], m[2]);
}

#define TINY_W 64
#define TINY_H 36

// A 64 by 36 thumbnail of the cropped luma of a frame on a 10-bit scale (box means), for finding the time offset
// between two files.
extern "C" __global__ void tiny_luma(Frame f, int x0, int y0, int w, int h, float *out)
{
    const int tx = blockIdx.x, ty = blockIdx.y;
    const int xa = x0 + tx * w / TINY_W, xb = x0 + (tx + 1) * w / TINY_W, ya = y0 + ty * h / TINY_H, yb = y0 + (ty + 1) * h / TINY_H;
    float s = 0;
    for (int y = ya + threadIdx.y; y < yb; y += blockDim.y)
        for (int x = xa + threadIdx.x; x < xb; x += blockDim.x) s += (float)(sample(f.y, f.ypitch, x, y, f) << (10 - f.depth));
    __shared__ float part[256];
    const int t = threadIdx.y * blockDim.x + threadIdx.x;
    part[t] = s;
    __syncthreads();
    for (int o = 128; o; o >>= 1) { if (t < o) part[t] += part[t + o]; __syncthreads(); }
    if (t == 0) out[ty * TINY_W + tx] = part[0] / (float)((xb - xa) * (yb - ya));
}

// cambi.c's preprocessing of an 8-bit picture: the values shifted to 10 bits, then its anti-dithering filter, the mean
// of each pixel with its right, lower and lower-right neighbours (pairs on the last row and column, the corner kept).
extern "C" __global__ void cambi_from8(const uint16_t *img, int pitch, int w, int h, uint16_t *out)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    auto at = [&](int xx, int yy) { return (unsigned)img[(size_t)yy * pitch + xx] << 2; };
    unsigned v;
    if (x < w - 1 && y < h - 1) v = (at(x, y) + at(x + 1, y) + at(x, y + 1) + at(x + 1, y + 1)) >> 2;
    else if (y < h - 1) v = (at(x, y) + at(x, y + 1)) >> 1;
    else if (x < w - 1) v = (at(x, y) + at(x + 1, y)) >> 1;
    else v = at(x, y);
    out[(size_t)y * w + x] = (uint16_t)v;
}

// A plane of integers into a libvmaf picture, whose samples are one byte at 8 bits and two above.
extern "C" __global__ void to_picture(const uint16_t *src, int w, int h, void *dst, int pitch, int bytes)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    char *row = (char *)dst + (size_t)y * pitch;
    if (bytes == 1) ((uint8_t *)row)[x] = (uint8_t)src[(size_t)y * w + x];
    else ((uint16_t *)row)[x] = src[(size_t)y * w + x];
}
