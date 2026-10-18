// GPU kernels for encode-score. Compiled to a fatbin and launched through the CUDA driver API.
#include <cstdint>

// ---- CAMBI (Contrast Aware Multiscale Banding Index), a port of libvmaf's cambi.c that gives the same scores ----
#include "cambi_lut.h"

// 1 where a pixel equals its right and bottom neighbours (the last column and row count as equal), as cambi.c's
// get_derivative_data_for_row.
extern "C" __global__ void cambi_derivative(const uint16_t *img, int pitch, int w, int h, uint8_t *d)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    const uint16_t *row = img + (size_t)y * pitch;
    bool hz = x == w - 1 || row[x] == row[x + 1];
    bool vt = y == h - 1 || row[x] == row[x + pitch];
    d[(size_t)y * w + x] = hz && vt;
}

// The spatial mask: 1 where more than `index` pixels of the 7x7 square around a pixel (zero outside the picture) have
// zero derivative.
extern "C" __global__ void cambi_mask(const uint8_t *d, int w, int h, int index, uint8_t *mask)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    int sum = 0;
    for (int dy = -3; dy <= 3; dy++) {
        int yy = y + dy;
        if (yy < 0 || yy >= h) continue;
        for (int dx = -3; dx <= 3; dx++) {
            int xx = x + dx;
            if (xx >= 0 && xx < w) sum += d[(size_t)yy * w + xx];
        }
    }
    mask[(size_t)y * w + x] = sum > index;
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

#define CAMBI_TX 32
#define CAMBI_TY 8
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


// The c-values of cambi.c's c_value_pixel, counting the whole window for every pixel but with the nine counts packed into
// two 64-bit registers (12 bits each, enough for a window of up to 4095 pixels) so the loop stays out of local memory.
extern "C" __global__ void cambi_cvalues_packed(const uint16_t *img, const uint8_t *mask, CambiParams p, float *out)
{
    extern __shared__ uint16_t tile[];
    const int tw = CAMBI_TX + 2 * p.pad, th = CAMBI_TY + 2 * p.pad;
    const int x0 = blockIdx.x * CAMBI_TX - p.pad, y0 = blockIdx.y * CAMBI_TY - p.pad;
    for (int i = threadIdx.y * CAMBI_TX + threadIdx.x; i < tw * th; i += CAMBI_TX * CAMBI_TY) {
        int xx = x0 + i % tw, yy = y0 + i / tw;
        uint16_t v = CAMBI_NONE;
        if (xx >= 0 && xx < p.w && yy >= 0 && yy < p.h) {
            size_t k = (size_t)yy * p.w + xx;
            uint16_t s = img[k];
            if (mask[k] && (uint16_t)(s - p.band_base) < (uint16_t)p.band_size) v = s;
        }
        tile[i] = v;
    }
    __syncthreads();
    const int x = blockIdx.x * CAMBI_TX + threadIdx.x, y = blockIdx.y * CAMBI_TY + threadIdx.y;
    if (x >= p.w || y >= p.h) return;
    size_t k = (size_t)y * p.w + x;
    const int v = img[k];
    float c = 0.0f;
    if (mask[k] && (unsigned)(v - p.band_base) < (unsigned)p.band_size) {
        const unsigned lo = (unsigned)(v - CAMBI_DIFFS) & 0xFFFF;
        unsigned long long a = 0, b = 0;  // counts of v-4 .. v and v+1 .. v+4, 12 bits each
        const int span = 2 * p.pad + 1;
        for (int dy = 0; dy < span; dy++) {
            const uint16_t *row = tile + (threadIdx.y + dy) * tw + threadIdx.x;
#pragma unroll 4
            for (int dx = 0; dx < span; dx++) {
                unsigned d = ((unsigned)row[dx] - lo) & 0xFFFF;
                if (d < 5) a += 1ull << (12 * d);
                else if (d < 9) b += 1ull << (12 * (d - 5));
            }
        }
        unsigned n[9];
        for (int i = 0; i < 5; i++) n[i] = (a >> (12 * i)) & 4095;
        for (int i = 0; i < 4; i++) n[5 + i] = (b >> (12 * i)) & 4095;
        const int value = v + CAMBI_DIFFS;
        const unsigned p0 = n[CAMBI_DIFFS];
#pragma unroll
        for (int d = 0; d < CAMBI_DIFFS; d++) {
            if (value <= p.tvi[d] && value + d + 1 > p.vlt) {
                unsigned p1 = n[CAMBI_DIFFS + d + 1];
                unsigned p2 = (v - d - 1 >= p.band_base) ? n[CAMBI_DIFFS - d - 1] : 0;
                unsigned pm = p1 > p2 ? p1 : p2;
                float val = (float)(p.weight[d] * p0 * pm) * reciprocal_lut[pm + p0];
                if (val > c) c = val;
            }
        }
    }
    out[k] = c;
}

#define ROWS_BINS 32
#define ROWS_SPAN 24  // values of v per pass: v - 4 .. v + 4 must fit in the 32 bins

// The same c-values as cambi_cvalues by separable counting. For a pass over 32 consecutive values starting at `b0`,
// each warp keeps, for every column of the tile region, the counts of those values in the vertical window around its
// current row, then slides horizontally along the row with lane b holding the window count of value b0 + b, so each
// pixel's nine counts are a shuffle away. Warps take every ROWS_WARPS-th row and update their column counts by the rows
// that enter and leave. A tile needs one pass for every 24 values its pixels span.
template <int ROWS_TX, int ROWS_TY, int ROWS_WARPS>
__device__ void cvalues_rows(const uint16_t *img, const uint8_t *mask, const CambiParams &p, float *out)
{
    extern __shared__ uint16_t shared[];
    const int span = 2 * p.pad + 1, tw = ROWS_TX + 2 * p.pad, th = ROWS_TY + 2 * p.pad;
    uint16_t *tile = shared;
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    uint8_t *cols = (uint8_t *)(tile + tw * th) + warp * tw * ROWS_BINS;
    // per warp, the window counts of 32 consecutive pixels, one row of ROWS_BINS + 1 (padded against bank conflicts) each
    uint16_t *counts16 = (uint16_t *)((uint8_t *)(tile + tw * th) + ROWS_WARPS * tw * ROWS_BINS) + warp * 32 * (ROWS_BINS + 1);
    const int x0 = blockIdx.x * ROWS_TX - p.pad, y0 = blockIdx.y * ROWS_TY - p.pad;
    for (int i = threadIdx.x; i < tw * th; i += blockDim.x) {
        int xx = x0 + i % tw, yy = y0 + i / tw;
        uint16_t v = CAMBI_NONE;
        if (xx >= 0 && xx < p.w && yy >= 0 && yy < p.h) {
            size_t k = (size_t)yy * p.w + xx;
            uint16_t s = img[k];
            if (mask[k] && (uint16_t)(s - p.band_base) < (uint16_t)p.band_size) v = s;
        }
        tile[i] = v;
    }
    __syncthreads();
    // the range of values among this tile's own pixels that need a c-value
    __shared__ int vmin, vmax;
    if (threadIdx.x == 0) { vmin = 0x7FFFFFFF; vmax = -1; }
    __syncthreads();
    int lmin = 0x7FFFFFFF, lmax = -1;
    for (int i = threadIdx.x; i < ROWS_TX * ROWS_TY; i += blockDim.x) {
        int r = i / ROWS_TX, c = i % ROWS_TX;
        uint16_t v = tile[(r + p.pad) * tw + c + p.pad];
        if (v != CAMBI_NONE) { lmin = min(lmin, (int)v); lmax = max(lmax, (int)v); }
    }
    atomicMin(&vmin, lmin);
    atomicMax(&vmax, lmax);
    __syncthreads();
    if (vmax < 0) return;  // the caller cleared `out`
    for (int b0 = vmin - CAMBI_DIFFS; b0 + CAMBI_DIFFS <= vmax; b0 += ROWS_SPAN) {
        const int first = b0 + CAMBI_DIFFS, last = b0 + CAMBI_DIFFS + ROWS_SPAN - 1;  // pixel values this pass serves
        int prev = -1;
        for (int r = warp; r < ROWS_TY; r += ROWS_WARPS) {
            if (y0 + p.pad + r >= p.h) break;
            // column counts for window rows r .. r + span - 1 of the region
            for (int c = lane; c < tw; c += 32) {
                uint8_t *col = cols + c * ROWS_BINS;
                if (prev < 0) {
                    for (int b = 0; b < ROWS_BINS; b++) col[b] = 0;
                    for (int dy = 0; dy < span; dy++) {
                        unsigned b = (unsigned)(tile[(r + dy) * tw + c] - b0);
                        if (b < ROWS_BINS) col[b]++;
                    }
                } else {
                    for (int rr = prev; rr < r; rr++) {
                        unsigned out_b = (unsigned)(tile[rr * tw + c] - b0);
                        if (out_b < ROWS_BINS) col[out_b]--;
                        unsigned in_b = (unsigned)(tile[(rr + span) * tw + c] - b0);
                        if (in_b < ROWS_BINS) col[in_b]++;
                    }
                }
            }
            prev = r;
            __syncwarp();
            const uint16_t *row = tile + (r + p.pad) * tw + p.pad;
            // rows with no pixel to serve in this pass need only their column counts kept up to date
            bool any = false;
            for (int x = lane; x < ROWS_TX; x += 32) { int v = row[x]; any |= v != CAMBI_NONE && v >= first && v <= last; }
            if (!__any_sync(0xffffffff, any)) { __syncwarp(); continue; }
            // slide along the row: lane holds the window count of value b0 + lane. Every 32 pixels the counts go to
            // shared memory, transposed, and each lane computes the c-value of one of those pixels.
            unsigned n = 0;
            for (int c = 0; c < span; c++) n += cols[c * ROWS_BINS + lane];
            for (int xb = 0; xb < ROWS_TX; xb += 32) {
                for (int x = xb; x < xb + 32; x++) {
                    counts16[(x - xb) * (ROWS_BINS + 1) + lane] = (uint16_t)n;
                    n += cols[(x + span) * ROWS_BINS + lane] - cols[x * ROWS_BINS + lane];
                }
                __syncwarp();
                const int x = xb + lane, v = row[x];
                if (v != CAMBI_NONE && v >= first && v <= last && x0 + p.pad + x < p.w) {
                    const uint16_t *cnt = counts16 + lane * (ROWS_BINS + 1) + (v - CAMBI_DIFFS - b0);
                    const int value = v + CAMBI_DIFFS;
                    const unsigned p0 = cnt[CAMBI_DIFFS];
                    float c = 0.0f;
#pragma unroll
                    for (int d = 0; d < CAMBI_DIFFS; d++) {
                        if (value <= p.tvi[d] && value + d + 1 > p.vlt) {
                            unsigned p1 = cnt[CAMBI_DIFFS + d + 1];
                            unsigned p2 = (v - d - 1 >= p.band_base) ? cnt[CAMBI_DIFFS - d - 1] : 0;
                            unsigned pm = p1 > p2 ? p1 : p2;
                            float val = (float)(p.weight[d] * p0 * pm) * reciprocal_lut[pm + p0];
                            if (val > c) c = val;
                        }
                    }
                    out[(size_t)(y0 + p.pad + r) * p.w + x0 + p.pad + x] = c;
                }
                __syncwarp();
            }
            __syncwarp();
        }
        __syncthreads();
    }
}

extern "C" __global__ void cambi_cvalues_rows(const uint16_t *img, const uint8_t *mask, CambiParams p, float *out)
{
    cvalues_rows<128, 64, 6>(img, mask, p, out);
}

// The same for small pictures, where large tiles would leave most of the GPU idle.
extern "C" __global__ void cambi_cvalues_rows_small(const uint16_t *img, const uint8_t *mask, CambiParams p, float *out)
{
    cvalues_rows<32, 32, 8>(img, mask, p, out);
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

// The frame cropped at (x0, y0) to w x h: luma as float on a 10-bit scale, and luma and both chroma planes as integers
// at the pipeline's bit depth `pd` (8 or 10). An 8-bit frame in a 10-bit pipeline is shifted up by 2, as libvmaf does.
extern "C" __global__ void frame_planes(Frame f, int x0, int y0, int pd, float *y_out, uint16_t *y16_out, uint16_t *u_out,
                                        uint16_t *v_out, int w, int h)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    const unsigned l = sample(f.y, f.ypitch, x + x0, y + y0, f);
    y_out[(size_t)y * w + x] = (float)(l << (10 - f.depth));
    y16_out[(size_t)y * w + x] = l << (pd - f.depth);
    if ((x & 1) || (y & 1)) return;
    const int cx = (x + x0) / 2, cy = (y + y0) / 2, ci = (y / 2) * (w / 2) + x / 2;
    const unsigned u = f.semi ? sample(f.u, f.cpitch, 2 * cx, cy, f) : sample(f.u, f.cpitch, cx, cy, f);
    const unsigned v = f.semi ? sample(f.u, f.cpitch, 2 * cx + 1, cy, f) : sample(f.v, f.cpitch, cx, cy, f);
    u_out[ci] = u << (pd - f.depth);
    v_out[ci] = v << (pd - f.depth);
}

extern "C" __global__ void box_h(const float *in, float *h3, float *h9, int w, int h)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    const float *row = in + (size_t)y * w;
    float s3 = 0, s9 = 0;
    for (int d = -4; d <= 4; d++) {
        float v = row[min(max(x + d, 0), w - 1)];
        s9 += v;
        if (d >= -1 && d <= 1) s3 += v;
    }
    h3[(size_t)y * w + x] = s3;
    h9[(size_t)y * w + x] = s9;
}

// Vertical means of the horizontal sums: the 3 by 3 and 9 by 9 box means.
extern "C" __global__ void box_v(const float *h3, const float *h9, float *b3, float *b9, int w, int h)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    float s3 = 0, s9 = 0;
    for (int d = -4; d <= 4; d++) {
        size_t i = (size_t)min(max(y + d, 0), h - 1) * w + x;
        s9 += h9[i];
        if (d >= -1 && d <= 1) s3 += h3[i];
    }
    b3[(size_t)y * w + x] = s3 / 9.0f;
    b9[(size_t)y * w + x] = s9 / 81.0f;
}

__device__ __forceinline__ float block_sum(float v, float *scratch)
{
    for (int o = 16; o; o >>= 1) v += __shfl_down_sync(0xffffffff, v, o);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    __syncthreads();
    if (lane == 0) scratch[warp] = v;
    __syncthreads();
    float s = 0;
    if (threadIdx.x == 0) for (int i = 0; i < (int)(blockDim.x >> 5); i++) s += scratch[i];
    return s;  // valid in thread 0
}

#define WIN 16
#define WINDOW_STATS 8

// Per 16 by 16 window of one frame pair: fine and mid detail energy of the source and the encode, the energy each shares
// with the other, and the oriented detail of each (the difference of the eigenvalues of the summed structure tensor of
// the 3-pixel-smoothed picture). Also writes the mid detail planes for the texture measure. One block of 256 threads per
// window; out[window * WINDOW_STATS + i] is: 0 source fine, 1 source mid, 2 encode fine, 3 encode mid, 4 shared fine,
// 5 shared mid, 6 source oriented, 7 encode oriented.
extern "C" __global__ void window_stats(const float *yr, const float *b3r, const float *b9r,
                                        const float *yd, const float *b3d, const float *b9d,
                                        int w, int h, float *mid_r, float *mid_d, float *out)
{
    __shared__ float scratch[8];
    const int wx = blockIdx.x, wy = blockIdx.y;
    const int x = wx * WIN + (threadIdx.x & 15), y = wy * WIN + (threadIdx.x >> 4);
    const size_t i = (size_t)y * w + x;
    const float fr = yr[i] - b3r[i], mr = b3r[i] - b9r[i], fd = yd[i] - b3d[i], md = b3d[i] - b9d[i];
    mid_r[i] = mr;
    mid_d[i] = md;
    float gxr = 0, gyr = 0, gxd = 0, gyd = 0;
    if (x > 0 && x < w - 1) { gxr = b3r[i + 1] - b3r[i - 1]; gxd = b3d[i + 1] - b3d[i - 1]; }
    if (y > 0 && y < h - 1) { gyr = b3r[i + w] - b3r[i - w]; gyd = b3d[i + w] - b3d[i - w]; }
    const float vals[12] = {fr * fr, mr * mr, fd * fd, md * md, fr * fd, mr * md,
                            gxr * gxr, gyr * gyr, gxr * gyr, gxd * gxd, gyd * gyd, gxd * gyd};
    float s[12];
    for (int k = 0; k < 12; k++) s[k] = block_sum(vals[k], scratch);
    if (threadIdx.x == 0) {
        float *o = out + ((size_t)wy * gridDim.x + wx) * WINDOW_STATS;
        for (int k = 0; k < 6; k++) o[k] = s[k];
        o[6] = sqrtf((s[6] - s[7]) * (s[6] - s[7]) + 4 * s[8] * s[8]);
        o[7] = sqrtf((s[9] - s[10]) * (s[9] - s[10]) + 4 * s[11] * s[11]);
    }
}

#define PERSIST_STATS 5

// The texture measure for one centre frame, per window: from the source's mid detail planes of the 6 frames before and
// after (`ring` holds 13 planes, centre at slot `centre`), the means A of offsets +-2, 4, 6 and B of +-1, 3, 5, and
// T = (A + B) / 2. out: 0 sum(source mid * T), 1 sum(encode mid * T), 2 sum(A * B), 3 sum(A * A), 4 sum(B * B).
extern "C" __global__ void persist_stats(const float *ring, size_t plane, int centre, const float *mid_d, int w, float *out)
{
    __shared__ float scratch[8];
    const int wx = blockIdx.x, wy = blockIdx.y;
    const int x = wx * WIN + (threadIdx.x & 15), y = wy * WIN + (threadIdx.x >> 4);
    const size_t i = (size_t)y * w + x;
    auto at = [&](int off) { return ring[(size_t)((centre + off + 13) % 13) * plane + i]; };
    const float a = (at(-6) + at(-4) + at(-2) + at(2) + at(4) + at(6)) / 6.0f;
    const float b = (at(-5) + at(-3) + at(-1) + at(1) + at(3) + at(5)) / 6.0f;
    const float t = (a + b) / 2;
    const float vals[5] = {at(0) * t, mid_d[i] * t, a * b, a * a, b * b};
    float s[5];
    for (int k = 0; k < 5; k++) s[k] = block_sum(vals[k], scratch);
    if (threadIdx.x == 0) {
        float *o = out + ((size_t)wy * gridDim.x + wx) * PERSIST_STATS;
        for (int k = 0; k < 5; k++) o[k] = s[k];
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
extern "C" __global__ void color_sums(const uint16_t *u, const uint16_t *v, size_t n, int up, double *sums)
{
    double su = 0, sv = 0, ss = 0;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        const double uu = (double)(u[i] << up), vv = (double)(v[i] << up), du = uu - 512, dv = vv - 512;
        su += uu; sv += vv; ss += sqrt(du * du + dv * dv);
    }
    for (int o = 16; o; o >>= 1) {
        su += __shfl_down_sync(0xffffffff, su, o); sv += __shfl_down_sync(0xffffffff, sv, o); ss += __shfl_down_sync(0xffffffff, ss, o);
    }
    if ((threadIdx.x & 31) == 0) { atomicAdd(&sums[0], su); atomicAdd(&sums[1], sv); atomicAdd(&sums[2], ss); }
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
