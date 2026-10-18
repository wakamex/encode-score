// Accumulators for a run or a segment of it: totals, per-window distributions and the 0 to 100 score.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

// Distribution of a per-window percentage, kept as counts in 0.05-point bins from -500% to 500%, so a whole film's
// windows cost a fixed 160 KB and medians and tenth percentiles come out to within half a bin.
struct Histogram {
    static constexpr double kLo = -500, kStep = 0.05;
    std::vector<uint64_t> bins = std::vector<uint64_t>(20001, 0);
    uint64_t n = 0;
    void add(double pct)
    {
        int b = (int)std::lround((std::clamp(pct, kLo, -kLo) - kLo) / kStep);
        bins[b]++;
        n++;
    }
    double quantile(double q) const
    {
        if (!n) return NAN;
        const double target = q * (double)(n - 1);
        uint64_t seen = 0;
        for (size_t b = 0; b < bins.size(); b++) {
            seen += bins[b];
            if ((double)seen > target) return kLo + (double)b * kStep;
        }
        return -kLo;
    }
    void merge(const Histogram &o) { for (size_t b = 0; b < bins.size(); b++) bins[b] += o.bins[b]; n += o.n; }
};

struct Weights { double fidelity = 0.30, texture = 0.30, banding = 0.25, grain = 0.15; };

struct Accum {
    uint64_t frames = 0;
    double vmaf = 0, vmaf_min = 1e9, psnr[3] = {}, cambi = 0, cambi_source = 0;
    double color_r[3] = {}, color_d[3] = {};
    // amount of detail: whole picture and structured windows, for fine, mid, oriented; source then encode
    double all_r[3] = {}, all_d[3] = {}, st_r[3] = {}, st_d[3] = {};
    Histogram amount[3];
    // shared (matching) detail for fine and mid: the source's energy and the energy both share
    double sh_all_r[2] = {}, sh_all_x[2] = {}, sh_st_r[2] = {}, sh_st_x[2] = {};
    Histogram matching[2];
    // persistent texture
    double own = 0, mine = 0;
    Histogram lasting;

    void merge(const Accum &o)
    {
        frames += o.frames; vmaf += o.vmaf; vmaf_min = std::min(vmaf_min, o.vmaf_min);
        cambi += o.cambi; cambi_source += o.cambi_source; own += o.own; mine += o.mine;
        for (int i = 0; i < 3; i++) {
            psnr[i] += o.psnr[i]; color_r[i] += o.color_r[i]; color_d[i] += o.color_d[i];
            all_r[i] += o.all_r[i]; all_d[i] += o.all_d[i]; st_r[i] += o.st_r[i]; st_d[i] += o.st_d[i];
            amount[i].merge(o.amount[i]);
        }
        for (int i = 0; i < 2; i++) {
            sh_all_r[i] += o.sh_all_r[i]; sh_all_x[i] += o.sh_all_x[i]; sh_st_r[i] += o.sh_st_r[i]; sh_st_x[i] += o.sh_st_x[i];
            matching[i].merge(o.matching[i]);
        }
        lasting.merge(o.lasting);
    }

    static double clamp100(double v) { return std::min(std::max(v, 0.0), 100.0); }
    static double kept(double d, double r) { return r > 0 ? 100.0 * std::sqrt(d / r) : NAN; }

    struct Parts { double fidelity, texture, banding, grain, score; };
    Parts score(const Weights &w = {}) const
    {
        Parts p;
        p.fidelity = clamp100(vmaf / frames);
        p.texture = lasting.n >= 100 ? clamp100(lasting.quantile(0.5)) / 2 + clamp100(lasting.quantile(0.1)) / 2 : NAN;
        p.banding = clamp100(100.0 * (1 - (cambi - cambi_source) / frames / 5.0));
        p.grain = clamp100(100.0 - std::fabs(kept(all_d[0], all_r[0]) - 100.0));
        double total = 0, weight = 0;
        for (auto [v, k] : {std::pair{p.fidelity, w.fidelity}, {p.texture, w.texture}, {p.banding, w.banding}, {p.grain, w.grain}})
            if (!std::isnan(v)) { total += v * k; weight += k; }
        p.score = total / weight;
        return p;
    }
};
