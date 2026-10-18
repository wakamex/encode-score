#include "cache.hpp"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
extern "C" {
#include <libavutil/avutil.h>
}

// Raised whenever a change to the CAMBI computation changes its scores, so scores cached before it are not used.
static constexpr int kVersion = 1;
static constexpr size_t kSample = 1 << 20, kFlush = 240;

static uint64_t fnv1a(uint64_t h, const void *data, size_t n)
{
    for (size_t i = 0; i < n; i++) h = (h ^ ((const unsigned char *)data)[i]) * 0x100000001b3ull;
    return h;
}

CambiCache::CambiCache(const std::string &path, int x0, int y0, int w, int h, int depth)
{
    const char *xdg = std::getenv("XDG_CACHE_HOME"), *home = std::getenv("HOME");
    std::string dir = xdg && *xdg ? xdg : home ? std::string(home) + "/.cache" : "";
    if (dir.empty()) return;
    mkdir(dir.c_str(), 0755);
    dir += "/encode-score";
    mkdir(dir.c_str(), 0755);

    FILE *f = std::fopen(path.c_str(), "rb");
    if (!f) return;
    std::vector<unsigned char> buf(kSample);
    std::fseek(f, 0, SEEK_END);
    const long size = std::ftell(f);
    char key[96];
    const int len = std::snprintf(key, sizeof key, "cambi %d %ld %d %d %d %d %d", kVersion, size, x0, y0, w, h, depth);
    uint64_t hash = fnv1a(0xcbf29ce484222325ull, key, (size_t)len);
    for (long at : {0L, std::max(size - (long)kSample, 0L)}) {
        std::fseek(f, at, SEEK_SET);
        hash = fnv1a(hash, buf.data(), std::fread(buf.data(), 1, kSample, f));
    }
    std::fclose(f);
    char name[40];
    std::snprintf(name, sizeof name, "/%016llx.cambi", (unsigned long long)hash);
    file_ = dir + name;

    fd_ = open(file_.c_str(), O_RDWR | O_CREAT | O_APPEND, 0644);
    if (fd_ < 0) return;
    struct stat st;
    fstat(fd_, &st);
    // a run stopped in the middle of a write leaves a partial record at the end, which is dropped
    const off_t whole = st.st_size / (off_t)sizeof(Record) * (off_t)sizeof(Record);
    if (whole != st.st_size && ftruncate(fd_, whole) != 0) { close(fd_); fd_ = -1; return; }
    std::vector<Record> records(whole / sizeof(Record));
    if (pread(fd_, records.data(), whole, 0) == whole)
        for (const Record &r : records) scores_[r.pts] = r.score;
}

CambiCache::~CambiCache()
{
    flush();
    if (fd_ >= 0) close(fd_);
}

std::optional<double> CambiCache::find(int64_t pts) const
{
    auto it = pts == AV_NOPTS_VALUE ? scores_.end() : scores_.find(pts);
    if (it == scores_.end()) return std::nullopt;
    return it->second;
}

void CambiCache::add(int64_t pts, double score)
{
    if (fd_ < 0 || pts == AV_NOPTS_VALUE) return;
    scores_[pts] = score;
    pending_.push_back({pts, score});
    if (pending_.size() >= kFlush) flush();
}

void CambiCache::flush()
{
    if (fd_ >= 0 && !pending_.empty() && write(fd_, pending_.data(), pending_.size() * sizeof(Record)) < 0) {
        close(fd_);
        fd_ = -1;
    }
    pending_.clear();
}
