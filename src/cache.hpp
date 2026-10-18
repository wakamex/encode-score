// The source's CAMBI per frame, kept on disk between runs. It depends on the source alone, not on the encodes scored
// against it, so a later run against the same source reads it instead of computing it again.
#pragma once
#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

class CambiCache {
public:
    // The scores of `path` cropped to `w` by `h` at (`x0`, `y0`) and scored at `depth` bits. The file is identified by
    // its size and its first and last mebibyte, so a renamed or moved source keeps its scores.
    CambiCache(const std::string &path, int x0, int y0, int w, int h, int depth);
    ~CambiCache();
    CambiCache(const CambiCache &) = delete;
    // The score of the frame with presentation timestamp `pts`, if an earlier run computed it.
    std::optional<double> find(int64_t pts) const;
    void add(int64_t pts, double score);
    const std::string &file() const { return file_; }

private:
    struct Record { int64_t pts; double score; };
    void flush();
    std::string file_;
    std::unordered_map<int64_t, double> scores_;
    std::vector<Record> pending_;
    int fd_ = -1;
};
