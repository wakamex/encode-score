#include "gpu.hpp"

// the kernels, compiled by the build into a C array (xxd -i kernels.fatbin)
#include "kernels_fatbin.h"

void Kernels::load() { cu_check(cuModuleLoadData(&module_, kernels_fatbin), "loading kernels"); }

CUfunction Kernels::get(const char *name) const
{
    CUfunction f;
    cu_check(cuModuleGetFunction(&f, module_, name), name);
    return f;
}

void Kernels::launch(CUfunction f, unsigned w, unsigned h, void **args, CUstream stream)
{
    const unsigned bx = 32, by = 8;
    cu_check(cuLaunchKernel(f, (w + bx - 1) / bx, (h + by - 1) / by, 1, bx, by, 1, 0, stream, args, nullptr), "kernel launch");
}
