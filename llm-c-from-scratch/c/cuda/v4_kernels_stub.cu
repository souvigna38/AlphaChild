#include <cstdio>

/* Phase 12 placeholder: real kernels will fuse RMSNorm, SwiGLU MoE, sliding attn. */
extern "C" int ds4_cuda_stub_version(void) {
    return 12;
}

int main(void) {
    std::printf("v4 CUDA stub OK (version %d)\n", ds4_cuda_stub_version());
    return 0;
}
