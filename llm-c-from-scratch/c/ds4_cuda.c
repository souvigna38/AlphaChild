#include "ds4_cuda.h"

#include "rmsnorm.h"

#ifdef DS4_HAVE_CUDA
extern int ds4_cuda_device_count_impl(void);
extern int ds4_cuda_rmsnorm_forward_launch(float *out, const float *inp, const float *weight, int n, int C, float eps);
#endif

int ds4_cuda_device_count(void) {
#ifdef DS4_HAVE_CUDA
    return ds4_cuda_device_count_impl();
#else
    return 0;
#endif
}

int ds4_cuda_available(void) {
    return ds4_cuda_device_count() > 0;
}

void ds4_rmsnorm_forward_cuda(float *out, const float *inp, const float *weight, int n, int C, float eps) {
#ifdef DS4_HAVE_CUDA
    if (n > 0 && C > 0 && ds4_cuda_rmsnorm_forward_launch(out, inp, weight, n, C, eps) == 0) {
        return;
    }
#endif
    ds4_rmsnorm_forward(out, (float *)inp, weight, n, C, eps);
}
