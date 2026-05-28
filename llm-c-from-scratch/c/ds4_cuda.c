#include "ds4_cuda.h"

#include "rmsnorm.h"
#include "swiglu.h"

#ifdef DS4_HAVE_CUDA
extern int ds4_cuda_device_count_impl(void);
extern int ds4_cuda_rmsnorm_forward_launch(float *out, const float *inp, const float *weight, int n, int C, float eps);
extern int ds4_cuda_swiglu_forward_launch(
    float *out,
    const float *x,
    int hidden,
    int intermediate,
    const float *gate_up_w,
    const float *down_w,
    float swiglu_limit);
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

void ds4_swiglu_forward_cuda(
    float *out,
    const float *x,
    int hidden,
    int intermediate,
    const float *gate_up_w,
    const float *down_w,
    float swiglu_limit) {
#ifdef DS4_HAVE_CUDA
    if (hidden > 0 && intermediate > 0 &&
        ds4_cuda_swiglu_forward_launch(out, x, hidden, intermediate, gate_up_w, down_w, swiglu_limit) == 0) {
        return;
    }
#endif
    ds4_swiglu_forward(out, x, hidden, intermediate, gate_up_w, down_w, swiglu_limit);
}
