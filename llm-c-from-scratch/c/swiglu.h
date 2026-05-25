/* SwiGLU FFN (DeepSeek-V4) — gate/up clamp + down proj. See nano SwiGLUExpert. */
#ifndef DEEPSEEK_V4_SWIGLU_H
#define DEEPSEEK_V4_SWIGLU_H

void ds4_swiglu_forward(
    float *out,
    const float *x,
    int hidden,
    int intermediate,
    const float *gate_up_w,
    const float *down_w,
    float swiglu_limit);

#endif /* DEEPSEEK_V4_SWIGLU_H */
