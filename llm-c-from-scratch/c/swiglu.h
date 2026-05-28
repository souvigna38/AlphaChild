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

void ds4_swiglu_forward_save(
    float *out,
    const float *x,
    int hidden,
    int intermediate,
    const float *gate_up_w,
    const float *down_w,
    float swiglu_limit,
    float *gate_up_act,
    float *h_act);

/* Saves gate/up pre-clamp in gate_up_act (2*I) and h (I); accumulates into dx, dgu, dd. */
void ds4_swiglu_backward(
    float *dx,
    float *d_gate_up_w,
    float *d_down_w,
    const float *dout,
    const float *x,
    int hidden,
    int intermediate,
    const float *gate_up_w,
    const float *down_w,
    const float *gate_up_act,
    const float *h_act,
    float swiglu_limit);

#endif /* DEEPSEEK_V4_SWIGLU_H */
