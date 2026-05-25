#ifndef ROPE_H
#define ROPE_H

#include "deepseek_v4_config.h"

/* Partial interleaved RoPE (nano apply_partial_rope). */

void ds4_rope_cos_sin_buffer(float *cos, float *sin, int T, int rope_dim, float theta);

/* Apply RoPE to trailing rope_dim slice of vector length head_dim. cos/sin length rope_dim/2. */
void ds4_apply_partial_rope_vec(float *hd, int head_dim, int rope_dim, const float *cos_t, const float *sin_t);

/* Per-token cos/sin rows: cos/sin are (T, rope_dim/2). hd is one head vector. */
void ds4_apply_partial_rope_vec_t(float *hd, int head_dim, int rope_dim, const float *cos_row, const float *sin_row);

/* Output projection path uses -sin (nano forward). */
void ds4_apply_partial_rope_vec_t_neg_sin(float *hd, int head_dim, int rope_dim, const float *cos_row, const float *sin_row);

#endif /* ROPE_H */
