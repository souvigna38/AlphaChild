/*
 * DeepSeekMoE forward — heavily commented (notebook 13).
 *
 * Reading order:
 *   1) llmc/deepseek_v2.py :: DeepSeekMoE.forward
 *   2) vendor/llm.c/train_gpt2.c :: gelu_forward + matmul MLP (GPT-2 has NO router)
 *   3) make test_moe
 */
#include "moe.h"

#include <math.h>
#include <stddef.h>
#include <string.h>

/* y = W @ x, W (out_dim, in_dim) row-major */
static void linear(const float *W, const float *x, float *y, int out_dim, int in_dim) {
    for (int o = 0; o < out_dim; o++) {
        float sum = 0.0f;
        const float *row = W + (size_t)o * (size_t)in_dim;
        for (int i = 0; i < in_dim; i++) {
            sum += row[i] * x[i];
        }
        y[o] = sum;
    }
}

/* SiLU (Swish): x * sigmoid(x) — PyTorch F.silu */
static float silu(float x) {
    return x / (1.0f + expf(-x));
}

/*
 * One SwiGLU expert (DeepSeek/Llama style):
 *   hidden = silu(x @ W1^T) * (x @ W3^T)
 *   out    = hidden @ W2^T
 */
static void swiglu_expert(
    float *out,
    const float *x,
    int C,
    int intermediate,
    const float *w1,
    const float *w2,
    const float *w3,
    float *hidden_silu,
    float *hidden_gate) {
    linear(w1, x, hidden_silu, intermediate, C);
    linear(w3, x, hidden_gate, intermediate, C);
    for (int i = 0; i < intermediate; i++) {
        hidden_silu[i] = silu(hidden_silu[i]) * hidden_gate[i];
    }
    linear(w2, hidden_silu, out, C, intermediate);
}

/* Stable softmax over n logits */
static void softmax(const float *logits, float *probs, int n) {
    float maxv = logits[0];
    for (int i = 1; i < n; i++) {
        if (logits[i] > maxv) {
            maxv = logits[i];
        }
    }
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        probs[i] = expf(logits[i] - maxv);
        sum += probs[i];
    }
    float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
    for (int i = 0; i < n; i++) {
        probs[i] *= inv;
    }
}

/* Top-k by partial selection sort (E is small, e.g. 8) */
static void topk(
    const float *probs,
    int *out_idx,
    float *out_w,
    int n,
    int k) {
    int used[32];
    if (n > 32) {
        n = 32;
    }
    for (int i = 0; i < n; i++) {
        used[i] = 0;
    }
    for (int t = 0; t < k; t++) {
        int best = -1;
        float bestv = -1.0f;
        for (int i = 0; i < n; i++) {
            if (!used[i] && probs[i] > bestv) {
                bestv = probs[i];
                best = i;
            }
        }
        out_idx[t] = best;
        out_w[t] = bestv;
        used[best] = 1;
    }
    float s = 0.0f;
    for (int t = 0; t < k; t++) {
        s += out_w[t];
    }
    if (s > 0.0f) {
        for (int t = 0; t < k; t++) {
            out_w[t] /= s;
        }
    }
}

size_t dsv2_moe_scratch_bytes(const Dsv2MoeConfig *cfg, int T) {
    (void)T;
    int C = cfg->n_embd;
    int E = cfg->n_routed_experts;
    int I = cfg->moe_intermediate;
    /* Per token: logits, probs, two SwiGLU hidden buffers, acc, tmp */
    size_t per_token = (size_t)E + (size_t)E + (size_t)I * 2 + (size_t)C * 2;
    return per_token * sizeof(float);
}

void dsv2_moe_forward(
    float *out,
    const float *x,
    const Dsv2MoeConfig *cfg,
    int T,
    const float *gate_w,
    const float *expert_w1,
    const float *expert_w2,
    const float *expert_w3,
    const float *shared_w1,
    const float *shared_w2,
    const float *shared_w3,
    int *last_top_expert,
    float *last_top_weight,
    float *scratch) {
    const int C = cfg->n_embd;
    const int E = cfg->n_routed_experts;
    const int I = cfg->moe_intermediate;
    const int k = cfg->num_experts_per_tok;
    const int n_shared = cfg->n_shared_experts;

    float *logits = scratch;
    float *probs = logits + E;
    float *hid_a = probs + E;
    float *hid_b = hid_a + I;
    float *tmp = hid_b + I;

    /* top-k indices on stack (E <= 32 in our tiny configs) */
    int topi[32];
    float topw[32];
    if (E > 32 || k > 32) {
        return;
    }

    const size_t expert_w1_stride = (size_t)I * (size_t)C;
    const size_t expert_w2_stride = (size_t)C * (size_t)I;
    const size_t expert_w3_stride = expert_w1_stride;

    for (int t = 0; t < T; t++) {
        const float *xt = x + (size_t)t * (size_t)C;
        float *yt = out + (size_t)t * (size_t)C;

        /* Step 1 — router: logits = gate @ x, then softmax (Python: gate + softmax) */
        linear(gate_w, xt, logits, E, C);
        softmax(logits, probs, E);

        /* Step 2 — top-k experts + renormalize weights (Python: topk + divide by sum) */
        topk(probs, topi, topw, E, k);
        if (last_top_expert != NULL && last_top_weight != NULL) {
            for (int j = 0; j < k; j++) {
                last_top_expert[t * k + j] = topi[j];
                last_top_weight[t * k + j] = topw[j];
            }
        }

        /* Step 3 — weighted sum of routed expert outputs */
        memset(yt, 0, (size_t)C * sizeof(float));
        for (int j = 0; j < k; j++) {
            int e = topi[j];
            const float *w1 = expert_w1 + (size_t)e * expert_w1_stride;
            const float *w2 = expert_w2 + (size_t)e * expert_w2_stride;
            const float *w3 = expert_w3 + (size_t)e * expert_w3_stride;
            swiglu_expert(tmp, xt, C, I, w1, w2, w3, hid_a, hid_b);
            float w = topw[j];
            for (int c = 0; c < C; c++) {
                yt[c] += w * tmp[c];
            }
        }

        /* Step 4 — shared expert(s) always run (Python: out = out + shared(flat)) */
        for (int s = 0; s < n_shared; s++) {
            const float *w1 = shared_w1 + (size_t)s * expert_w1_stride;
            const float *w2 = shared_w2 + (size_t)s * expert_w2_stride;
            const float *w3 = shared_w3 + (size_t)s * expert_w3_stride;
            swiglu_expert(tmp, xt, C, I, w1, w2, w3, hid_a, hid_b);
            for (int c = 0; c < C; c++) {
                yt[c] += tmp[c];
            }
        }
    }
}
