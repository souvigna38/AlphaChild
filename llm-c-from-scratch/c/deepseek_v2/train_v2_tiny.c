/*
 * train_v2_tiny.c — Phase 4 educational CPU trainer (DeepSeek-V2 tiny)
 *
 * Mirrors notebook 15 (train) and 16 (sample) + llm.c train_gpt2.c structure.
 *
 * Modes:
 *   ./bin/train_v2_tiny              forward-only loss demo (random weights)
 *   ./bin/train_v2_tiny -train-head  SGD on tied embedding/lm_head only (demo)
 *   ./bin/train_v2_tiny -train-1layer N  Phase 5: MLA+RMSNorm+wte (1 layer, MoE frozen)
 *   ./bin/train_v2_tiny -sample -ckpt path.bin  greedy generation
 *
 * Full MoE backward in C is Phase 5b — use PyTorch Trainer for all experts.
 */

#include "data.h"
#include "model.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void usage(const char *argv0) {
    fprintf(
        stderr,
        "Usage: %s [-train-head STEPS] [-train-1layer STEPS] [-sample] [-ckpt file] [-data path]\n",
        argv0);
}

static void sample_text(Dsv2Model *m, Dsv2Dataset *ds, int max_new, int T) {
    int prompt[] = {'R', 'O', 'M', 'E', 'O', ':'};
    int plen = 6;
    int *ctx = (int *)malloc((size_t)(T + max_new) * sizeof(int));
    for (int i = 0; i < plen; i++) {
        unsigned char c = (unsigned char)prompt[i];
        if (c < 256 && ds->vocab.stoi[c] >= 0) {
            ctx[i] = ds->vocab.stoi[c];
        } else {
            ctx[i] = 0;
        }
    }
    int len = plen;
    size_t act_bytes = dsv2_model_activations_bytes(&m->cfg, 1, T);
    float *acts = (float *)malloc(act_bytes);
    float *logits = (float *)malloc((size_t)T * (size_t)m->cfg.vocab_size * sizeof(float));

    printf("\n--- sample (greedy) ---\n");
    for (int i = 0; i < plen; i++) {
        putchar(ds->vocab.chars[ctx[i]]);
    }
    for (int n = 0; n < max_new; n++) {
        int cur = len < T ? len : T;
        dsv2_model_forward(logits, ctx + len - cur, m, 1, cur, acts);
        int next = dsv2_sample_greedy_last(logits, 1, cur, m->cfg.vocab_size);
        ctx[len++] = next;
        putchar(ds->vocab.chars[next]);
    }
    printf("\n");
    free(ctx);
    free(acts);
    free(logits);
}

int main(int argc, char **argv) {
    const char *data_path = "../data/tiny_shakespeare.txt";
    const char *ckpt = NULL;
    int train_head_steps = 0;
    int train_1layer_steps = 0;
    int do_sample = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-train-head") == 0 && i + 1 < argc) {
            train_head_steps = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-train-1layer") == 0 && i + 1 < argc) {
            train_1layer_steps = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-sample") == 0) {
            do_sample = 1;
        } else if (strcmp(argv[i], "-ckpt") == 0 && i + 1 < argc) {
            ckpt = argv[++i];
        } else if (strcmp(argv[i], "-data") == 0 && i + 1 < argc) {
            data_path = argv[++i];
        } else {
            usage(argv[0]);
            return 1;
        }
    }

    printf("=== DeepSeek-V2 tiny C trainer (phase 4) ===\n");
    printf("Data: %s\n", data_path);

    Dsv2Dataset ds;
    if (dsv2_load_text_dataset(&ds, data_path) != 0) {
        fprintf(stderr, "Failed to load dataset\n");
        return 1;
    }
    printf("vocab=%d tokens=%d\n", ds.vocab.vocab_size, ds.n_tokens);

    Dsv2ModelConfig cfg = {
        .vocab_size = ds.vocab.vocab_size,
        .block_size = 32,
        .n_layer = train_1layer_steps > 0 ? 1 : 2,
        .n_embd = 64,
        .n_head = 4,
        .kv_lora_rank = 16,
        .n_routed_experts = 4,
        .n_shared_experts = 1,
        .num_experts_per_tok = 2,
        .moe_intermediate = 96,
        .rms_eps = 1e-6f,
    };

    Dsv2Model model;
    if (ckpt && dsv2_model_load_checkpoint(&model, ckpt) == 0) {
        printf("Loaded checkpoint %s\n", ckpt);
    } else {
        if (ckpt) {
            printf("Checkpoint load failed, using random init\n");
        }
        dsv2_model_init(&model, &cfg, 42u);
    }

    if (do_sample) {
        sample_text(&model, &ds, 120, cfg.block_size);
        dsv2_model_free(&model);
        dsv2_dataset_free(&ds);
        return 0;
    }

    int B = 4;
    const int T = cfg.block_size;
    int *idx = (int *)malloc((size_t)B * (size_t)T * sizeof(int));
    int *targets = (int *)malloc((size_t)B * (size_t)T * sizeof(int));
    size_t act_bytes = dsv2_model_activations_bytes(&cfg, B, T);
    float *acts = (float *)malloc(act_bytes);
    float *logits = (float *)malloc((size_t)B * (size_t)T * (size_t)cfg.vocab_size * sizeof(float));
    unsigned int seed = 123u;

    printf("\n--- forward loss (like llm.c train loop body) ---\n");
    for (int step = 0; step < 10; step++) {
        dsv2_get_batch(ds.tokens, ds.n_tokens, idx, targets, B, T, &seed);
        dsv2_model_forward(logits, idx, &model, B, T, acts);
        float loss = dsv2_cross_entropy_loss(logits, targets, B, T, cfg.vocab_size, cfg.n_embd);
        printf("step %d loss %.4f\n", step, loss);
    }

    if (train_head_steps > 0) {
        printf("\n--- head-only SGD (tied wte/lm_head, blocks frozen) ---\n");
        printf("Use PyTorch notebook 15 for full-model training.\n");
        for (int step = 0; step < train_head_steps; step++) {
            dsv2_get_batch(ds.tokens, ds.n_tokens, idx, targets, B, T, &seed);
            dsv2_train_head_step(&model, idx, targets, B, T, 0.05f, acts, logits);
            dsv2_model_forward(logits, idx, &model, B, T, acts);
            float loss = dsv2_cross_entropy_loss(logits, targets, B, T, cfg.vocab_size, cfg.n_embd);
            if (step % 10 == 0 || step == train_head_steps - 1) {
                printf("head step %d loss %.4f\n", step, loss);
            }
        }
    }

    if (train_1layer_steps > 0) {
        printf("\n--- Phase 5: 1-layer C train (MLA backward, MoE frozen) ---\n");
        B = 1;
        size_t grad_bytes = 2 * dsv2_model_param_bytes(&cfg);
        float *grad = (float *)malloc(grad_bytes);
        for (int step = 0; step < train_1layer_steps; step++) {
            dsv2_get_batch(ds.tokens, ds.n_tokens, idx, targets, 1, T, &seed);
            float loss = dsv2_model_train_step_1layer(&model, idx, targets, 1, T, 0.01f, acts, logits, grad);
            if (step % 10 == 0 || step == train_1layer_steps - 1) {
                printf("1layer step %d loss %.4f\n", step, loss);
            }
        }
        free(grad);
    }

    printf("\nNext: python scripts/export_v2_tiny.py && %s -sample -ckpt checkpoints/v2_tiny.bin\n", argv[0]);
    printf("Phase 4 OK\n");

    free(idx);
    free(targets);
    free(acts);
    free(logits);
    dsv2_model_free(&model);
    dsv2_dataset_free(&ds);
    return 0;
}
