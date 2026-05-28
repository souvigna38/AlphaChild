/*
 * train_v2_tiny.c — Phase 4–6 educational CPU trainer (DeepSeek-V2 tiny)
 *
 * Mirrors notebook 15 (train) and 16 (sample) + llm.c train_gpt2.c structure.
 *
 * Modes:
 *   ./bin/train_v2_tiny              forward-only loss demo (random weights)
 *   ./bin/train_v2_tiny -train-head  SGD on tied embedding/lm_head only (demo)
 *   ./bin/train_v2_tiny -train-1layer N  Phase 5: MLA+RMSNorm+wte (1 layer, MoE frozen)
 *   ./bin/train_v2_tiny -train-full N    Phase 5b: SGD, all layers (B=1 default)
 *   ./bin/train_v2_tiny -train-adam N [-batch B] [-save ckpt.bin]  Phase 6: AdamW + B>1
 *   ./bin/train_v2_tiny -sample -ckpt path.bin  greedy generation
 */

#include "adamw.h"
#include "data.h"
#include "model.h"

#include "block.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void usage(const char *argv0) {
    fprintf(
        stderr,
        "Usage: %s [-train-head STEPS] [-train-1layer STEPS] [-train-full STEPS]\n"
        "       [-train-adam STEPS] [-batch B] [-lr LR] [-save ckpt.bin]\n"
        "       [-sample] [-ckpt file] [-data path]\n",
        argv0);
}

static void sample_text(Dsv2Model *m, Dsv2Dataset *ds, int max_new, int T) {
    int prompt[] = {'R', 'O', 'M', 'E', 'O', ':'};
    int plen = 6;
    int *ctx = (int *)malloc((size_t)(T + max_new) * sizeof(int));
    for (int i = 0; i < plen; i++) {
        unsigned char c = (unsigned char)prompt[i];
        if (ds->vocab.stoi[c] >= 0) {
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

static void run_full_train(
    Dsv2Model *model,
    Dsv2Dataset *ds,
    Dsv2ModelConfig *cfg,
    int B,
    int T,
    int steps,
    float lr,
    float grad_clip,
    Dsv2AdamW *adam,
    unsigned int *seed) {
    int *idx = (int *)malloc((size_t)B * (size_t)T * sizeof(int));
    int *targets = (int *)malloc((size_t)B * (size_t)T * sizeof(int));
    size_t work_bytes = dsv2_model_train_working_bytes(cfg, B, T);
    char *work = (char *)malloc(work_bytes);
    float *grad = dsv2_train_grad_ptr(work, cfg, B, T);
    Dsv2TrainConfig tc = {.lr = lr, .grad_clip = grad_clip, .adam = adam};

    for (int step = 0; step < steps; step++) {
        dsv2_get_batch(ds->tokens, ds->n_tokens, idx, targets, B, T, seed);
        float loss = dsv2_model_train_step_full(model, idx, targets, B, T, &tc, (float *)work, NULL, grad, NULL);
        if (step % 5 == 0 || step == steps - 1) {
            printf("train step %d loss %.4f\n", step, loss);
        }
    }

    free(work);
    free(idx);
    free(targets);
}

static Dsv2ModelConfig default_cfg(int vocab_size, int n_layer) {
    Dsv2ModelConfig cfg = {
        .vocab_size = vocab_size,
        .block_size = 32,
        .n_layer = n_layer,
        .n_embd = 64,
        .n_head = 4,
        .kv_lora_rank = 16,
        .n_routed_experts = 4,
        .n_shared_experts = 1,
        .num_experts_per_tok = 2,
        .moe_intermediate = 96,
        .rms_eps = 1e-6f,
    };
    return cfg;
}

int main(int argc, char **argv) {
    const char *data_path = "../data/tiny_shakespeare.txt";
    const char *ckpt = NULL;
    const char *save_path = NULL;
    int train_head_steps = 0;
    int train_1layer_steps = 0;
    int train_full_steps = 0;
    int train_adam_steps = 0;
    int batch_size = 1;
    float lr = 3e-3f;
    int do_sample = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-train-head") == 0 && i + 1 < argc) {
            train_head_steps = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-train-1layer") == 0 && i + 1 < argc) {
            train_1layer_steps = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-train-full") == 0 && i + 1 < argc) {
            train_full_steps = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-train-adam") == 0 && i + 1 < argc) {
            train_adam_steps = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-batch") == 0 && i + 1 < argc) {
            batch_size = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-lr") == 0 && i + 1 < argc) {
            lr = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "-save") == 0 && i + 1 < argc) {
            save_path = argv[++i];
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

    if (batch_size < 1) {
        batch_size = 1;
    }

    printf("=== DeepSeek-V2 tiny C trainer (phase 4–6) ===\n");
    printf("Data: %s\n", data_path);

    Dsv2Dataset ds;
    if (dsv2_load_text_dataset(&ds, data_path) != 0) {
        fprintf(stderr, "Failed to load dataset\n");
        return 1;
    }
    printf("vocab=%d tokens=%d\n", ds.vocab.vocab_size, ds.n_tokens);

    int n_layer = (train_1layer_steps > 0) ? 1 : 2;
    Dsv2ModelConfig cfg = default_cfg(ds.vocab.vocab_size, n_layer);

    Dsv2Model model = {0};
    if (ckpt && dsv2_model_load_checkpoint(&model, ckpt) == 0) {
        printf("Loaded checkpoint %s\n", ckpt);
        cfg = model.cfg;
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

    if (train_full_steps > 0) {
        printf("\n--- Phase 5b: full C train (SGD, MLA + MoE backward, B=1) ---\n");
        cfg.n_layer = 2;
        dsv2_model_free(&model);
        dsv2_model_init(&model, &cfg, 42u);
        run_full_train(&model, &ds, &cfg, 1, T, train_full_steps, 0.001f, 1.0f, NULL, &seed);
        printf("Phase 5b OK\n");
    }

    if (train_adam_steps > 0) {
        printf("\n--- Phase 6: AdamW full train (B=%d, lr=%g) ---\n", batch_size, lr);
        cfg.n_layer = 2;
        dsv2_model_free(&model);
        dsv2_model_init(&model, &cfg, 42u);
        Dsv2AdamW adam;
        size_t np = dsv2_model_param_bytes(&cfg) / sizeof(float);
        dsv2_adamw_init(&adam, np, lr, 0.01f);
        run_full_train(&model, &ds, &cfg, batch_size, T, train_adam_steps, lr, 1.0f, &adam, &seed);
        if (save_path) {
            if (dsv2_model_save_checkpoint(&model, save_path) == 0) {
                printf("Saved checkpoint %s\n", save_path);
            } else {
                fprintf(stderr, "Failed to save %s\n", save_path);
            }
        }
        dsv2_adamw_free(&adam);
        printf("Phase 6 OK — sample: %s -sample -ckpt %s\n", argv[0], save_path ? save_path : "(use -save)");
    }

    if (!train_adam_steps) {
        printf("\nNext: %s -train-adam 40 -batch 4 -save ../checkpoints/v2_c.bin\n", argv[0]);
        printf("Or: python scripts/export_v2_tiny.py && %s -sample -ckpt ../checkpoints/v2_tiny.bin\n", argv[0]);
    }

    free(idx);
    free(targets);
    free(acts);
    free(logits);
    dsv2_model_free(&model);
    dsv2_dataset_free(&ds);
    return 0;
}
