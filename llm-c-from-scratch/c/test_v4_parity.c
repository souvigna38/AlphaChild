#include "deepseek_v4_config.h"
#include "v4_parity.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define PARITY_SEED 12345u
#define PARITY_T 4

static const int g_ids[PARITY_T] = {1, 4, 7, 12};

static int write_fixture(const char *path, const float *logits, int T, int V) {
    FILE *f = fopen(path, "w");
    if (!f) {
        return -1;
    }
    fprintf(f, "# v4 parity golden seed=%u T=%d V=%d\n", PARITY_SEED, T, V);
    for (int t = 0; t < T; t++) {
        for (int v = 0; v < V; v++) {
            fprintf(f, "%.9g\n", logits[(size_t)t * (size_t)V + (size_t)v]);
        }
    }
    fclose(f);
    return 0;
}

static int load_fixture(const char *path, float *logits, int n) {
    FILE *f = fopen(path, "r");
    if (!f) {
        return -1;
    }
    char line[128];
    int i = 0;
    while (fgets(line, sizeof(line), f)) {
        if (line[0] == '#' || line[0] == '\n') {
            continue;
        }
        if (i >= n) {
            fclose(f);
            return -1;
        }
        logits[i++] = (float)strtod(line, NULL);
    }
    fclose(f);
    return (i == n) ? 0 : -1;
}

int main(int argc, char **argv) {
    const char *fixture_path = "../tests/fixtures/v4_parity_logits.txt";
    int write_mode = 0;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--write-fixture") == 0) {
            write_mode = 1;
        } else if (strcmp(argv[i], "--fixture") == 0 && i + 1 < argc) {
            fixture_path = argv[++i];
        }
    }

    DeepSeekV4Config cfg;
    ds4_config_init_tiny(&cfg);
    const int T = PARITY_T;
    const int V = cfg.vocab_size;

    Ds4ModelWeights mw = {0};
    ds4_parity_fill_model(&mw, &cfg, PARITY_SEED);

    float *logits = (float *)calloc((size_t)T * (size_t)V, sizeof(float));
    ds4_parity_forward_logits(logits, g_ids, T, &cfg, &mw);

    if (write_mode) {
        if (write_fixture(fixture_path, logits, T, V) != 0) {
            fprintf(stderr, "failed to write %s\n", fixture_path);
            ds4_parity_free_model(&mw, &cfg);
            free(logits);
            ds4_config_free(&cfg);
            return 1;
        }
        printf("wrote fixture %s (%d logits)\n", fixture_path, T * V);
        ds4_parity_free_model(&mw, &cfg);
        free(logits);
        ds4_config_free(&cfg);
        return 0;
    }

    float *golden = (float *)calloc((size_t)T * (size_t)V, sizeof(float));
    if (load_fixture(fixture_path, golden, T * V) != 0) {
        fprintf(stderr, "missing fixture %s (run with --write-fixture)\n", fixture_path);
        free(golden);
        ds4_parity_free_model(&mw, &cfg);
        free(logits);
        ds4_config_free(&cfg);
        return 1;
    }

    float max_err = 0.0f;
    for (int i = 0; i < T * V; i++) {
        float e = fabsf(logits[i] - golden[i]);
        if (e > max_err) {
            max_err = e;
        }
    }
    printf("parity max_abs_err=%.6g (T=%d V=%d seed=%u)\n", max_err, T, V, PARITY_SEED);
    if (max_err > 1e-5f) {
        fprintf(stderr, "logits differ from golden fixture\n");
        free(golden);
        ds4_parity_free_model(&mw, &cfg);
        free(logits);
        ds4_config_free(&cfg);
        return 1;
    }

    free(golden);
    ds4_parity_free_model(&mw, &cfg);
    free(logits);
    ds4_config_free(&cfg);
    printf("OK — v4 forward parity golden (deterministic weights)\n");
    return 0;
}
