#include "deepseek_v4_config.h"
#include "v4_parity.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define PARITY_SEED 12345u
#define PARITY_T 4

static const int g_ids[PARITY_T] = {1, 4, 7, 12};

static int write_fixture(const char *path, const float *sums, int nL) {
    FILE *f = fopen(path, "w");
    if (!f) {
        return -1;
    }
    fprintf(f, "# v4 layer stream checksums seed=%u T=%d\n", PARITY_SEED, PARITY_T);
    for (int L = 0; L < nL; L++) {
        fprintf(f, "%.9g\n", sums[L]);
    }
    fclose(f);
    return 0;
}

static int load_fixture(const char *path, float *sums, int nL) {
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
        if (i >= nL) {
            fclose(f);
            return -1;
        }
        sums[i++] = (float)strtod(line, NULL);
    }
    fclose(f);
    return (i == nL) ? 0 : -1;
}

int main(int argc, char **argv) {
    const char *fixture_path = "../tests/fixtures/v4_layer_checksums.txt";
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
    const int nL = cfg.num_hidden_layers;

    Ds4ModelWeights mw = {0};
    ds4_parity_fill_model(&mw, &cfg, PARITY_SEED);

    float *sums = (float *)calloc((size_t)nL, sizeof(float));
    ds4_parity_forward_layer_checksums(sums, nL, g_ids, PARITY_T, &cfg, &mw);

    if (write_mode) {
        if (write_fixture(fixture_path, sums, nL) != 0) {
            fprintf(stderr, "failed to write %s\n", fixture_path);
            free(sums);
            ds4_parity_free_model(&mw, &cfg);
            ds4_config_free(&cfg);
            return 1;
        }
        printf("wrote layer fixture %s (%d layers)\n", fixture_path, nL);
        free(sums);
        ds4_parity_free_model(&mw, &cfg);
        ds4_config_free(&cfg);
        return 0;
    }

    float *golden = (float *)calloc((size_t)nL, sizeof(float));
    if (load_fixture(fixture_path, golden, nL) != 0) {
        fprintf(stderr, "missing fixture %s (run with --write-fixture)\n", fixture_path);
        free(golden);
        free(sums);
        ds4_parity_free_model(&mw, &cfg);
        ds4_config_free(&cfg);
        return 1;
    }

    float max_err = 0.0f;
    for (int L = 0; L < nL; L++) {
        float e = fabsf(sums[L] - golden[L]);
        if (e > max_err) {
            max_err = e;
        }
        printf("  layer %d checksum=%.6g (golden %.6g)\n", L, sums[L], golden[L]);
    }
    if (max_err > 1e-5f) {
        fprintf(stderr, "layer checksums differ from golden\n");
        free(golden);
        free(sums);
        ds4_parity_free_model(&mw, &cfg);
        ds4_config_free(&cfg);
        return 1;
    }

    free(golden);
    free(sums);
    ds4_parity_free_model(&mw, &cfg);
    ds4_config_free(&cfg);
    printf("OK — v4 per-layer stream checksums (seed=%u T=%d)\n", PARITY_SEED, PARITY_T);
    return 0;
}
