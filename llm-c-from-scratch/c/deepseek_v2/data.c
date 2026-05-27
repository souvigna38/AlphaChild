#include "data.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int cmp_char(const void *a, const void *b) {
    return (*(const char *)a - *(const char *)b);
}

int dsv2_load_text_dataset(Dsv2Dataset *ds, const char *path) {
    memset(ds, 0, sizeof(*ds));
    FILE *f = fopen(path, "r");
    if (!f) {
        return -1;
    }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *text = (char *)malloc((size_t)sz + 1);
    if (!text) {
        fclose(f);
        return -1;
    }
    if (fread(text, 1, (size_t)sz, f) != (size_t)sz) {
        free(text);
        fclose(f);
        return -1;
    }
    text[sz] = '\0';
    fclose(f);

    int *seen = (int *)calloc(256, sizeof(int));
    int n_unique = 0;
    for (long i = 0; i < sz; i++) {
        unsigned char c = (unsigned char)text[i];
        if (!seen[c]) {
            seen[c] = 1;
            n_unique++;
        }
    }

    ds->vocab.chars = (char *)malloc((size_t)n_unique);
    ds->vocab.stoi = (int *)calloc(256, sizeof(int));
    ds->vocab.vocab_size = n_unique;

    int k = 0;
    for (int c = 0; c < 256; c++) {
        if (seen[c]) {
            ds->vocab.chars[k] = (char)c;
            ds->vocab.stoi[c] = k;
            k++;
        }
    }
    qsort(ds->vocab.chars, (size_t)n_unique, 1, cmp_char);
    for (int i = 0; i < n_unique; i++) {
        ds->vocab.stoi[(unsigned char)ds->vocab.chars[i]] = i;
    }
    free(seen);

    ds->n_tokens = (int)sz;
    ds->tokens = (int *)malloc((size_t)ds->n_tokens * sizeof(int));
    for (int i = 0; i < ds->n_tokens; i++) {
        ds->tokens[i] = ds->vocab.stoi[(unsigned char)text[i]];
    }
    free(text);
    return 0;
}

void dsv2_dataset_free(Dsv2Dataset *ds) {
    free(ds->tokens);
    free(ds->vocab.chars);
    free(ds->vocab.stoi);
    memset(ds, 0, sizeof(*ds));
}

void dsv2_get_batch(const int *tokens, int n_tokens, int *idx, int *targets, int B, int T, unsigned int *seed) {
    for (int b = 0; b < B; b++) {
        *seed = *seed * 1103515245u + 12345u;
        int max_start = n_tokens - T - 1;
        int start = (int)((*seed >> 16) % (unsigned int)(max_start > 0 ? max_start : 1));
        for (int t = 0; t < T; t++) {
            idx[b * T + t] = tokens[start + t];
            targets[b * T + t] = tokens[start + t + 1];
        }
    }
}
