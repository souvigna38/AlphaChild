/* Character-level data loader for train_v2_tiny.c (matches llmc/data.py). */
#ifndef DEEPSEEK_V2_DATA_H
#define DEEPSEEK_V2_DATA_H

typedef struct {
    char *chars;
    int vocab_size;
    int *stoi;
} Dsv2Vocab;

typedef struct {
    int *tokens;
    int n_tokens;
    Dsv2Vocab vocab;
} Dsv2Dataset;

int dsv2_load_text_dataset(Dsv2Dataset *ds, const char *path);
void dsv2_dataset_free(Dsv2Dataset *ds);

/* Sample random batch: idx and targets (B,T), caller frees buffers */
void dsv2_get_batch(const int *tokens, int n_tokens, int *idx, int *targets, int B, int T, unsigned int *seed);

#endif /* DEEPSEEK_V2_DATA_H */
