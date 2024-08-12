#ifndef __FBANK_H__
#define __FBANK_H__

#define SAMPLE_LEN (16000UL)
#define NUM_FRAMES (99UL)
#define NUM_FILT (32UL)

void fbank(const float input[SAMPLE_LEN], float output[NUM_FRAMES][NUM_FILT]);
void fbank_prep(float *input, size_t len);
char const *fbank_label_idx_to_str(size_t label);

#endif // __FBANK_H__