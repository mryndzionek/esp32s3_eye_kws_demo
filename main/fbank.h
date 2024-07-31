#ifndef __FBANK_H__
#define __FBANK_H__

#define SAMPLE_LEN (16000UL)
#define NUM_FRAMES (99UL)
#define NUM_FILT (32UL)

void fbank(float input[SAMPLE_LEN], float output[NUM_FRAMES][NUM_FILT]);
void fbank_norm(float inputoutput[NUM_FILT]);
void fbank_speech_detect(float input[NUM_FRAMES][NUM_FILT], size_t *label, float *logit);
void fbank_print_min_max(float input[NUM_FRAMES][NUM_FILT]);
char const *fbank_label_idx_to_str(size_t label);

#endif // __FBANK_H__