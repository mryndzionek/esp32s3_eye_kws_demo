#ifndef __FAST_GRNN__
#define __FAST_GRNN__

#include <stddef.h>

#define BRICK_SIZE (11)

void nn_norm(float input[99][32]);
void rnn0_process(const float input[99][32], float output[9][64]);
void fast_rnn1_process(const float input[9][64], float output[9][32]);
void fc_process(const float input[9][32], float output[9][6]);
void get_max_logit(const float input[9][6], float *max_logit, size_t *max_idx);
void nn_process(const float input[99][32], float *max_logit, size_t *max_idx);

#endif // __FAST_GRNN__
