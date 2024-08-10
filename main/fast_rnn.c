#include "fast_rnn.h"

#include <stdio.h>
#include <math.h>
#include <string.h>

#include "rnn0_params.h"
#include "rnn1_params.h"
#include "fc_params.h"

#define EULER_NUMBER_F (2.71828182846f)

static float sigmoidf(float n)
{
    return (1 / (1 + powf(EULER_NUMBER_F, -n)));
}

static void fast_rnn0_process(const float input[9][32], const float hidden[9][64], float output[9][64])
{
    for (size_t t = 0; t < 9; t++)
    {
        for (size_t i = 0; i < 32; i++)
        {
            for (size_t j = 0; j < 64; j++)
            {
                output[t][j] += RNN0_W[j][i] * input[t][i];
            }
        }

        for (size_t i = 0; i < 64; i++)
        {
            for (size_t j = 0; j < 64; j++)
            {
                output[t][j] += RNN0_U[j][i] * hidden[t][i];
            }
        }

        for (size_t j = 0; j < 64; j++)
        {
            output[t][j] += RNN0_BIAS_UPDATE[j];
            output[t][j] = tanhf(output[t][j]);
            output[t][j] *= sigmoidf(RNN0_ALPHA);
            output[t][j] += sigmoidf(RNN0_BETA) * hidden[t][j];
        }
    }
}

void rnn0_process(const float input[99][32], float output[9][64])
{
    float frame[9][32] = {{0.0f}};
    float hidden[9][64] = {{0.0f}};

    for (size_t k = 0; k < BRICK_SIZE; k++)
    {
        for (size_t i = 0; i < 9; i++)
        {
            const float *src = input[k + (i * BRICK_SIZE)];
            for (size_t j = 0; j < 32; j++)
            {
                frame[i][j] = src[j];
            }
        }

        memset(output, 0, sizeof(float) * 9 * 64);
        fast_rnn0_process(frame, hidden, output);
        memcpy(hidden, output, sizeof(float) * 9 * 64);
    }
}

void fast_rnn1_process(const float input[9][64], float output[9][32])
{
    for (size_t t = 0; t < 9; t++)
    {
        for (size_t j = 0; j < 32; j++)
        {
            for (size_t i = 0; i < 64; i++)
            {
                output[t][j] += RNN1_W[j][i] * input[t][i];
            }
        }

        if (t > 0)
        {
            for (size_t i = 0; i < 32; i++)
            {
                for (size_t j = 0; j < 32; j++)
                {
                    output[t][j] += RNN1_U[j][i] * output[t - 1][i];
                }
            }
        }

        for (size_t j = 0; j < 32; j++)
        {
            output[t][j] += RNN1_BIAS_UPDATE[j];
            output[t][j] = tanhf(output[t][j]);
            output[t][j] *= sigmoidf(RNN1_ALPHA);
            output[t][j] += sigmoidf(RNN1_BETA) * (t > 0 ? output[t - 1][j] : 0.0f);
        }
    }
}

void fc_process(const float input[9][32], float output[9][6])
{
    memset(output, 0, 9 * 6 * sizeof(float));

    for (size_t t = 0; t < 9; t++)
    {
        for (size_t j = 0; j < 6; j++)
        {
            for (size_t i = 0; i < 32; i++)
            {
                output[t][j] += input[t][i] * FC_W[j][i];
            }
            output[t][j] += FC_B[j];
        }
    }
}

void get_max_logit(const float input[9][6], float *max_logit, size_t *max_idx)
{
    *max_logit = input[0][0];
    *max_idx = 0;

    for (size_t t = 0; t < 9; t++)
    {
        for (size_t j = 0; j < 6; j++)
        {
            if (input[t][j] > *max_logit)
            {
                *max_logit = input[t][j];
                *max_idx = j;
            }
        }
    }
}

void nn_process(const float input[99][32], float *max_logit, size_t *max_idx)
{
    float output[9][64] = {{0.0f}};
    float output2[9][32] = {{0.0f}};
    float output3[9][6] = {{0.0f}};

    rnn0_process(input, output);
    fast_rnn1_process(output, output2);
    fc_process(output2, output3);
    get_max_logit(output3, max_logit, max_idx);
}