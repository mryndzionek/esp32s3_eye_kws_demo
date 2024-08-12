#include "fast_rnn.h"

#include <stdio.h>
#include <math.h>
#include <string.h>

#include "fastgrnn_rnn0_params.h"
#include "fastgrnn_rnn1_params.h"
#include "fastgrnn_fc_params.h"

#define EULER_NUMBER_F (2.71828182846f)

// clang-format off

static const float INPUT_MEANS[32] = {
     2.23279933e+00,  4.38483419e+00,  5.66743371e+00,  5.92484381e+00,  6.03141914e+00,  6.20821445e+00,  6.30910170e+00,  6.73419374e+00,
     6.64555068e+00,  6.59554122e+00,  6.71033896e+00,  6.76621793e+00,  6.67212248e+00,  6.78998775e+00,  6.95998504e+00,  6.98039170e+00,
     7.10836355e+00,  7.12655034e+00,  7.13869750e+00,  7.34861176e+00,  7.36918365e+00,  7.40887014e+00,  7.54533199e+00,  7.54548052e+00,
     7.49976708e+00,  7.43216719e+00,  7.36070535e+00,  7.32231995e+00,  7.27774656e+00,  7.23354726e+00,  6.99680107e+00,  5.94456227e+00,
};

static const float INPUT_STDEVS[32] = {
     5.86431247e+00,  6.21624540e+00,  6.49265223e+00,  6.55890561e+00,  6.60957581e+00,  6.76110713e+00,  6.82724513e+00,  6.85262507e+00,
     6.75889975e+00,  6.61042828e+00,  6.50297669e+00,  6.42205669e+00,  6.34636835e+00,  6.29363123e+00,  6.28004656e+00,  6.25042910e+00,
     6.20799454e+00,  6.16919058e+00,  6.18850649e+00,  6.24144682e+00,  6.21020983e+00,  6.18121356e+00,  6.20207482e+00,  6.17127658e+00,
     6.09244152e+00,  6.02534586e+00,  5.97316724e+00,  5.93246433e+00,  5.90074802e+00,  5.88158807e+00,  5.83245015e+00,  5.64465783e+00,
};

// clang-format on

static inline float sigmoidf(float n)
{
    return (1 / (1 + powf(EULER_NUMBER_F, -n)));
}

static void fast_rnn0_process(const float input[9][32], const float hidden[9][64], float output[9][64])
{
    float z;
    float c;

    for (size_t t = 0; t < 9; t++)
    {
        for (size_t j = 0; j < 64; j++)
        {
            for (size_t i = 0; i < 32; i += 4)
            {
                output[t][j] += GRNN0_W[j][i] * input[t][i];
                output[t][j] += GRNN0_W[j][i + 1] * input[t][i + 1];
                output[t][j] += GRNN0_W[j][i + 2] * input[t][i + 2];
                output[t][j] += GRNN0_W[j][i + 3] * input[t][i + 3];
            }
        }

        for (size_t j = 0; j < 64; j++)
        {
            for (size_t i = 0; i < 64; i += 4)
            {
                output[t][j] += GRNN0_U[j][i] * hidden[t][i];
                output[t][j] += GRNN0_U[j][i + 1] * hidden[t][i + 1];
                output[t][j] += GRNN0_U[j][i + 2] * hidden[t][i + 2];
                output[t][j] += GRNN0_U[j][i + 3] * hidden[t][i + 3];
            }
        }

        for (size_t j = 0; j < 64; j++)
        {
            z = output[t][j] + GRNN0_BIAS_GATE[j];
            z = sigmoidf(z);
            c = output[t][j] + GRNN0_BIAS_UPDATE[j];
            c = tanhf(c);

            output[t][j] = z * hidden[t][j] + (sigmoidf(GRNN0_ZETA) * (1.0 - z) + sigmoidf(GRNN0_NU)) * c;
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
    float z;
    float c;

    for (size_t t = 0; t < 9; t++)
    {
        for (size_t j = 0; j < 32; j++)
        {
            for (size_t i = 0; i < 64; i += 4)
            {
                output[t][j] += GRNN1_W[j][i] * input[t][i];
                output[t][j] += GRNN1_W[j][i + 1] * input[t][i + 1];
                output[t][j] += GRNN1_W[j][i + 2] * input[t][i + 2];
                output[t][j] += GRNN1_W[j][i + 3] * input[t][i + 3];
            }
        }

        if (t > 0)
        {
            for (size_t j = 0; j < 32; j++)
            {
                for (size_t i = 0; i < 32; i += 4)
                {
                    output[t][j] += GRNN1_U[j][i] * output[t - 1][i];
                    output[t][j] += GRNN1_U[j][i + 1] * output[t - 1][i + 1];
                    output[t][j] += GRNN1_U[j][i + 2] * output[t - 1][i + 2];
                    output[t][j] += GRNN1_U[j][i + 3] * output[t - 1][i + 3];
                }
            }
        }

        for (size_t j = 0; j < 32; j++)
        {
            z = output[t][j] + GRNN1_BIAS_GATE[j];
            z = sigmoidf(z);
            c = output[t][j] + GRNN1_BIAS_UPDATE[j];
            c = tanhf(c);

            output[t][j] = z * (t > 0 ? output[t - 1][j] : 0.0f) + (sigmoidf(GRNN1_ZETA) * (1.0 - z) + sigmoidf(GRNN1_NU)) * c;
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
            for (size_t i = 0; i < 32; i += 4)
            {
                output[t][j] += input[t][i] * FC_W[j][i];
                output[t][j] += input[t][i + 1] * FC_W[j][i + 1];
                output[t][j] += input[t][i + 2] * FC_W[j][i + 2];
                output[t][j] += input[t][i + 3] * FC_W[j][i + 3];
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

void nn_norm(float input[99][32])
{
    for (size_t i = 0; i < 99; i++)
    {
        for (size_t j = 0; j < 32; j++)
        {
            input[i][j] = (input[i][j] - INPUT_MEANS[j]) / INPUT_STDEVS[j];
        }
    }
}
