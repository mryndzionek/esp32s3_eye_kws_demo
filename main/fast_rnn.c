#include "sha_rnn_intf.h"

#include <stdio.h>
#include <math.h>
#include <string.h>

#include "fastrnn_rnn0_params.h"
#include "fastrnn_rnn1_params.h"
#include "fastrnn_fc_params.h"

#define EULER_NUMBER_F (2.71828182846f)

// clang-format off

static const float INPUT_MEANS[32] = {
    2.232799e+00, 4.384834e+00, 5.667434e+00, 5.924844e+00, 6.031419e+00, 6.208214e+00, 6.309102e+00, 6.734194e+00,
    6.645551e+00, 6.595541e+00, 6.710339e+00, 6.766218e+00, 6.672122e+00, 6.789988e+00, 6.959985e+00, 6.980392e+00,
    7.108364e+00, 7.126550e+00, 7.138697e+00, 7.348612e+00, 7.369184e+00, 7.408870e+00, 7.545332e+00, 7.545481e+00,
    7.499767e+00, 7.432167e+00, 7.360705e+00, 7.322320e+00, 7.277747e+00, 7.233547e+00, 6.996801e+00, 5.944562e+00,
};

static const float INPUT_STDEVS[32] = {
    5.864312e+00, 6.216245e+00, 6.492652e+00, 6.558906e+00, 6.609576e+00, 6.761107e+00, 6.827245e+00, 6.852625e+00,
    6.758900e+00, 6.610428e+00, 6.502977e+00, 6.422057e+00, 6.346368e+00, 6.293631e+00, 6.280047e+00, 6.250429e+00,
    6.207995e+00, 6.169191e+00, 6.188506e+00, 6.241447e+00, 6.210210e+00, 6.181214e+00, 6.202075e+00, 6.171277e+00,
    6.092442e+00, 6.025346e+00, 5.973167e+00, 5.932464e+00, 5.900748e+00, 5.881588e+00, 5.832450e+00, 5.644658e+00,
};

// clang-format on

static inline float sigmoidf(float n)
{
    return (1 / (1 + powf(EULER_NUMBER_F, -n)));
}

static void rnn0_process(const float input[9][32], const float hidden[9][64], float output[9][64])
{
    for (size_t t = 0; t < 9; t++)
    {
        for (size_t j = 0; j < 64; j++)
        {
            for (size_t i = 0; i < 32; i += 4)
            {
                output[t][j] += RNN0_W[j][i] * input[t][i];
                output[t][j] += RNN0_W[j][i + 1] * input[t][i + 1];
                output[t][j] += RNN0_W[j][i + 2] * input[t][i + 2];
                output[t][j] += RNN0_W[j][i + 3] * input[t][i + 3];
            }
        }

        for (size_t j = 0; j < 64; j++)
        {
            for (size_t i = 0; i < 64; i += 4)
            {
                output[t][j] += RNN0_U[j][i] * hidden[t][i];
                output[t][j] += RNN0_U[j][i + 1] * hidden[t][i + 1];
                output[t][j] += RNN0_U[j][i + 2] * hidden[t][i + 2];
                output[t][j] += RNN0_U[j][i + 3] * hidden[t][i + 3];
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

void sha_rnn_rnn0_process(const sha_rnn_input_t input, sha_rnn_rnn1_input_t output)
{
    float frame[9][32] = {{0.0f}};
    float hidden[9][64] = {{0.0f}};

    for (size_t k = 0; k < SHARNN_BRICK_SIZE; k++)
    {
        for (size_t i = 0; i < 9; i++)
        {
            const float *src = input[k + (i * SHARNN_BRICK_SIZE)];
            for (size_t j = 0; j < 32; j++)
            {
                frame[i][j] = src[j];
            }
        }

        memset(output, 0, sizeof(float) * 9 * 64);
        rnn0_process(frame, hidden, output);
        memcpy(hidden, output, sizeof(float) * 9 * 64);
    }
}

void sha_rnn_rnn1_process(const sha_rnn_rnn1_input_t input, sha_rnn_fc_input_t output)
{
    for (size_t t = 0; t < 9; t++)
    {
        for (size_t j = 0; j < 32; j++)
        {
            for (size_t i = 0; i < 64; i += 4)
            {
                output[t][j] += RNN1_W[j][i] * input[t][i];
                output[t][j] += RNN1_W[j][i + 1] * input[t][i + 1];
                output[t][j] += RNN1_W[j][i + 2] * input[t][i + 2];
                output[t][j] += RNN1_W[j][i + 3] * input[t][i + 3];
            }
        }

        if (t > 0)
        {
            for (size_t j = 0; j < 32; j++)
            {
                for (size_t i = 0; i < 32; i += 4)
                {
                    output[t][j] += RNN1_U[j][i] * output[t - 1][i];
                    output[t][j] += RNN1_U[j][i + 1] * output[t - 1][i + 1];
                    output[t][j] += RNN1_U[j][i + 2] * output[t - 1][i + 2];
                    output[t][j] += RNN1_U[j][i + 3] * output[t - 1][i + 3];
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

void sha_rnn_fc_process(const sha_rnn_fc_input_t input, sha_rnn_output_t output)
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

void sha_rnn_get_max_logit(const sha_rnn_output_t input, float *max_logit, size_t *max_idx)
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

void sha_rnn_process(const sha_rnn_input_t input, float *max_logit, size_t *max_idx)
{
    float output[9][64] = {{0.0f}};
    float output2[9][32] = {{0.0f}};
    float output3[9][6] = {{0.0f}};

    sha_rnn_rnn0_process(input, output);
    sha_rnn_rnn1_process(output, output2);
    sha_rnn_fc_process(output2, output3);
    sha_rnn_get_max_logit(output3, max_logit, max_idx);
}

void sha_rnn_norm(sha_rnn_input_t input)
{
    for (size_t i = 0; i < 99; i++)
    {
        for (size_t j = 0; j < 32; j++)
        {
            input[i][j] = (input[i][j] - INPUT_MEANS[j]) / INPUT_STDEVS[j];
        }
    }
}
