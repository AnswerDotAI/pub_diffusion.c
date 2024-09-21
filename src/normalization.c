#include "normalization.h"
#include <stdlib.h>
#include <math.h>

GroupNorm* groupnorm_create(int num_groups, int num_channels, float epsilon) {
    GroupNorm* gn = (GroupNorm*)malloc(sizeof(GroupNorm));
    if (gn == NULL) return NULL;

    gn->num_groups = num_groups;
    gn->num_channels = num_channels;
    gn->epsilon = epsilon;

    int32_t weight_shape[] = {num_channels};
    gn->weight = tensor_create(1, weight_shape);
    gn->bias = tensor_create(1, weight_shape);

    if (gn->weight == NULL || gn->bias == NULL) {
        groupnorm_free(gn);
        return NULL;
    }

    // Initialize weight to 1 and bias to 0
    for (int i = 0; i < num_channels; i++) {
        int32_t index[] = {i};
        tensor_set(gn->weight, index, 1.0f);
        tensor_set(gn->bias, index, 0.0f);
    }

    return gn;
}

void groupnorm_free(GroupNorm* gn) {
    if (gn == NULL) return;
    tensor_free(gn->weight);
    tensor_free(gn->bias);
    free(gn);
}

Tensor* groupnorm_forward(GroupNorm* gn, Tensor* input) {
    if (input->ndim != 4) {
        // GroupNorm expects 4D input (batch, channels, height, width)
        return NULL;
    }

    int batch_size = input->shape[0];
    int channels = input->shape[1];
    int height = input->shape[2];
    int width = input->shape[3];
    int pixels = height * width;
    int channels_per_group = channels / gn->num_groups;

    Tensor* output = tensor_create(4, input->shape);
    if (output == NULL) return NULL;

    for (int b = 0; b < batch_size; b++) {
        for (int g = 0; g < gn->num_groups; g++) {
            // Compute mean and variance for each group
            float sum = 0.0f, sq_sum = 0.0f;
            for (int c = 0; c < channels_per_group; c++) {
                for (int p = 0; p < pixels; p++) {
                    int32_t index[] = {b, g * channels_per_group + c, p / width, p % width};
                    float val = tensor_get(input, index);
                    sum += val;
                    sq_sum += val * val;
                }
            }
            float mean = sum / (channels_per_group * pixels);
            float var = (sq_sum / (channels_per_group * pixels)) - (mean * mean);
            float std = sqrtf(var + gn->epsilon);

            // Normalize, scale, and shift
            for (int c = 0; c < channels_per_group; c++) {
                int channel = g * channels_per_group + c;
                int32_t w_index[] = {channel};
                float weight = tensor_get(gn->weight, w_index);
                float bias = tensor_get(gn->bias, w_index);

                for (int p = 0; p < pixels; p++) {
                    int32_t index[] = {b, channel, p / width, p % width};
                    float val = tensor_get(input, index);
                    float norm_val = ((val - mean) / std) * weight + bias;
                    tensor_set(output, index, norm_val);
                }
            }
        }
    }

    return output;
}
