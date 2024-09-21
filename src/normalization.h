#ifndef NORMALIZATION_H
#define NORMALIZATION_H

#include "tensor.h"

typedef struct {
    int num_groups;
    int num_channels;
    float epsilon;
    Tensor* weight;
    Tensor* bias;
} GroupNorm;

GroupNorm* groupnorm_create(int num_groups, int num_channels, float epsilon);
void groupnorm_free(GroupNorm* gn);
Tensor* groupnorm_forward(GroupNorm* gn, Tensor* input);

#endif // NORMALIZATION_H
