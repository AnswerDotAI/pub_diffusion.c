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

typedef struct {
    int normalized_shape;
    float epsilon;
    Tensor* weight;
    Tensor* bias;
} LayerNorm;

LayerNorm* layernorm_create(int normalized_shape, float epsilon);
void layernorm_free(LayerNorm* ln);
Tensor* layernorm_forward(LayerNorm* ln, Tensor* input);


GroupNorm* groupnorm_create(int num_groups, int num_channels, float epsilon);
void groupnorm_free(GroupNorm* gn);
Tensor* groupnorm_forward(GroupNorm* gn, Tensor* input);

// SiLU activation function
float silu(float x);
Tensor* silu_forward(Tensor* input);


#endif // NORMALIZATION_H
