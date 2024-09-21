
#ifndef TENSOR_H
#define TENSOR_H

#include <stdint.h>

typedef struct {
    float* data;
    int32_t* shape;
    int32_t ndim;
    int32_t size;
} Tensor;

Tensor* tensor_create(int32_t ndim, int32_t* shape);
void tensor_free(Tensor* tensor);
float tensor_get(Tensor* tensor, int32_t* indices);
void tensor_set(Tensor* tensor, int32_t* indices, float value);


Tensor* conv2d(Tensor* input, Tensor* kernel, int stride, int padding);


#endif // TENSOR_H
