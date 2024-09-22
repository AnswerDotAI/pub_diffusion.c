
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
Tensor* add_bias(Tensor* input, Tensor* bias);
Tensor* conv2d(Tensor* input, Tensor* weight, int stride, int padding);

void tensor_free(Tensor* tensor);
float tensor_get(Tensor* tensor, int32_t* indices);
void tensor_set(Tensor* tensor, int32_t* indices, float value);


Tensor* conv2d(Tensor* input, Tensor* kernel, int stride, int padding);

// implementation pending
Tensor* upsample_conv2d(Tensor* input, Tensor* weight, Tensor* bias, int scale_factor);
Tensor* tensor_concat(Tensor* t1, Tensor* t2, int dim);


#endif // TENSOR_H
