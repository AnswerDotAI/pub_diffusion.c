
#include "tensor.h"
#include <stdlib.h>
#include <string.h>

Tensor* tensor_create(int32_t ndim, int32_t* shape) {
    Tensor* tensor = (Tensor*)malloc(sizeof(Tensor));
    if (tensor == NULL) return NULL;

    tensor->ndim = ndim;
    tensor->shape = (int32_t*)malloc(ndim * sizeof(int32_t));
    if (tensor->shape == NULL) {
        free(tensor);
        return NULL;
    }
    memcpy(tensor->shape, shape, ndim * sizeof(int32_t));

    tensor->size = 1;
    for (int i = 0; i < ndim; i++) {
        tensor->size *= shape[i];
    }

    tensor->data = (float*)calloc(tensor->size, sizeof(float));
    if (tensor->data == NULL) {
        free(tensor->shape);
        free(tensor);
        return NULL;
    }

    return tensor;
}

void tensor_free(Tensor* tensor) {
    if (tensor == NULL) return;
    free(tensor->data);
    free(tensor->shape);
    free(tensor);
}

float tensor_get(Tensor* tensor, int32_t* indices) {
    int32_t index = 0;
    int32_t stride = 1;
    for (int i = tensor->ndim - 1; i >= 0; i--) {
        index += indices[i] * stride;
        stride *= tensor->shape[i];
    }
    return tensor->data[index];
}

void tensor_set(Tensor* tensor, int32_t* indices, float value) {
    int32_t index = 0;
    int32_t stride = 1;
    for (int i = tensor->ndim - 1; i >= 0; i--) {
        index += indices[i] * stride;
        stride *= tensor->shape[i];
    }
    tensor->data[index] = value;
}

Tensor* conv2d(Tensor* input, Tensor* kernel, int stride, int padding) {
    int batch = input->shape[0];
    int in_channels = input->shape[1];
    int in_height = input->shape[2];
    int in_width = input->shape[3];
    
    int out_channels = kernel->shape[0];
    int kernel_height = kernel->shape[2];
    int kernel_width = kernel->shape[3];
    
    int out_height = (in_height + 2 * padding - kernel_height) / stride + 1;
    int out_width = (in_width + 2 * padding - kernel_width) / stride + 1;
    
    int32_t out_shape[] = {batch, out_channels, out_height, out_width};
    Tensor* output = tensor_create(4, out_shape);
    
    for (int b = 0; b < batch; b++) {
        for (int oc = 0; oc < out_channels; oc++) {
            for (int oh = 0; oh < out_height; oh++) {
                for (int ow = 0; ow < out_width; ow++) {
                    float sum = 0.0f;
                    for (int ic = 0; ic < in_channels; ic++) {
                        for (int kh = 0; kh < kernel_height; kh++) {
                            for (int kw = 0; kw < kernel_width; kw++) {
                                int ih = oh * stride + kh - padding;
                                int iw = ow * stride + kw - padding;
                                
                                if (ih >= 0 && ih < in_height && iw >= 0 && iw < in_width) {
                                    int32_t input_indices[] = {b, ic, ih, iw};
                                    int32_t kernel_indices[] = {oc, ic, kh, kw};
                                    sum += tensor_get(input, input_indices) * tensor_get(kernel, kernel_indices);
                                }
                            }
                        }
                    }
                    int32_t out_indices[] = {b, oc, oh, ow};
                    tensor_set(output, out_indices, sum);
                }
            }
        }
    }
    
    return output;
}
