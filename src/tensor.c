
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

Tensor* add_bias(Tensor* input, Tensor* bias) {
    // Ensure input is 4D (batch, channels, height, width) and bias is 1D (channels)
    if (input->ndim != 4 || bias->ndim != 1 || input->shape[1] != bias->shape[0]) {
        return NULL;  // Invalid input
    }

    Tensor* output = tensor_create(input->ndim, input->shape);
    if (output == NULL) return NULL;

    int batch = input->shape[0];
    int channels = input->shape[1];
    int height = input->shape[2];
    int width = input->shape[3];

    for (int b = 0; b < batch; b++) {
        for (int c = 0; c < channels; c++) {
            float bias_val = tensor_get(bias, (int32_t[]){c});
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    int32_t idx[] = {b, c, h, w};
                    float val = tensor_get(input, idx) + bias_val;
                    tensor_set(output, idx, val);
                }
            }
        }
    }

    return output;
}



Tensor* upsample_conv2d(Tensor* input, Tensor* weight, Tensor* bias, int scale_factor) {
    // Check input dimensions
    if (input->ndim != 4 || weight->ndim != 4 || bias->ndim != 1) {
        return NULL;  // Invalid input dimensions
    }

    int batch = input->shape[0];
    int in_channels = input->shape[1];
    int in_height = input->shape[2];
    int in_width = input->shape[3];
    int out_channels = weight->shape[0];
    int kernel_height = weight->shape[2];
    int kernel_width = weight->shape[3];

    // Calculate output dimensions
    int out_height = in_height * scale_factor;
    int out_width = in_width * scale_factor;

    // Create output tensor
    int32_t out_shape[] = {batch, out_channels, out_height, out_width};
    Tensor* output = tensor_create(4, out_shape);
    if (output == NULL) return NULL;

    // Perform upsampling and convolution
    for (int b = 0; b < batch; b++) {
        for (int oc = 0; oc < out_channels; oc++) {
            for (int oh = 0; oh < out_height; oh++) {
                for (int ow = 0; ow < out_width; ow++) {
                    float sum = 0.0f;
                    
                    // Calculate corresponding input position
                    int ih = oh / scale_factor;
                    int iw = ow / scale_factor;

                    // Perform convolution
                    for (int ic = 0; ic < in_channels; ic++) {
                        for (int kh = 0; kh < kernel_height; kh++) {
                            for (int kw = 0; kw < kernel_width; kw++) {
                                int ih_offset = ih + kh - kernel_height / 2;
                                int iw_offset = iw + kw - kernel_width / 2;

                                // Check boundaries
                                if (ih_offset >= 0 && ih_offset < in_height && iw_offset >= 0 && iw_offset < in_width) {
                                    float input_val = tensor_get(input, (int32_t[]){b, ic, ih_offset, iw_offset});
                                    float weight_val = tensor_get(weight, (int32_t[]){oc, ic, kh, kw});
                                    sum += input_val * weight_val;
                                }
                            }
                        }
                    }

                    // Add bias
                    sum += tensor_get(bias, (int32_t[]){oc});

                    // Set output value
                    tensor_set(output, (int32_t[]){b, oc, oh, ow}, sum);
                }
            }
        }
    }

    return output;
}

Tensor* tensor_concat(Tensor* t1, Tensor* t2, int dim) {
    // Check if tensors have the same number of dimensions
    if (t1->ndim != t2->ndim) {
        return NULL;  // Cannot concatenate tensors with different dimensions
    }

    // Check if dimensions other than the concatenation dimension match
    for (int i = 0; i < t1->ndim; i++) {
        if (i != dim && t1->shape[i] != t2->shape[i]) {
            return NULL;  // Mismatched dimensions
        }
    }

    // Calculate new shape
    int32_t new_shape[t1->ndim];
    for (int i = 0; i < t1->ndim; i++) {
        if (i == dim) {
            new_shape[i] = t1->shape[i] + t2->shape[i];
        } else {
            new_shape[i] = t1->shape[i];
        }
    }

    // Create new tensor
    Tensor* result = tensor_create(t1->ndim, new_shape);
    if (result == NULL) return NULL;

    // Calculate strides for t1, t2, and result
    int32_t t1_strides[t1->ndim];
    int32_t t2_strides[t2->ndim];
    int32_t result_strides[t1->ndim];
    t1_strides[t1->ndim - 1] = t2_strides[t2->ndim - 1] = result_strides[t1->ndim - 1] = 1;
    for (int i = t1->ndim - 2; i >= 0; i--) {
        t1_strides[i] = t1_strides[i + 1] * t1->shape[i + 1];
        t2_strides[i] = t2_strides[i + 1] * t2->shape[i + 1];
        result_strides[i] = result_strides[i + 1] * new_shape[i + 1];
    }

    // Copy data from t1 and t2 to result
    for (int i = 0; i < result->size; i++) {
        int32_t position[t1->ndim];
        int temp = i;
        for (int j = 0; j < t1->ndim; j++) {
            position[j] = temp / result_strides[j];
            temp %= result_strides[j];
        }

        int source_offset;
        if (position[dim] < t1->shape[dim]) {
            // Copy from t1
            source_offset = 0;
            for (int j = 0; j < t1->ndim; j++) {
                source_offset += position[j] * t1_strides[j];
            }
            result->data[i] = t1->data[source_offset];
        } else {
            // Copy from t2
            position[dim] -= t1->shape[dim];
            source_offset = 0;
            for (int j = 0; j < t2->ndim; j++) {
                source_offset += position[j] * t2_strides[j];
            }
            result->data[i] = t2->data[source_offset];
        }
    }

    return result;
}

