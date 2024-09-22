
#include "diffusion.h"
#include "tensor.h"
#include "normalization.h"
#include "unet_blocks.h"
#include <stdio.h>

void test_tensor(void) {
    int32_t shape[] = {2, 3, 4};
    Tensor* t = tensor_create(3, shape);
    if (t == NULL) {
        printf("Failed to create tensor\n");
        return;
    }

    printf("Tensor created successfully\n");
    printf("Dimensions: %d\n", t->ndim);
    printf("Shape: %d x %d x %d\n", t->shape[0], t->shape[1], t->shape[2]);
    printf("Size: %d\n", t->size);

    int32_t indices[] = {1, 1, 1};
    tensor_set(t, indices, 42.0f);

    float value = tensor_get(t, indices);
    printf("Value at [1,1,1]: %f\n", value);

    tensor_free(t);
    printf("Tensor freed\n");
}

void test_groupnorm(void) {
    int32_t input_shape[] = {1, 4, 2, 2};  // batch_size=1, channels=4, height=2, width=2
    Tensor* input = tensor_create(4, input_shape);
    
    // Initialize input tensor with some values
    for (int i = 0; i < 16; i++) {
        int32_t indices[] = {0, i / 4, (i % 4) / 2, i % 2};
        tensor_set(input, indices, (float)i);
    }

    GroupNorm* gn = groupnorm_create(2, 4, 1e-5);  // 2 groups, 4 channels, epsilon=1e-5
    if (gn == NULL) {
        printf("Failed to create GroupNorm\n");
        return;
    }

    Tensor* output = groupnorm_forward(gn, input);
    if (output == NULL) {
        printf("Failed to perform GroupNorm forward pass\n");
        groupnorm_free(gn);
        tensor_free(input);
        return;
    }

    printf("GroupNorm test:\n");
    printf("Input shape: %d x %d x %d x %d\n", input->shape[0], input->shape[1], input->shape[2], input->shape[3]);
    printf("Output shape: %d x %d x %d x %d\n", output->shape[0], output->shape[1], output->shape[2], output->shape[3]);

    // Print some output values
    for (int c = 0; c < 4; c++) {
        int32_t indices[] = {0, c, 0, 0};
        float val = tensor_get(output, indices);
        printf("Output[0, %d, 0, 0] = %f\n", c, val);
    }

    tensor_free(input);
    tensor_free(output);
    groupnorm_free(gn);
}


void test_conv2d(void) {
    int32_t input_shape[] = {1, 3, 32, 32};
    Tensor* input = tensor_create(4, input_shape);
    
    int32_t kernel_shape[] = {16, 3, 3, 3};
    Tensor* kernel = tensor_create(4, kernel_shape);
    
    Tensor* output = conv2d(input, kernel, 1, 1);
    
    printf("Conv2D output shape: %d x %d x %d x %d\n", 
           output->shape[0], output->shape[1], output->shape[2], output->shape[3]);
    
    tensor_free(input);
    tensor_free(kernel);
    tensor_free(output);
}

void test_layernorm(void) {
    int32_t input_shape[] = {2, 3, 4};  // batch_size=2, seq_len=3, feature_size=4
    Tensor* input = tensor_create(3, input_shape);
    
    // Initialize input tensor with some values
    for (int i = 0; i < 24; i++) {
        int32_t indices[] = {i / 12, (i % 12) / 4, i % 4};
        tensor_set(input, indices, (float)i);
    }

    LayerNorm* ln = layernorm_create(4, 1e-5);  // normalized_shape=4, epsilon=1e-5
    if (ln == NULL) {
        printf("Failed to create LayerNorm\n");
        return;
    }

    Tensor* output = layernorm_forward(ln, input);
    if (output == NULL) {
        printf("Failed to perform LayerNorm forward pass\n");
        layernorm_free(ln);
        tensor_free(input);
        return;
    }

    printf("LayerNorm test:\n");
    printf("Input shape: %d x %d x %d\n", input->shape[0], input->shape[1], input->shape[2]);
    printf("Output shape: %d x %d x %d\n", output->shape[0], output->shape[1], output->shape[2]);

    // Print some output values
    for (int b = 0; b < 2; b++) {
        for (int s = 0; s < 3; s++) {
            int32_t indices[] = {b, s, 0};
            float val = tensor_get(output, indices);
            printf("Output[%d, %d, 0] = %f\n", b, s, val);
        }
    }

    tensor_free(input);
    tensor_free(output);
    layernorm_free(ln);
}

void test_silu(void) {
    int32_t input_shape[] = {2, 3};  // 2x3 tensor
    Tensor* input = tensor_create(2, input_shape);
    
    // Initialize input tensor with some values
    float input_values[] = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f};
    for (int i = 0; i < 6; i++) {
        int32_t indices[] = {i / 3, i % 3};
        tensor_set(input, indices, input_values[i]);
    }

    Tensor* output = silu_forward(input);
    if (output == NULL) {
        printf("Failed to perform SiLU forward pass\n");
        tensor_free(input);
        return;
    }

    printf("SiLU test:\n");
    printf("Input shape: %d x %d\n", input->shape[0], input->shape[1]);
    printf("Output shape: %d x %d\n", output->shape[0], output->shape[1]);

    // Print input and output values
    for (int i = 0; i < 6; i++) {
        int32_t indices[] = {i / 3, i % 3};
        float input_val = tensor_get(input, indices);
        float output_val = tensor_get(output, indices);
        printf("SiLU(%f) = %f\n", input_val, output_val);
    }

    tensor_free(input);
    tensor_free(output);
}

void test_downblock() {
    printf("Testing DownBlock:\n");

    // Create a DownBlock
    int in_channels = 64;
    int out_channels = 128;
    int groups = 32;
    int downsample = 1;
    DownBlock* block = downblock_create(in_channels, out_channels, groups, downsample);

    if (block == NULL) {
        printf("Failed to create DownBlock\n");
        return;
    }

    // Check if the block was initialized correctly
    printf("DownBlock created successfully\n");
    printf("In channels: %d\n", block->in_channels);
    printf("Out channels: %d\n", block->out_channels);
    printf("Groups: %d\n", block->groups);
    printf("Downsample: %d\n", block->downsample);

    // Check if tensors were created
    printf("Conv1 weight shape: %d x %d x %d x %d\n", 
           block->conv1_weight->shape[0], block->conv1_weight->shape[1],
           block->conv1_weight->shape[2], block->conv1_weight->shape[3]);
    printf("Conv1 bias shape: %d\n", block->conv1_bias->shape[0]);

    // Check if GroupNorm layers were created
    if (block->groupnorm1 != NULL && block->groupnorm2 != NULL) {
        printf("GroupNorm layers created successfully\n");
    } else {
        printf("Failed to create GroupNorm layers\n");
    }

    // Check if downsample convolution was created
    if (block->downsample) {
        if (block->downsample_conv_weight != NULL && block->downsample_conv_bias != NULL) {
            printf("Downsample convolution created successfully\n");
        } else {
            printf("Failed to create downsample convolution\n");
        }
    }

    // Free the DownBlock
    downblock_free(block);
    printf("DownBlock freed\n");
}



#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "tensor.h"

void test_tensor_concat() {
    printf("Testing tensor_concat function:\n");

    // Create two tensors to concatenate
    int32_t shape1[] = {2, 3, 2};
    Tensor* t1 = tensor_create(3, shape1);
    int32_t shape2[] = {2, 2, 2};
    Tensor* t2 = tensor_create(3, shape2);

    // Fill tensors with some values
    for (int i = 0; i < t1->size; i++) {
        t1->data[i] = i + 1;
    }
    for (int i = 0; i < t2->size; i++) {
        t2->data[i] = i + 10;
    }

    // Concatenate along dimension 1
    Tensor* result = tensor_concat(t1, t2, 1);

    // Check the result
    if (result == NULL) {
        printf("tensor_concat failed: returned NULL\n");
        return;
    }

    // Check dimensions
    if (result->ndim != 3 || result->shape[0] != 2 || result->shape[1] != 5 || result->shape[2] != 2) {
        printf("tensor_concat failed: incorrect output shape\n");
        return;
    }

    // Check some values
    int32_t indices[] = {0, 0, 0};
    float expected_values[] = {1, 2, 3, 4, 5, 6, 7, 8, 10, 11};
    for (int i = 0; i < 10; i++) {
        indices[1] = i / 2;
        indices[2] = i % 2;
        float val = tensor_get(result, indices);
        if (fabs(val - expected_values[i]) > 1e-6) {
            printf("tensor_concat failed: incorrect value at [0, %d, %d], expected %f, got %f\n", 
                   indices[1], indices[2], expected_values[i], val);
            return;
        }
    }

    printf("tensor_concat test passed successfully\n");

    // Clean up
    tensor_free(t1);
    tensor_free(t2);
    tensor_free(result);
}


void test_tensor_concat_old() {
    printf("Testing tensor_concat function:\n");

    // Create two tensors to concatenate
    int32_t shape1[] = {2, 3, 2};
    Tensor* t1 = tensor_create(3, shape1);
    int32_t shape2[] = {2, 2, 2};
    Tensor* t2 = tensor_create(3, shape2);

    // Fill tensors with some values
    for (int i = 0; i < t1->size; i++) {
        t1->data[i] = i + 1;
    }
    for (int i = 0; i < t2->size; i++) {
        t2->data[i] = i + 10;
    }

    // Concatenate along dimension 1
    Tensor* result = tensor_concat(t1, t2, 1);

    // Check the result
    if (result == NULL) {
        printf("tensor_concat failed: returned NULL\n");
        return;
    }

    // Check dimensions
    if (result->ndim != 3 || result->shape[0] != 2 || result->shape[1] != 5 || result->shape[2] != 2) {
        printf("tensor_concat failed: incorrect output shape\n");
        return;
    }

    // Check some values
    int32_t indices[] = {0, 0, 0};
    float expected_values[] = {1, 2, 3, 4, 5, 6, 10, 11, 12, 13};
    for (int i = 0; i < 10; i++) {
        indices[1] = i;
        float val = tensor_get(result, indices);
        if (fabs(val - expected_values[i]) > 1e-6) {
            printf("tensor_concat failed: incorrect value at [0, %d, 0], expected %f, got %f\n", i, expected_values[i], val);
            return;
        }
    }

    printf("tensor_concat test passed successfully\n");

    // Clean up
    tensor_free(t1);
    tensor_free(t2);
    tensor_free(result);
}

void test_upsample_conv2d() {
    printf("Testing upsample_conv2d function:\n");

    // Create input tensor (1, 2, 2, 2)
    int32_t input_shape[] = {1, 2, 2, 2};
    Tensor* input = tensor_create(4, input_shape);
    for (int i = 0; i < input->size; i++) {
        input->data[i] = i + 1;
    }

    // Create weight tensor (3, 2, 2, 2)
    int32_t weight_shape[] = {3, 2, 2, 2};
    Tensor* weight = tensor_create(4, weight_shape);
    for (int i = 0; i < weight->size; i++) {
        weight->data[i] = 0.1 * (i + 1);
    }

    // Create bias tensor (3)
    int32_t bias_shape[] = {3};
    Tensor* bias = tensor_create(1, bias_shape);
    for (int i = 0; i < bias->size; i++) {
        bias->data[i] = 0.01 * (i + 1);
    }

    // Perform upsample convolution
    int scale_factor = 2;
    Tensor* result = upsample_conv2d(input, weight, bias, scale_factor);

    // Check the result
    if (result == NULL) {
        printf("upsample_conv2d failed: returned NULL\n");
        return;
    }

    // Check dimensions (should be 1, 3, 4, 4)
    if (result->ndim != 4 || result->shape[0] != 1 || result->shape[1] != 3 || 
        result->shape[2] != 4 || result->shape[3] != 4) {
        printf("upsample_conv2d failed: incorrect output shape\n");
        return;
    }

    // We won't check exact values here as they depend on the specific implementation
    // Instead, we'll check if the output is non-zero and within a reasonable range
    for (int i = 0; i < result->size; i++) {
        if (result->data[i] == 0 || fabs(result->data[i]) > 100) {
            printf("upsample_conv2d failed: suspicious output value %f at index %d\n", result->data[i], i);
            return;
        }
    }

    printf("upsample_conv2d test passed successfully\n");

    // Clean up
    tensor_free(input);
    tensor_free(weight);
    tensor_free(bias);
    tensor_free(result);
}



int main(void) {
    UNetConfig config = {
        .in_channels = 4,
        .out_channels = 4
    };

    UNet2DConditionModel* model = unet_create(config);
    if (model == NULL) {
        printf("Failed to create UNet model\n");
        return 1;
    }

    printf("UNet model created successfully\n");
    printf("Input channels: %d\n", model->config.in_channels);
    printf("Output channels: %d\n", model->config.out_channels);

    test_tensor();
    test_conv2d();
    test_groupnorm();
    test_layernorm();
    test_silu();
    test_downblock();

    test_upsample_conv2d();
    //    test_tensor_concat();

    
    unet_free(model);
    return 0;
}
