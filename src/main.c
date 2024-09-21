
#include "diffusion.h"
#include "tensor.h"
#include "normalization.h"
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

    unet_free(model);
    return 0;
}
