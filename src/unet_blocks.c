#include "unet_blocks.h"
#include <stdlib.h>

DownBlock* downblock_create(int in_channels, int out_channels, int groups, int downsample) {
    DownBlock* block = (DownBlock*)malloc(sizeof(DownBlock));
    if (block == NULL) return NULL;

    block->in_channels = in_channels;
    block->out_channels = out_channels;
    block->groups = groups;
    block->downsample = downsample;

    // Create convolution weights and biases
    int32_t conv_shape[] = {out_channels, in_channels, 3, 3};  // Assuming 3x3 convolutions
    block->conv1_weight = tensor_create(4, conv_shape);
    block->conv1_bias = tensor_create(1, &out_channels);
    block->conv2_weight = tensor_create(4, conv_shape);
    block->conv2_bias = tensor_create(1, &out_channels);

    // Create GroupNorm layers
    block->groupnorm1 = groupnorm_create(groups, out_channels, 1e-5);
    block->groupnorm2 = groupnorm_create(groups, out_channels, 1e-5);

    // Create downsample convolution if needed
    if (downsample) {
        block->downsample_conv_weight = tensor_create(4, conv_shape);
        block->downsample_conv_bias = tensor_create(1, &out_channels);
    } else {
        block->downsample_conv_weight = NULL;
        block->downsample_conv_bias = NULL;
    }

    return block;
}

void downblock_free(DownBlock* block) {
    if (block == NULL) return;

    tensor_free(block->conv1_weight);
    tensor_free(block->conv1_bias);
    tensor_free(block->conv2_weight);
    tensor_free(block->conv2_bias);

    groupnorm_free(block->groupnorm1);
    groupnorm_free(block->groupnorm2);

    if (block->downsample) {
        tensor_free(block->downsample_conv_weight);
        tensor_free(block->downsample_conv_bias);
    }

    free(block);
}

Tensor* downblock_forward(DownBlock* block, Tensor* input) {
    // Implement the forward pass here
    // This is a placeholder implementation and needs to be completed
    // with actual convolution, normalization, and activation operations

    // First convolution
    Tensor* conv1_out = conv2d(input, block->conv1_weight, 1, 1);  // Assuming stride 1 and padding 1
    Tensor* bias1_out = add_bias(conv1_out, block->conv1_bias);
    Tensor* norm1_out = groupnorm_forward(block->groupnorm1, bias1_out);
    Tensor* act1_out = silu_forward(norm1_out);

    // Second convolution
    Tensor* conv2_out = conv2d(act1_out, block->conv2_weight, 1, 1);
    Tensor* bias2_out = add_bias(conv2_out, block->conv2_bias);
    Tensor* norm2_out = groupnorm_forward(block->groupnorm2, bias2_out);
    Tensor* act2_out = silu_forward(norm2_out);

    // Downsample if needed
    Tensor* output = act2_out;
    if (block->downsample) {
        Tensor* downsample_out = conv2d(input, block->downsample_conv_weight, 2, 1);  // Stride 2 for downsampling
        output = add_bias(downsample_out, block->downsample_conv_bias);
    }

    // Free intermediate tensors
    tensor_free(conv1_out);
    tensor_free(bias1_out);
    tensor_free(norm1_out);
    tensor_free(act1_out);
    tensor_free(conv2_out);
    tensor_free(bias2_out);
    tensor_free(norm2_out);

    return output;
}

UpBlock* upblock_create(int in_channels, int out_channels, int groups, int upsample) {
    UpBlock* block = (UpBlock*)malloc(sizeof(UpBlock));
    if (block == NULL) return NULL;

    block->in_channels = in_channels;
    block->out_channels = out_channels;
    block->groups = groups;
    block->upsample = upsample;

    // Create convolution weights and biases
    int32_t conv_shape[] = {out_channels, in_channels, 3, 3};  // Assuming 3x3 convolutions
    block->conv1_weight = tensor_create(4, conv_shape);
    block->conv1_bias = tensor_create(1, &out_channels);
    block->conv2_weight = tensor_create(4, conv_shape);
    block->conv2_bias = tensor_create(1, &out_channels);

    // Create GroupNorm layers
    block->groupnorm1 = groupnorm_create(groups, out_channels, 1e-5);
    block->groupnorm2 = groupnorm_create(groups, out_channels, 1e-5);

    // Create upsample convolution if needed
    if (upsample) {
        int32_t upsample_shape[] = {out_channels, in_channels, 2, 2};  // 2x2 for upsampling
        block->upsample_conv_weight = tensor_create(4, upsample_shape);
        block->upsample_conv_bias = tensor_create(1, &out_channels);
    } else {
        block->upsample_conv_weight = NULL;
        block->upsample_conv_bias = NULL;
    }

    return block;
}

void upblock_free(UpBlock* block) {
    if (block == NULL) return;

    tensor_free(block->conv1_weight);
    tensor_free(block->conv1_bias);
    tensor_free(block->conv2_weight);
    tensor_free(block->conv2_bias);

    groupnorm_free(block->groupnorm1);
    groupnorm_free(block->groupnorm2);

    if (block->upsample) {
        tensor_free(block->upsample_conv_weight);
        tensor_free(block->upsample_conv_bias);
    }

    free(block);
}

Tensor* upblock_forward(UpBlock* block, Tensor* input, Tensor* skip_input) {
    Tensor* x = input;

    // Upsample if needed
    if (block->upsample) {
        // Implement upsampling (e.g., using transpose convolution or interpolation)
        // For simplicity, let's assume we have a function called upsample_conv2d
        x = upsample_conv2d(x, block->upsample_conv_weight, block->upsample_conv_bias, 2);
    }

    // Concatenate with skip input
    x = tensor_concat(x, skip_input, 1);  // Assuming channel dimension is 1

    // First convolution
    x = conv2d(x, block->conv1_weight, 1, 1);
    x = add_bias(x, block->conv1_bias);
    x = groupnorm_forward(block->groupnorm1, x);
    x = silu_forward(x);

    // Second convolution
    x = conv2d(x, block->conv2_weight, 1, 1);
    x = add_bias(x, block->conv2_bias);
    x = groupnorm_forward(block->groupnorm2, x);
    x = silu_forward(x);

    return x;
}
