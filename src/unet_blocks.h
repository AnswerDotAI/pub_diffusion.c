#ifndef UNET_BLOCKS_H
#define UNET_BLOCKS_H

#include "tensor.h"
#include "normalization.h"

typedef struct {
    // Convolution layers
    Tensor* conv1_weight;
    Tensor* conv1_bias;
    Tensor* conv2_weight;
    Tensor* conv2_bias;
    
    // Group Normalization layers
    GroupNorm* groupnorm1;
    GroupNorm* groupnorm2;
    
    // Downsample convolution (optional, for reducing spatial dimensions)
    Tensor* downsample_conv_weight;
    Tensor* downsample_conv_bias;
    
    // Configuration
    int in_channels;
    int out_channels;
    int groups;  // For GroupNorm
    int downsample;  // Boolean flag: 1 if downsample, 0 otherwise
} DownBlock;

typedef struct {
    // Convolution layers
    Tensor* conv1_weight;
    Tensor* conv1_bias;
    Tensor* conv2_weight;
    Tensor* conv2_bias;
    
    // Group Normalization layers
    GroupNorm* groupnorm1;
    GroupNorm* groupnorm2;
    
    // Upsample convolution (optional, for increasing spatial dimensions)
    Tensor* upsample_conv_weight;
    Tensor* upsample_conv_bias;
    
    // Configuration
    int in_channels;
    int out_channels;
    int groups;  // For GroupNorm
    int upsample;  // Boolean flag: 1 if upsample, 0 otherwise
} UpBlock;

// Function declarations for UpBlock
UpBlock* upblock_create(int in_channels, int out_channels, int groups, int upsample);
void upblock_free(UpBlock* block);
Tensor* upblock_forward(UpBlock* block, Tensor* input, Tensor* skip_input);


// Function declarations
DownBlock* downblock_create(int in_channels, int out_channels, int groups, int downsample);
void downblock_free(DownBlock* block);
Tensor* downblock_forward(DownBlock* block, Tensor* input);

#endif // UNET_BLOCKS_H
