
#ifndef DIFFUSION_H
#define DIFFUSION_H

#include "tensor.h"
#include "normalization.h"

typedef struct {
    int in_channels;
    int out_channels;
} UNetConfig;

typedef struct {
    UNetConfig config;
} UNet2DConditionModel;

UNet2DConditionModel* unet_create(UNetConfig config);
void unet_free(UNet2DConditionModel* model);

#endif // DIFFUSION_H
