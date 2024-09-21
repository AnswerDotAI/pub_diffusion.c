
#include "diffusion.h"
#include <stdlib.h>
#include <stdio.h>

UNet2DConditionModel* unet_create(UNetConfig config) {
    UNet2DConditionModel* model = (UNet2DConditionModel*)malloc(sizeof(UNet2DConditionModel));
    if (model == NULL) {
        return NULL;
    }
    model->config = config;
    return model;
}

void unet_free(UNet2DConditionModel* model) {
    if (model != NULL) {
        free(model);
    }
}
