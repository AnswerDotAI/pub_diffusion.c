# diffusion.c

## CUDA Mode IRL Hackathon Project

A C-based implementation of a Stable Diffusion model, inspired by llm.c.

### Project Overview

diffusion.c is a project to implement a Stable Diffusion model entirely in C/CUDA. Our goal is to create a lightweight, efficient, and highly controllable version of Stable Diffusion, for training and inference, which can run without the overhead of large machine learning frameworks.


# UNet2DConditionModel Implementation Plan

## 1. Core Data Structures
- [x] Define Tensor structure
- [x] Implement basic Tensor operations (creation, destruction, indexing)

## 2. Basic UNet2DConditionModel Structure
- [x] Define UNetConfig and UNet2DConditionModel structures
- [x] Implement creation and destruction functions

## 3. Core Operations
- [x] Implement conv2d function
- [x] Implement normalization layers (GroupNorm and LayerNorm)
- [x] Implement activation function (SiLU)
- [ ] Implement tensor_concat function (current implementation not working)
- [ ] Implement attention mechanism

## 4. UNet2DConditionModel Components
- [x] Implement down blocks
- [x] Implement up blocks
- [ ] Implement mid block

## 5. UNet2DConditionModel Forward Pass
- [ ] Connect all components (conv2d, normalization, attention, etc.)
- [ ] Implement time embedding
- [ ] Implement cross-attention with condition

## 6. Backward Pass and Optimization
- [ ] Implement gradient calculation for all operations
- [ ] Implement optimizer (e.g., AdamW)

## 7. Diffusion Process
- [ ] Implement forward diffusion (noise addition)
- [ ] Implement reverse diffusion (denoising)

## 8. Testing
- [x] Implement basic tests for tensor operations
- [x] Implement tests for normalization and activation functions
- [x] Implement tests for UNet blocks
- [ ] Fix and complete test for tensor_concat
- [ ] Implement test for attention mechanism
- [ ] Implement integration tests for full model
- [ ] Implement comparison with PyTorch reference implementation


### How to Build and Run

```
$ cd src/
$ make
$ ./unet_test
```

### Who

This is AnswerAI hack by Nate Cooper, Alexis Gallagher, with special moral support provided Sanyam Bhutani!

