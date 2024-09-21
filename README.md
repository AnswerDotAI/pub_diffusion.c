# diffusion.c

## CUDA Mode IRL Hackathon Project

A C-based implementation of a Stable Diffusion model, inspired by llm.c.

### Project Overview

diffusion.c is a project to implement a Stable Diffusion model entirely in C/CUDA. Our goal is to create a lightweight, efficient, and highly controllable version of Stable Diffusion, for training and inference, which can run without the overhead of large machine learning frameworks.

### Current Progress

So far, we have implemented:

1. Basic tensor operations (creation, destruction, indexing)
2. A simple UNet2DConditionModel structure
3. 2D Convolution operation
4. Group Normalization layer
   - Verified against PyTorch implementation for accuracy

### Next Steps

Our roadmap for completing the project includes:

1. Implement additional core operations:
   - Activation functions (e.g., SiLU/Swish)
   - Attention mechanism

2. Extend UNet2DConditionModel:
   - Implement down blocks
   - Implement up blocks
   - Implement mid block

3. Implement forward pass of UNet2DConditionModel:
   - Connect all components (conv2d, normalization, attention, etc.)
   - Implement time embedding
   - Implement cross-attention with condition

4. Implement backward pass and optimization:
   - Implement gradient calculation for all operations
   - Implement optimizer (e.g., AdamW)

5. Implement diffusion process:
   - Implement forward diffusion (noise addition)
   - Implement reverse diffusion (denoising)

6. Create comprehensive test suite:
   - Unit tests for individual components
   - Integration tests for full model
   - Implement comparison with PyTorch reference implementation

7. Optimize and refine:
   - Profile code to identify bottlenecks
   - Optimize critical sections
   - Implement CUDA kernels for GPU acceleration

### How to Build and Run

```
$ cd src/
$ make
$ ./unet_test
```

