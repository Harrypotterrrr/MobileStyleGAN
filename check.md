- [x] ConstantInput
- [x] EqualLinear
- [ ] StyledConv
- [x] NoiseInjection
- [ ] Blur
- [x] make_kernel
- [x] upfirdn2d_cpu
- [ ] ModulatedConv2d
- [ ] StyledConv2d
- [ ] MappingNetwork
- [x] PixelNorm

- PixelNorm
- EqualLinear -> Affine
- NoiseInjection -> tf.zeros(1)

- lazy regularization: accelerate training and decrease the memory
- Path length regularization: increase ppl and generate more smooth images
- Nogrowing , new & Darch: network frame optimize

- R1 regularization: stablize GAN very effective
- Exponential Moving Average: stablize Generator
- Equalized Learning Rate: learn with large LR
- Blur Knernel: blur in Upsampling stage, stablize the training. enable noise works in styleGAN
- Noise Input: detail generation from noise
- Minibatch Standard Deviation Layer: increase the diversity

- Noise reguliarzation: allows image information to be encoded into style variables as much as possible, so that the style variable enable to control the genetated style
- Noise: only complement detail