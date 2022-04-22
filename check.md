- [x] ConstantInput
- [x] NonlinearTransform
- [x] make_kernel
- [x] NoiseInjection
- [x] upfirdn2d_cpu
- [x] PixelNorm

- [x] Upsample
- [ ] FusedLeakyReLU
- [x] ToRGB
- [x] StyledConv
- [x] Blur
- [x] ModulatedConv
- [ ] StyledConv2d
- [ ] MappingNetwork
- [x] LinearTransform
- [ ] LinearTransform2d
- [ ] Discriminator

- PixelNorm
- EqualLinear -> Affine
- NoiseInjection -> tf.zeros(1)
- fused_leaky_relu
- self.weight -> w

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

- QA
  - pad on one side
  - Now we apply demodulation to remove the effect of s from the statistics of the output feature maps
  - upfirdn2d (upsample downsample not multiplier)
  - dictionary of output influence performance
  - no bias in styleConv
  - NonlinearTransform  / * lr_mul
  - fused_leaky_relu GPU 
  - NoiseInjection zero(1)
  - Group convolution

- add some comment

