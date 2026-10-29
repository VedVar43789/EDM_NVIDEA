# Notes

## Project Log

- 2026-10-28: Repo created, structure set up, notebooks dropped in. Started reading the EDM paper and the Oct 28 slides.
- 2026-10-29: TA check-in with Ben in HDSI 336.
- (add dated entries as I work)

## CIFAR-10 Notebook: Part 1 Predictions

Record these before running the notebook.

- **What will D(x; sigma) look like at sigma = 80?**
  Answer: At sigma = 80 the input is basically pure noise, so the denoiser has almost no signal to work with. The best it can do is output something close to the average CIFAR-10 image, which should look like a gray or brownish blur with no visible object. This matches what the EDM paper shows in Figure 1b and what the course slides showed on MNIST, where the high-noise estimate was an average digit shape.

- **At which noise level do you expect the content of the final image to be settled?**
  Answer: I expect the class and rough layout to lock in somewhere around sigma between 1 and 3. That is the range where the EDM paper and the trajectory sketches show the solution curves bending the most. Above that range the estimate is still a blur, and below that range the sampler is mostly adding fine detail and texture rather than changing what the image is of.

## CIFAR-10 Part 3 Trajectory Questions

1. **At which noise level does the class first show in D(x; sigma)? Is it earlier with w = 4?**
   Answer: TODO (fill in after running)

2. **From which noise level on does the image barely change?**
   Answer: TODO (fill in after running)

3. **How do the w = 4 estimates differ from the w = 1 ones at low noise?**
   Answer: TODO (fill in after running)

## CIFAR-10 Part 5 CFG Sweep Discussion

**Which FID would I choose (FID over all classes vs FID_class), and what does each measure miss?**

Conceptual framing (fill in actual numbers after Part 5 runs):

For this project I would lean toward FID_class as the primary metric when comparing across w, with the overall FID as a secondary check. Here is why.

Overall FID compares the full set of 1,000 generated images against all 50,000 CIFAR-10 training images. The problem is that CFG pushes samples toward sharper, more canonical class examples, which lowers the variety inside each class. So the all-class FID gets worse as w grows, but not because the images are worse. It is because the generated distribution has shrunk compared to the real one. That penalty is confounded with the actual effect I care about.

FID_class compares each class only against real images of that class. It isolates whether the samples look like real cats or real cars, independently of class balance or cross-class variety. The downside is that with only 100 images per class, the Gaussian estimate is very noisy, so absolute FID_class values are much higher than published numbers. I can only compare FID_class across w, not with external papers.

Both metrics miss things a human notices right away, like whether the image has consistent lighting, correct object parts, and no weird artifacts. Neither one checks that the requested class is actually shown either, which is why label accuracy sits beside them in the table.

TODO: fill in actual numbers for FID, FID_class, and label accuracy per w after the sweep finishes.

## CIFAR-10 Part 6 Training Observations

- **Does FID improve or get worse across snapshots on 100 and 1,000 images?**
  Answer: TODO (fill in after running)

- **Evidence of memorization in the nearest training image comparison:**
  Answer: TODO (fill in after running)

- **How does the CFG grid behavior differ from the pretrained model in Part 5?**
  Answer: TODO (fill in after running)

## MNIST Part 3 Controlled Comparison

Write 150 to 250 words answering the three questions below.

1. **Explain the difference between changing c (the condition) and changing w (the guidance strength).**
2. **Does my grid support the claim that larger w always helps? Include a failure or an ambiguous example rather than selecting only attractive digits.**
3. **Explain why target match and pixel spread together still do not establish image quality. Which samples need a human check?**

Answer: TODO (fill in after running my own comparison with a new digit and seed)

## MNIST Part 4 Architecture Questions

1. **Where does the requested digit enter the UNet (cemb), and where does the time enter?**
   Answer: Looking at UNet.forward in mnist_guidance_lab.py, the time t goes through self.temb, a two-layer MLP, and the requested digit y goes through self.cemb, an embedding table with 11 slots (10 digits plus the no-request label). The two are added together into a single conditioning vector e before any convolution runs. Then e is injected into every residual block through the Block.emb linear layer, which adds a per-channel bias to the feature map after the first conv. So the time and the condition enter the same way, at every block, as additive channel-wise shifts.

2. **Is this MNIST model pixel-space or latent-space? In latent diffusion, where would the encoder and decoder sit relative to the sampling loop?**
   Answer: This MNIST model is pixel-space. The U-Net takes a 1 by 28 by 28 noisy image directly and outputs a 1 by 28 by 28 clean estimate, with no encoder or decoder around it. In latent diffusion like Stable Diffusion, there is a frozen VAE wrapped around the sampling loop. The encoder sits before the loop, compressing the input image into a smaller latent (for Stable Diffusion, 4 channels at one eighth of the image resolution). The sampling loop then runs entirely in that latent space. After the final denoising step, the decoder sits after the loop, mapping the clean latent back into a full-resolution RGB image.

3. **What representation enters a DiT? Does replacing a U-Net by a transformer alone change the definition of guidance?**
   Answer: A Diffusion Transformer takes a sequence of patch tokens as input, the same way Vision Transformers do. The noisy image is split into patches, each patch is flattened and linearly embedded, and the time and class conditioning are added either as extra tokens or through adaptive layer norm. The transformer then processes the sequence of tokens with self-attention. Replacing a U-Net with a transformer does not change what guidance means. CFG still mixes a conditional and an unconditional prediction with the same formula, u + w (c - u). The denoiser is a black box as far as guidance is concerned, so switching the backbone from convolutional to attention-based does not touch the sampler-level math.

4. **For a text prompt, an exact pose, and a masked local edit, name an appropriate input or interface. Which requests can this digit-label model express without new training? Why can we not request a new class by using label 11?**
   Answer: Text prompts go through a text encoder like CLIP, with the output feeding into cross-attention layers in the U-Net or DiT. Exact pose is handled by ControlNet or a similar side-network that takes a pose skeleton as an extra conditioning image. Masked local edits are done with inpainting, where the mask and the original image are concatenated to the noisy input channels so the model knows which pixels to keep and which to regenerate.

   This MNIST model can only express the 10 digit classes it was trained on, plus the no-request null condition. It cannot handle any of the three richer requests above without new training, because none of its interfaces accept text, poses, or masks. As for label 11, the embedding table self.cemb has exactly 11 slots (10 digits plus the null token at index 10). Index 11 is out of range and would throw an index error. Even if the table were larger, any index beyond 10 was never updated during training, so its vector would be random noise and the sampler would produce garbage. New classes need either retraining with that new label present or a method that does not route through the embedding table at all.

## Open Questions for Mentor

- **Is the preliminary results section expected inside the Introduction or as its own section in the checkpoint report?**
- **How polished does the README need to be by Nov 6 compared to Dec 4?**
- **Is the optional full EDM training run worth attempting given DSMLP GPU time limits? If yes, roughly how many hours should I plan to let it run?**
- **Any specific attribute Prof. Wang wants me to study for the Q2 direction, or should I propose one?**
- **Preferred repo structure (notebooks vs scripts, folder layout)?**
- **For FID with only 1,000 images, is there a specific convention I should document in the report so readers do not compare my numbers to published EDM numbers?**

## What Worked

- TODO (fill in as I go)

## What Did Not Work

- TODO (fill in as I go)