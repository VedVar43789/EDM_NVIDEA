# Notes

> Working notebook for the DSC 180A Quarter 1 project on diffusion steering. Holds project log entries, predictions, trajectory and sweep observations, training notes, written answers to the MNIST architecture questions, open questions for the mentor, and what worked or did not work across runs.

## Index

- [Project Log](#project-log)
- [CIFAR-10 Notebook: Part 1 Predictions](#cifar-10-notebook-part-1-predictions)
- [CIFAR-10 Part 3 Trajectory Questions](#cifar-10-part-3-trajectory-questions)
- [CIFAR-10 Part 5 CFG Sweep Discussion](#cifar-10-part-5-cfg-sweep-discussion)
- [CIFAR-10 Part 6 Training Observations](#cifar-10-part-6-training-observations)
- [MNIST Part 3 Controlled Comparison](#mnist-part-3-controlled-comparison)
- [MNIST Part 4 Architecture Questions](#mnist-part-4-architecture-questions)
- [Open Questions for Mentor](#open-questions-for-mentor)
- [What Worked](#what-worked)
- [What Did Not Work](#what-did-not-work)

---

## Project Log

- 2026-10-28: Repo created, structure set up, notebooks dropped in. Started reading the EDM paper and the Oct 28 slides.
- 2026-10-29: TA check-in with Ben in HDSI 336.
- 2026-10-30: Meeting with Prof. Yusu Wang this week. Agreed to shift Q2 direction from image steering (NA-RFM reproduction on Oxford-IIIT Pet) to continuous diffusion for text generation. The EDM design-space view still applies: preconditioning, noise schedule, and sampler questions all come back in a continuous embedding space. Starting reading list: Diffusion-LM (Li et al. 2022), DiffuSeq (Gong et al. 2023), CDCD (Dieleman et al. 2022).
- 2026-11-01: Ran both notebooks end to end on Google Colab A100. CIFAR-10 full run (Parts 1 to 6) took about 50 minutes. MNIST notebook ran on GPU in about 15 minutes. All figures and metrics saved.
- 2026-11-02: Drafted checkpoint report in Overleaf. Filled in Preliminary Results with actual numbers. Polished README with reproduction commands.

---

# CIFAR NOTEBOOK QUESTION

## CIFAR-10 Notebook: Part 1 Predictions

Record these before running the notebook.

- **What will D(x; sigma) look like at sigma = 80?**

  **Answer:** At sigma = 80 the input is basically pure noise, so the denoiser has almost no signal to work with. The best it can do is output something close to the average CIFAR-10 image, which should look like a gray or brownish blur with no visible object. This matches what the EDM paper shows in Figure 1b and what the course slides showed on MNIST, where the high-noise estimate was an average digit shape.
- **At which noise level do you expect the content of the final image to be settled?**

  **Answer:** I expect the class and rough layout to lock in somewhere around sigma between 1 and 3. That is the range where the EDM paper and the trajectory sketches show the solution curves bending the most. Above that range the estimate is still a blur, and below that range the sampler is mostly adding fine detail and texture rather than changing what the image is of.

## CIFAR-10 Part 3 Trajectory Questions

1. **At which noise level does the class first show in D(x; sigma)? Is it earlier with w = 4?**

   **Answer:** In the conditional run with w = 1, a recognizable cat shape starts emerging around sigma = 4 to 5, with the ears and face shape becoming clear around sigma = 2. With w = 4, the cat features show up noticeably earlier, around sigma = 7 to 10. CFG amplifies the class direction, so the sampler commits to the target class much sooner in the trajectory.
2. **From which noise level on does the image barely change?**

   **Answer:** The image barely changes below sigma = 0.5 for both runs. By that point the object identity, pose, and coarse texture are locked in, and the remaining steps mostly polish fine detail. This matches my prediction that content locks in between sigma = 1 and 3, with the fine-detail phase below sigma = 1.
3. **How do the w = 4 estimates differ from the w = 1 ones at low noise?**

   **Answer:** At low noise, the w = 4 estimates are sharper, more saturated, and more stereotypically cat-like than the w = 1 estimates. Features like whiskers, eyes, and ear tips are more pronounced. The flip side is less variation across the three samples in w = 4, since CFG squeezes the samples toward a more canonical cat.

## CIFAR-10 Part 5 CFG Sweep Discussion

**Which FID would I choose (FID over all classes vs FID_class), and what does each measure miss?**

For this project I would lean toward FID_class as the primary metric when comparing across w, with the overall FID as a secondary check. Here is why.

Overall FID compares the full set of 1,000 generated images against all 50,000 CIFAR-10 training images. The problem is that CFG pushes samples toward sharper, more canonical class examples, which lowers the variety inside each class. So the all-class FID gets worse as w grows, but not because the images are worse. It is because the generated distribution has shrunk compared to the real one. That penalty is confounded with the actual effect I care about.

FID_class compares each class only against real images of that class. It isolates whether the samples look like real cats or real cars, independently of class balance or cross-class variety. The downside is that with only 100 images per class, the Gaussian estimate is very noisy, so absolute FID_class values are much higher than published numbers. I can only compare FID_class across w, not with external papers.

Both metrics miss things a human notices right away, like whether the image has consistent lighting, correct object parts, and no weird artifacts. Neither one checks that the requested class is actually shown either, which is why label accuracy sits beside them in the table.

Actual observed numbers (1,000 images per w):

| w | Label Accuracy | FID | FID_class |
| --- | --- | --- | --- |
| 0 | 11.5% | 31.18 | - |
| 1 | 95.3% | 31.33 | 112.1 |
| 1.5 | 98.7% | 31.45 | 107.6 |
| 2 | 98.8% | 33.05 | 108.9 |
| 3 | 99.5% | 36.06 | 112.5 |
| 4 | 98.9% | 39.06 | 117.6 |

The sweet spot for FID_class sits at w = 1.5, while overall FID is flat from w = 0 to w = 1.5 and then climbs steadily. Label accuracy saturates near 99 percent by w = 2.

## CIFAR-10 Part 6 Training Observations

- **Does FID improve or get worse across snapshots on 100 and 1,000 images?**

  **Answer:** FID gets worse across snapshots on both sizes. For the 100-image run, FID rises from 78.7 at step 4000 to 110.8 at step 20000. For the 1,000-image run, FID rises from 90.2 at step 4000 to 129.8 at step 20000. Training loss keeps dropping (0.04 to 0.017 on 100 images, 0.181 to 0.071 on 1000 images), which confirms the loss and FID move in opposite directions here. The model is fitting the data better in terms of loss, but the data it is fitting is so small that fitting it harder makes the generations worse.
- **Evidence of memorization in the nearest training image comparison:**

  **Answer:** The generated samples at the final snapshot sit visibly close to their nearest training images in pixel space. Many of the 100-image samples look like near-duplicates of training images. The 1,000-image run is less obviously memorized visually, but samples still land near training neighbors. This is the behavior the EDM paper predicts for the low-data regime: the optimal denoiser essentially stores and retrieves training examples, so sampling just produces near-copies.
- **How does the CFG grid behavior differ from the pretrained model in Part 5?**

  **Answer:** The small-subset CFG grids are much lower quality than the pretrained Part 5 grids. The classes are harder to recognize at any w, because the network has not seen enough examples of each class to form a general concept. CFG still visibly amplifies whatever class signal exists (samples at higher w are more saturated and committed), but there is less class signal to amplify in the first place, so the effect is muted. The 1,000-image model is better than the 100-image model but still well below the pretrained model.

# MNIST NOTEBOOK QUESTION

## MNIST Part 3 Controlled Comparison

Write 150 to 250 words answering the three questions below.

1. **Explain the difference between changing c (the condition) and changing w (the guidance strength).**
2. **Does my grid support the claim that larger w always helps? Include a failure or an ambiguous example rather than selecting only attractive digits.**
3. **Explain why target match and pixel spread together still do not establish image quality. Which samples need a human check?**

**Answer:** Changing c swaps which class the sampler aims for. With the same noise, the condition grid shows that a column of 3s and a column of 8s share coarse layout but differ wherever the digit identity requires it. The condition picks the target, while the noise still fixes most of the pose and stroke style. Changing w scales how strongly the sampler commits to that target. At w=0 the sampler ignores the condition entirely and produces random digits, so target_match collapses to 0.175, roughly chance. At w=1 the sampler follows the condition as trained and target_match jumps to 0.975. At w=2 and w=5 the matches hit 1.0 but the strokes get bolder, thicker, and more uniform across samples.

My grid does not support the claim that larger w always helps. By w=5 every 3 looks nearly identical, which is a diversity failure even though target_match is perfect. Also, at w=2 one of the 3s has a slightly malformed top loop that the classifier still reads as a 3, which flags an ambiguity between accuracy and quality.

Target_match and pixel_spread miss stroke smoothness, closure of loops, and overall legibility. The w=0 samples and the few borderline w=2 digits need a human check, since the classifier is confident even when the digit looks off.

## MNIST Part 4 Architecture Questions

1. **Where does the requested digit enter the UNet (cemb), and where does the time enter?**

   **Answer:** Looking at UNet.forward in mnist_guidance_lab.py, the time t goes through self.temb, a two-layer MLP, and the requested digit y goes through self.cemb, an embedding table with 11 slots (10 digits plus the no-request label). The two are added together into a single conditioning vector e before any convolution runs. Then e is injected into every residual block through the Block.emb linear layer, which adds a per-channel bias to the feature map after the first conv. So the time and the condition enter the same way, at every block, as additive channel-wise shifts.
2. **Is this MNIST model pixel-space or latent-space? In latent diffusion, where would the encoder and decoder sit relative to the sampling loop?**

   **Answer:** This MNIST model is pixel-space. The U-Net takes a 1 by 28 by 28 noisy image directly and outputs a 1 by 28 by 28 clean estimate, with no encoder or decoder around it. In latent diffusion like Stable Diffusion, there is a frozen VAE wrapped around the sampling loop. The encoder sits before the loop, compressing the input image into a smaller latent (for Stable Diffusion, 4 channels at one eighth of the image resolution). The sampling loop then runs entirely in that latent space. After the final denoising step, the decoder sits after the loop, mapping the clean latent back into a full-resolution RGB image.
3. **What representation enters a DiT? Does replacing a U-Net by a transformer alone change the definition of guidance?**

   **Answer:** A Diffusion Transformer takes a sequence of patch tokens as input, the same way Vision Transformers do. The noisy image is split into patches, each patch is flattened and linearly embedded, and the time and class conditioning are added either as extra tokens or through adaptive layer norm. The transformer then processes the sequence of tokens with self-attention. Replacing a U-Net with a transformer does not change what guidance means. CFG still mixes a conditional and an unconditional prediction with the same formula, u + w (c - u). The denoiser is a black box as far as guidance is concerned, so switching the backbone from convolutional to attention-based does not touch the sampler-level math.
4. **For a text prompt, an exact pose, and a masked local edit, name an appropriate input or interface. Which requests can this digit-label model express without new training? Why can we not request a new class by using label 11?**

   **Answer:** Text prompts go through a text encoder like CLIP, with the output feeding into cross-attention layers in the U-Net or DiT. Exact pose is handled by ControlNet or a similar side-network that takes a pose skeleton as an extra conditioning image. Masked local edits are done with inpainting, where the mask and the original image are concatenated to the noisy input channels so the model knows which pixels to keep and which to regenerate.

    This MNIST model can only express the 10 digit classes it was trained on, plus the no-request null condition. It cannot handle any of the three richer requests above without new training, because none of its interfaces accept text, poses, or masks. As for label 11, the embedding table self.cemb has exactly 11 slots (10 digits plus the null token at index 10). Index 11 is out of range and would throw an index error. Even if the table were larger, any index beyond 10 was never updated during training, so its vector would be random noise and the sampler would produce garbage. New classes need either retraining with that new label present or a method that does not route through the embedding table at all.

## Open Questions for Mentor

- **Is the preliminary results section expected inside the Introduction or as its own section in the checkpoint report?**
- **How polished does the README need to be by Nov 6 compared to Dec 4?**
- **Is the optional full EDM training run worth attempting given DSMLP GPU time limits? If yes, roughly how many hours should I plan to let it run?**
- **Any specific attribute Prof. Wang wants me to study for the Q2 direction, or should I propose one?**
- **Preferred repo structure (notebooks vs scripts, folder layout)?**
- **For FID with only 1,000 images, is there a specific convention I should document in the report so readers do not compare my numbers to published EDM numbers?**

## What Worked

- Running the CIFAR-10 notebook on an A100 cut Part 5 from the ~45 minute T4 estimate down to ~5 minutes, which freed up time for the training runs in Part 6.
- Hash-checked downloads on first run meant no restart headaches when the Colab session reconnected.
- Treating Part 6 as a pipeline validation (expecting memorization) rather than a quality run saved me from worrying about the rising FID numbers.
- Using the same seed (0) across the trajectory comparisons made the w = 1 vs w = 4 difference easier to see.

## What Did Not Work

- First attempt at Part 5 on a T4 session timed out after Colab disconnected at the 90-minute idle limit. Switching to A100 and keeping the tab active fixed it.
- Initial MNIST run on CPU took over an hour. Switching DEVICE from "cpu" to "cuda" at the top of the notebook cut it to about 15 minutes.
- The nearest-training-image comparison was initially confusing because the 1,000-image samples looked blurrier than the 100-image ones. I had to realize this is because 20k updates is enough for the 100-image model to fully memorize but not the 1,000-image model.