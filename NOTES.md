# Notes

## Project Log

- 2026-10-28: Repo created, structure set up, notebooks dropped in, basic code written.
- 2026-10-29: TA check-in with Ben.
- (add dated entries as you work)

## CIFAR-10 Notebook: Part 1 Predictions

Record these before running the notebook.

- What will D(x; sigma) look like at sigma = 80?
  Answer: TODO (hint: at very high noise the input is basically pure noise, so the best estimate is close to the dataset mean, a gray blur)

- At which noise level do you expect the content of the final image to be settled?
  Answer: TODO (will fill in after watching the trajectory)

## CIFAR-10 Part 3 Trajectory Questions

1. At which noise level does the class first show in D(x; sigma)? Is it earlier with w = 4?
   Answer: TODO

2. From which noise level on does the image barely change?
   Answer: TODO

3. How do the w = 4 estimates differ from the w = 1 ones at low noise?
   Answer: TODO

## CIFAR-10 Part 5 CFG Sweep Discussion

Which FID would you choose (FID over all classes vs FID_class), and what does each measure miss?

Answer: TODO (hint: all-class FID penalizes loss of variety across the full dataset, FID_class only checks each class separately so it hides class imbalance; both are noisy at 1,000 images)

## CIFAR-10 Part 6 Training Observations

- Does FID improve or get worse across snapshots on 100 and 1,000 images?
  Answer: TODO

- Evidence of memorization in the nearest training image comparison:
  Answer: TODO (compare median distance of generated samples vs real test images to their nearest training image)

- How does the CFG grid behavior differ from the pretrained model in Part 5?
  Answer: TODO

## MNIST Part 3 Controlled Comparison

Write 150 to 250 words answering the three questions below.

1. Explain the difference between changing c (the condition) and changing w (the guidance strength).
2. Does your grid support the claim that larger w always helps? Include a failure or an ambiguous example rather than selecting only attractive digits.
3. Explain why target match and pixel spread together still do not establish image quality. Which samples need a human check?

Answer: TODO

## MNIST Part 4 Architecture Questions

1. Where does the requested digit enter the UNet (cemb), and where does the time enter?
   Answer: TODO

2. Is this MNIST model pixel-space or latent-space? In latent diffusion, where would the encoder and decoder sit relative to the sampling loop?
   Answer: TODO

3. What representation enters a DiT? Does replacing a U-Net by a transformer alone change the definition of guidance?
   Answer: TODO

4. For a text prompt, an exact pose, and a masked local edit, name an appropriate input or interface. Which requests can this digit-label model express without new training? Why can we not request a new class by using label 11?
   Answer: TODO

## Open Questions for Mentor

- Is the preliminary results section expected inside the Introduction or as its own section in the checkpoint report?
- How polished does the README need to be by Nov 6 vs Dec 4?
- Is the optional full EDM training run worth attempting given DSMLP GPU time limits?
- Any specific attribute Prof. Wang wants me to study for Q2?
- Preferred repo structure (notebooks vs scripts, folder layout)?

## What Worked

- TODO (fill in as you go)

## What Did Not Work

- TODO (fill in as you go)