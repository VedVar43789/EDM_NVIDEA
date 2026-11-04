# AI Use Documentation

As per course policy, this file documents my AI usage across the DSC 180A Q1 project in more detail than the README contribution statement. It covers prompts used, what the AI was asked to do, how outputs were incorporated, and how they were tested or verified.

**Tools used:** Claude by Anthropic (primary, via claude.ai web app), Cursor (code edits), and occasional ChatGPT (quick lookups).

**Scope:** Prompts are grouped by task.

---

## 1. Understanding the Assignment

**Context:** The course page, the Oct 28 TA slides, the Oct 28 assignment PDF, and the two Colab notebooks each described part of what was due by Nov 6. I needed one general overview of everything that needs to be completed by Friday, November 6, 2026.

**Prompts I used:**

- "this is our quarter 1 project guidelines, can you explain in concise what needs to be done"
- "give me an entire quarter 1 project overview given all these things now. explain everything in very simple words, include the paper and github. also give a timeline at end given the check in, checkpoint and final submission"
- "what should be in the expected GitHub repo?"
- "what all to do before the TA check in tomorrow?"

**How the output was used:** I used the consolidated breakdowns to build my own two-week plan and to prepare for the Week 5 TA check-in.

**How I verified it:** I cross-checked each item against the official course page, the Oct 28 assignment PDF, and the notebook headers. Anything AI-generated that was not in the original files was dropped.

---

## 2. Paper Summaries

**Context:** Reading EDM, Diffusion-LM, DiffuSeq, and CDCD end to end before writing the report was not feasible given the week. I used AI for simple short summaries.

**Example prompt:**

- "give me a very detailed summary of this paper in very simple words as I am a beginner in this topic. talk about problem statement, what they did and everything else important, keep it concise, use simple language, basic examples and analogies wherever needed" (EDM paper)

**How the output was used:** As a starting framework for understanding each paper and for writing the Literature Review section of the report. The Q2 reading list summaries also shaped the conversation with Prof. Wang.

**How I verified it:** For EDM, I read the paper abstract, Section 2, and the main results tables myself, and corrected the summary wherever it was off. The 1.79 and 1.97 FID values, the 35 network evaluations per image, and the four EDM design axes were all verified in the paper. For the Q2 reading list, I used the summaries as starting points for discussion with Prof. Wang along with my team, not as final understanding.

---

## 3. Code: Notebook Cells

**Context:** Each notebook had one or two cells I needed to write myself (CIFAR-10 Part 3 "Your turn" trajectory, MNIST Part 3 "Your controlled comparison"). I drafted these with AI help, and clarified their runtime against different kind of GPUs.

**Example prompts:**

- "give code for CIFAR-10 notebook, and help me understand what exactly is going on"
- "how long would CIFAR-10 take to run on T4 GPU"
- "how long would CIFAR-10 take to run on A100 GPU"

**Example output (abridged, for the CIFAR-10 Part 3 cell):**

```python
TARGET_CLASS = "cat"
SEED = 0
labels = torch.full((3,), CLASSES.index(TARGET_CLASS))

_, history_cond = edm_sample(
    lambda x, s: D(cond, x, s, labels),
    n=3, seed=SEED, record=True
)
show_trajectory(history_cond, title=f"Conditional model (w = 1): trajectory toward {TARGET_CLASS}")

_, history_cfg = edm_sample(
    lambda x, s: D_cfg(x, s, labels, w=4),
    n=3, seed=SEED, record=True
)
show_trajectory(history_cfg, title=f"CFG w = 4: trajectory toward {TARGET_CLASS}")
```

**How the output was used:** I pasted the draft into the notebook, ran it, and inspected the resulting trajectory figure. Having their expected runtime, helped me to understand which GPU would be good for these files. I initially tried T4 GPU, but it timed out after running for like 45 minutes. I ultimately opted for A100 on Google Collab as that one ran the entire notebook in just an hour.

**How I tested it:**

- Visually confirmed that the w=1 and w=4 trajectories made sense (cat emerged earlier at w=4, as predicted by the EDM discussion)
- Confirmed both runs used the same seed so the comparison was fair
- Only kept the version that produced the expected trajectory structure on the basis of what we had learned in class

---

## 4. Code: GitHub Repo Scaffolding

**Context:** I needed a folder structure, README skeleton, requirements.txt, .gitignore.

**Example prompt:**

- "help me generate the entire GitHub repo skeleton"

**How the output was used:** I used it as a starting scaffold via Cursor. The scaffolding output created empty files and skeletons that I then filled in with my own content as experiments progressed.

**How I tested it:**

- Cloned the repo fresh and confirmed the folder structure matched
- Ran `pip install -r requirements.txt` in a fresh venv to confirm dependencies installed cleanly
- Verified `.gitignore` correctly excluded `downloads/`, `outputs/`, and checkpoint files from commits

---

## 5. Debugging

**Context:** My first Part 5 CFG sweep on a T4 Colab session timed out due to idle disconnect.

**Example prompt:**

- "my colab session keeps disconnecting during part 5 after ~45 min, what should I do"

**How the output was used:** Based on the response, I switched to A100 on Colab Pro, kept the browser tab active, and reran. The sweep finished in about 5-10 minutes on A100.

**How I tested it:**

- Spot-checked the metrics against the notebook stdout to make sure nothing was truncated

---

## 6. Report Writing

**Context:** The full report (abstract, intro, data, preliminary results, takeaways) needed to be written.

**Workflow:**

1. I wrote a rough first-pass of each section myself based on my understanding of the experiments and papers
2. I asked Claude to tighten the language, cut redundancy, and keep the tone conversational
3. I reviewed the output, rewrote anything that did not sound like me or did not match what I actually did
4. Iterated until the section read naturally

**Example prompts:**

- "help me format the my existing draft of the report for this assignment. make sure to keep the language conversational and professional"
- "this current draft is great, bit it is too long and complex. I want it a bit shorter"

**How the output was used:** As the polished version of my own first-pass drafts.

**How I verified it:**

- Every number in the Preliminary Results section (accuracies, FID, FID_class, confidence, loss) was typed directly by me by looking at the results from my notebook runs, not from any AI estimate
- Every claim about EDM, CFG, or the Q2 papers was cross-checked against the actual paper or course slides
- Read the final PDF end to end once compiled to confirm nothing sounded off

---

## 7. LaTeX Fixes

**Context:** Compile issues (citations rendering as "?", figure path mismatches between my files and the `\includegraphics` references, subcaption layout).

**Example prompts:**

- "all citations appear as question mark in the pdf"
- "the images are scaled weirdly"

**How the output was used:** AI generated the full `reference.bib` entries and fixed the `\includegraphics` paths to match my actual file names.

**How I tested it:**

- Recompiled from scratch in Overleaf, confirmed all citations resolved to real numbered references
- Checked the final PDF to confirm every figure rendered correctly, no missing image boxes
- Verified every bibtex entry pointed to a real paper on arXiv or the publication venue

---

## 8. Q2 Direction (Not AI-Assisted)

Separate from AI use, I took the initiative this week to set up a meeting with Prof. Yusu Wang and our team to discuss the Q2 project direction. That conversation led to the pivot from image steering (an NA-RFM reproduction on Oxford-IIIT Pet) to continuous diffusion for text generation. I used AI afterward to help find and summarize the three starting papers (Diffusion-LM, DiffuSeq, CDCD), but the decision came from the meeting itself.

---

## 9. Where AI Was Not Used

- Running the notebooks (done on my Google Colab Pro A100 account)
- Generating the figures (output of my notebook runs)
- Interpreting what the results mean
- Made [NOTES.md](http://NOTES.md) to trach all my ork and progress, along with answers to some of the notebook questions
- Making the first draft of the report in raw language
- Re-reading each file and report for any errors
- The meeting with Prof. Wang on the Q2 direction
- Submitting to Gradescope and maintaining the GitHub repo