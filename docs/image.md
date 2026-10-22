# Image generator: what changed in 2.10 and 2.14.2, and how it was measured

This page covers the from-scratch image generator behind the GUI *Image* tab,
`ImageGenEngine` and the `auralite image-train` / `image-generate` /
`image-eval` commands. § 1–9 are the 2.10 work on the diffusion model (and the
`image-train` / `image-generate` commands, new then); § 10–15 are 2.14.2: flow
matching and a GAN as further model types, a quality score without pretrained
networks, and the progress stages. Every number was measured on this
repository's code on a 2-vCPU VM (one physical core with two hyper-threads,
torch 2.14.1 CPU). The commands to reproduce them are in § 9.

Quality is measured on MNIST at 16 px, a data set small enough to train on this
machine:

* 5,000 training images exported as PNG files and loaded through the normal
  folder path;
* the *Tiny* preset (715 k parameters) for 10 epochs, which is 1,570 steps at
  batch 32;
* 1,000 samples scored with a **classifier-FID**: the Fréchet distance between
  penultimate-layer features of a small MNIST classifier (98.9 % test
  accuracy) on the samples and on the 10,000 test images.

Lower is better. 1,000 real training images score 8.0, which is the floor at
this sample size. Every row is a single training run (one seed), so treat
differences of a few tens of FID points between neighbouring rows as
indicative rather than proven. The held-out loss of § 3 moves by less than 1 %
between those rows, so it cannot rank them either.

## 1. Training speed

### Attention fell back to the slow kernel

The U-Net's attention blocks computed q, k and v with a 1x1 convolution and
reshaped the result to `(batch, heads, head_dim, tokens)` before a transpose.
`F.scaled_dot_product_attention` received tensors whose last dimension had a
stride of `tokens` instead of 1. The fused kernels (flash attention on CPU;
flash and memory-efficient attention on CUDA) require a unit stride, so every
call silently used the math fallback, which materialises the full attention
matrix. In the profile of a *Tiny* training step that fallback was 53 % of the
time.

2.10 keeps the same parameters, so old checkpoints load unchanged. The
projections now run as `F.linear` on a token-major `(batch, tokens, channels)`
view, so q, k and v arrive with a unit stride. The outputs match the old block
to 1e-5.

| SDPA forward+backward, batch 32, 256 tokens, 1 thread | old layout | 2.10 layout | speed-up |
|---|---|---|---|
| 32 channels, 4 heads (head_dim 8) | 117.8 ms | 38.4 ms | 3.1x |
| 32 channels, 1 head (head_dim 32) | 32.9 ms | 13.4 ms | 2.5x |
| 64 channels, 4 heads (head_dim 16) | 124.2 ms | 39.4 ms | 3.2x |
| 64 channels, 2 heads (head_dim 32) | 69.6 ms | 26.2 ms | 2.7x |
| 128 channels, 4 heads (head_dim 32) | 142.3 ms | 52.7 ms | 2.7x |

### channels-last memory format

On CPU the convolution kernels (oneDNN) are faster on NHWC tensors. The
model, its EMA copy and every batch are converted to `torch.channels_last`
by default on CPU (`channels_last=None` means automatic). Set it to `True` to
use it elsewhere. CUDA was not measured, so it stays off there by default.

### Automatic attention heads

The default `attn_heads` is now 0, which means one head per 32 channels. Head
dimensions of 8 or 16 are slow (see the table above). This change cuts a *Tiny*
step from 195 to 154 ms (table below), and in its run the model scored FID 245
against 275 for 4 heads (table in § 2). Checkpoints store their head count, so
models trained with 4 heads keep 4.

### Result

| Training step (AdamW, clip, EMA), 2 threads | 2.9 | 2.10 attention | + channels-last | + auto heads |
|---|---|---|---|---|
| *Tiny*, batch 32 | 403 ms | 232 ms | 195 ms | 154 ms |
| *Small*, batch 8 | 644 ms | 618 ms | 438 ms | 447 ms |

In total, a *Tiny* step is 2.6x faster and a *Small* step 1.45x. The rows
come from `benchmarks/image_bench.py step` (one process, alternating rounds).

With one thread the *Tiny* row reads 650 → 387 → 323 → 243 ms and the
*Small* row 1,123 → 1,044 → 773 → 761 ms. For *Small* at batch 32 (one
thread, separate processes) a step goes from 4,148 to 2,676 ms. Peak RAM drops
from 1,533 MB to 1,330 MB because the attention matrices are no longer
materialised.

### Threads

The GUI and the CLI always ran with the text engine's thread policy:
`AURALITE_NUM_THREADS`, default every logical CPU. The image engine imported
on its own did not, and torch's default is the number of *physical* cores. On
this VM that meant one thread: 243 ms per *Tiny* step instead of 154 ms. Since
2.10 the image engine applies the same policy (`configure_cpu_threads`, which
never lowers a higher count set by the caller).

Two more speed-ups:

* `torch.compile` gives nothing on this model on CPU: 330.4 ms against 330.7 ms
  eager, after a 73 s compile. It stays an option.
* **Images are decoded once.** The folder is decoded into a single uint8 tensor
  using a thread pool. This needs 1 byte per pixel instead of the old
  per-image float32 cache, which used 4. Batches are then gathered and flipped
  in one vectorised step, so there is no per-item `Dataset.__getitem__` or
  `DataLoader` collation. The tensor is reused while the folder is unchanged,
  so *continue training* does not decode again.

  JPEGs are opened in draft mode. The decoder downsamples by 2, 4 or 8 in the
  DCT domain to at least twice the target size. On 3000x2000 JPEGs loaded at
  32 px, a file took 53.3 ms decoded in full and 6.9 ms in draft mode (7.7x),
  and 3.9 ms with the thread pool (`image_bench.py decode`; smooth synthetic
  photos, so real ones are slower in both modes).

## 2. Fixes that change results

| MNIST-16, *Tiny*, 1,570 steps, DDIM 50 steps | classifier-FID | mean pixel | pixel std |
|---|---|---|---|
| real images | 8.0 | 0.131 | 0.264 |
| 2.9 behaviour | 308.2 | 0.249 | 0.304 |
| + rescaled noise schedule | 304.9 | 0.159 | 0.259 |
| + zero-initialised output layers, EMA warmup | 275.2 | 0.155 | 0.264 |
| + automatic heads (the 2.10 default) | 245.3 | 0.156 | 0.263 |

### The linear and sigmoid schedules did not reach pure noise

The linear and sigmoid beta ranges (1e-4 → 0.02) are DDPM's values for
T = 1000. With fewer steps the last step still contains signal, while sampling
starts from pure noise:

| T (steps) | 50 | 100 | 200 | 400 | 600 | 1000 |
|---|---|---|---|---|---|---|
| signal left at t = T, 2.9 (`sqrt(alpha_bar_T)`) | 78 % | 60 % | 36 % | 13 % | 4.8 % | 0.6 % |
| 2.10 (range × 1000/T) | 0.6 % | 0.6 % | 0.6 % | 0.6 % | 0.6 % | 0.6 % |

The engine defaults and the *Tiny* preset use linear with T = 200. The sampler
therefore asked the network to denoise inputs it had never seen in training.
The samples came out too bright and grey: mean pixel 0.249 against 0.131 for
real digits. With the rescaled range it is 0.159.

FID barely moves at this training budget (308 → 305), because digit shape is
the limiting factor. The cosine schedule (all other presets) was already
correct.

Checkpoints now record `schedule_version: 2`. Files without it load with the
old betas, so existing models sample exactly as before.

### EMA started from the random initialisation

The EMA copy used a fixed decay from step 1. It therefore kept `decay ** steps`
of the random initial weights: with 0.999 that is 61 % after 500 steps and 21 %
after 1,570. Sampling uses the EMA weights by default, so short runs produced
grey mush. The decay now ramps up as `min(decay, (1 + n) / (10 + n))`, and the
update is a single multi-tensor `lerp`.

| same run, decay 0.999 | classifier-FID | held-out loss | mean pixel |
|---|---|---|---|
| fixed decay (2.9) | 1502.6 | 0.0730 | 0.368 |
| with warmup (2.10) | 275.2 | 0.0307 | 0.155 |

Checkpoints store the number of EMA updates. Older files resume with the full
decay.

### Zero-initialised output layers

The last convolution of every residual branch and attention block, and the
output convolution, start at zero (as in DDPM and ADM). Each block starts as
the identity. In the table the FID went from 305 to 275 for this row, which
also includes the EMA warmup; the held-out loss is unchanged (0.0307).

### Smaller fixes

* **LR schedule with gradient accumulation.** The cosine schedule counted
  micro-batches against a total of optimizer steps. With
  `accumulation_steps=4` it ended at 87 % of the base LR instead of 10 %. It
  now counts optimizer steps.
* A partial accumulation group at the end of an epoch is applied instead of
  being dropped.
* **Continue training ignored the new settings.** It reused the old config, so
  new epochs, learning rate and so on had no effect, and the data set could be
  loaded at another size. Now the architecture, diffusion and class settings
  come from the model and everything else from the new params. A log line
  names any setting that the model overrides.
* The `torch.compile` probe now runs a real training step: full batch, with
  autograd. Compiled runs drop the last partial batch, so they never
  recompile.
* `tensor_to_uint8` rounds instead of truncating. A value of 0.0 maps to 128,
  not 127.
* **EXIF orientation** is applied when images are loaded, so portrait phone
  photos are no longer learned sideways.
* Unreadable files are skipped with a warning instead of aborting training.
* `grad_clip=0` now means no clipping. It used to scale every gradient to
  zero.
* Unknown training params are logged as ignored. `auralite image-train`
  rejects them and suggests the closest valid name.

## 3. Validation and early stopping

`val_split` holds out a fraction of the images, chosen with a fixed seed, so
the split does not depend on `seed`. The GUI and the CLI default to 5 %, and
the engine API defaults to 0.

After every epoch, each held-out image is noised at the same 8 stratified
timesteps with the same seeded noise. It is then scored with the training loss
weighting by the EMA network, which is the network that samples. The number is
therefore deterministic, and comparable across epochs and across runs with the
same diffusion settings.

* It appears in the progress callback (the 4th argument and `info["val_loss"]`).
* It is plotted next to the training loss in the GUI.
* It is summarised in `last_train_summary`: best value and epoch, steps,
  seconds and images per second. `val_history` holds every epoch's value.

`early_stopping_patience` stops after that many epochs without improvement.
With `restore_best_weights` (default on) it also restores the model and EMA
weights from the best epoch.

The loss is an honest generalisation check: it rises when a small data set is
memorised. It is not a substitute for looking at samples, though. In the table
in § 2, rows with FID 305, 275 and 245 have held-out losses of 0.0308, 0.0307
and 0.0303.

## 4. Samplers: DPM-Solver++(2M)

`sampler="dpm++2m"` is the multistep DPM-Solver++ (Lu et al. 2022), run on the
discrete schedule with timesteps spaced uniformly in log-SNR. It works in data
prediction, clamped to [-1, 1]. Each step reuses the previous x0 prediction
for a second-order update, so it costs one network call, the same as DDIM. It
is the new default in the GUI and the CLI (20 steps). DDIM and DDPM are
unchanged.

**Exact-denoiser test.** The data is N(0, 0.3²) and the network is the exact
posterior mean. The ODE then maps N(0, 1) to N(0, 0.3²), and any error in the
sample std is pure discretisation error.

| sample std error | 5 steps | 10 | 20 | 50 | 100 |
|---|---|---|---|---|---|
| DDIM, linear | -64.6 % | -37.3 % | -19.1 % | -8.0 % | -4.1 % |
| DPM-Solver++(2M), linear | -37.2 % | -1.9 % | +0.9 % | +0.1 % | +0.1 % |
| DDIM, cosine | -43.2 % | -21.2 % | -10.5 % | -4.3 % | -2.2 % |
| DPM-Solver++(2M), cosine | -45.4 % | -9.8 % | +1.7 % | +0.3 % | 0.0 % |

**Trained model** (MNIST-16, the zero-init row of § 2):

| classifier-FID | 10 steps | 20 steps | 50 steps |
|---|---|---|---|
| DDIM | 384.9 | 302.3 | 275.2 |
| DPM-Solver++(2M) | 271.9 | 269.0 | 269.9 |

With 10 steps, DPM-Solver++ beats DDIM with 50, which is 5x fewer network
calls. Below about 8 steps neither sampler is good.

## 5. Class-conditional generation and guidance

`class_conditional=True` (GUI: *Classes from sub-folders*; CLI:
`image-train --classes`) treats every first-level sub-folder as a class. The
names are stored in the checkpoint.

* **Training.** The label embedding is added to the timestep embedding. 10 % of
  the labels are replaced by a "no class" label (`class_dropout`), so the same
  network also learns the unconditional model. Images directly in the root
  folder train as "no class".
* **Generation.** `generate(class_label="cats", guidance_scale=2.0)` takes a
  name, an index, or `"all"` (one of each, cycling).
  * A scale above 1 applies classifier-free guidance, `u + w * (c - u)`. Both
    halves run in one batched call, so it costs twice as much per step.
  * `class_label=None` samples the unconditional model.
* **Interfaces.** The GUI has a *Class* picker and a *Guidance* field. The CLI
  has `image-generate --class NAME --guidance 2`.

MNIST-16 with one sub-folder per digit and the same budget (1,570 steps), with
DPM-Solver++ at 20 steps. *Class accuracy* is the share of samples that the
classifier assigns to the requested digit:

| MNIST-16, 1,570 steps, DPM-Solver++ 20 steps | classifier-FID | class accuracy | classifier confidence |
|---|---|---|---|
| unconditional model (2.10 defaults) | 235.2 | — | 0.831 |
| class-conditional model, no class | 272.0 | — | 0.825 |
| class-conditional, guidance 1 (none) | 99.7 | 75.1 % | 0.915 |
| class-conditional, guidance 2 (default) | **52.7** | 94.1 % | 0.979 |
| class-conditional, guidance 4 | 171.5 | 98.5 % | 0.991 |

At this budget, asking for a class is the largest quality gain on this page:
FID 52.7 against 235 for the unconditional model. Too much guidance trades
diversity for fidelity: at scale 4 almost every sample is the right digit,
but FID is three times worse than at scale 2.

## 6. Options that were measured but stay off

* `scale_shift_norm=True` (AdaGN, timestep FiLM after the norm): FID 250.3
  against 275.2 with 4 heads. The EMA held-out loss is 1.8 % lower but the
  raw-weight loss 1.1 % higher, and steps are 5 % slower. The evidence is
  mixed in a single run, so it is available but not the default.
* `torch.compile`: no gain on CPU (§ 1).

## 7. Command line

    auralite image-train --images photos/ --preset small --epochs 40 --model gen.pt \
        --samples after.png            # --classes, --val-split 0.05, --resume gen.pt
    auralite image-generate --model gen.pt -n 16 --steps 20 --out grid.png \
        [--sampler ddim] [--class cats --guidance 2] [--out-dir each/]

`--params` takes a JSON file with any key of `ImageGenEngine.DEFAULTS`, and a
misspelt key exits with code 2 and the nearest name. `--preset` accepts
`tiny`, `small`, `medium` or `large`.

## 8. Compatibility

* **Checkpoints.** 2.9 checkpoints (`.pt` and `.safetensors`) load and sample
  exactly as before:
  * the legacy betas, through the missing `schedule_version`;
  * their stored head count;
  * no class embedding;
  * EMA treated as warmed up.
* **What changes for new models.**
  * The GUI defaults change to 20 DPM-Solver++ steps, automatic heads, and
    5 % validation.
  * The engine API keeps `sampler="ddim"` and `val_split=0`.
  * `train()` still returns the list of epoch losses.
* **The `.safetensors` format** now appears in the GUI's save and load dialogs.
  It needs the optional `safetensors` package (`pip install safetensors`, or
  the `[safetensors]` / `[full]` extra). Without it, choosing a `.safetensors`
  name shows an error that says so, and nothing is written; `.pt` needs
  nothing extra.

## 9. Reproducing

    python benchmarks/image_bench.py attn step sampler decode
    python benchmarks/image_bench.py step --preset small --batch 8

The MNIST quality runs used the 5,000-image subset described at the top
(seed 0 for the subset, seed 1 for training). The classifier was a 3-conv CNN
trained for 3 epochs on the 60,000 MNIST training images at 16 px. FID used
1,000 samples drawn with seeds 0, 250, 500 and 750, 250 images each.

2.14.2 (§ 10–14): the same subset and sampling seeds with a newly trained
classifier of the same design; the training seed is 1 unless a table names
others. The time per training step and per sampled image of each model type:

    python benchmarks/image_bench.py types --size 16 --files 96 --repeats 3

# 2.14.2: three model types, a quality score, detailed progress

2.14.2 adds two ways to train an image generator next to the diffusion model
of § 1–9, a way to measure a model without pretrained networks, and progress
for every stage of a run. The measurements below were made with a new MNIST
classifier (98.6 % test accuracy, the same three-convolution design as § 9),
so their FID values are not comparable with the 2.10 tables above. On this
classifier 1,000 real training images score **14.1** (the floor at this sample
size). Every run trained on the same 5,000 images at 16 px for 10 epochs
(1,570 steps at batch 32); 1,000 samples (seeds 0, 250, 500, 750) were scored.

## 10. Model types

`model_type` (GUI: *Model Type*; CLI: `image-train --type`) picks what is
trained. All three types read the same folders (sub-folders as classes),
validate, stop early, continue training, autosave and save `.pt` or
`.safetensors` the same way.

| | `diffusion` (default) | `flow` | `gan` |
|---|---|---|---|
| network | self-attention U-Net | the same U-Net | ResNet generator + discriminator |
| training objective | DDPM noise prediction (ε / x0 / v), Min-SNR | rectified flow: velocity on a straight noise→image path | relativistic GAN loss with R1 + R2 |
| sampling | 10–50 network calls (DPM-Solver++, DDIM, DDPM) | 10–50 calls (DPM-Solver++, Euler, Heun) | **one call per image** |
| validation | held-out denoising loss | held-out velocity loss | patch SWD of samples vs. held-out images (§ 13) |
| class guidance | classifier-free | classifier-free | a class label per image (projection discriminator) |
| presets | *Tiny*, *Small*, *Medium*, *Large* | *Flow Tiny*, *Flow Small* | *GAN Tiny*, *GAN Small* |
| checkpoint `format` | `auralite-image-ddpm-v2` | `auralite-image-flow-v1` | `auralite-image-gan-v1` |

Results on MNIST-16 at the same budget (*Tiny* presets, 1,570 steps; lower
FID is better):

| model | classifier-FID | sampling 1,000 images | training (1,570 steps) |
|---|---|---|---|
| diffusion, DPM-Solver++ 20 steps | 187.5 | 27.7 s | 251 s |
| **flow matching, DPM-Solver++ 20 steps** | **74.4** | 27.3 s | 253 s |
| GAN, the defaults (mean of seeds 1–3: 195, 131, 141) | 155.5 | **0.2 s** | 271–282 s |

Flow matching gave the best images of the three at this budget: a 2.5 times
lower FID than diffusion with the same network, the same number of steps and
the same sampling cost. The GAN came second and samples about 130 times
faster than both. In the samples the difference is plain: flow matching draws
clean digits, the GAN recognisable but softer ones, and the diffusion model
at this budget many broken strokes.

The cost of a training step and of a sampled image (`image_bench.py types`,
3 channels, 16 px, batch 32, 2 threads):

| model | parameters | ms per training step | ms per sampled image |
|---|---|---|---|
| diffusion (*Tiny*) | 715 k | 164 | 29.2 (DPM-Solver++, 20 steps) |
| flow matching (*Flow Tiny*) | 715 k | 167 | 28.2 (DPM-Solver++, 20 steps) |
| GAN (*GAN Tiny*), R1 + R2 every step | G 172 k + D 164 k | 200 | 0.2–0.5 |
| GAN, R1 + R2 every 4 steps | G 172 k + D 164 k | 101 | 0.2 |

## 11. Flow matching (`model_type="flow"`)

`auralite/image_flow.py` trains the U-Net with **rectified flow** (Liu et al.
2022; Lipman et al. 2022), the objective of Stable Diffusion 3 and FLUX:

* the training example is `x_t = (1 - t) * x0 + t * noise` on the straight path
  from the image (t = 0) to pure noise (t = 1);
* the network predicts the velocity `noise - x0` (plain MSE, no weighting);
* `t` is drawn from a **logit-normal** distribution (`flow_time_sampling`,
  Esser et al. 2024), which puts more training on the middle of the path;
  `uniform` is the alternative;
* sampling integrates `dx/dt = v(x, t)` from t = 1 to 0. **DPM-Solver++(2M)**
  applies because the path is a Gaussian diffusion with `alpha = 1 - t`,
  `sigma = t`; **Euler** (what DDIM is on this path, so `ddim` is accepted as
  its name) and **Heun** (second order, two calls per step) are the
  alternatives. A diffusion-only sampler (`ddpm`) is refused with a message.

**Exact-velocity test.** Data N(0, 0.3²) with the exact velocity field: the
ODE maps N(0, 1) onto the data, so the error of the sample std is the
solver's own error (4,000 samples; the noise itself is 0.7 % below 1):

| sample std error | 4 steps | 8 | 10 | 20 | 50 |
|---|---|---|---|---|---|
| Euler (1 call per step) | -41.3 % | -22.3 % | -18.3 % | -9.9 % | -4.6 % |
| Heun (2 × steps − 1 calls) | -29.8 % | -8.4 % | -5.6 % | -2.0 % | -1.1 % |
| DPM-Solver++(2M) (1 call per step) | -38.6 % | -11.8 % | -6.6 % | **-0.2 %** | -0.3 % |

**Trained model** (MNIST-16, *Flow Tiny* settings, 1,570 steps):

| sampler | network calls | classifier-FID |
|---|---|---|
| Euler, 20 steps | 20 | 89.1 |
| Heun, 10 steps | 19 | 75.0 |
| DPM-Solver++(2M), 20 steps | 20 | **74.4** |

DPM-Solver++ is the default in the GUI and the CLI for both diffusion and flow
models. The validation loss of a flow model is the velocity loss at 8 fixed
times with fixed noise, so it is comparable across epochs like the diffusion
one.

## 12. GAN (`model_type="gan"`)

`auralite/image_gan.py` follows **R3GAN** (Huang et al., NeurIPS 2024, "The GAN
is dead; long live the GAN! A Modern GAN Baseline"): a loss that makes GAN
training well-behaved, so none of the classic tricks are needed.

* **Loss.** Relativistic pairing (RpGAN): the discriminator scores a generated
  image *relative to* a real one, `softplus(D(fake) - D(real))`, which removes
  the mode-dropping minima of the classic loss. `gan_loss` also offers `hinge`
  (SAGAN, BigGAN) and `nonsaturating` (the original GAN loss).
* **R1 + R2.** Zero-centred gradient penalties on real (R1) and generated (R2)
  images, `gamma / 2 * ||grad D||²` (`gan_gamma`). They are what makes the
  relativistic loss converge. They need a second backward pass, so they run
  every `gan_reg_every` steps with the weight scaled by the interval (lazy
  regularisation, StyleGAN2). With a cosine LR schedule `gamma` decays with
  it to 10 %, as R3GAN lowers its gamma over training.
* **Networks.** A symmetric ResNet generator and discriminator without
  normalisation layers: every block is a 1x1 → grouped 3x3 (16 channels per
  group) → 1x1 inverted bottleneck, resolutions change in separate transition
  layers with bilinear resampling, and **fix-up initialisation** (the last
  convolution of each block starts at zero, the other two are scaled by
  `L^-0.25`) replaces the normalisation. The generator starts from a learned
  4x4 basis (3x3 for 24/48/96 px) modulated by the latent; the discriminator
  ends with a depthwise 4x4 convolution and a linear layer. Width per
  resolution: `base_channels` doubling up to 4x; `channel_mults` and
  `num_res_blocks` apply as for the U-Net.
* **Optimiser.** Adam with betas (0, 0.99) for both networks, at `gan_lr_g` and
  `gan_lr_d` (two time scales are possible), following `lr_schedule`.
* **Augmentation** (`gan_augment`, off by default). DiffAugment (Zhao et al.
  2020): brightness, saturation, contrast, translation by up to 1/8 and cutout
  of half the size, applied to real *and* generated images before the
  discriminator and differentiable, so the generator learns through it and
  does not reproduce it. `ada` gives each operation ADA's **adaptive
  probability** (Karras et al. 2020): every 4 steps `r_t`, the mean sign of
  `D(real) - D(fake)` (how consistently the discriminator ranks a training
  image above a generated one), is compared with `gan_ada_target` (0.6);
  above it the probability rises, below it falls, by up to 1 per 100,000
  images. `diffaug` keeps it at 1. Augmentation exists against a
  discriminator that memorises a small data set over a long run (rising
  validation SWD); at the budgets measured here it did not pay off, even
  with 500 images (below), so it is off by default.
* **EMA generator and truncation.** Samples come from an EMA of the
  generator's weights (with the 2.10 warmup). `generate(truncation=1.5)`
  (GUI: *Truncation*; CLI: `--truncation`) resamples latent values beyond
  ±1.5: cleaner, less varied images (BigGAN's truncation trick).
* **Classes.** The label embedding is concatenated to the latent in the
  generator, and the discriminator projects its features onto a class
  embedding (projection discriminator). `class_dropout` trains a "no class"
  label too, which is what `class_label=None` samples.
* **Validation.** There is no likelihood to hold out, so each epoch the EMA
  generator draws images from fixed latents and the patch SWD (§ 13) against
  the held-out images is the validation value (`val_metric: "swd"` in the
  summary), used by early stopping and best-weights restore.

The GAN does not use `lr`, `weight_decay`, `grad_clip`, `accumulation_steps`,
`use_compile` or gradient checkpointing; a log line names the ones that were
set. `amp_dtype="fp16"` runs bf16 autocast (a gradient scaler does not fit the
double backward of R1/R2). A checkpoint holds the generator, its EMA and the
discriminator (to continue training) plus ADA's probability;
`save_model(path, include_discriminator=False)` writes half the size for
generation only, and continuing such a file trains a new discriminator.

**Finding the defaults** (MNIST-16, 1,570 steps, base 32: G 172 k + D 164 k
parameters; classifier-FID of the EMA generator and of the raw weights; one
change per row):

| change (from lr 1e-3, γ 0.05, R1 + R2 every step, ADA) | FID, EMA generator | raw weights |
|---|---|---|
| learning rate 2e-4 (R3GAN's value, for batch 512) | 394 | — |
| learning rate 1e-3 | 212 | 472 |
| learning rate 2e-3 | 478 (diverged) | 1012 |
| γ 0.01 | 208 | 284 |
| **γ 0.5** | **147** | 459 |
| γ 1 | 269 | 397 |
| γ 0.5 with a cosine LR schedule | 250 | 373 |
| R1 + R2 every 4 steps (lazy) | 201 | 243 |
| augmentation `off` | 205 | 202 |
| augmentation `diffaug` (always on) | 474 | 442 |
| `base_channels` 64 (G 536 k + D 520 k) | 216 | 250 |

GAN runs this short vary a lot with the seed, so the candidates were run
with seeds 1–3:

| configuration | seeds 1, 2, 3 | mean |
|---|---|---|
| γ 0.05, R1 + R2 every 4 steps, ADA | 201, 229, 240 | 223 |
| γ 0.5, every step, ADA | 147, 153, 265 | 189 |
| **γ 0.5, every step, no augmentation** (the defaults) | 195, 131, 141 | **156** |

With 500 images (a subset of the 5,000, 100 epochs: the same 1,600
steps) the GAN without augmentation scored 138 and with ADA 186; ADA's
probability had risen to 0.22. At these budgets the discriminator does not
get to memorise even 500 images, and the augmentation only makes its task
harder.

The defaults are therefore **learning rate 1e-3 for both networks, γ 0.5,
R1 + R2 every step, no augmentation, a constant LR, EMA**.
`gan_reg_every=4` trains twice as fast per step (101 against 200 ms in the
table of § 10) for a somewhat worse result. Sampling from the EMA generator
(the default; *EMA weights* in the Generate row) is the safer choice at a
learning rate this high: it scored better than the raw weights in 15 of the
23 runs that measured both, often by a factor of two (147 against 459,
212 against 472); where the raw weights won, they won by 2–38 %.

## 13. Quality without pretrained networks: patch SWD

FID needs an Inception network trained on ImageNet: weights AuraLite does not
ship and cannot download in an offline training run. `image_metrics.py`
implements the **multi-scale sliced Wasserstein distance** of ProGAN (Karras
et al. 2018), which needs nothing:

* every image becomes a Laplacian pyramid down to 8 px (16 px → levels 16 and
  8): the finest levels hold edges and texture, the coarsest the layout;
* from each level, 128 random 7x7 patches per image become descriptors
  (the same positions in both sets), normalised with the statistics of the
  **real** patches — ProGAN normalises each set on its own, which hides a
  colour or brightness shift;
* the distance is the average 1-D Wasserstein distance over 4 x 128 random
  directions, times 1000, per level and averaged.

Patch positions and directions are seeded, so the same two sets always give
the same number. Two sets of *real* images score above zero, so
`ImageGenEngine.evaluate(folder, n=256)` also reports that floor, from two
disjoint sets of real images:

    auralite image-eval --model gen.pt --images photos/ -n 256 [--json]
    # GUI: Generate → 📏 Measure Quality (against the selected folder)

What it can and cannot tell (MNIST-16, 1,000 images per set; real training
vs. real test images score **52.9**):

| 1,000 images | classifier-FID | SWD |
|---|---|---|
| real training digits | 14.1 | 52.9 |
| flow matching, Euler 20 steps | 89.1 | 56.0 |
| flow matching, DPM-Solver++ 20 steps | 74.4 | 61.6 |
| diffusion, DPM-Solver++ 20 steps | 187.5 | 75.3 |
| GAN, γ 0.05 (EMA generator) | 211.7 | 73.3 |
| GAN, γ 0.5 (EMA generator) | 147.3 | 80.9 |
| GAN, γ 0.05 (raw weights) | 472.4 | 209.5 |
| GAN, learning rate 2e-3 (diverged) | 478.3 | 220.9 |

Over all 51 results of these runs (every GAN setting, raw and EMA, and the
U-Net models), the rank correlation between SWD and the classifier-FID is
0.69; among the 37 with an FID below 300 it is 0.53.

SWD separates a bad model from a usable one — it fell from 210 to 73 as the
FID fell from 472 to 212 — but near the floor its sampling noise is larger
than the difference between good models. Compare models on the same images
with the same `n`, read the floor next to the score, and look at the samples.

## 14. Progress: every stage on one bar

`train()` of the image engine now reports the stages before training like the
text engine (2.14.1): decoding the images (done/total, images per second,
time left), building the networks and compiling, then every batch with the
epoch as its stage, and the validation (held-out images, or the images
generated for the GAN's SWD). In the GUI one progress bar shows the stage
before the first step — a stage without a count (building, compiling)
animates the bar — and the whole run from then on, with the stage's
done/total, speed and time left on the line under it. This holds for the
*Image* tab and the training tab, where 2.14.1 drew a second bar for the
stages. `auralite image-train` prints the same stages (one line updated in
place on a terminal) and the GAN's G and D losses per batch, with ADA's
augmentation probability when `gan_augment` is `ada`.

Stop while the images are decoded ends at once: `train()` returns `[]`, the
summary says `stopped_during: "decode"`, and the model in memory is the one
it was. `auralite image-train` now stops on Ctrl+C too: after the current
batch, saving what was trained (during the decoding nothing is saved and
the exit code is 130); a second Ctrl+C aborts.

## 15. Compatibility (2.14.2)

* Every 2.9–2.14.1 image checkpoint loads and samples as before; a missing
  `model_type` means diffusion, and diffusion checkpoints keep the 2.10 format
  name, so 2.14.1 still loads the diffusion models 2.14.2 writes.
* Flow and GAN checkpoints have their own format names: older versions refuse
  them ("Not an AuraLite image-generator checkpoint") instead of sampling them
  as a diffusion model.
* `model_type` and `latent_dim` belong to the architecture: *continue training*
  keeps the type of the model in memory whatever the new settings say (the log
  and the GUI's status line say so).
* `ImageGenEngine.ddpm` is the DDPM of a diffusion model and `None` for the
  other types; `process` holds the DDPM or the flow. `build_model` still
  builds the U-Net; `build_generator` / `build_discriminator` build the GAN.
* Presets carry `model_type`; the CLI accepts their short names (`tiny`,
  `flow-small`, `gan-tiny`, ...) and `--type gan --preset tiny` picks
  *GAN Tiny*.
