# Image generator: what changed in 2.10 and how it was measured

This page covers the from-scratch diffusion model behind the GUI *Image* tab,
`ImageGenEngine` and the new `auralite image-train` / `auralite image-generate`
commands. Every number was measured on this repository's code on a 2-vCPU VM
(one physical core with two hyper-threads, torch 2.14.1 CPU). The commands to
reproduce them are in § 9.

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
