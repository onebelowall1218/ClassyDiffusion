# Development Guide

> **How do I actually work with this project?**
> The whole project is one notebook, `prompt-based-diffusion-model-with-attention.ipynb`. Cell numbers below are 0-indexed. Search by function name if your editor numbers cells differently.

---

## Prerequisites

| Need | Notes |
|------|-------|
| Python | 3.10 (the notebook metadata records 3.10.13) |
| TensorFlow 2 + Keras | Must provide `keras.layers.GroupNormalization`. ⚠️ **Exact version: UNKNOWN / NEEDS VERIFICATION.** The notebook ran on Kaggle Docker image `30646` |
| GPU | Strongly recommended. The UNet has 64.8M params, and there are two copies (network + EMA) |
| Internet | The first run downloads CIFAR-10 via `tensorflow_datasets` |

## Installation

**Option A: Kaggle (recommended).** Open the [Kaggle notebook](https://www.kaggle.com/code/anjumulazim/prompt-based-diffusion-model-with-attention) → **Copy & Edit** → enable a GPU accelerator and internet.

**Option B: local.**

```bash
pip install tensorflow tensorflow-datasets numpy matplotlib tqdm pandas
jupyter notebook prompt-based-diffusion-model-with-attention.ipynb
```

> [!NOTE]
> There is no `requirements.txt`. The list above is taken from the notebook's imports. `pandas`, `tqdm`, and `shutil` are imported but not used by the model.

## Configuration

All knobs live in **cell 5** as module-level globals:

| Variable | Value | Effect |
|----------|-------|--------|
| `img_size` / `img_channels` | 32 / 3 | Input resolution |
| `num_classes` | 4 | Size of the one-hot conditioning vector (also hard-coded as `shape=(4,)` in `build_model`) |
| `total_timesteps` | 1000 | Diffusion steps T |
| `batch_size` | 32 | The logged `fit` call uses `batch_size*2` = **64** |
| `num_epochs` | 50 | The logged `fit` call uses `num_epochs//2` = **25** |
| `learning_rate` | 2e-4 | Adam |
| `first_conv_channels` × `channel_multiplier` | 64 × [1,2,4,8] | UNet widths 64/128/256/512 |
| `has_attention` | [F, F, T, T] | Attention at the 256- and 512-channel levels |
| `num_res_blocks` / `norm_groups` | 2 / 8 | Blocks per level / GroupNorm groups |

Other hard-coded values:

- `GaussianDiffusion`: `beta_start=1e-4`, `beta_end=0.02`
- `DiffusionModel`: `ema=0.999`, label-dropout keep probability `0.9`
- Classes used: `included_labels = [0,1,2,3]` → `["airplane", "car", "bird", "cat"]`

> [!WARNING]
> **Changing the classes means editing 4 places:** `included_labels`, `num_classes`, the `shape=(4,)` in `build_model`, and the two hard-coded class-name lists (`embedding_to_name`, `show_image_with_labels`).

## Environment Variables

None. All paths are hard-coded:

| Path | Used for |
|------|----------|
| `/kaggle/working/` | `tfds` `data_dir` (dataset cache) |
| `/kaggle/working/models/…` | Checkpoint load and save |

Running outside Kaggle means editing these paths.

---

## Running Locally

The saved notebook was **not executed top to bottom**, as its execution counts show. Use this order instead:

**🏋️ Train from scratch**

1. Run cells **0–18**: imports, config, data, `GaussianDiffusion`, UNet blocks.
2. Run cells **20–24**: `DiffusionModel`, build both UNets, compile.
3. **Skip cell 25** (`load_model`).
4. Run cells **37–39**: re-augment, shuffle, build `train_ds`.
5. **Skip cell 42** (`load_model`), then run cell **44**: `model.fit(...)`.
6. Run cell **46** to save both networks under `/kaggle/working/models/`.
7. Run `model.plot_images()` to check the results.

**🎨 Sample from existing checkpoints**

1. Put the checkpoints under `/kaggle/working/models/`. **They are not in this repo.**
2. Run cells 0–1, 3, 5, 17–18, 20–25.
3. Call `model.plot_images()` for random classes, or `text_to_image(model, one_hot_array)` (cells 31–34) for specific ones.

## Important APIs

**`GaussianDiffusion`** (cell 17)

| Method | Input → Output | Role |
|--------|----------------|------|
| `q_sample(x_start, t, noise)` | x₀, t, ε → xₜ | Forward noising in one shot |
| `predict_start_from_noise(x_t, t, noise)` | xₜ, t, ε̂ → x̂₀ | Inverts the forward equation |
| `q_posterior(x_start, x_t, t)` | → mean, var, log-var of q(xₜ₋₁ \| xₜ, x₀) | Posterior |
| `p_sample(pred_noise, x, t, clip_denoised)` | ε̂, xₜ, t → xₜ₋₁ | One reverse step |

**`DiffusionModel`** (cell 20)

| Method | Role |
|--------|------|
| `train_step(data)` | Sample t and ε → noise → predict → MSE → Adam → EMA update |
| `apply_random_mask(emb, p)` | Keep each label element with probability p (label dropout) |
| `generate_images(num_images=16)` | Random labels → 1000-step reverse loop using `ema_network` |
| `plot_images(num_rows=2, num_cols=8)` | Generate and show a titled grid |

**Standalone helpers**

| Function | Role |
|----------|------|
| `build_model(img_size, img_channels, widths, has_attention, …)` | Returns a Keras UNet with inputs `image_input`, `time_input`, `text_input` |
| `text_to_image(model, text_embeddings)` | Sample one image per row of a `(N, 4)` one-hot array |

Example: draw one image of each class.

```python
labels = np.eye(4)[[0, 1, 2, 3]]   # airplane, car, bird, cat
samples, _ = text_to_image(model, labels)
show_image_with_labels(samples, labels)
```

## Important Code Paths

1. **`train_step`**: the whole learning algorithm fits in about 15 lines. Read it first.
2. **`GaussianDiffusion.__init__` + `p_sample`**: where the math lives. `posterior_mean_coef1/2` implement the DDPM posterior.
3. **`ResidualBlock`**: where the timestep and class conditioning are injected.
4. **`build_model` skip handling**: `skips` is pushed 12 times on the way down and popped 12 times on the way up (4 levels × 3).

---

## Testing

There is **no test suite**. Validation is visual (`plot_images`) plus the loss printed by `fit`.

Cheap sanity checks you can add in a cell:

```python
# the forward process at t=999 should be ~N(0,1)
x = processed_filtered_images[:64]
xt = gdf_util.q_sample(x, tf.fill([64], tf.constant(999, tf.int64)), tf.random.normal(tf.shape(x)))
print(float(tf.math.reduce_mean(xt)), float(tf.math.reduce_std(xt)))

# the UNet output shape should match the input
print(network([x[:2], tf.constant([0, 500], tf.int64), np.eye(4, dtype='float32')[[0, 1]]]).shape)  # (2, 32, 32, 3)
```

## Debugging

| Symptom | Where to look |
|---------|---------------|
| Samples are noise or saturated | Check you're sampling from `ema_network` and that `clip_denoised=True` |
| Wrong class in output | Class-name list order must match `included_labels` order (CIFAR-10: 0 airplane, 1 automobile, 2 bird, 3 cat) |
| Samples ignore the class | Label dropout is too high, or the label isn't reaching `text_input`. Try the fixed labels in cell 33 |

> [!TIP]
> `generate_images` takes 1000 `predict()` calls. For quick checks, sample 4 images, not 16.

## Deployment

None. There is no serving code, exported model, or CI in the repository.

## Common Problems

| Problem | Fix |
|---------|-----|
| `OSError` / "No file or directory" at `load_model` | The checkpoints aren't in the repo. Skip cells 25 and 42, or provide your own |
| `AttributeError: GroupNormalization` | TF/Keras is too old. Upgrade TensorFlow |
| GPU out of memory | Lower `batch_size` in the `fit` call (it currently doubles `batch_size`) |
| Kaggle session times out mid-training | There's no checkpoint callback. Train in shorter chunks and save between them |
| Paths fail locally | Replace the `/kaggle/working/` paths |
