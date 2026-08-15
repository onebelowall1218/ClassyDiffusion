# ClassyDiffusion — a diffusion model built from scratch 🎨

A **class-conditional image-generation diffusion model** implemented from first principles in PyTorch — **DDPM + UNet + self-attention**, ~64M parameters — trained to generate 32×32 CIFAR-10 images conditioned on a class label.

No diffusion library. The forward noising, reverse denoising, noise schedule, UNet, and conditioning are all written by hand — the point was to *understand* diffusion, not import it.

> 📓 The full model and training loop live in
> [`prompt-based-diffusion-model-with-attention.ipynb`](prompt-based-diffusion-model-with-attention.ipynb)
> · also on [Kaggle](https://www.kaggle.com/code/anjumulazim/prompt-based-diffusion-model-with-attention)

---

## What's inside
- **Forward process** — progressively adds Gaussian noise over T steps (`q(xₜ|x₀)`).
- **Reverse process** — a UNet learns to predict and remove that noise, step by step.
- **UNet + attention** — down/mid/up blocks with timestep and class embeddings.
- **Class conditioning** — generate a specific class on demand.
- **Constraint-aware training** — architecture and batch size tuned to fit a single GPU.

## How it works
```
class label ─┐
             ▼
xₜ ──► UNet (down → mid+Attention → up, + timestep & class embeddings) ──► predicted noise εθ
             ▲
   noise schedule βₜ   (forward: add noise · reverse: remove noise)
```

## Results
| | |
|---|---|
| Parameters | ~64M |
| Dataset | CIFAR-10 (4 classes, ~20k images) |
| Resolution | 32×32 |
| Hardware / time | Kaggle P100 · ~12h |
| Sampler | DDPM |

<!-- TODO(anjum): add a generated-samples image so recruiters SEE the output. Save it as samples.png and it'll render here:
![samples](samples.png) -->

## Run it
The project is a self-contained notebook — the easiest path is Kaggle (free P100):

1. Open the [Kaggle notebook](https://www.kaggle.com/code/anjumulazim/prompt-based-diffusion-model-with-attention) and click **Copy & Edit**, or
2. Run locally:
   ```bash
   git clone https://github.com/onebelowall1218/ClassyDiffusion
   cd ClassyDiffusion
   pip install torch torchvision numpy matplotlib
   jupyter notebook prompt-based-diffusion-model-with-attention.ipynb
   ```

## Stack
`PyTorch` · `NumPy` · `CIFAR-10` · DDPM · UNet · self-attention
