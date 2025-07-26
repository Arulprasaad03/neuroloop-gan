# 🧠 NeuroLoop GAN: Learning Structured Representations with Wasserstein GAN

**NeuroLoop GAN** is a deep generative model built on **Wasserstein GAN with Gradient Penalty (WGAN-GP)**, designed to generate structured, realistic patterns in cyclic or looped domains. Inspired by recurrent biological and cognitive signal loops, this architecture is suitable for tasks such as structured face synthesis, temporal emotion transitions, and medical signal generation.

---

## 📌 Key Features

- ✅ **Wasserstein GAN with Gradient Penalty** for stable adversarial training.
- 🎯 **Latent feedback loop** in the generator, allowing previous latent states to influence the next generation step.
- 🧬 **Modular architecture**: clear separation of generator, discriminator, feedback engine, monitoring, and dataset loading.
- 🖼️ Generates high-resolution images (e.g., 64x64 CelebA-HQ faces).
- 🧪 Designed to explore structured latent spaces for tasks like emotion evolution or EEG pattern replay.

---

## 🧠 What is NeuroLoop?

NeuroLoop introduces an **inner latent feedback loop** to the GAN generator, allowing previous latent states to influence the next generation step — loosely mimicking recurrent processing in neural systems.

> Think of it as:  
> _"Not just generate an image — generate the memory of generating one."_  
> A looped evolution of feature space → better consistency → better realism.

---

## 🚀 Tech Stack

| Component         | Library       |
|------------------|---------------|
| GAN Base         | PyTorch       |
| Loop Encoding    | Custom Python Modules (`models/feedback.py`) |
| Visualization    | Matplotlib, torchvision |
| Preprocessing    | PIL, NumPy, torchvision.transforms |
| Training Control | TensorBoard (via `torch.utils.tensorboard`), tqdm |

---

## 📁 Project Structure

```text
NeuroLoop-GAN/
├── models/
│   ├── generator.py           # Generator with looped latent architecture
│   ├── discriminator.py       # WGAN-GP-compatible discriminator
│   ├── feedback.py            # Feedback engine for latent loop
│   └── blocks.py              # Residual blocks for generator
├── train/
│   └── train_loopgan.py       # Main training script
├── utils/
│   ├── dataset.py             # CelebA-HQ dataset loader
│   └── monitor.py             # Mode collapse monitoring
├── configs/
│   └── loopgan_config.py      # Training and model configuration
├── results/
│   └── generated_samples/     # Generated outputs during training
├── requirements.txt           # Python dependencies
├── main.py                    # Entry point for training
├── README.md
└── .gitignore
```

---

## 💻 How to Run

1. **Clone the repository**
    ```bash
    git clone https://github.com/yourusername/NeuroLoop-GAN.git
    cd NeuroLoop-GAN
    ```

2. **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3. **Prepare the dataset**
    - Download CelebA-HQ images and place them in `D:/celeba_hq_256` (default path).
    - Ensure images are in `.jpg`, `.jpeg`, or `.png` format.

4. **Start training**
    ```bash
    python main.py
    ```
    - Training resumes by default (see `main.py`).
    - Adjust configuration in `configs/loopgan_config.py` as needed.

---

## 🧪 Sample Output
![alt text](image-1.png)
![alt text](image.png)
![alt text](image-2.png)
![alt text](image-3.png)
---

## ⚙️ Configuration

All training and model parameters are set in `configs/loopgan_config.py`:
- `image_size`, `batch_size`, `epochs`, `z_dim`, learning rates, gradient penalty, diversity weight, etc.
- Output directories for results, models, and TensorBoard logs.

---

## 📊 Performance Tuning Tips

- **Spectral normalization** is used in both generator and discriminator for improved gradient flow.
- **Latent loop depth** and feedback fusion can be toggled via `cfg.use_feedback_fusion`.
- **Diversity loss** is implemented to encourage output variety.
- **Mode collapse monitoring** via ResNet-18 feature similarity (`utils/monitor.py`).
- **Gradient penalty** and **learning rate scheduling** are used for stability.

---

## 📎 Papers and Inspirations

- [Wasserstein GAN](https://arxiv.org/abs/1701.07875)
- [Improved Training of Wasserstein GANs](https://arxiv.org/abs/1704.00028)

---

## 📌 Status

> 🚧 In development — refining training dynamics and loop initialization strategies.
> Deployment-ready model & streamlit demo coming soon.

---

## 🤝 Contributing

Pull requests and feedback are welcome. For major changes, please open an issue first.

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for