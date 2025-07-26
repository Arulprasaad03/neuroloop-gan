import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter
from models.generator import Generator
from models.discriminator import Discriminator
from models.feedback import FeedbackEngine
from utils.monitor import ModeCollapseMonitor
from configs import cfg
from tqdm import tqdm
import os
from utils.dataset import CelebAHQDataset
from collections import deque
from torch.nn.functional import cosine_similarity
import sys

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Feedback adapter: map 1024 -> cfg.z_dim
feedback_adapter = nn.Linear(1024, cfg.z_dim).to(device)

def compute_gradient_penalty(D, real_samples, fake_samples, lambda_gp=cfg.lambda_gp):
    """Compute gradient penalty for WGAN-GP."""
    alpha = torch.rand(real_samples.size(0), 1, 1, 1, device=device)
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
    d_interpolates = D(interpolates, labels=None)
    grad_outputs = torch.ones_like(d_interpolates)
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambda_gp
    return gradient_penalty

def train_loopgan(resume_epoch=0):
    os.makedirs(cfg.result_dir, exist_ok=True)
    os.makedirs(cfg.model_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=cfg.tensorboard_dir)
    print(f"Using device: {device}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

    try:
        dataset = CelebAHQDataset(image_size=cfg.image_size)
        dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=cfg.pin_memory)
        
        # Debug data normalization and shape
        sample_image = dataset[0]
        print(f"Sample image shape: {sample_image.shape}")
        print(f"Sample image min: {sample_image.min().item()}, max: {sample_image.max().item()}")
        if sample_image.shape != torch.Size([3, cfg.image_size, cfg.image_size]):
            print(f"Error: Expected image shape [3, {cfg.image_size}, {cfg.image_size}], got {sample_image.shape}")
            return
        if not (-1.1 <= sample_image.min().item() <= -0.9 and 0.9 <= sample_image.max().item() <= 1.1):
            print("Warning: Images not normalized to [-1, 1]. Check CelebAHQDataset transforms.")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Ensure DATASET_DIR in dataset.py is set to 'D:/celeba_hq_256' and contains images")
        return

    G = Generator(z_dim=cfg.z_dim, image_size=cfg.image_size).to(device)
    D = Discriminator().to(device)
    
    # Debug generator output shape
    with torch.no_grad():
        test_z = torch.randn(1, cfg.z_dim).to(device)
        test_output = G(test_z, feedback=None)
        print(f"Generator output shape: {test_output.shape}")
        if test_output.shape[2:] != (cfg.image_size, cfg.image_size):
            print(f"Error: Generator output shape {test_output.shape[2:]} does not match expected ({cfg.image_size}, {cfg.image_size})")
            return

    opt_G = optim.Adam(G.parameters(), lr=cfg.lr_G, betas=(cfg.beta1, cfg.beta2))
    opt_D = optim.Adam(D.parameters(), lr=cfg.lr_D, betas=(cfg.beta1, cfg.beta2))

    loss_window = 100
    d_loss_queue = deque(maxlen=loss_window)
    g_loss_queue = deque(maxlen=loss_window)
    checkpoint_path = f"{cfg.model_dir}/model_epoch_{resume_epoch}.pth"
    n_critic = cfg.n_critic

    if resume_epoch > 0 and os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            G.load_state_dict(checkpoint['G'])
            D.load_state_dict(checkpoint['D'])
            opt_G.load_state_dict(checkpoint['opt_G'])
            opt_D.load_state_dict(checkpoint['opt_D'])
            n_critic = checkpoint.get('n_critic', cfg.n_critic)
            d_loss_queue = checkpoint.get('d_loss_queue', deque(maxlen=loss_window))
            g_loss_queue = checkpoint.get('g_loss_queue', deque(maxlen=loss_window))
            print(f"Resumed from checkpoint: {checkpoint_path}, n_critic: {n_critic}")
        except Exception as e:
            print(f"Error loading checkpoint {checkpoint_path}: {e}")

    feedback_engine = FeedbackEngine(None, use_feedback=cfg.use_feedback_fusion)
    collapse_monitor = ModeCollapseMonitor(device=device, threshold=0.98)
    collapse_count = 0

    for epoch in range(resume_epoch, cfg.epochs):
        progress_bar = tqdm(dataloader, total=len(dataloader), desc=f"Epoch {epoch+1}/{cfg.epochs}", file=sys.stdout, dynamic_ncols=True, leave=False)
        noise_scale = max(cfg.noise_scale_init * (0.98 ** epoch), cfg.noise_scale_min)

        for i, real_images in enumerate(progress_bar):
            real_images = real_images.to(device)
            bs = real_images.size(0)
            if real_images.shape[1:] != torch.Size([3, cfg.image_size, cfg.image_size]):
                print(f"Error: Batch image shape {real_images.shape[1:]} does not match expected [3, {cfg.image_size}, {cfg.image_size}]")
                return
            real_images_noisy = real_images + noise_scale * torch.randn_like(real_images).to(device)
            d_loss_total = 0

            for _ in range(cfg.n_critic):
                z = torch.randn(bs, cfg.z_dim).to(device)
                with torch.no_grad():
                    fake_images = G(z, feedback=None).detach()
                    if fake_images.shape[1:] != torch.Size([3, cfg.image_size, cfg.image_size]):
                        print(f"Error: Fake image shape {fake_images.shape[1:]} does not match expected [3, {cfg.image_size}, {cfg.image_size}]")
                        return
                    fake_images_alt = G(z, feedback=None).detach()
                    if i == 0 and epoch == resume_epoch:
                        print(f"Fake image min: {fake_images.min().item()}, max: {fake_images.max().item()}")

                real_validity = D(real_images_noisy, labels=None)
                fake_validity = D(fake_images_alt, labels=None)

                # Compute gradient penalty
                gp = compute_gradient_penalty(D, real_images_noisy, fake_images_alt)
                real_validity_smooth = real_validity * 0.9
                fake_validity_smooth = fake_validity * 0.9
                d_loss = torch.mean(fake_validity_smooth) - torch.mean(real_validity_smooth) + gp

                opt_D.zero_grad()
                d_loss.backward()
                torch.nn.utils.clip_grad_norm_(D.parameters(), max_norm=1.0)
                opt_D.step()
                d_loss_total += d_loss.item()

                # Log components
                writer.add_scalar('Loss/Discriminator_Wasserstein', torch.mean(fake_validity_smooth) - torch.mean(real_validity_smooth), epoch * len(dataloader) + i)
                writer.add_scalar('Loss/Discriminator_GP', gp, epoch * len(dataloader) + i)
                writer.add_scalar('Score/Real', torch.mean(real_validity).item(), epoch * len(dataloader) + i)
                writer.add_scalar('Score/Fake', torch.mean(fake_validity).item(), epoch * len(dataloader) + i)

            d_loss_avg = d_loss_total / cfg.n_critic
            d_loss_queue.append(d_loss_avg)

            z = torch.randn(bs, cfg.z_dim).to(device)
            fake_images = G(z, feedback=None)
            fake_images_noisy = fake_images + noise_scale * torch.randn_like(fake_images).to(device)
            fake_validity = D(fake_images_noisy, labels=None)
            adv_loss = -torch.mean(fake_validity)

            similarities = 1 - cosine_similarity(fake_images.view(bs, -1).unsqueeze(1), fake_images.view(bs, -1).unsqueeze(0), dim=2)
            diversity_loss = torch.mean(torch.triu(similarities, diagonal=1))
            similarity_score = torch.mean(similarities).item()
            g_loss = adv_loss + cfg.diversity_weight * diversity_loss

            opt_G.zero_grad()
            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(G.parameters(), max_norm=1.0)
            opt_G.step()
            g_loss_queue.append(g_loss.item())

            if torch.isnan(d_loss) or torch.isinf(d_loss) or torch.isnan(g_loss) or torch.isinf(g_loss):
                print(f"NaN/Inf detected at epoch {epoch+1}, iteration {i}")
                return

            if len(d_loss_queue) >= loss_window and len(g_loss_queue) >= loss_window:
                avg_d_loss = sum(d_loss_queue) / len(d_loss_queue)
                avg_g_loss = sum(g_loss_queue) / len(g_loss_queue)
                n_critic = cfg.n_critic

            collapsed, msg, similarity = collapse_monitor.update_and_check(fake_images_noisy)
            writer.add_scalar('Collapse/Similarity', similarity, epoch)
            if collapsed:
                collapse_count += 1
                for param_group in opt_G.param_groups:
                    param_group['lr'] *= 0.7
                if collapse_count >= 20:
                    print(f"Would reset discriminator at epoch {epoch+1}, but resets disabled for debugging")
            else:
                collapse_count = 0

            progress_bar.set_postfix({
                "G_loss": f"{g_loss.item():.4f}",
                "D_loss": f"{d_loss_avg:.4f}",
                "Sim": f"{similarity_score:.2f}",
                "n_critic": n_critic,
                "Collapse": "Yes" if collapsed else "No",
                "CollapseMsg": msg
            })

            writer.add_scalar('LR/Generator', opt_G.param_groups[0]['lr'], epoch)
            writer.add_scalar('LR/Discriminator', opt_D.param_groups[0]['lr'], epoch)
            writer.add_scalar('Diversity/Weight', cfg.diversity_weight, epoch)
            writer.add_scalar('Diversity/Similarity', similarity_score, epoch)

            # Early stopping
            if collapse_count >= 30:
                print(f"Stopping training due to persistent mode collapse after {collapse_count} epochs")
                writer.close()
                return

        writer.add_scalar('Loss/Generator', avg_g_loss, epoch)
        writer.add_scalar('Loss/Discriminator', avg_d_loss, epoch)
        writer.add_scalar('SimilarityScore', similarity_score, epoch)
        writer.add_images('GeneratedSamples', fake_images_noisy[:8], epoch)

        if epoch % cfg.save_freq == 0 or epoch == cfg.epochs - 1:
            try:
                save_image(fake_images_noisy[:25], f"{cfg.result_dir}/epoch_{epoch}.png", nrow=5, normalize=True)
                z1, z2 = torch.randn(1, cfg.z_dim).to(device), torch.randn(1, cfg.z_dim).to(device)
                alpha = torch.linspace(0, 1, steps=8).view(-1, 1).to(device)
                z_interp = torch.lerp(z1, z2, alpha)
                gen_imgs = G(z_interp, feedback=None)
                save_image(gen_imgs, f"{cfg.result_dir}/interpolate_{epoch}.png", nrow=8, normalize=True)
            except Exception as e:
                print(f"Error saving image: {e}")

            torch.save({
                'G': G.state_dict(),
                'D': D.state_dict(),
                'opt_G': opt_G.state_dict(),
                'opt_D': opt_D.state_dict(),
                'n_critic': n_critic,
                'd_loss_queue': d_loss_queue,
                'g_loss_queue': g_loss_queue
            }, f"{cfg.model_dir}/model_epoch_{epoch}.pth")

        progress_bar.close()

    writer.close()