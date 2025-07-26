import torch
import os
from torchvision.utils import make_grid, save_image
from configs import cfg
from models.generator import Generator
from models.discriminator import Discriminator
from models.feedback import FeedbackEngine

def generate_images(checkpoint_path, save_path='D:/Web project/neuroloop gan/samples/generated_images.png', num_images=25):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    feedback_engine = FeedbackEngine(None, use_feedback=cfg.use_feedback_fusion)
    feedback_adapter = torch.nn.Linear(1024, cfg.z_dim).to(device)
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        generator.load_state_dict(checkpoint['G'])
        discriminator.load_state_dict(checkpoint['D'])
        print(f"[✓] Loaded checkpoint from {checkpoint_path}")
    else:
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    generator.eval()
    discriminator.eval()
    with torch.no_grad():
        z = torch.randn(num_images, cfg.z_dim).to(device)
        fake_images = generator(z, feedback=None)
        if cfg.use_feedback_fusion:
            raw_feedback = discriminator.extract_features(fake_images)
            feedback = feedback_engine(raw_feedback)
            if feedback is not None:
                feedback = feedback_adapter(feedback)
                feedback = feedback + 0.02 * torch.randn_like(feedback)
            fake_images = generator(z, feedback=feedback)
        else:
            fake_images = generator(z, feedback=None)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    grid = make_grid(fake_images, nrow=int(num_images ** 0.5), normalize=True, pad_value=1)
    save_image(grid, save_path)
    print(f"[✓] Saved {num_images} images to {save_path}")

if __name__ == "__main__":
    cfg.model_dir = 'D:/Web project/neuroloop gan/models'
    cfg.z_dim = 100
    cfg.use_feedback_fusion = True
    cfg.image_size = 64
    checkpoint_path = f"{cfg.model_dir}/model_epoch_30.pth"
    generate_images(checkpoint_path)