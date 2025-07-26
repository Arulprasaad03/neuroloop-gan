from train.train_loopgan import train_loopgan
from utils.dataset import CelebAHQDataset
from configs import cfg

if __name__ == "__main__":
    # Override cfg settings for consistency

    # Test dataset
    try:
        dataset = CelebAHQDataset(image_size=cfg.image_size)
        print(f"Loaded dataset with {len(dataset)} images")
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        print("Ensure DATASET_DIR in dataset.py is set to 'D:/celeba_hq_256' and contains images")
        exit(1)

    # Run training
    resume_epoch = 40
    train_loopgan(resume_epoch=resume_epoch)