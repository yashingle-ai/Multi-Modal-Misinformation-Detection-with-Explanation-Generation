"""
Fine-tune EfficientNet branch for CIFAKE forensics detection.
Focus on deepfake/synthetic image detection while keeping other models frozen.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import os
from PIL import Image
from torchvision import transforms
import random

# Import from existing modules
from training_pipeline import MultiModalMisinfoDetector


class CIFAKEDataset(Dataset):
    """
    Dataset specifically for CIFAKE images (REAL vs FAKE detection).
    No text captions available, so we use placeholder text.
    """
    
    def __init__(self, image_paths, labels, is_train=True):
        """
        Args:
            image_paths (list): List of image file paths
            labels (list): List of labels (0=REAL, 1=FAKE)
            is_train (bool): Whether to apply augmentations
        """
        self.image_paths = image_paths
        self.labels = labels
        self.is_train = is_train
        
        # CLIP-specific transforms (matching MisinfoDataset)
        if is_train:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.481, 0.457, 0.408], std=[0.268, 0.261, 0.275])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.481, 0.457, 0.408], std=[0.268, 0.261, 0.275])
            ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image)
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            # Return blank tensor on error
            image_tensor = torch.zeros(3, 224, 224)
        
        return image_tensor, torch.tensor(label, dtype=torch.long)


def load_cifake_data(cifake_root, sample_per_label=2500):
    """
    Load CIFAKE dataset with balanced sampling.
    
    Args:
        cifake_root (str): Path to CIFAKE root folder
        sample_per_label (int): Number of samples per label (REAL/FAKE)
    
    Returns:
        tuple: (train_paths, train_labels, val_paths, val_labels)
    """
    real_paths = []
    fake_paths = []
    
    # Load REAL images from test folder (train folder doesn't have REAL images)
    real_dir = os.path.join(cifake_root, "test", "REAL")
    if os.path.isdir(real_dir):
        print(f"Loading REAL images from: {real_dir}")
        for fname in os.listdir(real_dir):
            fpath = os.path.join(real_dir, fname)
            if os.path.isfile(fpath) and fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                real_paths.append(fpath)
        print(f"Found {len(real_paths)} REAL images")
    else:
        print(f"WARNING: REAL directory not found: {real_dir}")
    
    # Load FAKE images from both train and test folders
    for folder in ["train", "test"]:
        fake_dir = os.path.join(cifake_root, folder, "FAKE")
        if os.path.isdir(fake_dir):
            print(f"Loading FAKE images from: {fake_dir}")
            count_before = len(fake_paths)
            for fname in os.listdir(fake_dir):
                fpath = os.path.join(fake_dir, fname)
                if os.path.isfile(fpath) and fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    fake_paths.append(fpath)
            print(f"Found {len(fake_paths) - count_before} FAKE images in {folder}")
        else:
            print(f"WARNING: FAKE directory not found: {fake_dir}")
    
    print(f"\nTotal images found: {len(real_paths)} REAL, {len(fake_paths)} FAKE")
    
    # Check if we have any data
    if len(real_paths) == 0 or len(fake_paths) == 0:
        raise ValueError(f"Insufficient data! REAL: {len(real_paths)}, FAKE: {len(fake_paths)}")
    
    # Balance and sample
    random.seed(42)
    random.shuffle(real_paths)
    random.shuffle(fake_paths)
    
    n_samples = min(sample_per_label, len(real_paths), len(fake_paths))
    real_paths = real_paths[:n_samples]
    fake_paths = fake_paths[:n_samples]
    
    # Create labels (0=REAL, 1=FAKE)
    real_labels = [0] * len(real_paths)
    fake_labels = [1] * len(fake_paths)
    
    # Combine and shuffle
    all_paths = real_paths + fake_paths
    all_labels = real_labels + fake_labels
    
    # Shuffle together
    combined = list(zip(all_paths, all_labels))
    random.shuffle(combined)
    all_paths, all_labels = zip(*combined)
    all_paths = list(all_paths)
    all_labels = list(all_labels)
    
    # Split 80/20 for train/val
    split_idx = int(0.8 * len(all_paths))
    train_paths = all_paths[:split_idx]
    train_labels = all_labels[:split_idx]
    val_paths = all_paths[split_idx:]
    val_labels = all_labels[split_idx:]
    
    print(f"Loaded {len(train_paths)} training samples ({train_labels.count(0)} REAL, {train_labels.count(1)} FAKE)")
    print(f"Loaded {len(val_paths)} validation samples ({val_labels.count(0)} REAL, {val_labels.count(1)} FAKE)")
    
    return train_paths, train_labels, val_paths, val_labels


def freeze_backbones(model):
    """
    Freeze CLIP and RoBERTa models, keep EfficientNet and fusion layer trainable.
    """
    # Freeze CLIP
    for param in model.clip_model.parameters():
        param.requires_grad = False
    
    # Freeze RoBERTa
    for param in model.roberta_model.parameters():
        param.requires_grad = False
    
    # EfficientNet and fusion_layer are already trainable by default
    # (since we froze them in the model initialization)
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")


def train_epoch(model, dataloader, optimizer, criterion, device, scaler):
    """
    Train for one epoch with mixed precision.
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    progress_bar = tqdm(dataloader, desc="Training")
    
    for images, labels in progress_bar:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision forward pass
        with autocast():
            # Create dummy text inputs (CIFAKE has no captions)
            batch_size = images.size(0)
            dummy_text_ids = torch.zeros(batch_size, 77, dtype=torch.long).to(device)
            dummy_attention_mask = torch.ones(batch_size, 77, dtype=torch.long).to(device)
            
            clip_text = {'input_ids': dummy_text_ids, 'attention_mask': dummy_attention_mask}
            roberta_text = {'input_ids': dummy_text_ids, 'attention_mask': dummy_attention_mask}
            
            # Forward pass
            logits, clip_similarity, deepfake_probability = model(clip_text, images, roberta_text)
            loss = criterion(logits, labels)
        
        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Statistics
        total_loss += loss.item()
        _, predicted = torch.max(logits, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        
        progress_bar.set_postfix({
            'loss': loss.item(),
            'acc': 100 * correct / total,
            'deepfake_prob': deepfake_probability.mean().item()
        })
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy


def validate(model, dataloader, criterion, device):
    """
    Validate the model.
    """
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Validating")
        
        for images, labels in progress_bar:
            images = images.to(device)
            labels = labels.to(device)
            
            # Create dummy text inputs
            batch_size = images.size(0)
            dummy_text_ids = torch.zeros(batch_size, 77, dtype=torch.long).to(device)
            dummy_attention_mask = torch.ones(batch_size, 77, dtype=torch.long).to(device)
            
            clip_text = {'input_ids': dummy_text_ids, 'attention_mask': dummy_attention_mask}
            roberta_text = {'input_ids': dummy_text_ids, 'attention_mask': dummy_attention_mask}
            
            # Forward pass
            with autocast():
                logits, clip_similarity, deepfake_probability = model(clip_text, images, roberta_text)
                loss = criterion(logits, labels)
            
            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
            progress_bar.set_postfix({
                'loss': loss.item(),
                'acc': 100 * correct / total
            })
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy


def main():
    print("="*60)
    print("CIFAKE Forensics Training - EfficientNet Fine-tuning")
    print("="*60)
    
    # Configuration
    cifake_root = r'D:\ACM\data\archive'
    batch_size = 16
    learning_rate = 1e-4
    num_epochs = 10
    sample_per_label = 2500  # 5000 total samples
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Load data
    print("\nLoading CIFAKE dataset...")
    train_paths, train_labels, val_paths, val_labels = load_cifake_data(
        cifake_root, sample_per_label
    )
    
    # Create datasets
    train_dataset = CIFAKEDataset(train_paths, train_labels, is_train=True)
    val_dataset = CIFAKEDataset(val_paths, val_labels, is_train=False)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # Initialize model
    print("\nInitializing MultiModalMisinfoDetector...")
    model = MultiModalMisinfoDetector()
    model.to(device)
    
    # Freeze backbones (CLIP and RoBERTa)
    print("\nFreezing CLIP and RoBERTa backbones...")
    freeze_backbones(model)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate
    )
    
    # Mixed precision scaler
    scaler = GradScaler()
    
    print(f"\nTraining configuration:")
    print(f"- Batch size: {batch_size}")
    print(f"- Learning rate: {learning_rate}")
    print(f"- Epochs: {num_epochs}")
    print(f"- Optimizer: Adam")
    print(f"- Loss: CrossEntropyLoss")
    print(f"- Mixed precision: Enabled")
    
    # Training loop
    best_val_acc = 0.0
    
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60)
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-"*60)
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device, scaler
        )
        print(f"Train - Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%")
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        print(f"Val   - Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'efficientnet_cifake_best.pth')
            print(f"✓ New best model saved! Validation accuracy: {val_acc:.2f}%")
    
    print("\n" + "="*60)
    print("Training completed!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print("Model saved to: efficientnet_cifake_best.pth")
    print("="*60)


if __name__ == "__main__":
    main()
