"""
Train the Fusion Judge - Meta-learner that combines all forensic signals.
Trains only the fusion_layer while keeping all other models frozen.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
import pandas as pd
from PIL import Image
import numpy as np
from tqdm import tqdm
import os
from transformers import RobertaTokenizer, CLIPProcessor
import pickle

# Import the MultiModalMisinfoDetector from misinfo_forensics
import sys
sys.path.append(os.path.dirname(__file__))
from misinfo_forensics import MultiModalMisinfoDetector, MisinfoForensics


class FusionTrainingDataset(Dataset):
    """
    Dataset that computes all 5 forensic scores for fusion layer training.
    """
    
    def __init__(
        self,
        csv_file: str,
        forensics_system: MisinfoForensics,
        max_samples: int = None
    ):
        """
        Args:
            csv_file: Path to Final_Fusion_Train.csv
            forensics_system: Initialized MisinfoForensics with all models loaded
            max_samples: Limit dataset size for faster training
        """
        self.df = pd.read_csv(csv_file)
        if max_samples:
            self.df = self.df.head(max_samples)
        
        self.forensics = forensics_system
        
        print(f"Loaded {len(self.df)} samples for fusion training")
        print(f"Label distribution: {self.df['label'].value_counts().to_dict()}")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        text = str(row['text'])
        image_path = str(row['image_path'])
        label = int(row['label'])
        
        # Check if image exists
        if not os.path.exists(image_path):
            # Use a default/placeholder if missing
            print(f" Image not found: {image_path}, using zeros")
            return {
                'scores': torch.zeros(5, dtype=torch.float32),
                'label': torch.tensor(label, dtype=torch.long)
            }
        
        try:
            # Extract all 5 scores
            # 1. Text Analysis (RoBERTa dual heads)
            text_scores = self.forensics.analyze_text(text)
            ai_score = text_scores['ai_score']
            misinfo_score = text_scores['misinfo_score']
            
            # 2. Visual Forensics (EfficientNet)
            image_scores = self.forensics.analyze_image(image_path)
            deepfake_score = image_scores['deepfake_score']
            
            # 3. Consistency Check (CLIP)
            consistency_scores = self.forensics.analyze_consistency(text, image_path)
            clip_similarity = consistency_scores['clip_similarity']
            
            # 4. Vault Check
            vault_results = self.forensics.search_vault(image_path)
            vault_discrepancy = vault_results['vault_discrepancy']
            
            # Combine into 5-dim vector
            scores = torch.tensor([
                ai_score,
                misinfo_score,
                deepfake_score,
                clip_similarity,
                vault_discrepancy
            ], dtype=torch.float32)
            
        except Exception as e:
            print(f"⚠ Error processing sample {idx}: {e}")
            scores = torch.zeros(5, dtype=torch.float32)
        
        return {
            'scores': scores,
            'label': torch.tensor(label, dtype=torch.long)
        }


def train_fusion_layer(
    csv_file: str = "Final_Fusion_Train.csv",
    batch_size: int = 16,
    epochs: int = 10,
    lr: float = 1e-3,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Train the fusion layer while keeping all other models frozen.
    """
    
    print("=" * 70)
    print("FUSION JUDGE TRAINING")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Training CSV: {csv_file}")
    print(f"Batch Size: {batch_size}")
    print(f"Epochs: {epochs}")
    print(f"Learning Rate: {lr}")
    print("=" * 70)
    
    # Initialize the forensics system with all pretrained models
    print("\n[1/5] Loading pretrained models...")
    forensics = MisinfoForensics(
        ai_head_weights="ai_head_best.pth",
        misinfo_head_weights="roberta_detective_best.pth",
        efficientnet_weights="efficientnet_cifake_best.pth",
        clip_weights="clip_detective_best.pth",
        faiss_index_path="guardian_embeddings.pkl",
        device=device
    )
    
    # Freeze ALL models except fusion layer
    print("\n[2/5] Freezing all models except fusion layer...")
    forensics.detector.eval()  # Set to eval mode
    
    # Freeze everything
    for param in forensics.detector.parameters():
        param.requires_grad = False
    
    # Unfreeze only fusion layer
    for param in forensics.detector.fusion_layer.parameters():
        param.requires_grad = True
    
    # Count trainable parameters
    total_params = sum(p.numel() for p in forensics.detector.parameters())
    trainable_params = sum(p.numel() for p in forensics.detector.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters (fusion only): {trainable_params:,}")
    print(f"Frozen parameters: {total_params - trainable_params:,}")
    
    # Load dataset
    print(f"\n[3/5] Loading training data from {csv_file}...")
    if not os.path.exists(csv_file):
        print(f"✗ Error: {csv_file} not found!")
        print("Please run prepare_fusion_dataset.py first to create the training data.")
        return
    
    train_dataset = FusionTrainingDataset(
        csv_file=csv_file,
        forensics_system=forensics,
        max_samples=None  # Use all samples
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0  # Windows compatibility
    )
    
    # Optimizer (only fusion layer parameters)
    optimizer = torch.optim.AdamW(
        forensics.detector.fusion_layer.parameters(),
        lr=lr,
        weight_decay=0.01
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=lr * 0.1
    )
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Mixed precision training
    scaler = GradScaler()
    
    print("\n[4/5] Starting training...")
    print("=" * 70)
    
    best_accuracy = 0.0
    best_loss = float('inf')
    
    for epoch in range(1, epochs + 1):
        forensics.detector.fusion_layer.train()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        
        for batch in pbar:
            scores = batch['scores'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass through fusion layer only
            with autocast():
                logits = forensics.detector.forward_fusion(scores)
                loss = criterion(logits, labels)
            
            # Backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # Statistics
            running_loss += loss.item() * scores.size(0)
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{100 * correct / total:.2f}%"
            })
        
        # Epoch statistics
        epoch_loss = running_loss / total
        epoch_acc = 100 * correct / total
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"\nEpoch {epoch}/{epochs}")
        print(f"  Loss: {epoch_loss:.4f}")
        print(f"  Accuracy: {epoch_acc:.2f}%")
        print(f"  Learning Rate: {current_lr:.6f}")
        
        # Save best model
        if epoch_acc > best_accuracy:
            best_accuracy = epoch_acc
            best_loss = epoch_loss
            
            print(f"  ✓ New best accuracy! Saving checkpoint...")
            
            # Save complete integrated model
            torch.save({
                'epoch': epoch,
                'fusion_layer_state_dict': forensics.detector.fusion_layer.state_dict(),
                'full_model_state_dict': forensics.detector.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': epoch_loss,
                'accuracy': epoch_acc,
            }, 'forensics_master_final.pth')
            
            print(f"  Saved to forensics_master_final.pth")
        
        # Step scheduler
        scheduler.step()
        print("-" * 70)
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print(f"Best Accuracy: {best_accuracy:.2f}%")
    print(f"Best Loss: {best_loss:.4f}")
    print(f"Final model saved to: forensics_master_final.pth")
    print("=" * 70)
    
    return forensics


def test_fusion_model(model_path: str = "forensics_master_final.pth"):
    """
    Test the trained fusion model on a sample.
    """
    print("\n" + "=" * 70)
    print("TESTING FUSION MODEL")
    print("=" * 70)
    
    # Load the trained model
    forensics = MisinfoForensics()
    
    checkpoint = torch.load(model_path, map_location=forensics.device, weights_only=False)
    forensics.detector.load_state_dict(checkpoint['full_model_state_dict'])
    
    print(f"✓ Loaded model from epoch {checkpoint['epoch']}")
    print(f"  Training Accuracy: {checkpoint['accuracy']:.2f}%")
    print(f"  Training Loss: {checkpoint['loss']:.4f}")
    
    # Test on a sample
    print("\nRunning test analysis...")
    
    # Get first available image
    import glob
    images = glob.glob("guardian_processed/*.jpg")
    if images:
        test_image = images[0]
        test_text = "Breaking news: Major event occurring now"
        
        results = forensics.analyze(
            text=test_text,
            image_path=test_image,
            verbose=True
        )
        
        print(f"\n✓ Test complete!")
        print(f"  Verdict: {results['verdict_text']}")
        print(f"  Confidence: {results['confidence']:.1%}")
    else:
        print("⚠ No test images available")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Fusion Judge")
    parser.add_argument("--csv", type=str, default="Final_Fusion_Train.csv", help="Training CSV file")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--test", action="store_true", help="Test trained model")
    args = parser.parse_args()
    
    if args.test:
        test_fusion_model()
    else:
        train_fusion_layer(
            csv_file=args.csv,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr
        )
