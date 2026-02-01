"""
Dual-Head RoBERTa Training Script for AI Detection Head
Trains only the new ai_head while keeping misinfo_head frozen.
Optimized for 6GB VRAM with fast training (2-3 epochs).
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.cuda.amp import GradScaler
from transformers import RobertaModel, RobertaTokenizer, CLIPModel, get_cosine_schedule_with_warmup
from torchvision import models
import pandas as pd
import os
from tqdm import tqdm
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix


# ==================== DUAL-HEAD MODEL DEFINITION ====================

class MultiModalMisinfoDetector(nn.Module):
    """
    Multi-modal misinformation detector with DUAL RoBERTa classification heads.
    - misinfo_head: Trained on WELFake (fake news detection)
    - ai_head: New head for HC3 (human vs AI text detection)
    """
    
    def __init__(self, num_classes=2, roberta_model_name='roberta-base', clip_model_name='openai/clip-vit-base-patch32'):
        super(MultiModalMisinfoDetector, self).__init__()
        
        # Shared RoBERTa backbone (encoder)
        self.roberta_model = RobertaModel.from_pretrained(roberta_model_name)
        roberta_hidden_size = self.roberta_model.config.hidden_size  # 768
        
        # Vision-Language branch: CLIP
        self.clip_model = CLIPModel.from_pretrained(clip_model_name)
        clip_hidden_size = self.clip_model.config.vision_config.hidden_size  # 768
        
        # Image branch: EfficientNet
        self.efficientnet = models.efficientnet_b0(pretrained=True)
        efficientnet_features = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier = nn.Identity()
        
        # ========== DUAL CLASSIFICATION HEADS ==========
        # Head 1: Misinformation detection (trained on WELFake)
        self.misinfo_head = nn.Linear(roberta_hidden_size, num_classes)
        
        # Head 2: AI text detection (new, for HC3 dataset)
        self.ai_head = nn.Linear(roberta_hidden_size, num_classes)
        
        # Vision feature projections
        self.clip_projection = nn.Linear(clip_hidden_size, 256)
        self.efficientnet_projection = nn.Linear(efficientnet_features, 256)
        
        # ========== FUSION LAYER (THE JUDGE) ==========
        # Takes both misinfo_logits and ai_logits as input
        fusion_input_size = (num_classes * 2) + 256 + 256  # misinfo(2) + ai(2) + clip(256) + efficient(256)
        
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_input_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, input_ids, attention_mask, clip_images=None, efficient_images=None, return_dict=True):
        """
        Forward pass with dual heads.
        
        Returns:
            dict: {
                'misinfo_logits': Logits from misinformation head,
                'ai_logits': Logits from AI detection head,
                'vision_features': Combined vision features,
                'clip_similarity': CLIP similarity scores,
                'fusion_logits': Final judgment from fusion layer
            }
        """
        # Shared RoBERTa encoding
        roberta_output = self.roberta_model(input_ids=input_ids, attention_mask=attention_mask)
        roberta_features = roberta_output.pooler_output  # (batch_size, 768)
        
        # Dual classification heads
        misinfo_logits = self.misinfo_head(roberta_features)  # (batch_size, 2)
        ai_logits = self.ai_head(roberta_features)  # (batch_size, 2)
        
        # Vision features (if images provided)
        if clip_images is not None and efficient_images is not None:
            # CLIP vision features
            clip_output = self.clip_model.vision_model(pixel_values=clip_images)
            clip_features = self.clip_projection(clip_output.pooler_output)  # (batch_size, 256)
            
            # EfficientNet features
            efficient_features = self.efficientnet(efficient_images)
            efficient_features = self.efficientnet_projection(efficient_features)  # (batch_size, 256)
            
            # Compute CLIP text-image similarity
            text_features = self.clip_model.text_model(input_ids=input_ids, attention_mask=attention_mask)
            text_embeds = self.clip_model.text_projection(text_features.pooler_output)
            clip_embeds = self.clip_model.vision_projection(clip_output.pooler_output)
            
            # Normalize and compute similarity
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
            clip_embeds = clip_embeds / clip_embeds.norm(dim=-1, keepdim=True)
            clip_similarity = (text_embeds @ clip_embeds.T).diag()  # (batch_size,)
        else:
            # Use dummy features if no images
            batch_size = input_ids.size(0)
            clip_features = torch.zeros(batch_size, 256, device=input_ids.device)
            efficient_features = torch.zeros(batch_size, 256, device=input_ids.device)
            clip_similarity = torch.zeros(batch_size, device=input_ids.device)
        
        # Fusion: Combine both heads + vision features
        fusion_input = torch.cat([
            misinfo_logits,
            ai_logits,
            clip_features,
            efficient_features
        ], dim=1)
        
        fusion_logits = self.fusion_layer(fusion_input)
        
        if return_dict:
            return {
                'misinfo_logits': misinfo_logits,
                'ai_logits': ai_logits,
                'vision_features': torch.cat([clip_features, efficient_features], dim=1),
                'clip_similarity': clip_similarity,
                'fusion_logits': fusion_logits
            }
        else:
            return fusion_logits
    
    def freeze_misinfo_branch(self):
        """Freeze RoBERTa backbone and misinfo_head for AI head training."""
        for param in self.roberta_model.parameters():
            param.requires_grad = False
        for param in self.misinfo_head.parameters():
            param.requires_grad = False
    
    def freeze_image_branches(self):
        """Freeze CLIP and EfficientNet."""
        for param in self.clip_model.parameters():
            param.requires_grad = False
        for param in self.efficientnet.parameters():
            param.requires_grad = False
    
    def unfreeze_all(self):
        """Unfreeze all parameters."""
        for param in self.parameters():
            param.requires_grad = True


# ==================== DATASET FOR AI HEAD TRAINING ====================

class HC3Dataset(Dataset):
    """Dataset for Human vs AI text classification (HC3 format)."""
    
    def __init__(self, csv_path, tokenizer, max_length=256):
        """
        Args:
            csv_path: Path to CSV with 'text' and 'label' columns
                     label 0 = Human, label 1 = AI
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
        """
        self.df = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Clean data
        self.df['text'] = self.df['text'].astype(str).str.strip()
        self.df = self.df[self.df['text'].str.len() > 0].reset_index(drop=True)
        
        print(f"Loaded {len(self.df)} HC3 samples")
        print(f"Label distribution: {self.df['label'].value_counts().to_dict()}")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        text = self.df.iloc[idx]['text']
        label = int(self.df.iloc[idx]['label'])
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }


# ==================== TRAINING FUNCTIONS ====================

def train_epoch(model, dataloader, optimizer, scaler, scheduler, device):
    """Train AI head for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    
    progress_bar = tqdm(dataloader, desc="Training AI Head")
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision forward pass
        with torch.amp.autocast('cuda'):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                clip_images=None,  # No images for text-only training
                efficient_images=None,
                return_dict=True
            )
            # Use ONLY ai_logits for loss
            ai_logits = outputs['ai_logits']
            loss = criterion(ai_logits, labels)
        
        # Backward pass
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        # Track metrics
        total_loss += loss.item()
        predictions = torch.argmax(ai_logits, dim=-1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
        
        progress_bar.set_postfix({
            'loss': f'{total_loss / (progress_bar.n + 1):.4f}',
            'acc': f'{100 * correct / total:.2f}%',
            'lr': f'{scheduler.get_last_lr()[0]:.2e}'
        })
    
    return total_loss / len(dataloader), 100 * correct / total


def validate(model, dataloader, device):
    """Validate AI head."""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Validation")
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                clip_images=None,
                efficient_images=None,
                return_dict=True
            )
            
            ai_logits = outputs['ai_logits']
            loss = criterion(ai_logits, labels)
            
            total_loss += loss.item()
            predictions = torch.argmax(ai_logits, dim=-1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            progress_bar.set_postfix({
                'loss': f'{total_loss / (progress_bar.n + 1):.4f}'
            })
    
    accuracy = 100 * np.sum(np.array(all_predictions) == np.array(all_labels)) / len(all_labels)
    avg_loss = total_loss / len(dataloader)
    
    return avg_loss, accuracy, all_predictions, all_labels


# ==================== MAIN TRAINING SCRIPT ====================

def main():
    # ==================== Configuration ====================
    # Update this to your HC3 dataset path
    HC3_CSV_PATH = r'c:\Users\Lenovo\OneDrive\Desktop\deep Learning\roberta_train_synthetic.csv'
    
    # Path to your trained WELFake model
    WELFAKE_MODEL_PATH = r'c:\Users\Lenovo\OneDrive\Desktop\deep Learning\roberta_detective_best.pth'
    
    # RoBERTa model path
    ROBERTA_PATH = r'c:\Users\Lenovo\OneDrive\Desktop\deep Learning\models\roberta-base'
    CLIP_PATH = r'c:\Users\Lenovo\OneDrive\Desktop\deep Learning\models\clip-vit-b32'
    
    # Save path for AI head model
    SAVE_PATH = r'c:\Users\Lenovo\OneDrive\Desktop\deep Learning\ai_head_best.pth'
    
    BATCH_SIZE = 16
    MAX_LENGTH = 256
    LEARNING_RATE = 1e-3  # Higher LR since we're only training one layer
    NUM_EPOCHS = 3  # Fast training
    VALIDATION_SPLIT = 0.2
    WARMUP_RATIO = 0.1
    
    # ==================== Setup ====================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"AI Head Training for HC3 Dataset")
    print(f"{'='*60}")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # ==================== Load Tokenizer ====================
    print(f"\nLoading tokenizer...")
    if os.path.exists(ROBERTA_PATH):
        tokenizer = RobertaTokenizer.from_pretrained(ROBERTA_PATH, local_files_only=True)
        print(f"✓ Loaded from: {ROBERTA_PATH}")
    else:
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        print(f"✓ Loaded from HuggingFace")
    
    # ==================== Load Dataset ====================
    print(f"\nLoading HC3 dataset: {HC3_CSV_PATH}")
    full_dataset = HC3Dataset(HC3_CSV_PATH, tokenizer, MAX_LENGTH)
    
    # Split dataset
    val_size = int(VALIDATION_SPLIT * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"\nDataset split:")
    print(f"  Training: {len(train_dataset)} samples")
    print(f"  Validation: {len(val_dataset)} samples")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # ==================== Initialize Model ====================
    print(f"\nInitializing Dual-Head MultiModalMisinfoDetector...")
    
    model = MultiModalMisinfoDetector(
        num_classes=2,
        roberta_model_name=ROBERTA_PATH if os.path.exists(ROBERTA_PATH) else 'roberta-base',
        clip_model_name=CLIP_PATH if os.path.exists(CLIP_PATH) else 'openai/clip-vit-base-patch32'
    )
    
    # ==================== Load WELFake Weights ====================
    if os.path.exists(WELFAKE_MODEL_PATH):
        print(f"\n⚙ Loading WELFake trained weights...")
        checkpoint = torch.load(WELFAKE_MODEL_PATH, map_location=device)
        
        # Load state dict (handle potential key mismatches)
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Load only compatible layers (ignore ai_head if not present)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict, strict=False)
        
        print(f"✓ Loaded {len(pretrained_dict)}/{len(state_dict)} layers from WELFake model")
        if 'val_acc' in checkpoint:
            print(f"  WELFake Val Acc: {checkpoint['val_acc']:.2f}%")
    else:
        print(f"\n⚠ No WELFake model found at {WELFAKE_MODEL_PATH}")
        print(f"  Starting from scratch (not recommended)")
    
    model = model.to(device)
    
    # ==================== Freeze Strategy ====================
    print(f"\n🔒 Freezing parameters...")
    
    # Freeze everything first
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze ONLY the ai_head
    for param in model.ai_head.parameters():
        param.requires_grad = True
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"✓ Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")
    print(f"  RoBERTa backbone: FROZEN ❄️")
    print(f"  Misinfo head: FROZEN ❄️")
    print(f"  AI head: TRAINABLE 🔥")
    print(f"  Vision branches: FROZEN ❄️")
    
    # ==================== Setup Training ====================
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
        weight_decay=0.01
    )
    
    total_steps = len(train_loader) * NUM_EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    scaler = GradScaler()
    
    print(f"\nTraining configuration:")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Max length: {MAX_LENGTH}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Total steps: {total_steps}")
    print(f"  Warmup steps: {warmup_steps}")
    
    # ==================== Training Loop ====================
    best_val_loss = float('inf')
    best_val_acc = 0.0
    
    print(f"\n{'='*60}")
    print(f"Starting AI Head Training")
    print(f"{'='*60}\n")
    
    for epoch in range(NUM_EPOCHS):
        print(f"\n{'─'*60}")
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
        print(f"{'─'*60}")
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, scaler, scheduler, device
        )
        
        # Validate
        val_loss, val_acc, predictions, labels = validate(model, val_loader, device)
        
        # Print summary
        print(f"\n📊 Epoch {epoch + 1} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        
        # Classification report
        print(f"\n  Classification Report (AI Head):")
        print(classification_report(labels, predictions, target_names=['Human (0)', 'AI (1)'], digits=3))
        cm = confusion_matrix(labels, predictions)
        print(f"  Confusion Matrix:\n{cm}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'train_loss': train_loss,
                'train_acc': train_acc,
            }, SAVE_PATH)
            
            print(f"  ✓ Best AI head model saved! (Val Loss: {val_loss:.4f})")
    
    # ==================== Final Summary ====================
    print(f"\n{'='*60}")
    print(f"🎉 AI Head Training Complete!")
    print(f"{'='*60}")
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Model saved to: {SAVE_PATH}")
    print(f"\n💡 Next Steps:")
    print(f"  1. Your model now has TWO heads:")
    print(f"     - misinfo_head (WELFake trained)")
    print(f"     - ai_head (HC3 trained)")
    print(f"  2. Use fusion_logits for final predictions")
    print(f"  3. Test with: outputs = model(text, images)")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
