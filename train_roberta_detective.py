"""
RoBERTa Text Classification for Human vs. AI Text Detection
Simple, focused fine-tuning script optimized for binary classification.
Optimized for 6GB VRAM with mixed precision training.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.cuda.amp import GradScaler
from transformers import RobertaTokenizer, RobertaForSequenceClassification, get_cosine_schedule_with_warmup
import pandas as pd
import os
from tqdm import tqdm
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix


class TextDataset(Dataset):
    """Simple dataset for text classification."""
    
    def __init__(self, csv_path, tokenizer, max_length=256):
        """
        Args:
            csv_path: Path to CSV with 'text' and 'label' columns
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
        """
        self.df = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Clean data
        self.df['text'] = self.df['text'].astype(str).str.strip()
        self.df = self.df[self.df['text'].str.len() > 0].reset_index(drop=True)
        
        print(f"Loaded {len(self.df)} samples")
        print(f"Label distribution: {self.df['label'].value_counts().to_dict()}")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        text = self.df.iloc[idx]['text']
        label = int(self.df.iloc[idx]['label'])
        
        # Tokenize
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


def train_epoch(model, dataloader, optimizer, scaler, scheduler, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    progress_bar = tqdm(dataloader, desc="Training")
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
                labels=labels
            )
            loss = outputs.loss
        
        # Backward pass
        scaler.scale(loss).backward()
        
        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Optimizer step
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        # Track metrics
        total_loss += loss.item()
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
        
        progress_bar.set_postfix({
            'loss': f'{total_loss / (progress_bar.n + 1):.4f}',
            'acc': f'{100 * correct / total:.2f}%',
            'lr': f'{scheduler.get_last_lr()[0]:.2e}'
        })
    
    return total_loss / len(dataloader), 100 * correct / total


def validate(model, dataloader, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Validation")
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            
            total_loss += loss.item()
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            progress_bar.set_postfix({
                'loss': f'{total_loss / (progress_bar.n + 1):.4f}'
            })
    
    accuracy = 100 * np.sum(np.array(all_predictions) == np.array(all_labels)) / len(all_labels)
    avg_loss = total_loss / len(dataloader)
    
    return avg_loss, accuracy, all_predictions, all_labels


def main():
    # ==================== Configuration ====================
    CSV_PATH = r'c:\Users\Lenovo\OneDrive\Desktop\deep Learning\roberta_train_synthetic.csv'
    MODEL_PATH = r'c:\Users\Lenovo\OneDrive\Desktop\deep Learning\models\roberta-base'
    SAVE_PATH = r'c:\Users\Lenovo\OneDrive\Desktop\deep Learning\roberta_detective_best.pth'
    
    BATCH_SIZE = 16
    MAX_LENGTH = 256  # Reduced from 512 for faster training
    LEARNING_RATE = 2e-5
    NUM_EPOCHS = 10
    VALIDATION_SPLIT = 0.2
    PATIENCE = 3
    WARMUP_RATIO = 0.1
    
    # ==================== Setup ====================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"RoBERTa Fine-Tuning for Human vs. AI Text Detection")
    print(f"{'='*60}")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # ==================== Load Tokenizer ====================
    print(f"\nLoading tokenizer...")
    if os.path.exists(MODEL_PATH):
        tokenizer = RobertaTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
        print(f"✓ Loaded from: {MODEL_PATH}")
    else:
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        print(f"✓ Loaded from HuggingFace")
    
    # ==================== Load Dataset ====================
    print(f"\nLoading dataset: {CSV_PATH}")
    full_dataset = TextDataset(CSV_PATH, tokenizer, MAX_LENGTH)
    
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
    print(f"\nInitializing RoBERTa model...")
    if os.path.exists(MODEL_PATH):
        model = RobertaForSequenceClassification.from_pretrained(
            MODEL_PATH,
            num_labels=2,
            local_files_only=True
        )
    else:
        model = RobertaForSequenceClassification.from_pretrained(
            'roberta-base',
            num_labels=2
        )
    
    model = model.to(device)
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Trainable parameters: {trainable_params:,}")
    
    # ==================== Setup Training ====================
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.01,
        eps=1e-8
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
    print(f"  Early stopping patience: {PATIENCE}")
    
    # ==================== Training Loop ====================
    best_val_loss = float('inf')
    best_val_acc = 0.0
    patience_counter = 0
    
    print(f"\n{'='*60}")
    print(f"Starting Training")
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
        
        # Detailed metrics every epoch
        if val_acc > 55:  # Only show detailed metrics if doing better than random
            print(f"\n  Classification Report:")
            print(classification_report(labels, predictions, target_names=['Human (0)', 'AI (1)'], digits=3))
            cm = confusion_matrix(labels, predictions)
            print(f"  Confusion Matrix:\n{cm}")
        
        # Save best model
        improved = False
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            patience_counter = 0
            improved = True
            
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
            
            print(f"  ✓ New best model saved! (Val Loss: {val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"  ⚠ No improvement for {patience_counter} epoch(s)")
        
        # Early stopping
        if patience_counter >= PATIENCE:
            print(f"\n⏹ Early stopping triggered after {epoch + 1} epochs")
            break
    
    # ==================== Final Summary ====================
    print(f"\n{'='*60}")
    print(f"🎉 Training Complete!")
    print(f"{'='*60}")
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Model saved to: {SAVE_PATH}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
