"""
CLIP Detective - Fine-tuning CLIP for Misinformation Detection
Optimized for 6GB VRAM with mixed precision training
"""

import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from torch.cuda.amp import autocast, GradScaler
import optuna
from optuna.trial import TrialState
import pickle
import json


class GuardianCLIPDataset(Dataset):
    """Dataset for CLIP training with Guardian articles."""
    
    def __init__(self, csv_file: str, processor, matched_only: bool = False):
        self.data = pd.read_csv(csv_file)
        self.processor = processor
        
        # Filter to only matched pairs if specified (for training)
        if matched_only:
            original_len = len(self.data)
            self.data = self.data[self.data['label'] == 0].reset_index(drop=True)
            print(f"  Filtered to {len(self.data)} matched pairs (from {original_len} total)")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):

        row = self.data.iloc[idx]
        
        # Load image
        try:
            image = Image.open(row['image_path']).convert('RGB')
        except Exception as e:
            # Fallback to black image if loading fails
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        
        # Get text and label
        text = row['text']
        label = int(row['label'])
        
        return image, text, label


def collate_fn(batch, processor):
    """Custom collate function to process images and text together."""
    images, texts, labels = zip(*batch)
    
    # Process images and texts together
    inputs = processor(
        text=list(texts),
        images=list(images),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=77
    )
    
    labels = torch.tensor(labels, dtype=torch.long)
    
    return inputs, labels


class CLIPDetective(nn.Module):
    """CLIP model with frozen encoders and trainable projection heads."""
    
    def __init__(self, model_name: str = r"C:\Users\Lenovo\OneDrive\Desktop\hack\models\clip-vit-b32"):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(model_name)
        
        # Freeze base encoders to save memory
        self._freeze_base_encoders()
        
        # Unfreeze projection heads and fusion layer
        self._unfreeze_projection_heads()
        
    def _freeze_base_encoders(self):
        """Freeze vision and text encoder backbones."""
        # Freeze vision encoder
        for param in self.clip.vision_model.parameters():
            param.requires_grad = False
        
        # Freeze text encoder
        for param in self.clip.text_model.parameters():
            param.requires_grad = False
        
        print("✓ Frozen base vision and text encoders")
    
    def _unfreeze_projection_heads(self):
        """Unfreeze projection heads for fine-tuning."""
        # Unfreeze visual projection
        if hasattr(self.clip, 'visual_projection'):
            for param in self.clip.visual_projection.parameters():
                param.requires_grad = True
        
        # Unfreeze text projection
        if hasattr(self.clip, 'text_projection'):
            for param in self.clip.text_projection.parameters():
                param.requires_grad = True
        
        # Unfreeze logit scale (fusion parameter)
        if hasattr(self.clip, 'logit_scale'):
            self.clip.logit_scale.requires_grad = True
        
        print("✓ Unfrozen projection heads and fusion layer")
    
    def forward(self, inputs):
        """Forward pass returning image and text embeddings."""
        outputs = self.clip(**inputs, return_dict=True)
        return outputs
    
    def get_trainable_params(self):
        """Get count of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def clip_contrastive_loss(image_embeds, text_embeds, logit_scale, labels=None):
    """
    Compute standard CLIP contrastive loss.
    
    Assumes all pairs in the batch are MATCHED (label=0).
    CLIP automatically creates negative samples by comparing each image 
    to all other texts in the batch (and vice versa).
    
    In a batch of size N:
    - There are N matched pairs (diagonal of similarity matrix)
    - There are N*(N-1) automatic negative pairs (off-diagonal)
    """
    # Normalize embeddings
    image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
    text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
    
    # Calculate similarity matrix: [batch_size, batch_size]
    # logits[i, j] = similarity between image[i] and text[j]
    logit_scale = logit_scale.exp()
    logits_per_image = logit_scale * image_embeds @ text_embeds.t()  # image->text
    logits_per_text = logits_per_image.t()  # text->image
    
    # Ground truth: each image matches its corresponding text (diagonal)
    # targets[i] = i means image[i] should match text[i]
    batch_size = image_embeds.shape[0]
    targets = torch.arange(batch_size, device=image_embeds.device)
    
    # Cross-entropy loss on both directions
    # For each image, predict which text it matches (should be same index)
    loss_i2t = nn.functional.cross_entropy(logits_per_image, targets)
    
    # For each text, predict which image it matches (should be same index)
    loss_t2i = nn.functional.cross_entropy(logits_per_text, targets)
    
    # Average the bidirectional losses
    loss = (loss_i2t + loss_t2i) / 2
    
    return loss


def calculate_accuracy(image_embeds, text_embeds, labels):
    """Calculate accuracy for matched/mismatched detection."""
    # Normalize embeddings
    image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
    text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
    
    # Calculate cosine similarity for each pair
    similarities = (image_embeds * text_embeds).sum(dim=-1)
    
    # Predict based on similarity threshold
    # High similarity (>0.3) = matched (0), Low similarity (<=0.3) = mismatched (1)
    # Find optimal threshold dynamically based on median
    threshold = similarities.median().item()
    
    # predictions: if similarity > threshold, predict matched (0), else mismatched (1)
    predictions = (similarities <= threshold).long()
    
    accuracy = (predictions == labels).float().mean()
    return accuracy.item()


def train_epoch(model, dataloader, optimizer, scaler, device, processor):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        inputs, labels = batch
        
        # Move to device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision forward pass
        with autocast():
            outputs = model(inputs)
            
            # Get embeddings
            image_embeds = outputs.image_embeds
            text_embeds = outputs.text_embeds
            logit_scale = model.clip.logit_scale
            
            # Calculate loss (no labels needed - CLIP creates negatives automatically)
            loss = clip_contrastive_loss(image_embeds, text_embeds, logit_scale)
        
        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        
        # Gradient clipping for stability
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / len(dataloader)
    
    return avg_loss, 0.0  # Return 0 for training acc since we only have matched pairs


def validate(model, dataloader, device, processor):
    """Validate the model."""
    model.eval()
    total_loss = 0
    total_accuracy = 0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation")
        for batch in pbar:
            inputs, labels = batch
            
            # Move to device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = labels.to(device)
            
            with autocast():
                outputs = model(inputs)
                
                # Get embeddings
                image_embeds = outputs.image_embeds
                text_embeds = outputs.text_embeds
                logit_scale = model.clip.logit_scale
                
                # Calculate loss
                loss = clip_contrastive_loss(image_embeds, text_embeds, logit_scale)
            
            # Calculate accuracy
            accuracy = calculate_accuracy(image_embeds, text_embeds, labels)
            
            total_loss += loss.item()
            total_accuracy += accuracy
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{accuracy:.4f}'})
    
    avg_loss = total_loss / len(dataloader)
    avg_accuracy = total_accuracy / len(dataloader)
    
    return avg_loss, avg_accuracy


def train_clip_detective(trial=None, use_optuna=False):
    """Main training function with optional Optuna hyperparameter tuning."""
    
    # Hyperparameters
    if use_optuna and trial:
        learning_rate = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
        batch_size = trial.suggest_categorical("batch_size", [8, 12, 16])
        num_epochs = trial.suggest_int("epochs", 5, 15)
    else:
        learning_rate = 1e-4  # Increased from 5e-6
        batch_size = 16  # Increased from 12
        num_epochs = 10  # Increased from 5
    
    print("=" * 60)
    print("CLIP Detective Training")
    print("=" * 60)
    print(f"Learning Rate: {learning_rate}")
    print(f"Batch Size: {batch_size}")
    print(f"Epochs: {num_epochs}")
    print("=" * 60)
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Load processor and model
    print("\nLoading CLIP model and processor...")
    processor = CLIPProcessor.from_pretrained(r"C:\Users\Lenovo\OneDrive\Desktop\hack\models\clip-vit-b32")
    model = CLIPDetective(r"C:\Users\Lenovo\OneDrive\Desktop\hack\models\clip-vit-b32")
    model = model.to(device)
    
    trainable_params = model.get_trainable_params()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")
    
    # Load datasets
    print("\nLoading datasets...")
    # Training: Only use MATCHED pairs (label=0)
    # CLIP contrastive loss creates negatives automatically within each batch
    train_dataset = GuardianCLIPDataset("clip_train.csv", processor, matched_only=True)
    
    # Validation: Use ALL pairs (matched + mismatched) to measure true performance
    val_dataset = GuardianCLIPDataset("clip_val.csv", processor, matched_only=False)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Set to 0 for Windows compatibility
        collate_fn=lambda batch: collate_fn(batch, processor)
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda batch: collate_fn(batch, processor)
    )
    
    # Optimizer and gradient scaler for mixed precision
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=learning_rate,
        weight_decay=0.01
    )
    
    # Learning rate scheduler for better convergence
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=learning_rate * 0.01
    )
    
    scaler = GradScaler()
    
    # Training loop
    best_val_loss = float('inf')
    best_val_accuracy = 0.0
    
    print("\n" + "=" * 60)
    print("Starting Training")
    print("=" * 60)
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 60)
        
        # Train
        train_loss, _ = train_epoch(model, train_loader, optimizer, scaler, device, processor)
        print(f"Train Loss: {train_loss:.4f}")
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, device, processor)
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        # Step the learning rate scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Learning Rate: {current_lr:.2e}")
        
        # Save best model based on validation accuracy (not loss)
        if val_acc > best_val_accuracy:
            best_val_loss = val_loss
            best_val_accuracy = val_acc
            
            try:
                checkpoint_path = os.path.join(os.getcwd(), 'clip_detective_best.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_loss': val_loss,
                    'val_accuracy': val_acc,
                    'train_loss': train_loss,
                }, checkpoint_path)
                
                # Verify the file was actually written
                if os.path.exists(checkpoint_path):
                    file_size = os.path.getsize(checkpoint_path) / (1024 * 1024)  # MB
                    print(f"✓ Saved best model to {checkpoint_path} ({file_size:.2f} MB)")
                    print(f"  val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}")
                else:
                    print(f"⚠ Warning: Checkpoint save reported success but file not found!")
            except Exception as e:
                print(f"✗ Error saving checkpoint: {e}")
        
        # Report to Optuna if using
        if use_optuna and trial:
            trial.report(val_loss, epoch)
            
            # Handle pruning
            if trial.should_prune():
                raise optuna.TrialPruned()
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"Best Val Loss: {best_val_loss:.4f}")
    print(f"Best Val Accuracy: {best_val_accuracy:.4f}")
    print("=" * 60)
    
    return best_val_loss


def optuna_objective(trial):
    """Objective function for Optuna hyperparameter tuning."""
    return train_clip_detective(trial, use_optuna=True)


def run_hyperparameter_tuning(n_trials=10):
    """Run Optuna hyperparameter tuning."""
    print("=" * 60)
    print("Starting Hyperparameter Tuning with Optuna")
    print(f"Number of trials: {n_trials}")
    print("=" * 60)
    
    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner()
    )
    
    study.optimize(optuna_objective, n_trials=n_trials)
    
    print("\n" + "=" * 60)
    print("Hyperparameter Tuning Complete!")
    print("=" * 60)
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best val loss: {study.best_value:.4f}")
    print("\nBest hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    print("=" * 60)


def generate_embeddings_database(model_path: str = "clip_detective_best.pth", 
                                 json_file: str = "vector_db_seed.json",
                                 output_file: str = "guardian_embeddings.pkl"):
    """
    Generate embeddings for all articles and create a vector database.
    
    Args:
        model_path: Path to the fine-tuned model weights
        json_file: Path to vector_db_seed.json containing article metadata
        output_file: Path to save the embeddings database
    """
    print("\n" + "=" * 60)
    print("Generating Embeddings Database")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Ensure absolute path
    if not os.path.isabs(model_path):
        model_path = os.path.join(os.getcwd(), model_path)
    
    # Check if model file exists
    if not os.path.exists(model_path):
        print(f"\n✗ Error: Model checkpoint not found at: {model_path}")
        print(f"  Current directory: {os.getcwd()}")
        print(f"  Available .pth files:")
        pth_files = [f for f in os.listdir(os.getcwd()) if f.endswith('.pth')]
        if pth_files:
            for f in pth_files:
                print(f"    - {f}")
        else:
            print(f"    (none found)")
        print("\nPlease run training first or specify correct model path.")
        return
    
    # Load the fine-tuned model
    print(f"\nLoading fine-tuned model from {model_path}...")
    processor = CLIPProcessor.from_pretrained(r"C:\Users\Lenovo\OneDrive\Desktop\hack\models\clip-vit-b32")
    model = CLIPDetective(r"C:\Users\Lenovo\OneDrive\Desktop\hack\models\clip-vit-b32")
    
    # Load the fine-tuned weights
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"✓ Loaded model from epoch {checkpoint['epoch']} with val_acc: {checkpoint['val_accuracy']:.4f}")
    
    # Load article metadata
    print(f"\nLoading article metadata from {json_file}...")
    with open(json_file, 'r', encoding='utf-8') as f:
        articles = json.load(f)
    
    print(f"Found {len(articles)} articles")
    
    # Generate embeddings
    print("\nGenerating embeddings...")
    embeddings_db = {
        'article_ids': [],
        'text_contents': [],
        'image_paths': [],
        'image_embeddings': [],
        'text_embeddings': [],
        'metadata': {
            'model_path': model_path,
            'total_articles': len(articles),
            'embedding_dim': None,
            'val_accuracy': checkpoint.get('val_accuracy', None)
        }
    }
    
    with torch.no_grad():
        for article in tqdm(articles, desc="Processing articles"):
            try:
                # Load image
                image = Image.open(article['image_local_path']).convert('RGB')
                text = article['text_content']
                
                # Process inputs
                inputs = processor(
                    text=[text],
                    images=[image],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=77
                )
                
                # Move to device
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                # Generate embeddings
                with autocast():
                    outputs = model(inputs)
                    image_embed = outputs.image_embeds.cpu().numpy()[0]
                    text_embed = outputs.text_embeds.cpu().numpy()[0]
                
                # Normalize embeddings
                image_embed = image_embed / np.linalg.norm(image_embed)
                text_embed = text_embed / np.linalg.norm(text_embed)
                
                # Store in database
                embeddings_db['article_ids'].append(article['article_id'])
                embeddings_db['text_contents'].append(text)
                embeddings_db['image_paths'].append(article['image_local_path'])
                embeddings_db['image_embeddings'].append(image_embed)
                embeddings_db['text_embeddings'].append(text_embed)
                
            except Exception as e:
                print(f"\nError processing {article['article_id']}: {e}")
                continue
    
    # Convert lists to numpy arrays for efficient storage
    embeddings_db['image_embeddings'] = np.array(embeddings_db['image_embeddings'])
    embeddings_db['text_embeddings'] = np.array(embeddings_db['text_embeddings'])
    embeddings_db['metadata']['embedding_dim'] = embeddings_db['image_embeddings'].shape[1]
    
    print(f"\n✓ Generated embeddings for {len(embeddings_db['article_ids'])} articles")
    print(f"  Embedding dimension: {embeddings_db['metadata']['embedding_dim']}")
    
    # Save embeddings database
    print(f"\nSaving embeddings database to {output_file}...")
    with open(output_file, 'wb') as f:
        pickle.dump(embeddings_db, f)
    
    print(f"✓ Saved embeddings database ({os.path.getsize(output_file) / 1e6:.2f} MB)")
    
    # Create summary JSON
    summary_file = output_file.replace('.pkl', '_summary.json')
    summary = {
        'total_articles': len(embeddings_db['article_ids']),
        'embedding_dimension': embeddings_db['metadata']['embedding_dim'],
        'model_val_accuracy': embeddings_db['metadata']['val_accuracy'],
        'database_size_mb': os.path.getsize(output_file) / 1e6,
        'sample_articles': embeddings_db['article_ids'][:5]
    }
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✓ Saved summary to {summary_file}")
    
    print("\n" + "=" * 60)
    print("Embeddings Database Creation Complete!")
    print(f"  - Total articles: {len(embeddings_db['article_ids'])}")
    print(f"  - Embedding dim: {embeddings_db['metadata']['embedding_dim']}")
    print(f"  - Database file: {output_file}")
    print("=" * 60)
    
    return embeddings_db


def search_similar_articles(query_text: str = None, 
                           query_image_path: str = None,
                           embeddings_db_path: str = "guardian_embeddings.pkl",
                           top_k: int = 5,
                           search_mode: str = "text"):
    """
    Search for similar articles using text or image query.
    
    Args:
        query_text: Text query
        query_image_path: Path to query image
        embeddings_db_path: Path to embeddings database
        top_k: Number of results to return
        search_mode: "text", "image", or "multimodal"
    """
    print(f"\nSearching for similar articles (mode: {search_mode}, top_k: {top_k})...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load embeddings database
    with open(embeddings_db_path, 'rb') as f:
        embeddings_db = pickle.load(f)
    
    # Load model and processor
    processor = CLIPProcessor.from_pretrained(r"C:\Users\Lenovo\OneDrive\Desktop\hack\models\clip-vit-b32")
    model = CLIPDetective(r"C:\Users\Lenovo\OneDrive\Desktop\hack\models\clip-vit-b32")
    
    checkpoint = torch.load("clip_detective_best.pth", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Generate query embedding
    with torch.no_grad():
        if search_mode == "text" and query_text:
            inputs = processor(text=[query_text], return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            query_embed = model.clip.get_text_features(**inputs).cpu().numpy()[0]
            db_embeddings = embeddings_db['text_embeddings']
            
        elif search_mode == "image" and query_image_path:
            image = Image.open(query_image_path).convert('RGB')
            inputs = processor(images=[image], return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            query_embed = model.clip.get_image_features(**inputs).cpu().numpy()[0]
            db_embeddings = embeddings_db['image_embeddings']
        
        else:
            raise ValueError("Invalid search mode or missing query")
        
        # Normalize query embedding
        query_embed = query_embed / np.linalg.norm(query_embed)
        
        # Calculate cosine similarities
        similarities = np.dot(db_embeddings, query_embed)
        
        # Get top-k results
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Print results
        print(f"\nTop {top_k} similar articles:")
        print("-" * 60)
        
        results = []
        for i, idx in enumerate(top_indices):
            result = {
                'rank': i + 1,
                'article_id': embeddings_db['article_ids'][idx],
                'similarity': float(similarities[idx]),
                'text': embeddings_db['text_contents'][idx][:100] + "...",
                'image_path': embeddings_db['image_paths'][idx]
            }
            results.append(result)
            
            print(f"{i+1}. [{result['similarity']:.4f}] {result['article_id']}")
            print(f"   {result['text']}")
            print()
        
        return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train CLIP Detective")
    parser.add_argument("--tune", action="store_true", help="Run hyperparameter tuning with Optuna")
    parser.add_argument("--trials", type=int, default=10, help="Number of Optuna trials")
    parser.add_argument("--generate-db", action="store_true", help="Generate embeddings database after training")
    parser.add_argument("--only-db", action="store_true", help="Only generate embeddings database (skip training)")
    parser.add_argument("--search", type=str, help="Search query text")
    args = parser.parse_args()
    
    if args.only_db:
        # Only generate embeddings database
        generate_embeddings_database()
    elif args.search:
        # Search mode
        search_similar_articles(query_text=args.search, search_mode="text")
    elif args.tune:
        # Run hyperparameter tuning
        run_hyperparameter_tuning(n_trials=args.trials)
        if args.generate_db:
            generate_embeddings_database()
    else:
        # Standard training
        train_clip_detective()
        if args.generate_db:
            generate_embeddings_database()
