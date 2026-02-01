"""
Training Pipeline for Multi-Modal Misinformation Detection
Demonstrates how to train models using the MisinfoDataset.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import CLIPModel, RobertaModel
import timm
from tqdm import tqdm
import numpy as np


class MultiModalMisinfoDetector(nn.Module):
    """
    Multi-modal misinformation detection model combining CLIP, RoBERTa, and EfficientNet.
    """
    
    def __init__(
        self,
        clip_model_name="openai/clip-vit-base-patch32",
        roberta_model_name="roberta-base",
        num_classes=2
    ):
        """
        Initialize the multi-modal detection model.
        
        Args:
            clip_model_name (str): Name of the CLIP model
            roberta_model_name (str): Name of the RoBERTa model
            num_classes (int): Number of output classes
        """
        super(MultiModalMisinfoDetector, self).__init__()
        
        # Load pretrained models
        self.clip_model = CLIPModel.from_pretrained(clip_model_name)
        self.roberta_model = RobertaModel.from_pretrained(roberta_model_name)
        self.efficientnet = timm.create_model(
            "efficientnet_b0",
            pretrained=True,
            num_classes=1
        )
        
        # Get embedding dimensions
        clip_dim = self.clip_model.config.projection_dim
        roberta_dim = self.roberta_model.config.hidden_size
        
        # Feature fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(clip_dim * 2 + roberta_dim + 2, 512),  # +1 for similarity, +1 for deepfake probability
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

        # Freeze backbone weights (CLIP, RoBERTa, EfficientNet)
        for param in self.clip_model.parameters():
            param.requires_grad = False
        for param in self.roberta_model.parameters():
            param.requires_grad = False
        for param in self.efficientnet.parameters():
            param.requires_grad = False
    
    def forward(self, clip_text, image_tensor, roberta_text):
        """
        Forward pass through the model.
        
        Args:
            clip_text (dict): CLIP tokenized text with 'input_ids' and 'attention_mask'
            image_tensor (torch.Tensor): Preprocessed image tensor
            roberta_text (dict): RoBERTa tokenized text with 'input_ids' and 'attention_mask'
            
        Returns:
            tuple: (logits, clip_similarity, deepfake_probability)
                - logits: Classification logits
                - clip_similarity: Cosine similarity between CLIP image and text embeddings
                - deepfake_probability: EfficientNet deepfake probability score
        """
        # CLIP image and text embeddings
        clip_outputs = self.clip_model(
            input_ids=clip_text['input_ids'],
            attention_mask=clip_text['attention_mask'],
            pixel_values=image_tensor,
            return_dict=True
        )
        
        clip_image_embeds = clip_outputs.image_embeds
        clip_text_embeds = clip_outputs.text_embeds
        
        # Normalize CLIP embeddings
        clip_image_embeds_norm = F.normalize(clip_image_embeds, p=2, dim=-1)
        clip_text_embeds_norm = F.normalize(clip_text_embeds, p=2, dim=-1)
        
        # Calculate cosine similarity between CLIP image and text embeddings
        clip_similarity = (clip_image_embeds_norm * clip_text_embeds_norm).sum(dim=-1, keepdim=True)
        
        # RoBERTa text embeddings
        roberta_outputs = self.roberta_model(
            input_ids=roberta_text['input_ids'],
            attention_mask=roberta_text['attention_mask'],
            return_dict=True
        )
        roberta_embeds = roberta_outputs.last_hidden_state[:, 0, :]  # [CLS] token

        # EfficientNet deepfake probability (pixel-level artifacts)
        efficientnet_logits = self.efficientnet(image_tensor)
        deepfake_probability = torch.sigmoid(efficientnet_logits)
        
        # Concatenate all features
        combined_features = torch.cat([
            clip_image_embeds,
            clip_text_embeds,
            roberta_embeds,
            clip_similarity,
            deepfake_probability
        ], dim=-1)
        
        # Classification
        logits = self.fusion_layer(combined_features)
        
        return logits, clip_similarity, deepfake_probability


def train_epoch(model, dataloader, optimizer, criterion, device):
    """
    Train the model for one epoch.
    
    Args:
        model: The model to train
        dataloader: DataLoader for training data
        optimizer: Optimizer for training
        criterion: Loss function
        device: Device to train on (cuda/cpu)
        
    Returns:
        tuple: (average_loss, accuracy)
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    progress_bar = tqdm(dataloader, desc="Training")
    
    for batch in progress_bar:
        # Unpack the dictionary from MisinfoDataset
        clip_text = {
            'input_ids': batch['clip_text']['input_ids'].to(device),
            'attention_mask': batch['clip_text']['attention_mask'].to(device)
        }
        
        roberta_text = {
            'input_ids': batch['roberta_text']['input_ids'].to(device),
            'attention_mask': batch['roberta_text']['attention_mask'].to(device)
        }
        
        image_tensor = batch['image_tensor'].to(device)
        labels = batch['label'].to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass: pass clip_text, image_tensor, and roberta_text to the model
        logits, clip_similarity, deepfake_probability = model(clip_text, image_tensor, roberta_text)
        
        # Calculate loss
        loss = criterion(logits, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        _, predicted = torch.max(logits, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': loss.item(),
            'acc': 100 * correct / total,
            'clip_sim': clip_similarity.mean().item(),
            'deepfake_prob': deepfake_probability.mean().item()
        })
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy


def evaluate(model, dataloader, criterion, device):
    """
    Evaluate the model on validation/test data.
    
    Args:
        model: The model to evaluate
        dataloader: DataLoader for evaluation data
        criterion: Loss function
        device: Device to evaluate on (cuda/cpu)
        
    Returns:
        tuple: (average_loss, accuracy, avg_clip_similarity)
    """
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    clip_similarities = []
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Evaluating")
        
        for batch in progress_bar:
            # Unpack the dictionary
            clip_text = {
                'input_ids': batch['clip_text']['input_ids'].to(device),
                'attention_mask': batch['clip_text']['attention_mask'].to(device)
            }
            
            roberta_text = {
                'input_ids': batch['roberta_text']['input_ids'].to(device),
                'attention_mask': batch['roberta_text']['attention_mask'].to(device)
            }
            
            image_tensor = batch['image_tensor'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            logits, clip_similarity, deepfake_probability = model(clip_text, image_tensor, roberta_text)
            
            # Calculate loss
            loss = criterion(logits, labels)
            
            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            clip_similarities.extend(clip_similarity.cpu().numpy())
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': loss.item(),
                'acc': 100 * correct / total,
                'deepfake_prob': deepfake_probability.mean().item()
            })
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    avg_clip_similarity = np.mean(clip_similarities)
    
    return avg_loss, accuracy, avg_clip_similarity


def training_loop_example(dataset, num_epochs=5, batch_size=8, learning_rate=1e-5):
    """
    Complete training loop example using MisinfoDataset.
    
    Args:
        dataset: MisinfoDataset instance
        num_epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        learning_rate (float): Learning rate for optimizer
    """
    print("="*60)
    print("Multi-Modal Misinformation Detection Training")
    print("="*60)
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0  # Set to 0 for Windows compatibility
    )
    
    print(f"DataLoader created with batch size: {batch_size}")
    print(f"Number of batches: {len(dataloader)}")
    
    # Initialize model
    print("\nInitializing model...")
    model = MultiModalMisinfoDetector()
    model.to(device)
    print("Model initialized and moved to device")
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    print(f"\nOptimizer: AdamW with learning rate {learning_rate}")
    print(f"Loss function: CrossEntropyLoss")
    
    # Training loop
    print(f"\nStarting training for {num_epochs} epochs...")
    print("="*60)
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-"*60)
        
        # Train for one epoch
        train_loss, train_acc = train_epoch(
            model, dataloader, optimizer, criterion, device
        )
        
        print(f"Training - Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%")
        
        # Example: Evaluate on validation set (using same data for demonstration)
        val_loss, val_acc, avg_clip_sim = evaluate(
            model, dataloader, criterion, device
        )
        
        print(f"Validation - Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%")
        print(f"Average CLIP Similarity: {avg_clip_sim:.4f}")
    
    print("\n" + "="*60)
    print("Training completed!")
    print("="*60)
    
    return model


# Simple training snippet demonstration
def simple_training_snippet():
    """
    Simplified training loop snippet showing the core logic.
    """
    print("\n" + "="*60)
    print("Simple Training Loop Snippet")
    print("="*60 + "\n")
    
    code_snippet = '''
# Import required libraries
from torch.utils.data import DataLoader
from misinformation_dataset import MisinfoDataset
from transformers import CLIPModel, RobertaModel
import torch.nn.functional as F
import timm
import torch

# Initialize dataset and dataloader
dataset = MisinfoDataset(data_list)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Initialize models
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
roberta_model = RobertaModel.from_pretrained("roberta-base")
efficientnet = timm.create_model("efficientnet_b0", pretrained=True, num_classes=1)

# Training loop
for batch in dataloader:
    # Unpack the dictionary from MisinfoDataset
    clip_text = batch['clip_text']
    roberta_text = batch['roberta_text']
    image_tensor = batch['image_tensor']
    video_tensor = batch['video_tensor']
    labels = batch['label']
    
    # Forward pass through CLIP model
    clip_outputs = clip_model(
        input_ids=clip_text['input_ids'],
        attention_mask=clip_text['attention_mask'],
        pixel_values=image_tensor
    )
    
    # Extract CLIP embeddings
    clip_image_embeds = clip_outputs.image_embeds
    clip_text_embeds = clip_outputs.text_embeds
    
    # Normalize embeddings
    clip_image_embeds_norm = F.normalize(clip_image_embeds, p=2, dim=-1)
    clip_text_embeds_norm = F.normalize(clip_text_embeds, p=2, dim=-1)
    
    # Calculate cosine similarity between CLIP image and text embeddings
    cosine_similarity = (clip_image_embeds_norm * clip_text_embeds_norm).sum(dim=-1)
    
    # Forward pass through RoBERTa model
    roberta_outputs = roberta_model(
        input_ids=roberta_text['input_ids'],
        attention_mask=roberta_text['attention_mask']
    )
    
    # Extract RoBERTa embeddings (use [CLS] token)
    roberta_embeds = roberta_outputs.last_hidden_state[:, 0, :]
    
    # EfficientNet deepfake probability
    deepfake_logits = efficientnet(image_tensor)
    deepfake_probability = torch.sigmoid(deepfake_logits)
    
    # Continue with your classification logic...
    # Example: Concatenate features and pass through classifier
    # combined_features = torch.cat([clip_image_embeds, clip_text_embeds, roberta_embeds, deepfake_probability], dim=-1)
    # logits = classifier(combined_features)
    '''
    
    print(code_snippet)


if __name__ == "__main__":
    # Show simple training snippet
    simple_training_snippet()
    
    print("\nNote: To run the full training pipeline, first create a proper dataset")
    print("with actual image and video files, then call:")
    print("  from misinformation_dataset import MisinfoDataset")
    print("  dataset = MisinfoDataset(your_data_list)")
    print("  model = training_loop_example(dataset)")
