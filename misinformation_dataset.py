"""
Unified Multimodal Dataset for Misinformation Detection
Handles text, images, and video frames for multi-modal learning.
"""

import torch
from torch.utils.data import Dataset
from transformers import CLIPTokenizer, AutoTokenizer
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
import os
from io import BytesIO
import random


class RandomJPEGCompression:
    """
    Custom transform that applies random JPEG compression to simulate compression artifacts.
    This helps detect adversarially compressed or manipulated images.
    """
    
    def __init__(self, quality_range=(40, 80)):
        """
        Initialize the RandomJPEGCompression transform.
        
        Args:
            quality_range (tuple): Min and max JPEG quality values
        """
        self.quality_range = quality_range
    
    def __call__(self, image):
        """
        Apply random JPEG compression to the image.
        
        Args:
            image (PIL.Image): Input image
            
        Returns:
            PIL.Image: Compressed image
        """
        if not isinstance(image, Image.Image):
            return image
        
        # Randomly select JPEG quality
        quality = random.randint(self.quality_range[0], self.quality_range[1])
        
        # Save and reload with JPEG compression
        buffer = BytesIO()
        image.save(buffer, format='JPEG', quality=quality, optimize=False)
        buffer.seek(0)
        compressed_image = Image.open(buffer)
        compressed_image.load()  # Force load to avoid buffer issues
        
        # Return as RGB to maintain consistency
        return compressed_image.convert('RGB')


class MisinfoDataset(Dataset):
    """
    PyTorch Dataset for multi-modal misinformation detection.
    Handles text, images, and video frames with appropriate preprocessing.
    """
    
    def __init__(
        self,
        data_list,
        clip_model_name="openai/clip-vit-base-patch32",
        roberta_model_name="roberta-base",
        max_text_length=77,
        image_size=224,
        is_train=True
    ):
        """
        Initialize the MisinfoDataset.
        
        Args:
            data_list (list): List of dictionaries containing:
                - 'text': Text content
                - 'image_path': Path to image file (optional)
                - 'video_path': Path to video file (optional)
                - 'label': Label (0 for genuine, 1 for misinformation)
            clip_model_name (str): Name of the CLIP model for tokenization
            roberta_model_name (str): Name of the RoBERTa model for tokenization
            max_text_length (int): Maximum text length for tokenization
            image_size (int): Size to resize images to (default: 224)
            is_train (bool): If True, apply data augmentations. If False, use validation transforms
        """
        self.data_list = data_list
        self.max_text_length = max_text_length
        self.image_size = image_size
        self.is_train = is_train
        
        # Initialize tokenizers
        print("Loading tokenizers...")
        self.clip_tokenizer = CLIPTokenizer.from_pretrained(clip_model_name)
        self.roberta_tokenizer = AutoTokenizer.from_pretrained(roberta_model_name)
        print("Tokenizers loaded successfully")
        
        # Create separate transforms for training and validation
        if is_train:
            # Training transforms with augmentations for robustness
            self.image_transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
                # Data augmentations for robustness
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.1
                ),
                transforms.RandomApply(
                    [transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.0))],
                    p=0.3
                ),
                transforms.Lambda(RandomJPEGCompression(quality_range=(40, 80))),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.481, 0.457, 0.408],
                    std=[0.268, 0.261, 0.275]
                )
            ])
        else:
            # Validation/Test transforms without augmentations
            self.image_transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.481, 0.457, 0.408],
                    std=[0.268, 0.261, 0.275]
                )
            ])
    
    def __len__(self):
        """Return the total number of samples in the dataset."""
        return len(self.data_list)
    
    def tokenize_text(self, text):
        """
        Tokenize text using both CLIP and RoBERTa tokenizers.
        
        Args:
            text (str): Input text to tokenize
            
        Returns:
            tuple: (clip_tokens, roberta_tokens)
        """
        # CLIP tokenization
        clip_tokens = self.clip_tokenizer(
            text,
            padding='max_length',
            max_length=self.max_text_length,
            truncation=True,
            return_tensors='pt'
        )
        
        # RoBERTa tokenization
        roberta_tokens = self.roberta_tokenizer(
            text,
            padding='max_length',
            max_length=self.max_text_length,
            truncation=True,
            return_tensors='pt'
        )
        
        return clip_tokens, roberta_tokens
    
    def load_and_preprocess_image(self, image_path):
        """
        Load and preprocess an image.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            torch.Tensor: Preprocessed image tensor of shape (3, 224, 224)
        """
        try:
            image = Image.open(image_path)
            image_tensor = self.image_transform(image)
            return image_tensor
        except Exception as e:
            print(f"Error loading image {image_path}: {str(e)}")
            # Return a blank tensor if image loading fails
            return torch.zeros(3, self.image_size, self.image_size)
    
    def extract_and_preprocess_video_frames(self, video_path, num_frames=3):
        """
        Extract equidistant frames from a video and preprocess them.
        
        Args:
            video_path (str): Path to the video file
            num_frames (int): Number of frames to extract (default: 3)
            
        Returns:
            torch.Tensor: Stacked tensor of preprocessed frames with shape (num_frames, 3, 224, 224)
        """
        try:
            # Open video file
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                raise ValueError(f"Cannot open video file: {video_path}")
            
            # Get total number of frames
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if total_frames < num_frames:
                print(f"Warning: Video has only {total_frames} frames, requested {num_frames}")
                num_frames = max(1, total_frames)
            
            # Calculate equidistant frame indices
            frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
            
            frames_tensors = []
            
            for frame_idx in frame_indices:
                # Set frame position
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if ret:
                    # Convert BGR (OpenCV) to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Convert to PIL Image
                    pil_image = Image.fromarray(frame_rgb)
                    
                    # Apply image transforms
                    frame_tensor = self.image_transform(pil_image)
                    frames_tensors.append(frame_tensor)
                else:
                    print(f"Warning: Could not read frame {frame_idx}")
                    # Add blank frame if reading fails
                    frames_tensors.append(torch.zeros(3, self.image_size, self.image_size))
            
            cap.release()
            
            # Stack frames: shape (num_frames, 3, 224, 224)
            video_tensor = torch.stack(frames_tensors)
            
            return video_tensor
            
        except Exception as e:
            print(f"Error processing video {video_path}: {str(e)}")
            # Return blank video tensor if processing fails
            return torch.zeros(num_frames, 3, self.image_size, self.image_size)
    
    def __getitem__(self, idx):
        """
        Get a single sample from the dataset.
        
        Args:
            idx (int): Index of the sample
            
        Returns:
            dict: Dictionary containing:
                - 'clip_text': CLIP tokenized text (input_ids, attention_mask)
                - 'roberta_text': RoBERTa tokenized text (input_ids, attention_mask)
                - 'image_tensor': Preprocessed image tensor (3, 224, 224)
                - 'video_tensor': Preprocessed video frames tensor (3, 3, 224, 224)
                - 'label': Label (0 or 1)
        """
        sample = self.data_list[idx]
        
        # Extract text and tokenize
        text = sample.get('text', '')
        clip_tokens, roberta_tokens = self.tokenize_text(text)
        
        # Process image if available
        image_path = sample.get('image_path', None)
        if image_path and os.path.exists(image_path):
            image_tensor = self.load_and_preprocess_image(image_path)
        else:
            # Return blank tensor if no image
            image_tensor = torch.zeros(3, self.image_size, self.image_size)
        
        # Process video if available
        video_path = sample.get('video_path', None)
        if video_path and os.path.exists(video_path):
            video_tensor = self.extract_and_preprocess_video_frames(video_path, num_frames=3)
        else:
            # Return blank tensor if no video
            video_tensor = torch.zeros(3, 3, self.image_size, self.image_size)
        
        # Get label
        label = sample.get('label', 0)
        
        return {
            'clip_text': {
                'input_ids': clip_tokens['input_ids'].squeeze(0),
                'attention_mask': clip_tokens['attention_mask'].squeeze(0)
            },
            'roberta_text': {
                'input_ids': roberta_tokens['input_ids'].squeeze(0),
                'attention_mask': roberta_tokens['attention_mask'].squeeze(0)
            },
            'image_tensor': image_tensor,
            'video_tensor': video_tensor,
            'label': torch.tensor(label, dtype=torch.long)
        }


def create_sample_dataset(split='train'):
    """
    Create a sample dataset for testing purposes with train/validation split support.
    
    Args:
        split (str): 'train' for training dataset with augmentations, 'val' for validation without augmentations
    
    Returns:
        tuple: (train_dataset, val_dataset) or single dataset based on split
    """
    # Sample data (replace with actual data)
    sample_data = [
        {
            'text': 'A cat sitting on a couch',
            'image_path': 'sample_images/cat.jpg',
            'video_path': 'sample_videos/cat_video.mp4',
            'label': 0  # Genuine
        },
        {
            'text': 'Breaking news: Major event happened',
            'image_path': 'sample_images/news.jpg',
            'video_path': 'sample_videos/news_video.mp4',
            'label': 1  # Misinformation
        }
    ]
    
    if split == 'both':
        # Return both training and validation datasets
        train_dataset = MisinfoDataset(sample_data, is_train=True)
        val_dataset = MisinfoDataset(sample_data, is_train=False)
        return train_dataset, val_dataset
    else:
        # Return single dataset
        is_train = split == 'train'
        dataset = MisinfoDataset(sample_data, is_train=is_train)
        return dataset


if __name__ == "__main__":
    print("="*60)
    print("Unified Multimodal Dataset - Test")
    print("="*60)
    
    # Create sample dataset with train/validation split
    print("\nCreating sample datasets (train and validation)...")
    train_dataset, val_dataset = create_sample_dataset(split='both')
    
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    
    # Test getting an item from training dataset
    print("\n" + "="*60)
    print("Training Dataset Sample (with augmentations)")
    print("="*60)
    try:
        sample = train_dataset[0]
        print("\nSample structure:")
        print(f"- clip_text input_ids shape: {sample['clip_text']['input_ids'].shape}")
        print(f"- roberta_text input_ids shape: {sample['roberta_text']['input_ids'].shape}")
        print(f"- image_tensor shape: {sample['image_tensor'].shape}")
        print(f"- video_tensor shape: {sample['video_tensor'].shape}")
        print(f"- label: {sample['label']}")
    except Exception as e:
        print(f"Error: {str(e)}")
    
    # Test getting an item from validation dataset
    print("\n" + "="*60)
    print("Validation Dataset Sample (without augmentations)")
    print("="*60)
    try:
        sample = val_dataset[0]
        print("\nSample structure:")
        print(f"- clip_text input_ids shape: {sample['clip_text']['input_ids'].shape}")
        print(f"- roberta_text input_ids shape: {sample['roberta_text']['input_ids'].shape}")
        print(f"- image_tensor shape: {sample['image_tensor'].shape}")
        print(f"- video_tensor shape: {sample['video_tensor'].shape}")
        print(f"- label: {sample['label']}")
    except Exception as e:
        print(f"Error: {str(e)}")
    
    print("\n" + "="*60)
    print("Augmentations Applied:")
    print("="*60)
    print("Training transforms:")
    print("  ✓ RandomHorizontalFlip (50% probability)")
    print("  ✓ ColorJitter (brightness, contrast, saturation, hue)")
    print("  ✓ GaussianBlur (30% probability)")
    print("  ✓ RandomJPEGCompression (quality: 40-80)")
    print("\nValidation transforms:")
    print("  ✓ No augmentations applied (clean preprocessing only)")

