"""
CLIP Similarity Engine for Multi-Modal Misinformation Detection
This module uses OpenAI's CLIP model to calculate similarity between images and text,
helping detect inconsistencies in multi-modal content.
"""

import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import os


class CLIPSimilarityEngine:
    """
    A class for computing similarity between images and text using CLIP model.
    """
    
    def __init__(self, model_name="openai/clip-vit-base-patch32", threshold=0.25):
        """
        Initialize the CLIP model and processor.
        
        Args:
            model_name (str): The name of the CLIP model to load
            threshold (float): Similarity threshold for Match/Mismatch classification
        """
        print(f"Loading CLIP model: {model_name}...")
        try:
            self.model = CLIPModel.from_pretrained(model_name)
            self.processor = CLIPProcessor.from_pretrained(model_name)
            self.threshold = threshold
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)
            print(f"Model loaded successfully on {self.device}")
        except Exception as e:
            raise RuntimeError(f"Failed to load CLIP model: {str(e)}")
    
    def load_image(self, image_path):
        """
        Load and validate an image from the given path.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            PIL.Image: Loaded image
            
        Raises:
            FileNotFoundError: If image file doesn't exist
            ValueError: If file is not a valid image
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        try:
            image = Image.open(image_path)
            # Convert to RGB if necessary (handles RGBA, grayscale, etc.)
            if image.mode != "RGB":
                image = image.convert("RGB")
            return image
        except Exception as e:
            raise ValueError(f"Failed to load image from {image_path}: {str(e)}")
    
    def calculate_similarity(self, image_path, text):
        """
        Calculate cosine similarity between an image and text.
        
        Args:
            image_path (str): Path to the image file
            text (str): Text string to compare with the image
            
        Returns:
            tuple: (similarity_score, label)
                - similarity_score (float): Cosine similarity between 0 and 1
                - label (str): 'Match' or 'Mismatch' based on threshold
        """
        try:
            # Load and validate image
            image = self.load_image(image_path)
            
            # Validate text input
            if not text or not isinstance(text, str):
                raise ValueError("Text input must be a non-empty string")
            
            # Process inputs
            inputs = self.processor(
                text=[text],
                images=image,
                return_tensors="pt",
                padding=True
            )
            
            # Move inputs to the same device as model
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Calculate embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                
            # Get image and text features
            image_features = outputs.image_embeds
            text_features = outputs.text_embeds
            
            # Normalize features
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # Calculate cosine similarity
            similarity = (image_features @ text_features.T).item()
            
            # Determine label based on threshold
            label = "Match" if similarity >= self.threshold else "Mismatch"
            
            return similarity, label
            
        except (FileNotFoundError, ValueError) as e:
            # Re-raise known exceptions
            raise
        except Exception as e:
            raise RuntimeError(f"Error calculating similarity: {str(e)}")
    
    def analyze_with_explanation(self, image_path, text):
        """
        Analyze image-text pair and provide detailed explanation.
        
        Args:
            image_path (str): Path to the image file
            text (str): Text string to compare with the image
            
        Returns:
            dict: Dictionary containing similarity score, label, and explanation
        """
        try:
            similarity, label = self.calculate_similarity(image_path, text)
            
            # Generate explanation
            explanation = self._generate_explanation(similarity, label)
            
            return {
                "image_path": image_path,
                "text": text,
                "similarity_score": round(similarity, 4),
                "label": label,
                "explanation": explanation
            }
        except Exception as e:
            return {
                "image_path": image_path,
                "text": text,
                "error": str(e)
            }
    
    def _generate_explanation(self, similarity, label):
        """
        Generate human-readable explanation for the similarity result.
        
        Args:
            similarity (float): Similarity score
            label (str): Match or Mismatch label
            
        Returns:
            str: Explanation text
        """
        if label == "Match":
            if similarity >= 0.7:
                return f"Strong match detected (score: {similarity:.4f}). The image and text are highly consistent."
            elif similarity >= 0.5:
                return f"Moderate match detected (score: {similarity:.4f}). The image and text show reasonable alignment."
            else:
                return f"Weak match detected (score: {similarity:.4f}). The image and text are barely above the threshold."
        else:
            if similarity < 0.1:
                return f"Strong mismatch detected (score: {similarity:.4f}). The image and text appear completely unrelated."
            else:
                return f"Mismatch detected (score: {similarity:.4f}). The image and text show inconsistencies that may indicate misinformation."


def main():
    """
    Example usage of the CLIP Similarity Engine.
    """
    # Initialize the engine
    engine = CLIPSimilarityEngine(threshold=0.25)
    
    # Example usage (replace with actual image path and text)
    print("\n" + "="*60)
    print("CLIP Similarity Engine - Example Usage")
    print("="*60)
    
    # Test cases
    test_cases = [
        {
            "image": "test_image.jpg",  # Replace with actual image path
            "text": "A cat sitting on a couch"
        }
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"\nTest Case {i}:")
        print(f"Image: {test['image']}")
        print(f"Text: {test['text']}")
        print("-" * 60)
        
        result = engine.analyze_with_explanation(test['image'], test['text'])
        
        if "error" in result:
            print(f"Error: {result['error']}")
        else:
            print(f"Similarity Score: {result['similarity_score']}")
            print(f"Label: {result['label']}")
            print(f"Explanation: {result['explanation']}")
        print("-" * 60)


if __name__ == "__main__":
    main()
