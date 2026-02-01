"""
Misinformation Forensics Master Inference Script
Orchestrates multi-modal analysis with explainable AI using Gemini.
"""

import os
import json
import pickle
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from typing import Dict, Tuple, Optional, Union, List
import warnings
warnings.filterwarnings('ignore')

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Transformers
from transformers import (
    RobertaTokenizer, 
    RobertaModel,
    CLIPProcessor, 
    CLIPModel,
    AutoImageProcessor
)

# Torchvision for EfficientNet
from torchvision import models
import torchvision.transforms as transforms

# Google Gemini
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("⚠ Warning: google-generativeai not installed. Install with: pip install google-generativeai")


class MultiModalMisinfoDetector(nn.Module):
    """
    Multi-Modal Misinformation Detector with dual RoBERTa heads,
    EfficientNet visual forensics, CLIP consistency, and fusion layer.
    """
    
    def __init__(self, roberta_model_name: str = "roberta-base"):
        super().__init__()
        
        # RoBERTa for text analysis
        self.roberta = RobertaModel.from_pretrained(roberta_model_name)
        hidden_size = self.roberta.config.hidden_size  # 768
        
        # Dual heads for RoBERTa
        self.ai_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)  # AI-generated (0) vs Human (1)
        )
        
        self.misinfo_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)  # Real (0) vs Fake (1)
        )
        
        # EfficientNet for visual forensics (pretrained on CIFAKE)
        self.efficientnet = models.efficientnet_b0(weights=None)
        self.efficientnet.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280, 2)  # Real (0) vs Fake (1)
        )
        
        # CLIP embeddings size (512 from ViT-B/32)
        clip_dim = 512
        
        # Fusion layer: combines all signals
        # Inputs: [ai_score, misinfo_score, deepfake_score, clip_similarity, vault_discrepancy]
        self.fusion_layer = nn.Sequential(
            nn.Linear(5, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)  # Final: Real (0) vs Fake (1)
        )
    
    def forward_text(self, input_ids, attention_mask):
        """Forward pass for text through RoBERTa with dual heads."""
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        
        ai_logits = self.ai_head(pooled)
        misinfo_logits = self.misinfo_head(pooled)
        
        return ai_logits, misinfo_logits
    
    def forward_image(self, image_tensor):
        """Forward pass for image through EfficientNet."""
        return self.efficientnet(image_tensor)
    
    def forward_fusion(self, scores_tensor):
        """Forward pass through fusion layer."""
        return self.fusion_layer(scores_tensor)


class MisinfoForensics:
    """
    Master forensics orchestrator that coordinates all models and generates
    explainable verdicts using Gemini AI.
    """
    
    def __init__(
        self,
        fusion_weights: str = "forensics_master_final.pth",
        ai_head_weights: str = "ai_head_best.pth",
        misinfo_head_weights: str = "roberta_detective_best.pth",
        efficientnet_weights: str = "efficientnet_cifake_best.pth",
        clip_model_dir: str = r"C:\Users\Lenovo\OneDrive\Desktop\hack\models\clip-vit-b32",
        clip_weights: str = "clip_detective_best.pth",
        faiss_index_path: str = "guardian_embeddings.pkl",
        gemini_api_key: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize the forensics system.
        
        Args:
            fusion_weights: Path to trained fusion weights (forensics_master_final.pth)
            ai_head_weights: Path to AI detection head weights (fallback if fusion not found)
            misinfo_head_weights: Path to misinfo head weights (fallback)
            efficientnet_weights: Path to EfficientNet deepfake weights (fallback)
            clip_model_dir: Directory containing CLIP model
            clip_weights: Path to fine-tuned CLIP weights (fallback)
            faiss_index_path: Path to Guardian embeddings database
            gemini_api_key: Google Gemini API key (optional, reads from .env if None)
            device: torch device (cuda/cpu)
        """
        
        self.device = torch.device(device)
        print(f"Using device: {self.device}")
        
        # Initialize Gemini if available
        self.gemini_available = False
        if gemini_api_key is None:
            gemini_api_key = os.getenv('GOOGLE_API_KEY')
        
        if GEMINI_AVAILABLE and gemini_api_key:
            try:
                genai.configure(api_key=gemini_api_key)
                self.gemini_model = genai.GenerativeModel('gemini-2.5-flash')
                self.gemini_available = True
                print("✓ Gemini AI configured")
            except Exception as e:
                print(f"⚠ Gemini initialization failed: {e}")
                print("  Falling back to rule-based explanations")
        else:
            if not GEMINI_AVAILABLE:
                print("⚠ google-generativeai not installed. Using fallback explanations.")
            elif not gemini_api_key:
                print("⚠ GOOGLE_API_KEY not found in environment. Using fallback explanations.")
        
        # Load RoBERTa tokenizer
        print("\nLoading RoBERTa tokenizer...")
        self.roberta_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        
        # Initialize detector model
        self.detector = MultiModalMisinfoDetector("roberta-base").to(self.device)
        
        # Try to load final trained fusion model first
        if os.path.exists(fusion_weights):
            print(f"\n🎯 Loading FINAL TRAINED MODEL from {fusion_weights}...")
            try:
                checkpoint = torch.load(fusion_weights, map_location=self.device, weights_only=False)
                
                # Load the full model state dict (all detectives + judge synchronized)
                if 'full_model_state_dict' in checkpoint:
                    self.detector.load_state_dict(checkpoint['full_model_state_dict'], strict=False)
                    print(f"  ✓ Loaded complete integrated model")
                    print(f"    Training Epoch: {checkpoint.get('epoch', 'N/A')}")
                    print(f"    Fusion Accuracy: {checkpoint.get('accuracy', 0):.2f}%")
                    print(f"    Fusion Loss: {checkpoint.get('loss', 0):.4f}")
                else:
                    print(f"  ⚠ No full_model_state_dict found, loading individual components...")
                    raise KeyError("Missing full_model_state_dict")
                
            except Exception as e:
                print(f"  ⚠ Error loading fusion weights: {e}")
                print("  Falling back to individual model loading...")
                self._load_individual_weights(
                    ai_head_weights, misinfo_head_weights, 
                    efficientnet_weights, clip_weights
                )
        else:
            print(f"\n⚠ Final fusion model not found: {fusion_weights}")
            print("  Loading individual pretrained models...")
            self._load_individual_weights(
                ai_head_weights, misinfo_head_weights,
                efficientnet_weights, clip_weights
            )
        
        self.detector.eval()
        
        # Load CLIP model for consistency check and text similarity
        print(f"\nLoading CLIP model from {clip_model_dir}...")
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_dir)
        self.clip_model = CLIPModel.from_pretrained(clip_model_dir).to(self.device)
        self.clip_model.eval()
        
        # Load Truth Vault (FAISS/embeddings database)
        self.vault_loaded = False
        if os.path.exists(faiss_index_path):
            print(f"\nLoading Truth Vault from {faiss_index_path}...")
            with open(faiss_index_path, 'rb') as f:
                self.vault_data = pickle.load(f)
            
            # Handle different database formats
            if 'embeddings' in self.vault_data:
                self.vault_embeddings = self.vault_data['embeddings']
                self.vault_metadata = self.vault_data['metadata']
            elif 'image_embeddings' in self.vault_data:
                self.vault_embeddings = self.vault_data['image_embeddings']
                # Build metadata from separate arrays
                self.vault_metadata = []
                for i in range(len(self.vault_data.get('text_contents', []))):
                    self.vault_metadata.append({
                        'title': self.vault_data['text_contents'][i] if i < len(self.vault_data['text_contents']) else 'Unknown',
                        'url': self.vault_data['image_paths'][i] if i < len(self.vault_data['image_paths']) else 'N/A',
                        'date': 'N/A'
                    })
            else:
                print(f"  ⚠ Unknown database format")
                self.vault_embeddings = None
                self.vault_metadata = None
            
            if self.vault_embeddings is not None:
                print(f"  ✓ Loaded {len(self.vault_metadata)} verified articles")
                self.vault_loaded = True
        else:
            print(f"⚠ Truth Vault not found: {faiss_index_path}")
            self.vault_embeddings = None
            self.vault_metadata = None
        
        # Image transforms for EfficientNet
        self.efficientnet_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _to_pil_image(self, image_or_path: Union[str, Image.Image]) -> Image.Image:
        if isinstance(image_or_path, Image.Image):
            return image_or_path.convert('RGB')
        return Image.open(str(image_or_path)).convert('RGB')
    
    def _load_individual_weights(
        self, 
        ai_head_weights: str,
        misinfo_head_weights: str,
        efficientnet_weights: str,
        clip_weights: str
    ):
        """Helper method to load individual model weights (fallback)."""
        
        # Load AI head weights (HC3)
        if os.path.exists(ai_head_weights):
            print(f"Loading AI detection head from {ai_head_weights}...")
            checkpoint = torch.load(ai_head_weights, map_location=self.device, weights_only=False)
            ai_state = {k.replace('ai_head.', ''): v for k, v in checkpoint['model_state_dict'].items() if 'ai_head' in k}
            self.detector.ai_head.load_state_dict(ai_state, strict=False)
            print(f"  ✓ Loaded from epoch {checkpoint.get('epoch', 'N/A')}")
        
        # Load misinfo head weights (WELFake)
        if os.path.exists(misinfo_head_weights):
            print(f"Loading misinfo detection head from {misinfo_head_weights}...")
            checkpoint = torch.load(misinfo_head_weights, map_location=self.device, weights_only=False)
            misinfo_state = {k.replace('misinfo_head.', ''): v for k, v in checkpoint['model_state_dict'].items() if 'misinfo_head' in k}
            self.detector.misinfo_head.load_state_dict(misinfo_state, strict=False)
            print(f"  ✓ Loaded from epoch {checkpoint.get('epoch', 'N/A')}")
        
        # Load EfficientNet weights (CIFAKE)
        if os.path.exists(efficientnet_weights):
            print(f"Loading EfficientNet from {efficientnet_weights}...")
            checkpoint = torch.load(efficientnet_weights, map_location=self.device, weights_only=False)
            
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                efficientnet_state = {k.replace('efficientnet.', ''): v for k, v in state_dict.items() if 'efficientnet' in k}
                self.detector.efficientnet.load_state_dict(efficientnet_state, strict=False)
                print(f"  ✓ Loaded from epoch {checkpoint.get('epoch', 'N/A')}")
            else:
                try:
                    self.detector.efficientnet.load_state_dict(checkpoint, strict=False)
                    print(f"  ✓ Loaded weights successfully")
                except RuntimeError:
                    classifier_state = {k: v for k, v in checkpoint.items() if 'classifier' in k}
                    if classifier_state:
                        self.detector.efficientnet.classifier.load_state_dict(classifier_state, strict=False)
                        print(f"  ✓ Loaded classifier head")
        
        # Load CLIP fine-tuned weights
        if os.path.exists(clip_weights):
            print(f"Loading fine-tuned CLIP weights from {clip_weights}...")
            try:
                checkpoint = torch.load(clip_weights, map_location=self.device, weights_only=False)
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                    clip_state = {k.replace('clip.', ''): v for k, v in state_dict.items() if k.startswith('clip.')}
                    if clip_state:
                        self.clip_model.load_state_dict(clip_state, strict=False)
                        print(f"  ✓ Loaded from epoch {checkpoint.get('epoch', 'N/A')}")
            except Exception as e:
                print(f"  ⚠ Could not load CLIP weights: {e}")
    
    def analyze_text(self, text: str) -> Dict[str, float]:
        """
        Step 1: Text Check with RoBERTa dual heads.
        
        Returns:
            Dictionary with ai_score and misinfo_score (probabilities for class 1)
        """
        # Tokenize
        inputs = self.roberta_tokenizer(
            text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True
        ).to(self.device)
        
        # Forward pass
        with torch.no_grad():
            ai_logits, misinfo_logits = self.detector.forward_text(
                inputs['input_ids'],
                inputs['attention_mask']
            )
            
            ai_probs = torch.softmax(ai_logits, dim=1)
            misinfo_probs = torch.softmax(misinfo_logits, dim=1)
            
            # Probability of class 1 (AI-generated / Fake)
            ai_score = ai_probs[0, 1].item()
            misinfo_score = misinfo_probs[0, 1].item()
        
        return {
            'ai_score': ai_score,
            'misinfo_score': misinfo_score
        }
    
    def analyze_image(self, image_path: Union[str, Image.Image]) -> Dict[str, float]:
        """
        Step 2: Visual Check with EfficientNet deepfake detection.
        
        Returns:
            Dictionary with deepfake_score (probability of fake)
        """
        # Load and preprocess image
        image = self._to_pil_image(image_path)
        image_tensor = self.efficientnet_transform(image).unsqueeze(0).to(self.device)
        
        # Forward pass
        with torch.no_grad():
            logits = self.detector.forward_image(image_tensor)
            probs = torch.softmax(logits, dim=1)
            deepfake_score = probs[0, 1].item()  # Probability of fake
        
        return {
            'deepfake_score': deepfake_score
        }
    
    def analyze_consistency(self, text: str, image_path: Union[str, Image.Image]) -> Dict[str, float]:
        """
        Step 3: Consistency Check with CLIP similarity.
        
        Returns:
            Dictionary with clip_similarity (cosine similarity)
        """
        # Load image
        image = self._to_pil_image(image_path)
        
        # Process inputs
        inputs = self.clip_processor(
            text=[text],
            images=image,
            return_tensors="pt",
            padding=True
        ).to(self.device)
        
        # Get embeddings
        with torch.no_grad():
            outputs = self.clip_model(**inputs)
            text_embeds = outputs.text_embeds
            image_embeds = outputs.image_embeds
            
            # Normalize
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
            image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
            
            # Cosine similarity
            similarity = (text_embeds @ image_embeds.T).item()
        
        return {
            'clip_similarity': similarity
        }
    
    def search_vault(self, image_path: Union[str, Image.Image], user_caption: str = None, top_k: int = 5) -> Dict:
        """
        Step 4: Vault Check - Search Guardian articles database.
        
        Args:
            image_path: Path to image to search
            user_caption: User's caption/headline to compare with original (optional)
            top_k: Number of top matches to return
        
        Returns:
            Dictionary with vault_discrepancy score, matched articles, and text similarity
        """
        if not self.vault_loaded:
            return {
                'vault_discrepancy': 0.0,
                'matches': [],
                'vault_available': False,
                'text_similarity': 0.0
            }
        
        # Get image embedding from CLIP
        image = self._to_pil_image(image_path)
        inputs = self.clip_processor(
            images=image,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            image_embeds = self.clip_model.get_image_features(**inputs)
            image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
            query_embedding = image_embeds.cpu().numpy()[0]
        
        # Calculate cosine similarities with all vault embeddings
        vault_embeddings_norm = self.vault_embeddings / np.linalg.norm(
            self.vault_embeddings, axis=1, keepdims=True
        )
        similarities = vault_embeddings_norm @ query_embedding
        
        # Get top K matches
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        top_similarities = similarities[top_indices]
        
        matches = []
        for idx, sim in zip(top_indices, top_similarities):
            idx = int(idx)  # Convert numpy int to Python int
            matches.append({
                'similarity': float(sim),
                'title': self.vault_metadata[idx]['title'],
                'url': self.vault_metadata[idx].get('url', 'N/A'),
                'date': self.vault_metadata[idx].get('date', 'N/A')
            })
        
        # Discrepancy score: if high similarity found, might be reused image
        max_similarity = float(top_similarities[0])
        vault_discrepancy = max_similarity if max_similarity > 0.85 else 0.0
        
        # Calculate text similarity if match found and user caption provided
        text_similarity = 0.0
        if user_caption and max_similarity > 0.85 and matches:
            original_headline = matches[0]['title']
            
            # Use CLIP text encoder to compute similarity
            with torch.no_grad():
                text_inputs = self.clip_processor(
                    text=[user_caption, original_headline],
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                ).to(self.device)
                
                text_embeds = self.clip_model.get_text_features(**text_inputs)
                text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
                
                # Cosine similarity between user caption and original headline
                text_similarity = float((text_embeds[0] @ text_embeds[1].T).item())
        
        return {
            'vault_discrepancy': vault_discrepancy,
            'matches': matches,
            'vault_available': True,
            'text_similarity': text_similarity
        }

    def analyze_video(
        self,
        video_path: str,
        text: Optional[str] = None,
        max_frames: int = 12,
        stride_seconds: float = 1.0
    ) -> Dict:
        """Analyze a video by sampling frames and aggregating forensic signals."""
        try:
            import cv2
        except Exception as e:
            raise RuntimeError(
                "opencv-python is required for video analysis. Install with: pip install opencv-python"
            ) from e

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            fps = 25.0
        frame_stride = max(1, int(round(fps * max(0.1, float(stride_seconds)))))

        frame_idx = 0
        sampled = 0
        deepfake_scores: List[float] = []
        clip_sims: List[float] = []
        best_vault = {
            'vault_discrepancy': 0.0,
            'matches': [],
            'vault_available': self.vault_loaded,
            'text_similarity': 0.0
        }

        best_frame: Optional[Image.Image] = None

        while sampled < max_frames:
            ok, frame = cap.read()
            if not ok:
                break

            if frame_idx % frame_stride != 0:
                frame_idx += 1
                continue

            frame_idx += 1
            sampled += 1

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)

            # Per-frame signals
            img_scores = self.analyze_image(pil)
            deepfake_scores.append(float(img_scores.get('deepfake_score', 0.0)))

            if text:
                cons = self.analyze_consistency(text, pil)
                clip_sims.append(float(cons.get('clip_similarity', 0.0)))

            vault = self.search_vault(pil, user_caption=text)
            if float(vault.get('vault_discrepancy', 0.0)) > float(best_vault.get('vault_discrepancy', 0.0)):
                best_vault = vault
                best_frame = pil

        cap.release()

        if not deepfake_scores:
            raise RuntimeError("No frames could be read from the video.")

        deepfake_mean = float(np.mean(deepfake_scores))
        clip_mean = float(np.mean(clip_sims)) if clip_sims else 0.0

        return {
            'deepfake_score': deepfake_mean,
            'clip_similarity': clip_mean,
            'vault_discrepancy': float(best_vault.get('vault_discrepancy', 0.0)),
            'text_similarity': float(best_vault.get('text_similarity', 0.0)),
            'vault_matches': best_vault.get('matches', []),
            'best_frame': best_frame
        }
    
    def fusion_verdict(self, scores: Dict[str, float]) -> Dict:
        """
        Step 5: Fusion Layer - Combine all scores for final verdict.
        
        Args:
            scores: Dictionary with all forensic scores
        
        Returns:
            Dictionary with binary label (0=REAL, 1=FAKE) and confidence from softmax
        """
        # Prepare input tensor for fusion layer
        # [ai_score, misinfo_score, deepfake_score, clip_similarity, vault_discrepancy]
        fusion_input = torch.tensor([
            scores.get('ai_score', 0.0),
            scores.get('misinfo_score', 0.0),
            scores.get('deepfake_score', 0.0),
            scores.get('clip_similarity', 0.0),
            scores.get('vault_discrepancy', 0.0)
        ], dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # Forward pass through fusion layer
        with torch.no_grad():
            logits = self.detector.forward_fusion(fusion_input)
            probs = torch.softmax(logits, dim=1)
            
            # Extract probabilities for both classes
            real_prob = probs[0, 0].item()
            fake_prob = probs[0, 1].item()
            
            # Binary label: 0 = REAL, 1 = FAKE
            verdict_label = 1 if fake_prob > 0.5 else 0
            
            # Confidence: probability of the predicted class
            confidence = fake_prob if verdict_label == 1 else real_prob
        
        return {
            'verdict': verdict_label,
            'confidence': confidence,
            'fake_probability': fake_prob,
            'real_probability': real_prob
        }
    
    def build_gemini_prompt(self, all_scores: Dict, vault_matches: list) -> str:
        """
        Build a clean, formatted prompt for Gemini API with all forensic scores.
        
        Args:
            all_scores: All forensic scores
            vault_matches: Top matches from vault search
        
        Returns:
            Formatted prompt string for Gemini API
        """
        verdict_text = "FAKE" if all_scores.get('verdict', 0) == 1 else "REAL"
        confidence = float(all_scores.get('confidence', 0.0) or 0.0)
        
        prompt = f"""You are a senior misinformation forensics analyst writing a detailed but concise report for a dashboard.

    Write the response in Markdown with the exact section headers below, using the provided numeric signals verbatim where relevant.

    Rules:
    - Be specific: cite key numbers (probabilities/similarities) and explain what they imply.
    - Rank the top signals (strongest to weakest) and explain how they contributed.
    - If a modality is missing (text/image/video), explicitly note what was skipped and how that limits confidence.
    - Avoid generic advice; focus on evidence-based reasoning.
    - Keep it readable: 120–220 words total.

    Use this format:
    ### Verdict
    <1–2 sentences with verdict + confidence and the core reason>

    ### Key Evidence (ranked)
    - <bullet 1>
    - <bullet 2>
    - <bullet 3>

    ### Cross-Checks & Caveats
    - <1–2 bullets about vault/consistency or missing signals>

    ### Recommended Next Step
    <1 sentence: what the user should do to verify>

FORENSIC ANALYSIS SCORES:

1. Final Verdict & Confidence:
   - Verdict: {verdict_text}
   - Confidence Score: {confidence:.1%} (derived from softmax probabilities)
   - REAL Probability: {all_scores.get('real_probability', 0.0):.2%}
   - FAKE Probability: {all_scores.get('fake_probability', 0.0):.2%}

2. AI-Text & Propaganda Probability:
   - AI-Generated Score: {all_scores.get('ai_score', 0.0):.2%} (RoBERTa classifier, higher = more AI-like)
   - Propaganda/Misinfo Score: {all_scores.get('misinfo_score', 0.0):.2%} (trained on WELFake dataset)

3. Deepfake Visual Score:
   - Deepfake Probability: {all_scores.get('deepfake_score', 0.0):.2%} (EfficientNet on CIFAKE dataset)

4. Consistency (CLIP) & Vault Discrepancy:
    - Image-Text Consistency: {float(all_scores.get('clip_similarity', 0.0) or 0.0):.4f} (cosine similarity, -1 to 1)
    - Historical Database Match: {float(all_scores.get('vault_discrepancy', 0.0) or 0.0):.2%} (image found in Guardian archive)
"""
        
        # Add vault match details if available
        if vault_matches and all_scores.get('vault_discrepancy', 0.0) > 0.5:
            top_match = vault_matches[0]
            text_sim = float(all_scores.get('text_similarity', 0.0) or 0.0)
            
            prompt += f"""
5. Truth Vault Cross-Check:
   - Match Found: "{top_match['title']}"
   - Image Similarity: {top_match['similarity']:.1%}
   - Text Similarity Score: {text_sim:.2%} (CLIP text encoder comparison)
   - Published: {top_match.get('date', 'N/A')}
   - Context: Image reused from different story
"""
        
        prompt += "\n\nTask: Produce the Markdown report using the structure above. Emphasize the strongest quantitative signals and any contradictions (e.g., high vault match but low text similarity, or strong text signal but weak visual signal)."
        
        return prompt
    
    def generate_gemini_explanation(self, all_scores: Dict, vault_matches: list) -> str:
        """
        Step 6: Gemini Live Explainer - Generate human-readable forensic summary.
        
        Args:
            all_scores: All forensic scores
            vault_matches: Top matches from vault search
        
        Returns:
            2-sentence forensic summary explaining the math behind the verdict
        """
        # Check if Gemini is available
        if not self.gemini_available:
            print("  ℹ Using fallback explanation (Gemini not available)")
            return self._generate_fallback_explanation(all_scores, vault_matches)
        
        # Build the detailed prompt with all scores
        prompt = self.build_gemini_prompt(all_scores, vault_matches)
        
        # Call Gemini API with error handling
        try:
            response = self.gemini_model.generate_content(prompt)
            
            # Check if response is valid
            if hasattr(response, 'text') and response.text:
                return response.text.strip()
            else:
                print("  ⚠ Gemini returned empty response")
                return self._generate_fallback_explanation(all_scores, vault_matches)
                
        except Exception as e:
            error_msg = str(e).lower()
            
            # Specific error handling
            if "api key" in error_msg or "authentication" in error_msg:
                print(f"  ⚠ Gemini API authentication failed: {e}")
                print("  Check your GOOGLE_API_KEY in .env file")
            elif "quota" in error_msg or "limit" in error_msg:
                print(f"  ⚠ Gemini API quota exceeded: {e}")
            elif "timeout" in error_msg:
                print(f"  ⚠ Gemini API timeout: {e}")
            else:
                print(f"  ⚠ Gemini API error: {e}")
            
            print("  Falling back to rule-based explanation")
            return self._generate_fallback_explanation(all_scores, vault_matches)
    
    def _generate_fallback_explanation(self, all_scores: Dict, vault_matches: list) -> str:
        """Fallback explanation when Gemini is unavailable."""
        verdict_text = "FAKE" if all_scores['verdict'] == 1 else "REAL"
        
        # Determine primary concern
        if all_scores['vault_discrepancy'] > 0.7:
            return (f"This content is classified as {verdict_text}. "
                   f"Our database found this image was previously published in a different context "
                   f"(\"{vault_matches[0]['title']}\"), suggesting potential misuse.")
        elif all_scores['deepfake_score'] > 0.7:
            return (f"This content is classified as {verdict_text}. "
                   f"The image shows strong signs of digital manipulation (deepfake probability: {all_scores['deepfake_score']:.1%}).")
        elif all_scores['ai_score'] > 0.7:
            return (f"This content is classified as {verdict_text}. "
                   f"The text exhibits characteristics typical of AI-generated content.")
        elif all_scores['misinfo_score'] > 0.7:
            return (f"This content is classified as {verdict_text}. "
                   f"The text uses language patterns commonly associated with misinformation.")
        elif all_scores['clip_similarity'] < 0.3:
            return (f"This content is classified as {verdict_text}. "
                   f"The image and caption show poor alignment, suggesting potential mismatching.")
        else:
            return (f"This content is classified as {verdict_text} with {all_scores['confidence']:.1%} confidence. "
                   f"Multiple signals from text analysis, image forensics, and database checks support this assessment.")
    
    def analyze(
        self,
        text: Optional[str] = None,
        image_path: Optional[str] = None,
        video_path: Optional[str] = None,
        verbose: bool = True
    ) -> Dict:
        """
        Complete forensic analysis pipeline.
        
        Args:
            text: News headline or article text
            image_path: Path to accompanying image
            verbose: Print detailed analysis steps
        
        Returns:
            Complete forensic report
        """
        if verbose:
            print("\n" + "=" * 70)
            print("MISINFORMATION FORENSICS ANALYSIS")
            print("=" * 70)
        
        if not text and not image_path and not video_path:
            raise ValueError("Provide at least one of: text, image_path, or video_path")

        # Step 1: Text Check (optional)
        text_scores = {'ai_score': 0.0, 'misinfo_score': 0.0}
        if text:
            if verbose:
                print("\n[Step 1] Text Analysis (RoBERTa Dual Heads)...")
            text_scores = self.analyze_text(text)
            if verbose:
                print(f"  • AI-Generated Score: {text_scores['ai_score']:.2%}")
                print(f"  • Misinfo/Propaganda Score: {text_scores['misinfo_score']:.2%}")
        elif verbose:
            print("\n[Step 1] Text Analysis (RoBERTa Dual Heads)...")
            print("  • Skipped (no text provided)")
        
        # Step 2-4: Visual / Consistency / Vault
        image_scores = {'deepfake_score': 0.0}
        consistency_scores = {'clip_similarity': 0.0}
        vault_results = {'vault_discrepancy': 0.0, 'matches': [], 'vault_available': self.vault_loaded, 'text_similarity': 0.0}

        if video_path:
            if verbose:
                print("\n[Step 2] Video Forensics (Frame Sampling)...")
            video_scores = self.analyze_video(video_path, text=text)
            image_scores['deepfake_score'] = video_scores.get('deepfake_score', 0.0)
            consistency_scores['clip_similarity'] = video_scores.get('clip_similarity', 0.0)
            vault_results['vault_discrepancy'] = video_scores.get('vault_discrepancy', 0.0)
            vault_results['matches'] = video_scores.get('vault_matches', [])
            vault_results['text_similarity'] = video_scores.get('text_similarity', 0.0)

            if verbose:
                print(f"  • Deepfake Probability (avg): {image_scores['deepfake_score']:.2%}")
                if text:
                    print(f"  • CLIP Similarity (avg): {consistency_scores['clip_similarity']:.4f}")
                print(f"  • Historical Discrepancy (max): {vault_results['vault_discrepancy']:.2%}")
        elif image_path:
            if verbose:
                print("\n[Step 2] Visual Forensics (EfficientNet)...")
            image_scores = self.analyze_image(image_path)
            if verbose:
                print(f"  • Deepfake Probability: {image_scores['deepfake_score']:.2%}")

            if text:
                if verbose:
                    print("\n[Step 3] Image-Text Consistency (CLIP)...")
                consistency_scores = self.analyze_consistency(text, image_path)
                if verbose:
                    print(f"  • CLIP Similarity: {consistency_scores['clip_similarity']:.4f}")
            elif verbose:
                print("\n[Step 3] Image-Text Consistency (CLIP)...")
                print("  • Skipped (no text provided)")

            if verbose:
                print("\n[Step 4] Truth Vault Search (Guardian Database)...")
            vault_results = self.search_vault(image_path, user_caption=text)
            if verbose:
                if vault_results['vault_available']:
                    print(f"  • Historical Discrepancy: {vault_results['vault_discrepancy']:.2%}")
                    if vault_results['matches']:
                        print(f"  • Top Match: \"{vault_results['matches'][0]['title']}\"")
                        print(f"    Image Similarity: {vault_results['matches'][0]['similarity']:.1%}")
                        if vault_results.get('text_similarity', 0.0) > 0:
                            print(f"    Text Similarity: {vault_results['text_similarity']:.2%}")
                else:
                    print("  • Vault not available")
        else:
            if verbose:
                print("\n[Step 2] Visual Forensics (EfficientNet)...")
                print("  • Skipped (no image/video provided)")
                print("\n[Step 3] Image-Text Consistency (CLIP)...")
                print("  • Skipped (no image/video provided)")
                print("\n[Step 4] Truth Vault Search (Guardian Database)...")
                print("  • Skipped (no image/video provided)")
        
        # Combine all scores
        all_scores = {
            **text_scores,
            **image_scores,
            **consistency_scores,
            'vault_discrepancy': vault_results['vault_discrepancy'],
            'text_similarity': vault_results.get('text_similarity', 0.0)
        }
        
        # Step 5: Verdict
        if verbose:
            print("\n[Step 5] Verdict...")

        # If we have both text + (image or video), use fusion.
        use_fusion = bool(text) and bool(image_path or video_path)
        if use_fusion:
            verdict_results = self.fusion_verdict(all_scores)
        else:
            # Fallback verdicts for missing modality
            if text and not (image_path or video_path):
                fake_prob = float(all_scores.get('misinfo_score', 0.0))
            elif (image_path or video_path) and not text:
                fake_prob = float(max(all_scores.get('deepfake_score', 0.0), all_scores.get('vault_discrepancy', 0.0)))
            else:
                fake_prob = 0.5
            fake_prob = max(0.0, min(1.0, fake_prob))
            real_prob = 1.0 - fake_prob
            verdict_label = 1 if fake_prob > 0.5 else 0
            confidence = fake_prob if verdict_label == 1 else real_prob
            verdict_results = {
                'verdict': verdict_label,
                'confidence': confidence,
                'fake_probability': fake_prob,
                'real_probability': real_prob
            }
        all_scores.update(verdict_results)
        
        if verbose:
            verdict_emoji = "🔴" if verdict_results['verdict'] == 1 else "🟢"
            verdict_text = "FAKE" if verdict_results['verdict'] == 1 else "REAL"
            print(f"  {verdict_emoji} Final Verdict: {verdict_text}")
            print(f"  • Confidence: {verdict_results['confidence']:.1%}")
        
        # Step 6: Gemini Explanation
        if verbose:
            print("\n[Step 6] Generating Forensic Summary...")
        explanation = self.generate_gemini_explanation(all_scores, vault_results['matches'])
        
        if verbose:
            print("\n" + "=" * 70)
            print("FORENSIC SUMMARY")
            print("=" * 70)
            print(explanation)
            print("=" * 70)
        
        return {
            'verdict': verdict_results['verdict'],
            'verdict_text': "FAKE" if verdict_results['verdict'] == 1 else "REAL",
            'confidence': verdict_results['confidence'],
            'scores': all_scores,
            'vault_matches': vault_results['matches'],
            'explanation': explanation
        }


def main():
    """Example usage of the forensics system."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Misinformation Forensics Analysis")
    parser.add_argument("--text", type=str, help="News headline or article text")
    parser.add_argument("--image", type=str, help="Path to image file")
    parser.add_argument("--video", type=str, help="Path to video file")
    parser.add_argument("--gemini-key", type=str, help="Google Gemini API key (optional, reads from .env)")
    parser.add_argument("--output", type=str, help="Save results to JSON file")
    args = parser.parse_args()

    if not (args.text or args.image or args.video):
        parser.error("Provide at least one of --text, --image, or --video")
    
    # Get API key from args or environment
    gemini_key = args.gemini_key or os.getenv('GOOGLE_API_KEY')
    
    # Initialize forensics system
    forensics = MisinfoForensics(
        gemini_api_key=gemini_key
    )
    
    # Run analysis
    results = forensics.analyze(
        text=args.text,
        image_path=args.image,
        video_path=args.video,
        verbose=True
    )
    
    # Save results if requested
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n✓ Results saved to {args.output}")


if __name__ == "__main__":
    main()
