"""
Prepare Final_Fusion_Train.csv by combining different types of misinformation.
Mixes out-of-context, AI-generated, and deepfake examples.
"""

import pandas as pd
import os
import random
from pathlib import Path


def prepare_fusion_dataset(
    output_csv: str = "Final_Fusion_Train.csv",
    samples_per_type: int = 500
):
    """
    Create a balanced dataset mixing different misinformation types.
    
    Data sources:
    1. clip_train.csv - Out-of-context (mismatched image-text pairs)
    2. hc_train.csv - AI-generated text
    3. CIFAKE images - Deepfake/manipulated images
    """
    
    print("=" * 70)
    print("PREPARING FUSION TRAINING DATASET")
    print("=" * 70)
    
    all_samples = []
    
    # ========================================================================
    # SOURCE 1: Out-of-Context (CLIP Training Data)
    # ========================================================================
    print("\n[1/3] Processing out-of-context samples (CLIP data)...")
    
    if os.path.exists("clip_train.csv"):
        clip_df = pd.read_csv("clip_train.csv")
        
        # Filter for mismatched pairs (label=1)
        mismatched = clip_df[clip_df['label'] == 1].copy()
        
        # Sample
        if len(mismatched) > samples_per_type:
            mismatched = mismatched.sample(n=samples_per_type, random_state=42)
        
        # Add source type
        mismatched['source_type'] = 'out_of_context'
        
        all_samples.append(mismatched[['image_path', 'text', 'label', 'source_type']])
        print(f"  ✓ Added {len(mismatched)} out-of-context samples")
        
        # Also add some REAL matched pairs for balance
        matched = clip_df[clip_df['label'] == 0].copy()
        if len(matched) > samples_per_type // 2:
            matched = matched.sample(n=samples_per_type // 2, random_state=42)
        matched['source_type'] = 'real_matched'
        all_samples.append(matched[['image_path', 'text', 'label', 'source_type']])
        print(f"  ✓ Added {len(matched)} real matched samples")
    else:
        print("  ⚠ clip_train.csv not found, skipping out-of-context samples")
    
    # ========================================================================
    # SOURCE 2: AI-Generated Text (HC3 Data)
    # ========================================================================
    print("\n[2/3] Processing AI-generated text samples (HC3 data)...")
    
    if os.path.exists("hc_train.csv"):
        hc_df = pd.read_csv("hc_train.csv")
        
        # Get AI-generated samples (label=0 in HC3)
        ai_generated = hc_df[hc_df['label'] == 0].copy()
        
        # Get human-written samples for balance
        human_written = hc_df[hc_df['label'] == 1].copy()
        
        # Sample
        if len(ai_generated) > samples_per_type:
            ai_generated = ai_generated.sample(n=samples_per_type, random_state=42)
        
        if len(human_written) > samples_per_type // 2:
            human_written = human_written.sample(n=samples_per_type // 2, random_state=42)
        
        # Need to pair with images - use random guardian images
        guardian_images = list(Path("guardian_processed").glob("*.jpg"))
        
        if guardian_images:
            # For AI-generated: pair with random images, label as FAKE (1)
            ai_generated['image_path'] = [
                str(random.choice(guardian_images)) for _ in range(len(ai_generated))
            ]
            ai_generated['label'] = 1  # FAKE
            ai_generated['source_type'] = 'ai_generated'
            
            # For human-written: pair with random images, label as REAL (0)
            human_written['image_path'] = [
                str(random.choice(guardian_images)) for _ in range(len(human_written))
            ]
            human_written['label'] = 0  # REAL
            human_written['source_type'] = 'human_text'
            
            all_samples.append(ai_generated[['image_path', 'text', 'label', 'source_type']])
            all_samples.append(human_written[['image_path', 'text', 'label', 'source_type']])
            
            print(f"  ✓ Added {len(ai_generated)} AI-generated samples")
            print(f"  ✓ Added {len(human_written)} human-written samples")
        else:
            print("  ⚠ No guardian images found, skipping HC3 samples")
    else:
        print("  ⚠ hc_train.csv not found, skipping AI-generated samples")
    
    # ========================================================================
    # SOURCE 3: Deepfakes (CIFAKE/Fake News Images)
    # ========================================================================
    print("\n[3/3] Processing deepfake/manipulated image samples...")
    
    # If you have CIFAKE data, add it here
    # For now, we'll create synthetic examples using WELFake data with images
    
    if os.path.exists("WELFake_Dataset.csv"):
        welfake_df = pd.read_csv("WELFake_Dataset.csv")
        
        # Sample fake and real news
        fake_news = welfake_df[welfake_df['label'] == 1].copy()
        real_news = welfake_df[welfake_df['label'] == 0].copy()
        
        if len(fake_news) > samples_per_type:
            fake_news = fake_news.sample(n=samples_per_type, random_state=42)
        
        if len(real_news) > samples_per_type // 2:
            real_news = real_news.sample(n=samples_per_type // 2, random_state=42)
        
        # Pair with random guardian images
        guardian_images = list(Path("guardian_processed").glob("*.jpg"))
        
        if guardian_images:
            fake_news['image_path'] = [
                str(random.choice(guardian_images)) for _ in range(len(fake_news))
            ]
            fake_news['source_type'] = 'fake_news'
            
            real_news['image_path'] = [
                str(random.choice(guardian_images)) for _ in range(len(real_news))
            ]
            real_news['source_type'] = 'real_news'
            
            all_samples.append(fake_news[['image_path', 'text', 'label', 'source_type']])
            all_samples.append(real_news[['image_path', 'text', 'label', 'source_type']])
            
            print(f"  ✓ Added {len(fake_news)} fake news samples")
            print(f"  ✓ Added {len(real_news)} real news samples")
        else:
            print("  ⚠ No guardian images found, skipping WELFake samples")
    else:
        print("  ⚠ WELFake_Dataset.csv not found, skipping fake news samples")
    
    # ========================================================================
    # COMBINE AND SHUFFLE
    # ========================================================================
    print("\n[4/4] Combining and shuffling dataset...")
    
    if not all_samples:
        print("✗ Error: No samples collected! Please ensure source datasets exist.")
        return
    
    # Combine all sources
    final_df = pd.concat(all_samples, ignore_index=True)
    
    # Shuffle
    final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save
    final_df.to_csv(output_csv, index=False)
    
    print(f"\n✓ Dataset created: {output_csv}")
    print(f"  Total samples: {len(final_df)}")
    print(f"\nLabel distribution:")
    print(f"  REAL (0): {len(final_df[final_df['label'] == 0])}")
    print(f"  FAKE (1): {len(final_df[final_df['label'] == 1])}")
    print(f"\nSource type distribution:")
    print(final_df['source_type'].value_counts().to_string())
    
    print("\n" + "=" * 70)
    print("DATASET PREPARATION COMPLETE!")
    print("=" * 70)
    
    # Show sample
    print("\nSample rows:")
    print(final_df.head(3).to_string())
    
    return final_df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare Fusion Training Dataset")
    parser.add_argument("--output", type=str, default="Final_Fusion_Train.csv", help="Output CSV file")
    parser.add_argument("--samples", type=int, default=500, help="Samples per misinformation type")
    args = parser.parse_args()
    
    prepare_fusion_dataset(
        output_csv=args.output,
        samples_per_type=args.samples
    )
