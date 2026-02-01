"""
Fast Guardian Data Preparation Pipeline
Prepares samples for CLIP training and Vector DB seeding
"""

import os
import json
import random
import csv
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple, Optional
import requests
from bs4 import BeautifulSoup
from PIL import Image
from io import BytesIO
from tqdm import tqdm


class GuardianPreprocessor:
    def __init__(self, source_dir: str, output_dir: str = "./guardian_processed/"):
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Storage for extracted data
        self.articles_data = []
        
    def get_random_json_files(self, n_samples: int = 15000) -> List[Path]:
        """Randomly select n_samples JSON files from the Guardian directory."""
        all_files = list(self.source_dir.glob("*.json"))
        
        if len(all_files) < n_samples:
            print(f"Warning: Only {len(all_files)} files found, using all of them")
            return all_files
        
        selected_files = random.sample(all_files, n_samples)
        print(f"Selected {len(selected_files)} random files from {len(all_files)} total files")
        return selected_files
    
    def extract_metadata_from_json(self, json_file: Path) -> Optional[Dict]:
        """Extract og:title and og:image from HTML file (with .json extension)."""
        try:
            
            # Read as plain text/HTML, not JSON
            with open(json_file, 'r', encoding='utf-8', errors='ignore') as f:
                html_content = f.read()
            
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Try multiple strategies to find title
            title = None
            
            # Strategy 1: og:title
            og_title = soup.find('meta', property='og:title')
            if og_title and og_title.get('content'):
                title = og_title['content'].strip()
            
            # Strategy 2: twitter:title
            if not title:
                twitter_title = soup.find('meta', attrs={'name': 'twitter:title'})
                if twitter_title and twitter_title.get('content'):
                    title = twitter_title['content'].strip()
            
            # Strategy 3: og:site_name
            if not title:
                og_site = soup.find('meta', property='og:site_name')
                if og_site and og_site.get('content'):
                    title = og_site['content'].strip()
            
            # Strategy 4: title tag
            if not title:
                title_tag = soup.find('title')
                if title_tag and title_tag.text:
                    title = title_tag.text.strip()
            
            # Strategy 5: h1 tag
            if not title:
                h1_tag = soup.find('h1')
                if h1_tag and h1_tag.text:
                    title = h1_tag.text.strip()
            
            # Try multiple strategies to find image
            image_url = None
            
            # Strategy 1: og:image
            og_image = soup.find('meta', property='og:image')
            if og_image and og_image.get('content'):
                image_url = og_image['content']
            
            # Strategy 2: twitter:image
            if not image_url:
                twitter_img = soup.find('meta', attrs={'name': 'twitter:image'})
                if twitter_img and twitter_img.get('content'):
                    image_url = twitter_img['content']
            
            # Strategy 3: link rel="image_src"
            if not image_url:
                img_link = soup.find('link', rel='image_src')
                if img_link and img_link.get('href'):
                    image_url = img_link['href']
            
            # Strategy 4: First valid img tag with reasonable size
            if not image_url:
                for img_tag in soup.find_all('img'):
                    src = img_tag.get('src') or img_tag.get('data-src')
                    if src and (src.startswith('http://') or src.startswith('https://')):
                        # Try to avoid tiny images (icons, logos)
                        width = img_tag.get('width')
                        height = img_tag.get('height')
                        if width and height:
                            try:
                                if int(width) >= 200 and int(height) >= 200:
                                    image_url = src
                                    break
                            except:
                                pass
                        else:
                            # If no size info, use first image
                            image_url = src
                            break
            
            # Validate we have both title and image
            if title and image_url and len(title) > 10:
                # Ensure image URL is absolute
                if not image_url.startswith('http'):
                    return None
                
                return {
                    'article_id': json_file.stem,
                    'title': title[:500],  # Limit title length
                    'image_url': image_url,
                    'source_file': str(json_file)
                }
            
            return None
            
        except Exception as e:
            # Silent fail for corrupt files
            return None
    
    def download_and_process_image(self, article_data: Dict) -> Optional[Dict]:
        """Download image, resize to 224x224, and save locally."""
        try:
            article_id = article_data['article_id']
            image_url = article_data['image_url']
            
            # Download image with retries
            max_retries = 2
            for attempt in range(max_retries):
                try:
                    response = requests.get(image_url, timeout=15, headers={
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                    }, allow_redirects=True)
                    
                    if response.status_code == 200:
                        break
                except:
                    if attempt == max_retries - 1:
                        return None
                    continue
            
            if response.status_code != 200:
                return None
            
            # Open and resize image
            img = Image.open(BytesIO(response.content))
            img = img.convert('RGB')  # Ensure RGB format
            img = img.resize((224, 224), Image.LANCZOS)
            
            # Save image
            image_path = self.output_dir / f"{article_id}.jpg"
            img.save(image_path, 'JPEG', quality=95)
            
            # Update article data with local path
            article_data['image_local_path'] = str(image_path)
            return article_data
            
        except Exception as e:
            return None
    
    def process_articles_in_batches(self, target_articles: int = 3000, batch_size: int = 10000, max_workers: int = 15) -> List[Dict]:
        """Process articles in batches until we reach target."""
        all_files = list(self.source_dir.glob("*.json"))
        random.shuffle(all_files)
        
        print(f"\nTotal files available: {len(all_files)}")
        print(f"Target: {target_articles} successful articles")
        
        processed_articles = []
        files_processed = 0
        
        while len(processed_articles) < target_articles and files_processed < len(all_files):
            # Get next batch
            batch_end = min(files_processed + batch_size, len(all_files))
            batch_files = all_files[files_processed:batch_end]
            
            print(f"\n[Batch] Processing files {files_processed} to {batch_end}")
            print(f"        Current progress: {len(processed_articles)}/{target_articles} articles")
            
            # Extract metadata from batch
            metadata_list = []
            for json_file in tqdm(batch_files, desc="Extracting metadata"):
                metadata = self.extract_metadata_from_json(json_file)
                if metadata:
                    metadata_list.append(metadata)
            
            print(f"        Found {len(metadata_list)} articles with metadata in this batch")
            
            # Download and process images
            if metadata_list:
                print(f"        Downloading images...")
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = [executor.submit(self.download_and_process_image, article) for article in metadata_list]
                    
                    for future in tqdm(as_completed(futures), total=len(futures), desc="Processing images"):
                        result = future.result()
                        if result:
                            processed_articles.append(result)
                            
                            # Check if we've reached target
                            if len(processed_articles) >= target_articles:
                                print(f"\n✓ Reached target of {target_articles} images!")
                                break
            
            files_processed = batch_end
            
            # Stop if we've reached target
            if len(processed_articles) >= target_articles:
                break
        
        print(f"\nFinal: Successfully processed {len(processed_articles)} images")
        self.articles_data = processed_articles
        return processed_articles
    
    def create_clip_training_csv(self, train_file: str = "clip_train.csv", val_file: str = "clip_val.csv"):
        """Create CSV files with matched and mismatched pairs for CLIP training and validation."""
        print("\n[3/3] Creating CLIP training and validation CSVs with 80/20 split...")
        
        if len(self.articles_data) < 10:
            print("Error: Not enough articles to create train/val split")
            return
        
        # 80/20 split - shuffle first to ensure randomness
        shuffled_articles = self.articles_data.copy()
        random.shuffle(shuffled_articles)
        
        split_idx = int(len(shuffled_articles) * 0.8)
        train_articles = shuffled_articles[:split_idx]
        val_articles = shuffled_articles[split_idx:]
        
        print(f"Split: {len(train_articles)} train articles, {len(val_articles)} val articles")
        print(f"No data leakage: Train and val sets are completely separate")
        
        # Create training pairs
        print(f"\nCreating training pairs...")
        train_pairs = []
        
        # 1 matched pair per image
        for article in train_articles:
            train_pairs.append({
                'image_path': article['image_local_path'],
                'text': article['title'],
                'label': 0,  # Matched
                'pair_type': 'matched'
            })
        
        # 1 mismatched pair per image (only using train articles)
        for i, article in enumerate(train_articles):
            other_idx = random.choice([j for j in range(len(train_articles)) if j != i])
            other_article = train_articles[other_idx]
            
            train_pairs.append({
                'image_path': article['image_local_path'],
                'text': other_article['title'],  # Different title from train set
                'label': 1,  # Mismatched
                'pair_type': 'mismatched'
            })
        
        # Shuffle and write training CSV
        random.shuffle(train_pairs)
        with open(train_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['image_path', 'text', 'label', 'pair_type'])
            writer.writeheader()
            writer.writerows(train_pairs)
        
        print(f"✓ Created {train_file} with {len(train_pairs)} rows")
        print(f"  - Matched pairs: {sum(1 for p in train_pairs if p['label'] == 0)}")
        print(f"  - Mismatched pairs: {sum(1 for p in train_pairs if p['label'] == 1)}")
        
        # Create validation pairs
        print(f"\nCreating validation pairs...")
        val_pairs = []
        
        # 1 matched pair per image
        for article in val_articles:
            val_pairs.append({
                'image_path': article['image_local_path'],
                'text': article['title'],
                'label': 0,  # Matched
                'pair_type': 'matched'
            })
        
        # 1 mismatched pair per image (only using val articles)
        for i, article in enumerate(val_articles):
            other_idx = random.choice([j for j in range(len(val_articles)) if j != i])
            other_article = val_articles[other_idx]
            
            val_pairs.append({
                'image_path': article['image_local_path'],
                'text': other_article['title'],  # Different title from val set
                'label': 1,  # Mismatched
                'pair_type': 'mismatched'
            })
        
        # Shuffle and write validation CSV
        random.shuffle(val_pairs)
        with open(val_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['image_path', 'text', 'label', 'pair_type'])
            writer.writeheader()
            writer.writerows(val_pairs)
        
        print(f"✓ Created {val_file} with {len(val_pairs)} rows")
        print(f"  - Matched pairs: {sum(1 for p in val_pairs if p['label'] == 0)}")
        print(f"  - Mismatched pairs: {sum(1 for p in val_pairs if p['label'] == 1)}")
    
    def create_vector_db_json(self, output_file: str = "vector_db_seed.json"):
        """Create JSON file for Vector DB seeding."""
        print(f"\nCreating Vector DB seed file...")
        
        vector_db_data = []
        for article in self.articles_data:
            vector_db_data.append({
                'article_id': article['article_id'],
                'text_content': article['title'],
                'image_local_path': article['image_local_path']
            })
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(vector_db_data, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Created {output_file} with {len(vector_db_data)} entries")


def main():
    """Main execution pipeline."""
    print("=" * 60)
    print("Guardian Data Preparation Pipeline")
    print("=" * 60)
    
    # Configuration
    SOURCE_DIR = r"D:\ACM\data\articles\articles\guardian_articles\guardian_json"
    TARGET_ARTICLES = 3000  # Need 3000 for 3k matched + 3k mismatched pairs
    BATCH_SIZE = 10000      # Process 10k files at a time
    MAX_WORKERS = 20        # Parallel download threads
    
    # Initialize preprocessor
    preprocessor = GuardianPreprocessor(source_dir=SOURCE_DIR)
    
    # Process articles in batches until target reached
    processed_articles = preprocessor.process_articles_in_batches(
        target_articles=TARGET_ARTICLES,
        batch_size=BATCH_SIZE,
        max_workers=MAX_WORKERS
    )
    
    if not processed_articles:
        print("\n❌ No articles were successfully processed. Exiting.")
        return
    
    print(f"\n{'='*60}")
    print(f"Processing complete: {len(processed_articles)} articles ready")
    print(f"{'='*60}")
    
    # Create outputs
    preprocessor.create_clip_training_csv("clip_train.csv", "clip_val.csv")
    preprocessor.create_vector_db_json("vector_db_seed.json")
    
    # Calculate split sizes
    train_size = int(len(processed_articles) * 0.8)
    val_size = len(processed_articles) - train_size
    
    print("\n" + "=" * 60)
    print("✓ Pipeline completed successfully!")
    print(f"  - Processed images: {len(processed_articles)}")
    print(f"  - Output directory: {preprocessor.output_dir}")
    print(f"  - CLIP train CSV: clip_train.csv ({train_size * 2} rows)")
    print(f"  - CLIP val CSV: clip_val.csv ({val_size * 2} rows)")
    print(f"  - Vector DB seed: vector_db_seed.json ({len(processed_articles)} entries)")
    print("=" * 60)


if __name__ == "__main__":
    main()
