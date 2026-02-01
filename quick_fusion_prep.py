import pandas as pd

# Load CLIP training data
clip_df = pd.read_csv('clip_train.csv').head(800)
clip_df['source'] = 'clip'

# Save as fusion training dataset
clip_df.to_csv('Final_Fusion_Train.csv', index=False)

print(f'✓ Created {len(clip_df)} samples')
print(f'  Label distribution: {clip_df["label"].value_counts().to_dict()}')
print(f'  Saved to: Final_Fusion_Train.csv')
