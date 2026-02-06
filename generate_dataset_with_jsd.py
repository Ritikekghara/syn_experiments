import torch
import torch.nn.functional as F
import pandas as pd
from torchvision import transforms
from PIL import Image
from torchvision import models
import torch.nn as nn
import os
from tqdm import tqdm
import glob


def compute_jsd_from_logits(logits_stack):
    """
    Computes the Jensen-Shannon Divergence (JSD) for a stack of predictions.
    
    Args:
        logits_stack (Tensor): Shape [K, Batch_Size, Num_Classes] 
                               where K is the number of MC Dropout passes.
    
    Returns:
        jsd_score (Tensor): Shape [Batch_Size]. The epistemic uncertainty score.
    """
    # 1. Convert logits to probabilities
    # Shape: [K, Batch_Size, Classes]
    probs = F.softmax(logits_stack, dim=-1)
    
    # 2. Compute the Mean Distribution (The "Consensus")
    # Shape: [Batch_Size, Classes]
    mean_dist = torch.mean(probs, dim=0)
    
    # 3. Compute Entropy of the Mean (Total Uncertainty)
    # H(Mean) = - sum(Mean * log(Mean))
    eps = 1e-8  # For numerical stability
    entropy_of_mean = -torch.sum(mean_dist * torch.log(mean_dist + eps), dim=-1)
    
    # 4. Compute Entropy of Each Individual Pass (Aleatoric Uncertainty)
    # H(P_i) = - sum(P_i * log(P_i))
    # Shape: [K, Batch_Size]
    individual_entropies = -torch.sum(probs * torch.log(probs + eps), dim=-1)
    
    # 5. Compute Mean of the Individual Entropies
    # Shape: [Batch_Size]
    mean_of_entropies = torch.mean(individual_entropies, dim=0)
    
    # 6. JSD = H(Mean) - Mean(H) (Epistemic Uncertainty)
    jsd_score = entropy_of_mean - mean_of_entropies
    
    # Clamp negative values (floating point errors) to 0
    return torch.clamp(jsd_score, min=0.0)


def get_mc_dropout_scores(model, image_tensor, k_passes=10):
    """
    Runs the model K times with Dropout enabled to estimate JSD.
    
    Args:
        model: The neural network model
        image_tensor: Input image tensor
        k_passes: Number of forward passes with dropout
    
    Returns:
        jsd_score: Jensen-Shannon Divergence score
        predictions_array: Array of all predictions from k_passes
    """
    # Enable Dropout only, freeze BatchNorm
    def enable_dropout_only(m):
        if type(m) == torch.nn.Dropout:
            m.train()
    
    model.eval()  # Freeze BN
    model.apply(enable_dropout_only)  # Unfreeze Dropout
    
    # Run K forward passes
    outputs = []
    predictions = []
    
    with torch.no_grad():
        for _ in range(k_passes):
            # output shape: [Batch, Classes]
            logits = model(image_tensor) 
            outputs.append(logits)
            
            # Get prediction (class with highest probability)
            probs = F.softmax(logits, dim=-1)
            predictions.append(probs.cpu().numpy())
            
    # Stack outputs: [K, Batch, Classes]
    logits_stack = torch.stack(outputs)
    
    # Compute JSD
    jsd = compute_jsd_from_logits(logits_stack)
    
    return jsd, predictions


def collect_real_images(base_path='data'):
    """
    Collect all real images from TRAIN directory only.
    
    Returns:
        List of dictionaries with image info
    """
    real_images = []
    
    # Only collect from train split (for retraining)
    splits = ['train']
    classes = {'BNV': 0, 'MEL': 1}
    
    for split in splits:
        for class_name, label in classes.items():
            class_dir = os.path.join(base_path, split, class_name)
            
            if not os.path.exists(class_dir):
                print(f"Warning: Directory not found: {class_dir}")
                continue
            
            # Get all images in this directory
            image_files = glob.glob(os.path.join(class_dir, '*.*'))
            image_files = [f for f in image_files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
            
            for img_path in image_files:
                real_images.append({
                    'filename': img_path,
                    'label': label,
                    'is_syn': 0,
                    'jsd_score': 0.0  # Real images are trusted, no uncertainty
                })
    
    return real_images


def collect_synthetic_images(syn_path='data/train_synthetic/malignant'):
    """
    Collect all synthetic images.
    
    Returns:
        List of image paths
    """
    if not os.path.exists(syn_path):
        print(f"Warning: Synthetic directory not found: {syn_path}")
        return []
    
    image_files = glob.glob(os.path.join(syn_path, '*.*'))
    image_files = [f for f in image_files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    
    return image_files


def main():
    # Configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Model checkpoint path - UPDATE THIS PATH
    model_path = 'outputs/resnet50.pth'  # Your baseline model trained on real data
    
    # Paths
    real_data_path = 'data'  # Base path for real images
    synthetic_data_path = 'data/train_synthetic/MEL'  # Path for synthetic images
    output_csv = 'dataset.csv'  # Output CSV file
    
    # Number of MC Dropout passes
    k_passes = 10
    
    print("="*60)
    print("JSD Score Generation for Real and Synthetic Images")
    print("="*60)
    
    # ==========================================
    # 1. Load the Baseline Model
    # ==========================================
    print("\n[1/4] Loading baseline model...")
    model = models.resnet50(pretrained=False)
    # Add Dropout layer before final classification
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.fc.in_features, 2)
    )
    
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        
        # Handle checkpoint format
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Load weights, handling the fc layer mismatch
        # Old checkpoint: fc.weight, fc.bias
        # New model: fc.0 (Dropout), fc.1.weight, fc.1.bias (Linear)
        if 'fc.weight' in state_dict:
            # Manually map old fc weights to new fc[1] (Linear layer)
            state_dict['fc.1.weight'] = state_dict.pop('fc.weight')
            state_dict['fc.1.bias'] = state_dict.pop('fc.bias')
        
        model.load_state_dict(state_dict)
        print(f"✓ Model loaded from: {model_path}")
    else:
        print(f"⚠ Warning: Model checkpoint not found at {model_path}")
        print("Using randomly initialized model (for testing purposes)")
    
    model.to(device)
    model.eval()
    
    # ==========================================
    # 2. Setup Image Transforms
    # ==========================================
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # ==========================================
    # 3. Collect Real Images
    # ==========================================
    print("\n[2/4] Collecting real images...")
    real_images_data = collect_real_images(base_path=real_data_path)
    print(f"✓ Found {len(real_images_data)} real images")
    
    # ==========================================
    # 4. Process Synthetic Images with MC Dropout
    # ==========================================
    print("\n[3/4] Processing synthetic images with MC Dropout...")
    synthetic_images_paths = collect_synthetic_images(syn_path=synthetic_data_path)
    print(f"✓ Found {len(synthetic_images_paths)} synthetic images")
    
    synthetic_images_data = []
    logits_data = []  # Store logits for each synthetic image
    
    if len(synthetic_images_paths) > 0:
        print(f"\nRunning {k_passes} forward passes per image with Dropout active...")
        
        for img_path in tqdm(synthetic_images_paths, desc="Processing synthetic images"):
            try:
                # Load and transform image
                img = Image.open(img_path).convert('RGB')
                img_tensor = transform(img).unsqueeze(0).to(device)  # [1, 3, 224, 224]
                
                # Calculate JSD with MC Dropout and get all logits
                def get_mc_dropout_with_logits(model, image_tensor, k_passes=10):
                    def enable_dropout_only(m):
                        if type(m) == torch.nn.Dropout:
                            m.train()
                    
                    model.eval()
                    model.apply(enable_dropout_only)
                    
                    outputs = []
                    with torch.no_grad():
                        for _ in range(k_passes):
                            logits = model(image_tensor)
                            outputs.append(logits)
                    
                    logits_stack = torch.stack(outputs)
                    jsd = compute_jsd_from_logits(logits_stack)
                    
                    return jsd, logits_stack
                
                jsd_score, logits_stack = get_mc_dropout_with_logits(
                    model, 
                    img_tensor, 
                    k_passes=k_passes
                )
                
                # Store results
                synthetic_images_data.append({
                    'filename': img_path,
                    'label': 1,  # Synthetic images are malignant
                    'is_syn': 1,
                    'jsd_score': jsd_score.item()
                })
                
                # Convert logits to probabilities (softmax): [K, 1, 2] -> [K, 2]
                probs = F.softmax(logits_stack, dim=-1)
                probs_list = probs.squeeze(1).cpu().numpy().tolist()
                logits_data.append({
                    'filename': img_path,
                    'probabilities': probs_list  # List of 10 pairs of probabilities
                })
                
            except Exception as e:
                print(f"\n⚠ Error processing {img_path}: {e}")
                continue
        
        print(f"\n✓ Processed {len(synthetic_images_data)} synthetic images")
        
        # Save probabilities to CSV in expanded format (each pass as separate columns)
        expanded_logits_data = []
        for logit_entry in logits_data:
            row = {'filename': logit_entry['filename']}
            probs = logit_entry['probabilities']
            # Create columns: benign_pass1, malignant_pass1, benign_pass2, malignant_pass2, ...
            for pass_num, prob_pair in enumerate(probs, 1):
                row[f'benign_pass{pass_num}'] = prob_pair[0]
                row[f'malignant_pass{pass_num}'] = prob_pair[1]
            expanded_logits_data.append(row)
        
        logits_df = pd.DataFrame(expanded_logits_data)
        logits_csv = 'synthetic_probabilities.csv'
        logits_df.to_csv(logits_csv, index=False)
        print(f"✓ Saved probabilities to: {logits_csv}")
        
        # Print statistics
        jsd_scores = [item['jsd_score'] for item in synthetic_images_data]
        if jsd_scores:
            print(f"\nJSD Score Statistics:")
            print(f"  Min:  {min(jsd_scores):.6f}")
            print(f"  Max:  {max(jsd_scores):.6f}")
            print(f"  Mean: {sum(jsd_scores)/len(jsd_scores):.6f}")
    
    # ==========================================
    # 5. Create Master CSV
    # ==========================================
    print("\n[4/4] Creating master CSV...")
    
    # Combine real and synthetic data
    all_data = real_images_data + synthetic_images_data
    
    # Create DataFrame
    df = pd.DataFrame(all_data)
    
    # Save to CSV
    df.to_csv(output_csv, index=False)
    
    print(f"\n✓ Saved dataset to: {output_csv}")
    print(f"\nDataset Summary:")
    print(f"  Total images:     {len(df)}")
    print(f"  Real images:      {len(df[df['is_syn'] == 0])}")
    print(f"  Synthetic images: {len(df[df['is_syn'] == 1])}")
    print(f"  BNV (0):       {len(df[df['label'] == 0])}")
    print(f"  MEL (1):    {len(df[df['label'] == 1])}")
    
    # Show sample rows
    print("\n" + "="*60)
    print("Sample rows from dataset.csv:")
    print("="*60)
    print(df.head(10).to_string(index=False))
    
    if len(synthetic_images_data) > 0:
        print("\n" + "="*60)
        print("Sample synthetic images with JSD scores:")
        print("="*60)
        syn_df = df[df['is_syn'] == 1].sort_values('jsd_score')
        print("\nMost Stable (Low JSD - High Quality):")
        print(syn_df.head(5)[['filename', 'jsd_score']].to_string(index=False))
        print("\nMost Unstable (High JSD - Low Quality):")
        print(syn_df.tail(5)[['filename', 'jsd_score']].to_string(index=False))
        
        # Display probabilities for first 5 synthetic images
        print("\n" + "="*60)
        print("Probabilities from 10 MC Dropout passes (first 5 synthetic images):")
        print("="*60)
        for i, logit_entry in enumerate(logits_data[:5]):
            print(f"\n[{i+1}] {logit_entry['filename']}")
            probs = logit_entry['probabilities']
            for pass_num, prob_pair in enumerate(probs):
                print(f"  Pass {pass_num+1:2d}: probs=[BEL={prob_pair[0]:.4f}, MEL={prob_pair[1]:.4f}]")
    
    print("\n" + "="*60)
    print("✓ COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()
