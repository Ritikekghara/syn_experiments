import random
from torch.utils.data import Sampler


class ThresholdBasedBatchSampler(Sampler):
    """
    Batch sampler that filters synthetic images by JSD threshold and maintains class ratio for real images.
    
    Behavior:
    - Load ONLY real image batches with maintained class ratio (benign:malignant = 120:40 = 3:1)
    - Load ONLY eligible synthetic image batches
    - No mixing of real and synthetic in same batch
    - If threshold is None, load only real images
    """
    
    def __init__(self, real_indices, synthetic_indices, synthetic_jsd_scores,
                 batch_size, threshold=None, num_benign=120):
        """
        Args:
            real_indices: List of indices for real images
            synthetic_indices: List of indices for synthetic images
            synthetic_jsd_scores: List of JSD scores for each synthetic image
            batch_size: Batch size
            threshold: JSD threshold (None = no synthetic, use only real)
            num_benign: Number of benign real images (first indices are benign, rest are malignant)
        """
        self.real_indices = real_indices
        self.synthetic_indices = synthetic_indices
        self.synthetic_jsd_scores = synthetic_jsd_scores
        self.batch_size = batch_size
        self.threshold = threshold
        self.num_benign = num_benign
        
        # Split real indices by class
        # Assumption: first num_benign indices are benign, rest are malignant
        self.benign_indices = real_indices[:num_benign]
        self.malignant_indices = real_indices[num_benign:]
        
        # Calculate ratio: benign:malignant = 120:40 = 3:1
        self.benign_ratio = num_benign / len(real_indices)  # 120/160 = 0.75
        self.malignant_ratio = (len(real_indices) - num_benign) / len(real_indices)  # 40/160 = 0.25
        
        # Calculate per-batch distribution
        self.benign_per_batch = int(batch_size * self.benign_ratio)
        self.malignant_per_batch = batch_size - self.benign_per_batch
        
        # Filter synthetic images by threshold
        if threshold is None:
            self.eligible_synth_indices = []
        else:
            self.eligible_synth_indices = [
                idx for idx, score in zip(synthetic_indices, synthetic_jsd_scores)
                if score <= threshold
            ]
        
        print(f"Threshold: {threshold} | Real: {len(real_indices)} "
              f"(Benign: {len(self.benign_indices)}, Malignant: {len(self.malignant_indices)}) | "
              f"Synthetic (total): {len(synthetic_indices)} | "
              f"Eligible Synthetic: {len(self.eligible_synth_indices)}")
        print(f"Real batch ratio: {self.benign_per_batch} benign + {self.malignant_per_batch} malignant = {self.batch_size}")
    
    def __iter__(self):
        """
        Yields batches with FIXED number based on real samples only.
        
        Number of batches = len(real_indices) / batch_size (ALWAYS based on real samples)
        
        1. If NO eligible synthetic (threshold=None):
           - Batch: 12 benign + 4 malignant = 16
        
        2. If eligible synthetic EXISTS:
           - Distribute synthetic evenly across ALL batches
           - Fill remaining slots with real maintaining 3:1 ratio
           
        Example 1 (threshold=None, 160 real, batch_size=16):
        - Batches: 160 / 16 = 10
        - Each: 12 benign + 4 malignant
        
        Example 2 (4 eligible synthetic, 160 real, batch_size=16):
        - Batches: 160 / 16 = 10 (SAME)
        - Synthetic per batch: 4 / 10 = 0.4 → distributed as [1,1,1,1,0,0,0,0,0,0]
        - Batch with 1 synth: 1 synth + 15 real (11 benign + 4 malignant)
        - Batch with 0 synth: 0 synth + 16 real (12 benign + 4 malignant)
        
        Example 3 (12 eligible synthetic, 160 real, batch_size=16):
        - Batches: 160 / 16 = 10
        - Synthetic per batch: 12 / 10 = 1.2 → distributed as [2,2,1,1,1,1,1,1,1,1]
        - Batch with 2 synth: 2 synth + 14 real (10 benign + 4 malignant)
        - Batch with 1 synth: 1 synth + 15 real (11 benign + 4 malignant)
        """
        # Shuffle all pools
        benign_shuffled = self.benign_indices.copy()
        malignant_shuffled = self.malignant_indices.copy()
        random.shuffle(benign_shuffled)
        random.shuffle(malignant_shuffled)
        
        benign_idx = 0
        malignant_idx = 0
        
        # Number of batches based ONLY on real samples
        num_batches = len(self.real_indices) // self.batch_size
        
        # If no eligible synthetic, use pure real batches
        if len(self.eligible_synth_indices) == 0:
            for batch_num in range(num_batches):
                batch = []
                
                # Add benign samples (12 for batch_size=16)
                for _ in range(self.benign_per_batch):
                    if benign_idx < len(benign_shuffled):
                        batch.append(benign_shuffled[benign_idx])
                        benign_idx += 1
                
                # Add malignant samples (4 for batch_size=16)
                for _ in range(self.malignant_per_batch):
                    if malignant_idx < len(malignant_shuffled):
                        batch.append(malignant_shuffled[malignant_idx])
                        malignant_idx += 1
                
                if len(batch) == self.batch_size:
                    yield batch
            return
        
        # ===== WITH ELIGIBLE SYNTHETIC: Distribute across fixed batches =====
        synth_shuffled = self.eligible_synth_indices.copy()
        random.shuffle(synth_shuffled)
        synth_idx = 0
        
        # Distribute synthetic evenly across num_batches
        for batch_num in range(num_batches):
            batch = []
            
            # Calculate how many synthetic for this batch (even distribution)
            synth_start = (batch_num * len(synth_shuffled)) // num_batches
            synth_end = ((batch_num + 1) * len(synth_shuffled)) // num_batches
            synth_count = synth_end - synth_start
            
            # Add synthetic samples
            for _ in range(synth_count):
                if synth_idx < len(synth_shuffled):
                    batch.append(synth_shuffled[synth_idx])
                    synth_idx += 1
            
            # Calculate real samples needed to fill the batch
            real_needed = self.batch_size - len(batch)
            
            # Maintain 3:1 benign:malignant ratio for real portion
            benign_needed = int(real_needed * self.benign_ratio)
            malignant_needed = real_needed - benign_needed
            
            # Add benign samples
            for _ in range(benign_needed):
                if benign_idx < len(benign_shuffled):
                    batch.append(benign_shuffled[benign_idx])
                    benign_idx += 1
            
            # Add malignant samples
            for _ in range(malignant_needed):
                if malignant_idx < len(malignant_shuffled):
                    batch.append(malignant_shuffled[malignant_idx])
                    malignant_idx += 1
            
            if len(batch) == self.batch_size:
                yield batch
    
    def __len__(self):
        """Returns number of batches (ALWAYS based on real samples only)"""
        return len(self.real_indices) // self.batch_size
