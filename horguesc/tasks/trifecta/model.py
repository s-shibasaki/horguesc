import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import itertools
import math
from horguesc.core.base.model import BaseModel

logger = logging.getLogger(__name__)

class TrifectaModel(BaseModel):
    """Model for predicting trifecta (exact order of top 3 finishers) in horse races.
    
    The model works as follows:
    1. Encodes horse features using the shared FeatureEncoder
    2. Processes each horse to get a performance representation
    3. Calculates scores for all possible trifecta combinations
    4. Outputs probabilities for each trifecta combination
    """
    
    def __init__(self, config, encoder=None):
        """Initialize the trifecta model.
        
        Args:
            config: Application configuration
            encoder: Shared feature encoder (FeatureEncoder instance)
        """
        super().__init__(config, encoder)
        
        # Get model parameters from config
        self.hidden_dim = config.getint('model.trifecta', 'hidden_dim', fallback=128)
        self.dropout_rate = config.getfloat('model.trifecta', 'dropout_rate', fallback=0.2)
        
        # Input dimension is the output dimension of the encoder
        self.input_dim = self.encoder.output_dim if self.encoder else 128
        
        # Horse performance network (processes encoded features to get horse performance representation)
        self.horse_network = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            # Output a vector representation of horse performance
            nn.Linear(self.hidden_dim // 2, self.hidden_dim // 4)
        )
        
        # Race context encoder (optional - captures race-wide information)
        self.context_encoder = nn.Sequential(
            nn.Linear(self.hidden_dim // 4, self.hidden_dim // 8),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 8, self.hidden_dim // 4)
        )
        
        # Combination scoring layer (scores trifecta combinations)
        # Input: concatenated horse performance vectors for 3 horses (3 * (hidden_dim // 4))
        self.combination_scorer = nn.Sequential(
            nn.Linear(3 * (self.hidden_dim // 4), self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dim // 2, 1)
        )
        
        # Loss function
        self.loss_fn = nn.CrossEntropyLoss()
        
        logger.info(f"TrifectaModel initialized: input_dim={self.input_dim}, hidden_dim={self.hidden_dim}")
    
    def forward(self, inputs):
        """Forward pass for the trifecta model.
        
        Args:
            inputs: Dictionary containing input features
                - Each feature is a tensor of shape [batch_size, max_horses]
                  or [batch_size, max_horses, feature_dim] if already embedded
                - Should include 'horse_count' tensor [batch_size] with the actual number of horses in each race
        
        Returns:
            dict: Contains:
                - 'logits': Raw scores for each trifecta combination [batch_size, num_combinations]
                - 'sanrentan_probabilities': Probabilities for each trifecta combination [batch_size, num_combinations]
                - 'horse_performances': Performance vectors for each horse [batch_size, max_horses, perf_dim]
                - Additional bet type probabilities from compute_bet_probabilities (only in eval/inference mode)
        """
        batch_size = next(iter(inputs.values())).shape[0]
        max_horses = next(iter(inputs.values())).shape[1]
        
        # Encode all horse features at once using the shared encoder
        # Shape of encoded_features: [batch_size, max_horses, encoder_output_dim]
        encoded_features = self.encoder(inputs)
        
        # Process each horse to get performance representations
        # Reshape to process all horses from all batches at once
        reshaped_features = encoded_features.view(-1, self.input_dim)
        all_horse_performances = self.horse_network(reshaped_features)
        
        # Reshape back to [batch_size, max_horses, performance_dim]
        performance_dim = all_horse_performances.shape[-1]
        horse_performances = all_horse_performances.view(batch_size, max_horses, performance_dim)
        
        # Optional: Generate race context vector and enhance horse performances
        # Average pool over all horses to get race-level context
        race_contexts = torch.mean(horse_performances, dim=1, keepdim=True)  # [batch_size, 1, perf_dim]
        enhanced_context = self.context_encoder(race_contexts)  # [batch_size, 1, perf_dim]
        
        # Add race context to each horse's performance (broadcasting)
        horse_performances = horse_performances + enhanced_context
        
        # Get actual horse counts for each race in the batch
        horse_counts = inputs.get('horse_count')
        if horse_counts is None:
            logger.warning("No 'horse_count' found in inputs, assuming all horses are valid")
            horse_counts = torch.full((batch_size,), max_horses, device=horse_performances.device)
        
        # Generate all possible trifecta combinations (top 3 horses in order)
        # For efficiency, we'll calculate scores for all possible 3-horse combinations
        trifecta_scores = self._compute_trifecta_scores(horse_performances, max_horses, horse_counts)
        
        # Apply softmax to get probabilities
        trifecta_probs = F.softmax(trifecta_scores, dim=1)
        
        # Create result dictionary with essential outputs
        results = {
            'logits': trifecta_scores,
            'sanrentan_probabilities': trifecta_probs,
            'horse_performances': horse_performances
        }
        
        # Only compute additional bet probabilities in eval/inference mode
        # This avoids unnecessary computation during training
        if not self.training:
            # Get wakuban (horse numbers) from inputs if available
            wakuban = inputs.get('raw_wakuban', None)
            
            # Compute probabilities for different bet types
            bet_probabilities = self.compute_bet_probabilities(trifecta_probs, max_horses, wakuban)
            
            # Add all bet probabilities to the results
            results.update(bet_probabilities)
            
            logger.debug("Computed additional bet probabilities (eval/inference mode)")
        else:
            logger.debug("Skipped additional bet probabilities calculation (training mode)")
        
        return results
    
    def compute_bet_probabilities(self, trifecta_probs, max_horses, wakuban=None):
        """Compute probabilities for different bet types using trifecta probabilities.
        
        Args:
            trifecta_probs: Trifecta probabilities [batch_size, num_combinations]
            max_horses: Maximum number of horses per race
            wakuban: Optional tensor of horse numbers [batch_size, max_horses]
                    If not provided, assumes horses are numbered 1 to max_horses
        
        Returns:
            dict: Dictionary containing probabilities for different bet types
        """
        batch_size = trifecta_probs.shape[0]
        device = trifecta_probs.device
        
        # Create the mapping from sanrentan to sanrenpuku
        sanrentan_to_sanrenpuku = self._create_sanrentan_to_sanrenpuku_mapping(
            max_horses, batch_size, device
        )
        
        # Calculate sanrenpuku (trio) probabilities
        # Shape: [batch_size, n_sanrenpuku]
        sanrenpuku_probs = torch.bmm(sanrentan_to_sanrenpuku, trifecta_probs.unsqueeze(2)).squeeze(2)
        
        # Create the mapping from sanrentan to umatan (exacta)
        sanrentan_to_umatan = self._create_sanrentan_to_umatan_mapping(
            max_horses, batch_size, device
        )
        
        # Calculate umatan (exacta) probabilities
        # Shape: [batch_size, n_umatan]
        umatan_probs = torch.bmm(sanrentan_to_umatan, trifecta_probs.unsqueeze(2)).squeeze(2)
        
        # Create the mapping from sanrentan to umaren (quinella)
        sanrentan_to_umaren = self._create_sanrentan_to_umaren_mapping(
            max_horses, batch_size, device
        )
        
        # Calculate umaren (quinella) probabilities
        # Shape: [batch_size, n_umaren]
        umaren_probs = torch.bmm(sanrentan_to_umaren, trifecta_probs.unsqueeze(2)).squeeze(2)
        
        # Create the mapping from sanrentan to tansho (win)
        sanrentan_to_tansho = self._create_sanrentan_to_tansho_mapping(
            max_horses, batch_size, device
        )
        
        # Calculate tansho (win) probabilities
        # Shape: [batch_size, max_horses]
        tansho_probs = torch.bmm(sanrentan_to_tansho, trifecta_probs.unsqueeze(2)).squeeze(2)
        
        # Create the mapping from sanrentan to fukusho (place)
        sanrentan_to_fukusho = self._create_sanrentan_to_fukusho_mapping(
            max_horses, batch_size, device
        )
        
        # Calculate fukusho (place) probabilities
        # Shape: [batch_size, max_horses]
        fukusho_probs = torch.bmm(sanrentan_to_fukusho, trifecta_probs.unsqueeze(2)).squeeze(2)
        
        # Create the mapping from sanrentan to wide (rumor)
        sanrentan_to_wide = self._create_sanrentan_to_wide_mapping(
            max_horses, batch_size, device
        )
        
        # Calculate wide (rumor) probabilities
        # Shape: [batch_size, n_wide]
        wide_probs = torch.bmm(sanrentan_to_wide, trifecta_probs.unsqueeze(2)).squeeze(2)
        
        # Create the mapping from sanrentan to wakuren (box quinella)
        sanrentan_to_wakuren = self._create_sanrentan_to_wakuren_mapping(
            max_horses, batch_size, device, wakuban
        )
        
        # Calculate wakuren (box quinella) probabilities if we have post position info
        if sanrentan_to_wakuren is not None:
            # Shape: [batch_size, n_wakuren]
            wakuren_probs = torch.bmm(sanrentan_to_wakuren, trifecta_probs.unsqueeze(2)).squeeze(2)
        
        # Return bet probabilities
        bet_probabilities = {
            'sanrenpuku_probabilities': sanrenpuku_probs,
            'umatan_probabilities': umatan_probs,
            'umaren_probabilities': umaren_probs,
            'tansho_probabilities': tansho_probs,
            'fukusho_probabilities': fukusho_probs,
            'wide_probabilities': wide_probs
        }
        
        # Add wakuren probabilities if we calculated them
        if sanrentan_to_wakuren is not None:
            bet_probabilities['wakuren_probabilities'] = wakuren_probs
        
        return bet_probabilities
    
    def _create_sanrentan_to_fukusho_mapping(self, max_horses, batch_size, device):
        """Create mapping from sanrentan (ordered trifecta) to fukusho (place).
        
        This mapping is used to convert trifecta probabilities to place bet probabilities.
        A fukusho (place) bet wins if the horse finishes in one of the top positions
        (typically top 3 for races with 8+ horses).
        
        Args:
            max_horses: Maximum number of horses per race
            batch_size: Number of races in the batch
            device: Device to use for computation
            
        Returns:
            torch.Tensor: Mapping matrix [batch_size, max_horses, n_sanrentan]
                          Where mapping[b, i, j] = 1 if sanrentan j has horse i+1 in top positions
        """
        # Generate all possible trifecta combinations (1-indexed horse numbers)
        sanrentan_combinations = list(itertools.permutations(range(1, max_horses + 1), 3))
        # Convert to tensor: [n_sanrentan, 3]
        sanrentan_tensor = torch.tensor(sanrentan_combinations, device=device)
        
        # Extract horses in top 3 positions from sanrentan combinations
        # Shape: [n_sanrentan, 3]
        top_three_horses = sanrentan_tensor
        
        # Create a mapping matrix of shape [max_horses, n_sanrentan]
        # We'll use one-hot encoding combined with comparison operations
        
        # First create a range tensor representing all horse numbers (1-indexed)
        horse_indices = torch.arange(1, max_horses + 1, device=device).unsqueeze(1)  # [max_horses, 1]
        
        # Reshape sanrentan tensor for broadcasting
        # Each row of this will be compared with horse_indices
        first_pos = top_three_horses[:, 0].unsqueeze(0)   # [1, n_sanrentan]
        second_pos = top_three_horses[:, 1].unsqueeze(0)  # [1, n_sanrentan]
        third_pos = top_three_horses[:, 2].unsqueeze(0)   # [1, n_sanrentan]
        
        # A horse is in fukusho if it's in any of the top 3 positions
        # Compare each horse number with each position
        # Shape: [max_horses, n_sanrentan]
        in_first = (horse_indices == first_pos).float()
        in_second = (horse_indices == second_pos).float()  
        in_third = (horse_indices == third_pos).float()
        
        # Combine the mappings with logical OR (implemented as sum + clamp)
        mapping = torch.clamp(in_first + in_second + in_third, 0, 1)
        
        # Expand for batch dimension
        # Shape: [batch_size, max_horses, n_sanrentan]
        sanrentan_to_fukusho = mapping.unsqueeze(0).expand(batch_size, -1, -1)
        
        return sanrentan_to_fukusho
    
    def _create_sanrentan_to_umaren_mapping(self, max_horses, batch_size, device):
        """Create mapping from sanrentan (ordered trifecta) to umaren (unordered quinella).
        
        This mapping is used to convert trifecta probabilities to quinella probabilities.
        A quinella bet considers the first and second place finishers regardless of order.
        
        Args:
            max_horses: Maximum number of horses per race
            batch_size: Number of races in the batch
            device: Device to use for computation
            
        Returns:
            torch.Tensor: Mapping matrix [batch_size, n_umaren, n_sanrentan]
                          Where mapping[b, i, j] = 1 if sanrentan j projects to umaren i
        """
        # Generate all possible trifecta combinations (1-indexed horse numbers)
        sanrentan_combinations = list(itertools.permutations(range(1, max_horses + 1), 3))
        # Convert to tensor: [n_sanrentan, 3]
        sanrentan_tensor = torch.tensor(sanrentan_combinations, device=device)
        
        # Generate all possible quinella combinations (1-indexed horse numbers)
        umaren_combinations = list(itertools.combinations(range(1, max_horses + 1), 2))
        # Convert to tensor: [n_umaren, 2]
        umaren_tensor = torch.tensor(umaren_combinations, device=device)
        
        # Extract first two positions from sanrentan combinations and sort them
        # This ensures we match quinella (unordered first two) regardless of order
        # Shape: [n_sanrentan, 2]
        sanrentan_first_two = torch.sort(sanrentan_tensor[:, :2], dim=1)[0]
        
        # Reshape for broadcasting comparison
        # Shape: [n_umaren, 1, 2]
        umaren_expanded = umaren_tensor.unsqueeze(1)
        # Shape: [1, n_sanrentan, 2]
        sanrentan_first_two_expanded = sanrentan_first_two.unsqueeze(0)
        
        # Compare each umaren with each sanrentan's sorted first two positions
        # Result shape: [n_umaren, n_sanrentan, 2]
        matches = umaren_expanded == sanrentan_first_two_expanded
        
        # An umaren matches a sanrentan if both horses match (order doesn't matter)
        # Shape: [n_umaren, n_sanrentan]
        all_match = matches.all(dim=2).float()
        
        # Expand for batch dimension
        # Shape: [batch_size, n_umaren, n_sanrentan]
        sanrentan_to_umaren = all_match.unsqueeze(0).expand(batch_size, -1, -1)
        
        return sanrentan_to_umaren
    
    def _create_sanrentan_to_umatan_mapping(self, max_horses, batch_size, device):
        """Create mapping from sanrentan (ordered trifecta) to umatan (exacta).
        
        This mapping is used to convert trifecta probabilities to exacta probabilities.
        An exacta bet considers only the first and second place finishers in order.
        
        Args:
            max_horses: Maximum number of horses per race
            batch_size: Number of races in the batch
            device: Device to use for computation
            
        Returns:
            torch.Tensor: Mapping matrix [batch_size, n_umatan, n_sanrentan]
                          Where mapping[b, i, j] = 1 if sanrentan j projects to umatan i
        """
        # Generate all possible trifecta combinations (1-indexed horse numbers)
        sanrentan_combinations = list(itertools.permutations(range(1, max_horses + 1), 3))
        # Convert to tensor: [n_sanrentan, 3]
        sanrentan_tensor = torch.tensor(sanrentan_combinations, device=device)
        
        # Generate all possible exacta combinations (1-indexed horse numbers)
        umatan_combinations = list(itertools.permutations(range(1, max_horses + 1), 2))
        # Convert to tensor: [n_umatan, 2]
        umatan_tensor = torch.tensor(umatan_combinations, device=device)
        
        # Extract first two positions from sanrentan combinations
        # Shape: [n_sanrentan, 2]
        sanrentan_first_two = sanrentan_tensor[:, :2]
        
        # Reshape for broadcasting comparison
        # Shape: [n_umatan, 1, 2]
        umatan_expanded = umatan_tensor.unsqueeze(1)
        # Shape: [1, n_sanrentan, 2]
        sanrentan_first_two_expanded = sanrentan_first_two.unsqueeze(0)
        
        # Compare each umatan with each sanrentan's first two positions
        # Result shape: [n_umatan, n_sanrentan, 2]
        matches = umatan_expanded == sanrentan_first_two_expanded
        
        # An umatan matches a sanrentan if both horses match in the same order
        # Shape: [n_umatan, n_sanrentan]
        all_match = matches.all(dim=2).float()
        
        # Expand for batch dimension
        # Shape: [batch_size, n_umatan, n_sanrentan]
        sanrentan_to_umatan = all_match.unsqueeze(0).expand(batch_size, -1, -1)
        
        return sanrentan_to_umatan
    
    def _create_sanrentan_to_sanrenpuku_mapping(self, max_horses, batch_size, device):
        """Create mapping from sanrentan (ordered trifecta) to sanrenpuku (unordered trio).
        
        This mapping is used to convert ordered trifecta probabilities to unordered trio probabilities.
        
        Args:
            max_horses: Maximum number of horses per race
            batch_size: Number of races in the batch
            device: Device to use for computation
            
        Returns:
            torch.Tensor: Mapping matrix [batch_size, n_sanrenpuku, n_sanrentan]
                          Where mapping[b, i, j] = 1 if sanrentan j belongs to sanrenpuku i
        """
        # Generate all possible trifecta combinations (1-indexed horse numbers)
        sanrentan_combinations = list(itertools.permutations(range(1, max_horses + 1), 3))
        # Convert to tensor: [n_sanrentan, 3]
        sanrentan_tensor = torch.tensor(sanrentan_combinations, device=device)
        
        # Generate all possible trio combinations (1-indexed horse numbers)
        sanrenpuku_combinations = list(itertools.combinations(range(1, max_horses + 1), 3))
        # Convert to tensor: [n_sanrenpuku, 3]
        sanrenpuku_tensor = torch.tensor(sanrenpuku_combinations, device=device)
        
        # Sort each sanrentan combination to make comparison easier
        # Shape: [n_sanrentan, 3]
        sorted_sanrentan = torch.sort(sanrentan_tensor, dim=1)[0]
        
        # Reshape for broadcasting comparison
        # Shape: [n_sanrenpuku, 1, 3]
        sanrenpuku_expanded = sanrenpuku_tensor.unsqueeze(1)
        # Shape: [1, n_sanrentan, 3]
        sorted_sanrentan_expanded = sorted_sanrentan.unsqueeze(0)
        
        # Compare each sanrenpuku with each sorted sanrentan
        # Result shape: [n_sanrenpuku, n_sanrentan, 3]
        matches = sanrenpuku_expanded == sorted_sanrentan_expanded
        
        # A sanrentan maps to a sanrenpuku if all 3 horses match (regardless of order)
        # Shape: [n_sanrenpuku, n_sanrentan]
        all_match = matches.all(dim=2).float()
        
        # Expand for batch dimension
        # Shape: [batch_size, n_sanrenpuku, n_sanrentan]
        sanrentan_to_sanrenpuku = all_match.unsqueeze(0).expand(batch_size, -1, -1)
        
        return sanrentan_to_sanrenpuku
    
    def _create_sanrentan_to_tansho_mapping(self, max_horses, batch_size, device):
        """Create mapping from sanrentan (ordered trifecta) to tansho (win).
        
        This mapping is used to convert trifecta probabilities to win probabilities.
        A tansho (win) bet considers only the first place finisher.
        
        Args:
            max_horses: Maximum number of horses per race
            batch_size: Number of races in the batch
            device: Device to use for computation
            
        Returns:
            torch.Tensor: Mapping matrix [batch_size, max_horses, n_sanrentan]
                          Where mapping[b, i, j] = 1 if sanrentan j has horse i+1 in first place
        """
        # Generate all possible trifecta combinations (1-indexed horse numbers)
        sanrentan_combinations = list(itertools.permutations(range(1, max_horses + 1), 3))
        # Convert to tensor: [n_sanrentan, 3]
        sanrentan_tensor = torch.tensor(sanrentan_combinations, device=device)
        
        # Extract first position from sanrentan combinations
        # Shape: [n_sanrentan]
        first_place_horses = sanrentan_tensor[:, 0]
        
        # Create a one-hot encoding for each horse number in first place
        # Shape: [max_horses, n_sanrentan]
        horse_indices = torch.arange(1, max_horses + 1, device=device).unsqueeze(1)
        first_place_expanded = first_place_horses.unsqueeze(0)
        mapping = (horse_indices == first_place_expanded).float()
        
        # Expand for batch dimension
        # Shape: [batch_size, max_horses, n_sanrentan]
        sanrentan_to_tansho = mapping.unsqueeze(0).expand(batch_size, -1, -1)
        
        return sanrentan_to_tansho
    
    def _create_sanrentan_to_wide_mapping(self, max_horses, batch_size, device):
        """Create mapping from sanrentan (ordered trifecta) to wide (rumor).
        
        This mapping is used to convert trifecta probabilities to wide bet probabilities.
        A wide bet wins if the two selected horses both finish in the top positions
        (typically top 3 for races with 7+ horses).
        
        Args:
            max_horses: Maximum number of horses per race
            batch_size: Number of races in the batch
            device: Device to use for computation
            
        Returns:
            torch.Tensor: Mapping matrix [batch_size, n_wide, n_sanrentan]
                          Where mapping[b, i, j] = 1 if sanrentan j has both horses from wide i in top positions
        """
        # Generate all possible trifecta combinations (1-indexed horse numbers)
        sanrentan_combinations = list(itertools.permutations(range(1, max_horses + 1), 3))
        # Convert to tensor: [n_sanrentan, 3]
        sanrentan_tensor = torch.tensor(sanrentan_combinations, device=device)
        
        # Generate all possible wide combinations (1-indexed horse numbers)
        wide_combinations = list(itertools.combinations(range(1, max_horses + 1), 2))
        # Convert to tensor: [n_wide, 2]
        wide_tensor = torch.tensor(wide_combinations, device=device)
        
        # Extract horses in top 3 positions from sanrentan combinations
        # Shape: [n_sanrentan, 3]
        top_three_horses = sanrentan_tensor
        
        # Reshape for broadcasting
        # Shape: [n_wide, 1, 2]
        wide_pairs = wide_tensor.unsqueeze(1)
        
        # Check if first horse in wide pair is in top 3
        # Shape: [n_wide, n_sanrentan]
        first_in_top3 = (wide_pairs[:, :, 0:1] == top_three_horses.unsqueeze(0)).any(dim=2)
        
        # Check if second horse in wide pair is in top 3
        # Shape: [n_wide, n_sanrentan]
        second_in_top3 = (wide_pairs[:, :, 1:2] == top_three_horses.unsqueeze(0)).any(dim=2)
        
        # A wide bet wins if both horses are in top 3
        # Shape: [n_wide, n_sanrentan]
        both_in_top3 = (first_in_top3 & second_in_top3).float()
        
        # Expand for batch dimension
        # Shape: [batch_size, n_wide, n_sanrentan]
        sanrentan_to_wide = both_in_top3.unsqueeze(0).expand(batch_size, -1, -1)
        
        return sanrentan_to_wide
    
    def _create_sanrentan_to_wakuren_mapping(self, max_horses, batch_size, device, wakuban=None):
        """Create mapping from sanrentan (ordered trifecta) to wakuren (box quinella).
    
        Wakuren is a bet on two frames/posts (not horses) to contain the first and second finishers,
        regardless of order. In Japanese racing, frame numbers range from 1 to 8.
    
        Args:
            max_horses: Maximum number of horses per race
            batch_size: Number of races in the batch
            device: Device to use for computation
            wakuban: Tensor with frame/post numbers for each horse [batch_size, max_horses]
                    If None, returns None as wakuren probabilities can't be calculated
        
        Returns:
            torch.Tensor or None: Mapping matrix [batch_size, n_wakuren, n_sanrentan]
                                Where mapping[b, i, j] = 1 if sanrentan j projects to wakuren i
        """
        if wakuban is None:
            return None
        
        # Move wakuban to the correct device
        wakuban = wakuban.to(device)
        
        # Generate all possible trifecta combinations (1-indexed horse numbers)
        sanrentan_combinations = list(itertools.permutations(range(1, max_horses + 1), 3))
        # Convert to tensor: [n_sanrentan, 3]
        sanrentan_tensor = torch.tensor(sanrentan_combinations, device=device)
        n_sanrentan = len(sanrentan_combinations)
        
        # Define the maximum frame number (in Japan, typically 8)
        max_frame = 8
        
        # Generate all possible wakuren combinations (frame pairs including same frame)
        wakuren_combinations = []
        for i in range(1, max_frame + 1):
            for j in range(i, max_frame + 1):  # Start from i to avoid duplicates
                wakuren_combinations.append((i, j))
    
        # Convert to tensor: [n_wakuren, 2]
        wakuren_tensor = torch.tensor(wakuren_combinations, device=device)
        n_wakuren = len(wakuren_combinations)
        
        # Extract indices for first and second place horses from sanrentan combinations
        # Subtract 1 to convert from 1-indexed to 0-indexed
        first_place_indices = sanrentan_tensor[:, 0] - 1  # [n_sanrentan]
        second_place_indices = sanrentan_tensor[:, 1] - 1  # [n_sanrentan]
        
        # Create batch indices for advanced indexing
        batch_indices = torch.arange(batch_size, device=device)
        
        # Expand indices for broadcasting
        # [batch_size, n_sanrentan]
        batch_indices_expanded = batch_indices.view(batch_size, 1).expand(batch_size, n_sanrentan)
        first_place_expanded = first_place_indices.view(1, n_sanrentan).expand(batch_size, n_sanrentan)
        second_place_expanded = second_place_indices.view(1, n_sanrentan).expand(batch_size, n_sanrentan)
        
        # Get frame numbers for first and second place horses
        # [batch_size, n_sanrentan]
        first_frames = wakuban[batch_indices_expanded, first_place_expanded]
        second_frames = wakuban[batch_indices_expanded, second_place_expanded]
        
        # Sort frame numbers (since wakuren is unordered)
        # [batch_size, n_sanrentan]
        min_frames = torch.min(first_frames, second_frames)
        max_frames = torch.max(first_frames, second_frames)
        
        # Reshape for broadcasting comparison
        # [batch_size, 1, n_sanrentan, 2]
        sorted_frame_pairs = torch.stack([min_frames, max_frames], dim=2).unsqueeze(1)
        # [1, n_wakuren, 1, 2]
        wakuren_expanded = wakuren_tensor.unsqueeze(0).unsqueeze(2)
        
        # Compare each wakuren with each sorted frame pair
        # [batch_size, n_wakuren, n_sanrentan, 2]
        matches = sorted_frame_pairs == wakuren_expanded
        
        # A match occurs when both frame numbers match
        # [batch_size, n_wakuren, n_sanrentan]
        all_match = matches.all(dim=3).float()
        
        return all_match
    
    def _compute_trifecta_scores(self, horse_performances, max_horses, horse_counts):
        """Compute scores for all possible trifecta combinations.
        
        This method efficiently calculates scores for all possible ordered combinations
        of 3 horses from max_horses, using vectorized operations instead of loops.
        It also masks out impossible combinations based on actual horse counts.
        
        Args:
            horse_performances: Tensor of horse performance vectors [batch_size, max_horses, perf_dim]
            max_horses: Maximum number of horses per race
            horse_counts: Tensor of actual horse counts for each race [batch_size]
            
        Returns:
            torch.Tensor: Scores for each trifecta combination [batch_size, num_combinations]
        """
        batch_size = horse_performances.shape[0]
        device = horse_performances.device
        
        # Generate all possible trifecta combinations (1-indexed horse numbers)
        # We consider horses with indices 1 through max_horses
        all_combinations = list(itertools.permutations(range(1, max_horses + 1), 3))
        num_combinations = len(all_combinations)
        
        # Create tensors to hold the indices of horses for each position in the trifecta
        # Subtract 1 to convert to 0-indexed for tensor indexing
        first_place_indices = torch.tensor([combo[0] - 1 for combo in all_combinations], device=device)
        second_place_indices = torch.tensor([combo[1] - 1 for combo in all_combinations], device=device)
        third_place_indices = torch.tensor([combo[2] - 1 for combo in all_combinations], device=device)
        
        # Gather horse performance vectors for each position in each combination
        # We'll create a batch dimension for the combinations to use batch gather
        # Repeat horse performances for each combination
        expanded_performances = horse_performances.unsqueeze(1).expand(
            batch_size, num_combinations, max_horses, horse_performances.shape[2]
        )
        
        # Create batch indices for gather operation
        batch_indices = torch.arange(batch_size, device=device).view(batch_size, 1).expand(batch_size, num_combinations)
        combo_indices = torch.arange(num_combinations, device=device).view(1, num_combinations).expand(batch_size, num_combinations)
        
        # Gather horse performances for each position
        first_horse_perfs = expanded_performances[batch_indices, combo_indices, first_place_indices]
        second_horse_perfs = expanded_performances[batch_indices, combo_indices, second_place_indices]
        third_horse_perfs = expanded_performances[batch_indices, combo_indices, third_place_indices]
        
        # Concatenate performances for all three positions
        # Shape: [batch_size, num_combinations, 3 * perf_dim]
        combined_perfs = torch.cat([first_horse_perfs, second_horse_perfs, third_horse_perfs], dim=2)
        
        # Calculate scores for each combination
        # Reshape for the scorer network
        reshaped_perfs = combined_perfs.view(-1, combined_perfs.shape[2])
        scores = self.combination_scorer(reshaped_perfs).view(batch_size, num_combinations)
        
        # Create a mask for invalid combinations
        # A combination is valid only if all horses in it are within the race's actual horse count
        
        # Expand horse_counts for broadcasting: [batch_size, 1]
        horse_counts_expanded = horse_counts.unsqueeze(1)
        
        # For each combination, check if all three positions use valid horses
        # The 1-indexed horse numbers are stored in all_combinations
        # Create tensors for all horse positions in combinations
        combo_first = torch.tensor([c[0] for c in all_combinations], device=device)
        combo_second = torch.tensor([c[1] for c in all_combinations], device=device)
        combo_third = torch.tensor([c[2] for c in all_combinations], device=device)
        
        # Create masks for each position (True if the horse number is valid)
        first_valid = combo_first.unsqueeze(0) <= horse_counts_expanded  # [batch_size, num_combinations]
        second_valid = combo_second.unsqueeze(0) <= horse_counts_expanded  # [batch_size, num_combinations]
        third_valid = combo_third.unsqueeze(0) <= horse_counts_expanded  # [batch_size, num_combinations]
        
        # Combine masks (all three positions must be valid)
        valid_combinations = first_valid & second_valid & third_valid  # [batch_size, num_combinations]
        
        # Apply mask by setting invalid combination scores to a large negative value
        # This ensures they have near-zero probability after softmax
        invalid_mask = ~valid_combinations
        scores = scores.masked_fill(invalid_mask, -1e9)
        
        return scores
    
    def compute_loss(self, outputs, targets):
        """Compute the loss for trifecta prediction.
        
        Args:
            outputs: Output from forward pass containing 'logits'
            targets: Dictionary containing 'target_sanrentan' with correct combination indices
            
        Returns:
            torch.Tensor: Loss value
        """
        logits = outputs['logits']
        
        # Targets should be tensor with shape [batch_size] containing the index of the correct trifecta
        if 'target_sanrentan' in targets:
            trifecta_targets = targets['target_sanrentan']
            
            # If targets aren't already on the device, move them
            if trifecta_targets.device != logits.device:
                trifecta_targets = trifecta_targets.to(logits.device)
            
            loss = self.loss_fn(logits, trifecta_targets)
            return loss
        else:
            logger.error("No 'target_sanrentan' found in targets")
            # Return a zero loss as a fallback (helps prevent training crashes)
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
    
    def compute_metrics(self, outputs, inputs, epoch=None):
        """Compute evaluation metrics for the trifecta model.
        
        Args:
            outputs: Output from forward pass
            inputs: Target values
            epoch: Current epoch number (optional)
            
        Returns:
            dict: Dictionary of metrics:
                - accuracy: Proportion of correctly predicted trifectas
                - top5_accuracy: Proportion of targets in top 5 predictions
                - top10_accuracy: Proportion of targets in top 10 predictions
                - top20_accuracy: Proportion of targets in top 20 predictions
                - mean_rank: Average rank of the correct trifecta
                - betting_metrics: Performance metrics for different betting strategies
        """
        metrics = {}
        
        if 'target_sanrentan' not in inputs:
            logger.warning("No 'target_sanrentan' found in targets, cannot compute metrics")
            return metrics
        
        # Get predictions and targets
        probabilities = outputs['sanrentan_probabilities']
        trifecta_targets = inputs['target_sanrentan'].to(probabilities.device)
        
        # Calculate accuracy (exact match)
        _, top1_predictions = torch.topk(probabilities, k=1, dim=1)
        correct = (top1_predictions.squeeze(-1) == trifecta_targets).float()
        accuracy = correct.mean().item()
        metrics['accuracy'] = accuracy
        
        # Calculate top-k accuracies
        for k in [5, 10, 20]:
            # Skip if we have fewer than k possible combinations
            if probabilities.shape[1] < k:
                logger.warning(f"Cannot compute top{k}_accuracy: fewer than {k} combinations available")
                metrics[f'top{k}_accuracy'] = float('nan')
                continue
                
            # Get top k predictions
            _, top_k_predictions = torch.topk(probabilities, k=k, dim=1)
            
            # Check if the target is in the top k predictions
            is_in_top_k = torch.any(top_k_predictions == trifecta_targets.unsqueeze(1), dim=1).float()
            top_k_accuracy = is_in_top_k.mean().item()
            metrics[f'top{k}_accuracy'] = top_k_accuracy
        
        # Calculate mean rank of correct prediction
        batch_size = probabilities.shape[0]
        sorted_indices = torch.argsort(probabilities, dim=1, descending=True)
        
        # For each sample, find the rank of the target
        ranks = torch.zeros(batch_size, dtype=torch.long, device=probabilities.device)
        for i in range(batch_size):
            # Find the position of the target in the sorted predictions
            target_idx = trifecta_targets[i].item()
            # Add 1 so rank starts from 1, not 0
            ranks[i] = (sorted_indices[i] == target_idx).nonzero().item() + 1
        
        mean_rank = ranks.float().mean().item()
        metrics['mean_rank'] = mean_rank
        
        # Run betting simulations if we have odds data
        if any(key.startswith('odds_') for key in inputs):
            betting_metrics = self._simulate_betting_strategies(outputs, inputs, epoch)
            metrics['betting_metrics'] = betting_metrics

        return metrics

    def _simulate_betting_strategies(self, outputs, inputs, epoch=None):
        """Run betting simulations using different strategies.
        
        Args:
            outputs: Model outputs containing probabilities for different bet types
            inputs: Dictionary containing odds and target (hit) information
            epoch: Current epoch number (optional)
            
        Returns:
            dict: Simulation results for different betting strategies
        """
        try:
            from horguesc.core.betting.simulator import BettingSimulator
            
            # Check if we have the necessary data
            has_odds = any(key.startswith('odds_') for key in inputs)
            has_targets = any(key.startswith('target_') for key in inputs)
            
            if not (has_odds and has_targets):
                logger.warning("Missing odds or target data, cannot run betting simulation")
                return {}
            
            # Extract odds and results data
            odds_data = {k: v for k, v in inputs.items() if k.startswith('odds_')}
            race_results = {k: v for k, v in inputs.items() if k.startswith('target_')}
            
            # Get race IDs if available
            race_ids = inputs.get('kyoso_id', None)
            if race_ids is None:
                # Create dummy race IDs if not provided
                batch_size = list(outputs.values())[0].shape[0]
                race_ids = [f'race_{i}' for i in range(batch_size)]
            
            # Initialize simulator
            simulator = BettingSimulator(config=self.config)
            simulator.add_default_strategies()
            
            # Set common data
            simulator.set_common_data(
                model_outputs=outputs,
                odds_data=odds_data,
                race_results=race_results,
                race_ids=race_ids
            )
            
            # Generate capital trend visualization path based on epoch and timestamp
            from datetime import datetime
            
            # Get visualization directory from config or use default
            vis_dir = self.config.get('paths', 'visualization_dir', fallback='visualizations')
            os.makedirs(vis_dir, exist_ok=True)
            
            # Create timestamp and epoch-based filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            epoch_str = f"epoch{epoch}" if epoch is not None else "inference"
            
            capital_trend_path = os.path.join(vis_dir, f"trifecta_capital_trend_{epoch_str}_{timestamp}.png")
            betting_details_path = os.path.join(vis_dir, f"trifecta_betting_details_{epoch_str}_{timestamp}.xlsx")

            # Run simulations for all strategies
            betting_results = simulator.simulate(capital_trend_path=capital_trend_path, betting_details_path=betting_details_path)
            
            # Convert results to a more compact format for metrics
            betting_metrics = {}
            for strategy_name, result in betting_results.items():
                betting_metrics[strategy_name] = {
                    'profit': result['total_profit'],
                    'total_bet': result['total_bet'],
                    'hit_rate': result['hit_rate']
                }
            
            return betting_metrics
        
        except ImportError as e:
            logger.warning(f"Could not import betting module: {e}")
            return {}
        except Exception as e:
            logger.error(f"Error running betting simulation: {e}")
            return {'error': str(e)}