"""Probabilistic router for thought branch selection"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from typing import Tuple


class ThoughtRouter(nn.Module):
    """Probabilistic router to select thought branches"""
    
    def __init__(self, hidden_size: int, num_thought_branches: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_thought_branches = num_thought_branches
        
        self.router = nn.Linear(hidden_size, num_thought_branches)
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)
        self.diversity_bonus = nn.Parameter(torch.ones(1) * 0.1)
    
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Route to thought branches probabilistically
        
        Args:
            hidden_states: Input tensor [batch_size, seq_len, hidden_size]
            
        Returns:
            selected_thoughts: Indices of selected branches [batch_size, num_selected]
            thought_probs: Probabilities for each branch [batch_size, num_branches]
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Compute routing logits from pooled representation
        router_logits = self.router(hidden_states.mean(dim=1))  # [B, num_branches]
        
        # Add stochastic diversity
        diversity_noise = torch.randn_like(router_logits) * self.diversity_bonus
        router_logits = router_logits + diversity_noise
        
        # Probabilistic sampling with temperature
        thought_probs = F.softmax(router_logits / self.temperature, dim=-1)
        
        # Select 2-3 branches per sample (non-deterministic)
        num_selected = random.randint(2, 3)
        selected_thoughts = torch.multinomial(
            thought_probs, 
            num_samples=num_selected, 
            replacement=False
        )
        
        return selected_thoughts, thought_probs
