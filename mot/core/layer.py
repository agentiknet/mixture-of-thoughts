"""Mixture of Thoughts Layer - main building block"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict

from mot.core.router import ThoughtRouter
from mot.core.branch import ThoughtBranch


class AttentionCombiner(nn.Module):
    """Intelligently combines outputs from different thought branches"""
    
    def __init__(self, hidden_size: int, num_heads: int = 8):
        super().__init__()
        self.hidden_size = hidden_size
        self.attention = nn.MultiheadAttention(
            hidden_size, 
            num_heads=num_heads, 
            batch_first=True
        )
    
    def forward(
        self, 
        thought_outputs: List[torch.Tensor], 
        router_weights: torch.Tensor
    ) -> torch.Tensor:
        """
        Combine thought outputs using attention
        
        Args:
            thought_outputs: List of tensors [batch_size, seq_len, hidden_size]
            router_weights: Router probabilities [num_thoughts]
            
        Returns:
            Combined output [batch_size, seq_len, hidden_size]
        """
        if len(thought_outputs) == 1:
            return thought_outputs[0]
        
        # Stack thoughts
        stacked_thoughts = torch.stack(thought_outputs, dim=1)  # [B, num_thoughts, seq_len, hidden]
        batch_size, num_thoughts, seq_len, hidden_size = stacked_thoughts.shape
        
        # Reshape for attention
        thoughts_flat = stacked_thoughts.view(batch_size, num_thoughts * seq_len, hidden_size)
        
        # Cross-attention between thoughts
        attended_thoughts, _ = self.attention(thoughts_flat, thoughts_flat, thoughts_flat)
        
        # Reshape back
        attended_thoughts = attended_thoughts.view(batch_size, num_thoughts, seq_len, hidden_size)
        
        # Weight by router probabilities
        router_weights_expanded = router_weights.unsqueeze(-1).unsqueeze(-1)  # [num_thoughts, 1, 1]
        weighted_thoughts = attended_thoughts * router_weights_expanded
        
        # Sum over thought dimension
        combined = weighted_thoughts.sum(dim=1)  # [B, seq_len, hidden]
        
        return combined


class MixtureOfThoughtsLayer(nn.Module):
    """Main MoT layer - replaces FFN in transformer"""
    
    def __init__(self, hidden_size: int, num_thoughts: int = 8):
        super().__init__()
        self.num_thoughts = num_thoughts
        self.hidden_size = hidden_size
        
        # Router for branch selection
        self.thought_router = ThoughtRouter(hidden_size, num_thoughts)
        
        # Independent thought branches
        self.thought_branches = nn.ModuleList([
            ThoughtBranch(hidden_size) for _ in range(num_thoughts)
        ])
        
        # Intelligent combiner
        self.thought_combiner = AttentionCombiner(hidden_size)
        
        # Diversity loss weight
        self.diversity_loss_weight = 0.1
    
    def forward(
        self, 
        hidden_states: torch.Tensor, 
        creativity_level: float = 1.0
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass through MoT layer
        
        Args:
            hidden_states: Input tensor [batch_size, seq_len, hidden_size]
            creativity_level: Boost for creativity (default: 1.0)
            
        Returns:
            output: Combined output [batch_size, seq_len, hidden_size]
            metrics: Dictionary of metrics for analysis
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # 1. Probabilistic branch selection
        selected_thoughts, router_probs = self.thought_router(hidden_states)
        
        # 2. Process each selected branch
        thought_outputs = []
        active_weights = []
        
        for batch_idx in range(batch_size):
            batch_thoughts = []
            batch_weights = []
            
            for thought_idx in selected_thoughts[batch_idx]:
                # Each branch processes with its own creativity
                branch_output = self.thought_branches[thought_idx](
                    hidden_states[batch_idx:batch_idx+1],
                    creativity_boost=creativity_level
                )
                batch_thoughts.append(branch_output)
                batch_weights.append(router_probs[batch_idx, thought_idx])
            
            # Normalize weights for this batch
            batch_weights = torch.stack(batch_weights)
            batch_weights = F.softmax(batch_weights, dim=0)
            
            thought_outputs.append(batch_thoughts)
            active_weights.append(batch_weights)
        
        # 3. Intelligently combine thoughts
        combined_outputs = []
        for batch_idx in range(batch_size):
            if len(thought_outputs[batch_idx]) > 1:
                combined = self.thought_combiner(
                    thought_outputs[batch_idx],
                    active_weights[batch_idx]
                )
            else:
                combined = thought_outputs[batch_idx][0]
            combined_outputs.append(combined)
        
        final_output = torch.cat(combined_outputs, dim=0)
        
        # Compute metrics for monitoring
        metrics = {
            'diversity_score': self._compute_diversity(thought_outputs),
            'router_entropy': self._compute_entropy(router_probs),
            'active_branches': [len(thoughts) for thoughts in thought_outputs]
        }
        
        return final_output, metrics
    
    def _compute_diversity(self, thought_outputs: List[List[torch.Tensor]]) -> float:
        """Compute diversity between branches using cosine similarity"""
        if len(thought_outputs[0]) < 2:
            return 0.0
        
        diversities = []
        for batch_thoughts in thought_outputs:
            if len(batch_thoughts) >= 2:
                similarities = []
                for i in range(len(batch_thoughts)):
                    for j in range(i+1, len(batch_thoughts)):
                        sim = F.cosine_similarity(
                            batch_thoughts[i].flatten(),
                            batch_thoughts[j].flatten(),
                            dim=0
                        )
                        similarities.append(sim.item())
                
                # Diversity = 1 - average similarity
                avg_similarity = sum(similarities) / len(similarities)
                diversities.append(1.0 - avg_similarity)
        
        return sum(diversities) / len(diversities) if diversities else 0.0
    
    def _compute_entropy(self, router_probs: torch.Tensor) -> float:
        """Compute router entropy (measure of selection diversity)"""
        entropy = -torch.sum(router_probs * torch.log(router_probs + 1e-8), dim=1)
        return entropy.mean().item()
