"""Complete Mixture of Thoughts Transformer"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from typing import Optional, List, Dict

from mot.core.layer import MixtureOfThoughtsLayer


class MoTConfig:
    """Configuration for MoT Transformer"""
    
    def __init__(
        self,
        vocab_size: int = 50257,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_thoughts: int = 8,
        max_position_embeddings: int = 1024,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_thoughts = num_thoughts
        self.max_position_embeddings = max_position_embeddings


class MixtureOfThoughtsTransformer(nn.Module):
    """Complete Transformer with Mixture of Thoughts layers"""
    
    def __init__(self, config: MoTConfig):
        super().__init__()
        self.config = config
        
        # Embeddings
        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, 
            config.hidden_size
        )
        
        # MoT layers
        self.mot_layers = nn.ModuleList([
            MixtureOfThoughtsLayer(config.hidden_size, num_thoughts=config.num_thoughts)
            for _ in range(config.num_hidden_layers)
        ])
        
        # Output head
        self.output_head = nn.Linear(config.hidden_size, config.vocab_size)
        
        # Global creativity control
        self.global_creativity = nn.Parameter(torch.ones(1))
    
    def forward(
        self, 
        input_ids: torch.Tensor, 
        creativity_schedule: Optional[List[float]] = None
    ) -> Dict:
        """
        Forward pass through MoT transformer
        
        Args:
            input_ids: Token indices [batch_size, seq_len]
            creativity_schedule: Optional per-layer creativity levels
            
        Returns:
            Dictionary with logits, metrics, and hidden states
        """
        batch_size, seq_len = input_ids.shape
        
        # Embeddings
        token_embeds = self.token_embeddings(input_ids)
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        position_embeds = self.position_embeddings(position_ids)
        
        hidden_states = token_embeds + position_embeds
        
        # Pass through MoT layers
        all_metrics = []
        for layer_idx, mot_layer in enumerate(self.mot_layers):
            # Adaptive creativity per layer
            if creativity_schedule:
                creativity = creativity_schedule[layer_idx]
            else:
                creativity = self.global_creativity.item()
            
            hidden_states, metrics = mot_layer(hidden_states, creativity_level=creativity)
            metrics['layer'] = layer_idx
            all_metrics.append(metrics)
        
        # Output logits
        logits = self.output_head(hidden_states)
        
        return {
            'logits': logits,
            'metrics': all_metrics,
            'final_hidden_states': hidden_states
        }
    
    def generate_parallel_thoughts(
        self, 
        input_ids: torch.Tensor, 
        num_parallel: int = 4, 
        max_length: int = 50
    ) -> List[Dict]:
        """
        Generate multiple thought trajectories in parallel
        
        Args:
            input_ids: Starting tokens [batch_size, seq_len]
            num_parallel: Number of parallel thoughts to generate
            max_length: Maximum tokens to generate
            
        Returns:
            List of generation results with metrics
        """
        self.eval()
        parallel_results = []
        
        for i in range(num_parallel):
            # Different seed for each generation
            torch.manual_seed(random.randint(0, 100000))
            
            # Variable creativity
            creativity_level = 0.5 + random.random() * 1.0  # Between 0.5 and 1.5
            creativity_schedule = [creativity_level] * len(self.mot_layers)
            
            with torch.no_grad():
                current_ids = input_ids.clone()
                generated_metrics = []
                
                for step in range(max_length):
                    outputs = self.forward(current_ids, creativity_schedule)
                    
                    # Sampling with temperature
                    temperature = 0.8 + random.random() * 0.4  # Between 0.8 and 1.2
                    next_token_logits = outputs['logits'][:, -1, :] / temperature
                    next_token = torch.multinomial(
                        F.softmax(next_token_logits, dim=-1), 
                        num_samples=1
                    )
                    
                    current_ids = torch.cat([current_ids, next_token], dim=1)
                    generated_metrics.extend(outputs['metrics'])
                
                parallel_results.append({
                    'generated_ids': current_ids,
                    'creativity_level': creativity_level,
                    'metrics': generated_metrics,
                    'branch_id': i
                })
        
        return parallel_results
