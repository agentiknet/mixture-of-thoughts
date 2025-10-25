"""Thought Branch with non-deterministic noise injection"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NoiseInjection(nn.Module):
    """Injects creative noise into representations"""
    
    def __init__(self, hidden_size: int, noise_scale: float = 0.1):
        super().__init__()
        self.noise_scale = nn.Parameter(torch.tensor(noise_scale))
        self.adaptive_noise = nn.Linear(hidden_size, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            # Adaptive noise based on content
            adaptive_scale = torch.sigmoid(self.adaptive_noise(x.mean(dim=1, keepdim=True)))
            noise = torch.randn_like(x) * self.noise_scale * adaptive_scale
            return x + noise
        return x


class ThoughtBranch(nn.Module):
    """A non-deterministic thought branch"""
    
    def __init__(self, hidden_size: int, intermediate_size: int = None):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size or hidden_size * 4
        
        # FFN with creative variations
        self.ffn1 = nn.Linear(self.hidden_size, self.intermediate_size)
        self.ffn2 = nn.Linear(self.intermediate_size, self.hidden_size)
        
        # Creative noise injection
        self.noise_injection = NoiseInjection(self.hidden_size)
        
        # Learned creativity factor
        self.creativity_factor = nn.Parameter(torch.ones(1) * 0.2)
        
        # Adaptive normalization
        self.layer_norm = nn.LayerNorm(self.hidden_size)
    
    def forward(self, x: torch.Tensor, creativity_boost: float = 1.0) -> torch.Tensor:
        """
        Forward pass with non-deterministic behavior
        
        Args:
            x: Input tensor [batch_size, seq_len, hidden_size]
            creativity_boost: Multiplier for creativity (default: 1.0)
            
        Returns:
            Output tensor with same shape as input
        """
        # Save for residual connection
        residual = x
        
        # Inject creative noise
        x_noisy = self.noise_injection(x)
        
        # FFN transformation with creative non-linearity
        hidden = F.gelu(self.ffn1(x_noisy))
        
        # Noise in latent space for more diversity
        if self.training:
            latent_noise = torch.randn_like(hidden) * self.creativity_factor * creativity_boost
            hidden = hidden + latent_noise
        
        # Output projection
        output = self.ffn2(hidden)
        
        # Residual connection + normalization
        output = self.layer_norm(output + residual)
        
        return output
