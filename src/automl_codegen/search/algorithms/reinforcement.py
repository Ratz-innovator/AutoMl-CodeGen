"""
Reinforcement Learning for Neural Architecture Search

Implementation of RL-based NAS using a controller network that learns to generate
architectures based on their performance feedback.

Based on: "Neural Architecture Search with Reinforcement Learning" (Zoph & Le, 2017)
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
import random
from collections import deque
import copy

from ..space.search_space import SearchSpace
from ..objectives.multi_objective import MultiObjectiveOptimizer

logger = logging.getLogger(__name__)


class ControllerRNN(nn.Module):
    """RNN controller that generates neural architectures."""
    
    def __init__(
        self,
        num_layers: int = 12,
        hidden_size: int = 100,
        num_layer_types: int = 6,
        num_filters: List[int] = [16, 32, 64, 128],
        max_layers: int = 20
    ):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.num_layer_types = num_layer_types
        self.num_filters = num_filters
        self.max_layers = max_layers
        
        # Embedding for decisions
        self.embedding = nn.Embedding(100, hidden_size)  # Large vocab for various decisions
        
        # LSTM controller
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        
        # Decision heads
        self.layer_type_head = nn.Linear(hidden_size, num_layer_types)
        self.filter_head = nn.Linear(hidden_size, len(num_filters))
        self.kernel_size_head = nn.Linear(hidden_size, 3)  # 1, 3, 5
        self.stride_head = nn.Linear(hidden_size, 2)  # 1, 2
        self.activation_head = nn.Linear(hidden_size, 3)  # ReLU, GELU, Swish
        
        # Start token
        self.start_token = nn.Parameter(torch.randn(1, 1, hidden_size))
        
    def forward(self, max_steps: int = 20) -> Tuple[List[Dict[str, Any]], List[torch.Tensor]]:
        """Generate an architecture by sampling from the controller."""
        batch_size = 1
        device = next(self.parameters()).device
        
        # Initialize hidden state
        hidden = (
            torch.zeros(1, batch_size, self.hidden_size, device=device),
            torch.zeros(1, batch_size, self.hidden_size, device=device)
        )
        
        # Start with start token
        input_token = self.start_token
        
        architecture_layers = []
        log_probs = []
        
        for step in range(max_steps):
            # LSTM forward pass
            output, hidden = self.lstm(input_token, hidden)
            
            # Sample layer type
            layer_type_logits = self.layer_type_head(output.squeeze(1))
            layer_type_probs = F.softmax(layer_type_logits, dim=-1)
            layer_type_dist = torch.distributions.Categorical(layer_type_probs)
            layer_type = layer_type_dist.sample()
            log_probs.append(layer_type_dist.log_prob(layer_type))
            
            # Convert to layer type name
            layer_types = ['conv2d', 'linear', 'relu', 'batchnorm', 'dropout', 'flatten']
            layer_type_name = layer_types[layer_type.item()]
            
            layer_spec = {'type': layer_type_name}
            
            # Sample parameters based on layer type
            if layer_type_name == 'conv2d':
                # Sample number of filters
                filter_logits = self.filter_head(output.squeeze(1))
                filter_probs = F.softmax(filter_logits, dim=-1)
                filter_dist = torch.distributions.Categorical(filter_probs)
                filter_idx = filter_dist.sample()
                log_probs.append(filter_dist.log_prob(filter_idx))
                layer_spec['out_channels'] = self.num_filters[filter_idx.item()]
                
                # Sample kernel size
                kernel_logits = self.kernel_size_head(output.squeeze(1))
                kernel_probs = F.softmax(kernel_logits, dim=-1)
                kernel_dist = torch.distributions.Categorical(kernel_probs)
                kernel_idx = kernel_dist.sample()
                log_probs.append(kernel_dist.log_prob(kernel_idx))
                kernel_sizes = [1, 3, 5]
                layer_spec['kernel_size'] = kernel_sizes[kernel_idx.item()]
                layer_spec['padding'] = layer_spec['kernel_size'] // 2
                
                # Sample stride
                stride_logits = self.stride_head(output.squeeze(1))
                stride_probs = F.softmax(stride_logits, dim=-1)
                stride_dist = torch.distributions.Categorical(stride_probs)
                stride_idx = stride_dist.sample()
                log_probs.append(stride_dist.log_prob(stride_idx))
                strides = [1, 2]
                layer_spec['stride'] = strides[stride_idx.item()]
                
            elif layer_type_name == 'linear':
                # Sample output features
                filter_logits = self.filter_head(output.squeeze(1))
                filter_probs = F.softmax(filter_logits, dim=-1)
                filter_dist = torch.distributions.Categorical(filter_probs)
                filter_idx = filter_dist.sample()
                log_probs.append(filter_dist.log_prob(filter_idx))
                output_features = [64, 128, 256, 512]
                layer_spec['out_features'] = output_features[min(filter_idx.item(), len(output_features)-1)]
                
            elif layer_type_name == 'dropout':
                # Fixed dropout rate for simplicity
                layer_spec['p'] = 0.5
                
            # Add layer to architecture
            architecture_layers.append(layer_spec)
            
            # Prepare input for next step (embedding of current decision)
            next_input = layer_type.unsqueeze(0)  # [1]
            input_token = self.embedding(next_input)  # [1, 1, hidden_size] - embedding already has correct shape
            
            # Early stopping condition
            if layer_type_name == 'flatten' or len(architecture_layers) >= max_steps - 2:
                # Add final classification layer
                architecture_layers.append({
                    'type': 'linear',
                    'out_features': 10  # Assume 10 classes for now
                })
                break
        
        return architecture_layers, log_probs


class ReinforcementSearch:
    """Reinforcement learning-based neural architecture search."""
    
    def __init__(
        self,
        search_space: SearchSpace,
        objectives: MultiObjectiveOptimizer,
        config: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None,
        **kwargs
    ):
        self.search_space = search_space
        self.objectives = objectives
        self.config = config or {}
        self.logger = logger or logging.getLogger(__name__)
        
        # RL hyperparameters
        self.controller_lr = self.config.get('controller_lr', 3e-4)
        self.baseline_decay = self.config.get('baseline_decay', 0.95)
        self.entropy_weight = self.config.get('entropy_weight', 1e-3)
        
        # Architecture limits
        self.max_layers = self.config.get('max_layers', 20)
        self.num_filters = [16, 32, 64, 128, 256]
        
        # Controller network
        self.controller = ControllerRNN(
            num_layers=self.max_layers,
            hidden_size=100,
            num_layer_types=6,
            num_filters=self.num_filters,
            max_layers=self.max_layers
        )
        
        # Optimizer
        self.controller_optimizer = optim.Adam(
            self.controller.parameters(),
            lr=self.controller_lr
        )
        
        # Baseline tracking
        self.baseline = None
        self.baseline_history = deque(maxlen=100)
        
        # Search history
        self.search_history = []
        self.best_architecture = None
        self.best_reward = float('-inf')
        
        self.logger.info("Initialized RL-based search with controller network")
    
    def search(
        self,
        num_episodes: int = 100,
        architectures_per_episode: int = 10,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run reinforcement learning search.
        
        Args:
            num_episodes: Number of training episodes
            architectures_per_episode: Number of architectures to sample per episode
            
        Returns:
            Best architecture and search results
        """
        self.logger.info(f"Starting RL search for {num_episodes} episodes")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.controller.to(device)
        
        for episode in range(num_episodes):
            episode_rewards = []
            episode_log_probs = []
            episode_architectures = []
            
            # Sample architectures for this episode
            for arch_idx in range(architectures_per_episode):
                # Generate architecture
                architecture_layers, log_probs = self.controller()
                
                # Create full architecture specification
                architecture = {
                    'task': 'image_classification',
                    'layers': architecture_layers,
                    'connections': 'sequential',
                    'input_shape': (3, 32, 32),
                    'algorithm': 'reinforcement'
                }
                
                # Evaluate architecture
                reward = self._evaluate_architecture(architecture)
                
                # Store results
                episode_rewards.append(reward)
                episode_log_probs.extend(log_probs)
                episode_architectures.append(architecture)
                
                # Update best architecture
                if reward > self.best_reward:
                    self.best_reward = reward
                    self.best_architecture = copy.deepcopy(architecture)
                    self.logger.info(f"New best architecture found with reward: {reward:.4f}")
            
            # Update controller
            self._update_controller(episode_rewards, episode_log_probs)
            
            # Update baseline
            avg_reward = np.mean(episode_rewards)
            self.baseline_history.append(avg_reward)
            if self.baseline is None:
                self.baseline = avg_reward
            else:
                self.baseline = (
                    self.baseline_decay * self.baseline + 
                    (1 - self.baseline_decay) * avg_reward
                )
            
            # Log progress
            self.logger.info(
                f"Episode {episode+1}/{num_episodes}: "
                f"Avg Reward: {avg_reward:.4f}, "
                f"Best Reward: {self.best_reward:.4f}, "
                f"Baseline: {self.baseline:.4f}"
            )
            
            # Store episode results
            self.search_history.append({
                'episode': episode,
                'rewards': episode_rewards,
                'architectures': episode_architectures,
                'avg_reward': avg_reward
            })
        
        return {
            'architecture': self.best_architecture,
            'reward': self.best_reward,
            'search_history': self.search_history
        }
    
    def _evaluate_architecture(self, architecture: Dict[str, Any]) -> float:
        """
        Evaluate an architecture and return a reward.
        
        This is a simplified evaluation - in practice, you'd train the model.
        """
        layers = architecture.get('layers', [])
        
        if not layers:
            return 0.0
        
        # Simplified reward based on architecture properties
        reward = 0.0
        
        # Reward for having both conv and linear layers
        has_conv = any(layer.get('type') == 'conv2d' for layer in layers)
        has_linear = any(layer.get('type') == 'linear' for layer in layers)
        has_activation = any(layer.get('type') in ['relu', 'gelu'] for layer in layers)
        
        if has_conv:
            reward += 0.3
        if has_linear:
            reward += 0.2
        if has_activation:
            reward += 0.2
        
        # Reward for reasonable depth
        num_conv = sum(1 for layer in layers if layer.get('type') == 'conv2d')
        if 2 <= num_conv <= 8:
            reward += 0.2
        elif num_conv > 8:
            reward -= 0.1  # Penalize too deep networks
        
        # Reward for parameter efficiency
        estimated_params = self._estimate_parameters(architecture)
        if estimated_params < 1e6:  # Less than 1M parameters
            reward += 0.1
        elif estimated_params > 10e6:  # More than 10M parameters
            reward -= 0.1
        
        # Add some noise to make it more realistic
        reward += np.random.normal(0, 0.05)
        
        # Ensure reward is in reasonable range
        return max(0.0, min(1.0, reward))
    
    def _estimate_parameters(self, architecture: Dict[str, Any]) -> int:
        """Estimate number of parameters in architecture."""
        layers = architecture.get('layers', [])
        total_params = 0
        current_channels = 3  # RGB input
        spatial_size = 32  # Assume 32x32 input
        
        for layer in layers:
            layer_type = layer.get('type')
            
            if layer_type == 'conv2d':
                out_channels = layer.get('out_channels', 64)
                kernel_size = layer.get('kernel_size', 3)
                stride = layer.get('stride', 1)
                
                # Parameters: out_channels * in_channels * kernel_size^2
                layer_params = out_channels * current_channels * kernel_size * kernel_size
                total_params += layer_params
                
                current_channels = out_channels
                spatial_size = spatial_size // stride
                
            elif layer_type == 'linear':
                out_features = layer.get('out_features', 128)
                
                # Estimate input features (flattened if coming from conv)
                if spatial_size > 1:
                    in_features = current_channels * spatial_size * spatial_size
                    spatial_size = 1  # After linear layer
                else:
                    in_features = current_channels
                
                layer_params = in_features * out_features
                total_params += layer_params
                
                current_channels = out_features
        
        return total_params
    
    def _update_controller(self, rewards: List[float], log_probs: List[torch.Tensor]):
        """Update controller using REINFORCE algorithm."""
        if not rewards or not log_probs:
            return
        
        # Calculate advantages
        rewards = np.array(rewards)
        baseline = self.baseline if self.baseline is not None else 0.0
        advantages = rewards - baseline
        
        # Normalize advantages
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Calculate loss
        policy_loss = 0
        for i, log_prob in enumerate(log_probs):
            # Repeat advantage for each decision in the architecture
            arch_idx = i // (len(log_probs) // len(rewards))
            arch_idx = min(arch_idx, len(advantages) - 1)
            advantage = advantages[arch_idx]
            
            policy_loss -= log_prob * advantage
        
        # Add entropy bonus for exploration
        entropy_loss = 0
        for log_prob in log_probs:
            entropy_loss -= self.entropy_weight * log_prob
        
        total_loss = policy_loss + entropy_loss
        
        # Update controller
        self.controller_optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.controller.parameters(), 5.0)
        
        self.controller_optimizer.step()
    
    def sample_architecture(self) -> Dict[str, Any]:
        """Sample a single architecture from the trained controller."""
        with torch.no_grad():
            architecture_layers, _ = self.controller()
            
            architecture = {
                'task': 'image_classification',
                'layers': architecture_layers,
                'connections': 'sequential',
                'input_shape': (3, 32, 32),
                'algorithm': 'reinforcement'
            }
            
            return architecture 