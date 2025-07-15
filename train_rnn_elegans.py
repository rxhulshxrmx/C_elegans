#!/usr/bin/env python3
"""
RNN Training for C. elegans Control
Trains RNN networks to control worm agents in 1D and 3D scenarios
Analyzes weight and correlation changes before/after training
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import time
import os

class SimpleRNN(nn.Module):
    """Simple RNN for worm control"""
    def __init__(self, input_size=3, hidden_size=16, output_size=2):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.output = nn.Linear(hidden_size, output_size)
        
        # Initialize weights for better learning
        for name, param in self.rnn.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
        nn.init.xavier_uniform_(self.output.weight)
        
    def forward(self, x, hidden=None):
        rnn_out, hidden = self.rnn(x, hidden)
        output = torch.tanh(self.output(rnn_out))  # Add activation
        return output, hidden
    
    def get_weights(self):
        """Extract all network weights for analysis"""
        weights = {}
        # Extract PyTorch tensor weights as numpy arrays
        weights['rnn_ih'] = self.rnn.weight_ih_l0.detach().cpu().numpy()
        weights['rnn_hh'] = self.rnn.weight_hh_l0.detach().cpu().numpy()
        weights['output'] = self.output.weight.detach().cpu().numpy()
        return weights

class WormEnvironment:
    """Simple worm environment for 1D and 3D scenarios"""
    def __init__(self, mode='1D'):
        self.mode = mode
        self.reset()
        
    def reset(self):
        if self.mode == '1D':
            self.worm_pos = np.array([0.0])
            self.reward_pos = np.array([5.0])
        else:  # 3D
            self.worm_pos = np.array([0.0, 0.0, 0.0])
            self.reward_pos = np.array([3.0, 2.0, 1.0])
        
        self.step_count = 0
        self.max_steps = 50  # Shorter episodes for better learning
        return self.get_state()
    
    def get_state(self):
        """Get current state: [position, distance_to_reward, concentration]"""
        distance = np.linalg.norm(self.worm_pos - self.reward_pos)
        concentration = 1.0 / (1.0 + distance)
        
        if self.mode == '1D':
            return np.array([self.worm_pos[0], distance, concentration])
        else:  # 3D
            return np.concatenate([self.worm_pos, [distance, concentration]])
    
    def step(self, action):
        """Take action and return next state, reward, done"""
        # Convert RNN output to movement
        if self.mode == '1D':
            movement = action[0] * 0.2  # Scale movement
            self.worm_pos[0] += movement
        else:  # 3D
            movement = action[:3] * 0.2
            self.worm_pos += movement
        
        # Calculate reward based on distance to target
        old_distance = np.linalg.norm(self.worm_pos - movement - self.reward_pos)
        new_distance = np.linalg.norm(self.worm_pos - self.reward_pos)
        
        # Reward for getting closer, penalty for getting farther
        reward = (old_distance - new_distance) * 10.0
        if new_distance < 0.5:
            reward += 10.0  # Bonus for reaching target
        
        self.step_count += 1
        done = self.step_count >= self.max_steps or new_distance < 0.3
        
        return self.get_state(), reward, done

def generate_expert_data(mode='1D', num_episodes=200, max_steps=30):
    """Generate expert demonstration data for imitation learning"""
    episodes = []
    env = WormEnvironment(mode)
    
    for episode in range(num_episodes):
        states, actions, rewards = [], [], []
        state = env.reset()
        
        for step in range(max_steps):
            # Expert policy: move toward reward
            if mode == '1D':
                direction = env.reward_pos[0] - env.worm_pos[0]
                action = np.array([np.sign(direction) * 0.8, 0.0])
            else:  # 3D
                direction = env.reward_pos - env.worm_pos
                direction = direction / (np.linalg.norm(direction) + 1e-6)
                action = np.concatenate([direction * 0.8, [0.0, 0.0]])
            
            # Add some noise for variety
            action += np.random.normal(0, 0.1, size=action.shape)
            action = np.clip(action, -1.0, 1.0)
            
            states.append(state)
            actions.append(action)
            
            next_state, reward, done = env.step(action)
            rewards.append(reward)
            
            state = next_state
            if done:
                # Pad sequences to fixed length
                while len(states) < max_steps:
                    states.append(state)  # Repeat last state
                    actions.append(actions[-1])  # Repeat last action
                    rewards.append(0.0)  # Zero reward for padding
                break
        
        # Ensure all episodes have exactly max_steps length
        states = np.array(states[:max_steps])
        actions = np.array(actions[:max_steps])
        rewards = np.array(rewards[:max_steps])
        
        episodes.append({
            'states': states,
            'actions': actions,
            'rewards': rewards
        })
    
    return episodes

class WormDataset(Dataset):
    """Dataset for training worm control"""
    def __init__(self, mode='1D', num_episodes=200):
        self.mode = mode
        self.episodes = generate_expert_data(mode, num_episodes)
    
    def __len__(self):
        return len(self.episodes)
    
    def __getitem__(self, idx):
        episode = self.episodes[idx]
        return (torch.FloatTensor(episode['states']),
                torch.FloatTensor(episode['actions']),
                torch.FloatTensor(episode['rewards']))

def train_rnn(mode='1D', epochs=100):
    """Train RNN network for worm control using behavioral cloning"""
    print(f"Training RNN for {mode} worm control...")
    
    # Setup
    input_size = 3 if mode == '1D' else 5
    output_size = 2 if mode == '1D' else 5
    model = SimpleRNN(input_size, 16, output_size)
    
    # Extract initial weights
    initial_weights = model.get_weights()
    
    # Create dataset and dataloader
    dataset = WormDataset(mode, num_episodes=300)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    # Training setup
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)  # Higher learning rate
    
    losses = []
    
    print(f"Training for {epochs} epochs...")
    for epoch in range(epochs):
        epoch_loss = 0.0
        
        for batch_idx, (states, actions, rewards) in enumerate(dataloader):
            optimizer.zero_grad()
            
            # Forward pass
            predicted_actions, _ = model(states)
            
            # Behavioral cloning: learn to imitate expert actions
            loss = criterion(predicted_actions, actions)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch}: Loss = {avg_loss:.4f}")
    
    # Extract final weights
    final_weights = model.get_weights()
    
    print(f"Training completed! Final loss: {losses[-1]:.4f}")
    
    return model, initial_weights, final_weights, losses

def analyze_weight_changes(initial_weights, final_weights, mode):
    """Analyze changes in network weights"""
    print(f"\nAnalyzing weight changes for {mode} mode...")
    
    changes = {}
    for key in initial_weights:
        initial = initial_weights[key]
        final = final_weights[key]
        change = final - initial
        changes[key] = {
            'initial': initial,
            'final': final,
            'change': change,
            'magnitude': np.linalg.norm(change),
            'relative_change': np.linalg.norm(change) / (np.linalg.norm(initial) + 1e-8)
        }
        print(f"{key}: Change magnitude = {changes[key]['magnitude']:.4f}, "
              f"Relative change = {changes[key]['relative_change']:.4f}")
    
    return changes

def analyze_correlations(initial_weights, final_weights, mode):
    """Analyze neuron correlation changes"""
    print(f"\nAnalyzing correlation changes for {mode} mode...")
    
    # Focus on hidden-to-hidden weights (recurrent connections)
    initial_hh = initial_weights['rnn_hh']
    final_hh = final_weights['rnn_hh']
    
    # Compute correlation matrices
    initial_corr = np.corrcoef(initial_hh)
    final_corr = np.corrcoef(final_hh)
    
    # Handle NaN values in correlation
    initial_corr = np.nan_to_num(initial_corr)
    final_corr = np.nan_to_num(final_corr)
    
    # Analyze changes
    corr_change = final_corr - initial_corr
    
    print(f"Initial correlation mean: {np.mean(initial_corr):.4f}")
    print(f"Final correlation mean: {np.mean(final_corr):.4f}")
    print(f"Correlation change magnitude: {np.linalg.norm(corr_change):.4f}")
    
    return initial_corr, final_corr, corr_change

def create_visualizations(mode, losses, weight_changes, correlations):
    """Create analysis visualizations"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'RNN Training Analysis - {mode} Mode', fontsize=16)
    
    # Training curve
    axes[0,0].plot(losses)
    axes[0,0].set_title('Training Loss')
    axes[0,0].set_xlabel('Epoch')
    axes[0,0].set_ylabel('Loss')
    axes[0,0].grid(True)
    
    # Weight changes
    layers = list(weight_changes.keys())
    changes = [weight_changes[layer]['magnitude'] for layer in layers]
    axes[0,1].bar(layers, changes)
    axes[0,1].set_title('Weight Change Magnitude by Layer')
    axes[0,1].set_ylabel('Change Magnitude')
    axes[0,1].tick_params(axis='x', rotation=45)
    
    # Initial vs Final correlations
    initial_corr, final_corr, corr_change = correlations
    
    im1 = axes[1,0].imshow(initial_corr, cmap='coolwarm', vmin=-1, vmax=1)
    axes[1,0].set_title('Initial Neuron Correlations')
    plt.colorbar(im1, ax=axes[1,0])
    
    im2 = axes[1,1].imshow(final_corr, cmap='coolwarm', vmin=-1, vmax=1)
    axes[1,1].set_title('Final Neuron Correlations')
    plt.colorbar(im2, ax=axes[1,1])
    
    plt.tight_layout()
    os.makedirs('results', exist_ok=True)
    filename = f'results/rnn_analysis_{mode.lower()}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Visualization saved: {filename}")
    
def test_trained_model(model, mode):
    """Test the trained model in the environment"""
    print(f"\nTesting trained model in {mode} environment...")
    
    env = WormEnvironment(mode)
    state = env.reset()
    
    total_reward = 0
    trajectory = [env.worm_pos.copy()]
    
    for step in range(50):
        # Convert state to tensor and get action
        state_tensor = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            action, _ = model(state_tensor)
            action = action.squeeze().numpy()
        
        # Take action
        next_state, reward, done = env.step(action)
        total_reward += reward
        trajectory.append(env.worm_pos.copy())
        
        state = next_state
        if done:
            break
    
    print(f"Test completed: Total reward = {total_reward:.2f}, Steps = {step+1}")
    print(f"Final distance to reward: {np.linalg.norm(env.worm_pos - env.reward_pos):.2f}")
    
    return total_reward, trajectory

def main():
    """Main research pipeline"""
    print("=== RNN Training for C. elegans Control ===")
    print("Goal: Train RNNs for 1D and 3D scenarios and analyze weight/correlation changes")
    
    results = {}
    
    # Train for both scenarios
    for mode in ['1D', '3D']:
        print(f"\n{'='*50}")
        print(f"TRAINING {mode} SCENARIO")
        print(f"{'='*50}")
        
        # Train the model
        model, initial_weights, final_weights, losses = train_rnn(mode, epochs=50)
        
        # Analyze changes
        weight_changes = analyze_weight_changes(initial_weights, final_weights, mode)
        correlations = analyze_correlations(initial_weights, final_weights, mode)
        
        # Test trained model
        test_reward, trajectory = test_trained_model(model, mode)
        
        # Create visualizations
        create_visualizations(mode, losses, weight_changes, correlations)
        
        # Store results
        results[mode] = {
            'model': model,
            'initial_weights': initial_weights,
            'final_weights': final_weights,
            'weight_changes': weight_changes,
            'correlations': correlations,
            'losses': losses,
            'test_reward': test_reward,
            'trajectory': trajectory
        }
    
    # Summary comparison
    print(f"\n{'='*50}")
    print("SUMMARY COMPARISON")
    print(f"{'='*50}")
    
    for mode in ['1D', '3D']:
        result = results[mode]
        print(f"\n{mode} Results:")
        print(f"  Final training loss: {result['losses'][-1]:.4f}")
        print(f"  Test reward: {result['test_reward']:.2f}")
        
        # Weight change summary
        total_change = sum(result['weight_changes'][layer]['magnitude'] 
                          for layer in result['weight_changes'])
        print(f"  Total weight change: {total_change:.4f}")
        
        # Correlation change summary
        initial_corr, final_corr, _ = result['correlations']
        corr_change = np.mean(final_corr) - np.mean(initial_corr)
        print(f"  Correlation change: {corr_change:.4f}")
    
    print(f"\nAnalysis complete! Check the generated PNG files for detailed visualizations.")

if __name__ == '__main__':
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    main() 