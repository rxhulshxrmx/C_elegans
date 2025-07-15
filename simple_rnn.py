#!/usr/bin/env python3
"""
Simple RNN Demo for C. elegans Weight Analysis
Demonstrates clear weight and correlation changes before/after training
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import copy
import os

class SimpleRNN(nn.Module):
    """Simple RNN for demonstration"""
    def __init__(self, input_size=2, hidden_size=8, output_size=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.output = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        out, _ = self.rnn(x)
        out = torch.tanh(self.output(out))
        return out

def extract_weights(model):
    """Extract weights as numpy arrays"""
    weights = {}
    with torch.no_grad():
        # RNN input-to-hidden weights
        weights['rnn_ih'] = model.rnn.weight_ih_l0.clone().cpu().numpy()
        # RNN hidden-to-hidden weights  
        weights['rnn_hh'] = model.rnn.weight_hh_l0.clone().cpu().numpy()
        # Output layer weights
        weights['output'] = model.output.weight.clone().cpu().numpy()
    return weights

def create_1d_task_data(num_samples=1000, seq_length=20):
    """Create 1D navigation task data"""
    X, y = [], []
    
    for _ in range(num_samples):
        # Random starting position and target
        start_pos = np.random.uniform(-5, 5)
        target_pos = np.random.uniform(-5, 5)
        
        sequence_x = []
        sequence_y = []
        current_pos = start_pos
        
        for step in range(seq_length):
            # Input: [current_position, distance_to_target]
            distance = target_pos - current_pos
            inputs = np.array([current_pos, distance])
            sequence_x.append(inputs)
            
            # Target: optimal action (move toward target)
            optimal_action = np.sign(distance) * 0.5
            sequence_y.append([optimal_action])
            
            # Simulate movement
            current_pos += optimal_action * 0.1
        
        X.append(sequence_x)
        y.append(sequence_y)
    
    return torch.FloatTensor(X), torch.FloatTensor(y)

def create_3d_task_data(num_samples=1000, seq_length=15):
    """Create 3D navigation task data"""
    X, y = [], []
    
    for _ in range(num_samples):
        # Random starting position and target in 3D
        start_pos = np.random.uniform(-3, 3, 3)
        target_pos = np.random.uniform(-3, 3, 3)
        
        sequence_x = []
        sequence_y = []
        current_pos = start_pos.copy()
        
        for step in range(seq_length):
            # Input: [current_position, distance_vector] (5D total)
            distance_vec = target_pos - current_pos
            inputs = np.concatenate([current_pos, distance_vec[:2]])  # Use 2D for simplicity
            sequence_x.append(inputs)
            
            # Target: optimal action (move toward target)
            distance_norm = np.linalg.norm(distance_vec)
            if distance_norm > 0:
                optimal_action = distance_vec[0] / distance_norm * 0.5  # Just X component
            else:
                optimal_action = 0.0
            sequence_y.append([optimal_action])
            
            # Simulate movement
            current_pos[0] += optimal_action * 0.1
        
        X.append(sequence_x)
        y.append(sequence_y)
    
    return torch.FloatTensor(X), torch.FloatTensor(y)

def train_model(model, X, y, epochs=100, lr=0.01):
    """Train the model and return training history"""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    losses = []
    print(f"Training for {epochs} epochs...")
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Forward pass
        predictions = model(X)
        loss = criterion(predictions, y)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.6f}")
    
    print(f"Training completed! Final loss: {losses[-1]:.6f}")
    return losses

def analyze_weight_changes(initial_weights, final_weights):
    """Analyze and visualize weight changes"""
    print("\nWeight Change Analysis:")
    print("-" * 40)
    
    changes = {}
    for key in initial_weights:
        initial = initial_weights[key]
        final = final_weights[key]
        change = final - initial
        
        change_magnitude = np.linalg.norm(change)
        initial_magnitude = np.linalg.norm(initial)
        relative_change = change_magnitude / (initial_magnitude + 1e-8)
        
        changes[key] = {
            'initial': initial,
            'final': final,
            'change': change,
            'magnitude': change_magnitude,
            'relative': relative_change
        }
        
        print(f"{key}:")
        print(f"  Shape: {initial.shape}")
        print(f"  Change magnitude: {change_magnitude:.6f}")
        print(f"  Relative change: {relative_change:.6f}")
        print()
    
    return changes

def analyze_correlations(initial_weights, final_weights):
    """Analyze correlation changes in hidden layer"""
    print("Correlation Analysis:")
    print("-" * 40)
    
    # Focus on hidden-to-hidden weights
    initial_hh = initial_weights['rnn_hh']
    final_hh = final_weights['rnn_hh']
    
    # Compute correlations
    initial_corr = np.corrcoef(initial_hh)
    final_corr = np.corrcoef(final_hh)
    
    # Handle NaN values
    initial_corr = np.nan_to_num(initial_corr)
    final_corr = np.nan_to_num(final_corr)
    
    print(f"Initial correlation mean: {np.mean(initial_corr):.6f}")
    print(f"Final correlation mean: {np.mean(final_corr):.6f}")
    print(f"Correlation std change: {np.std(final_corr) - np.std(initial_corr):.6f}")
    
    return initial_corr, final_corr

def create_analysis_plots(mode, losses, weight_changes, correlations):
    """Create comprehensive analysis plots"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'RNN Analysis - {mode} Navigation Task', fontsize=16)
    
    # Training curve
    axes[0,0].plot(losses)
    axes[0,0].set_title('Training Loss')
    axes[0,0].set_xlabel('Epoch')
    axes[0,0].set_ylabel('MSE Loss')
    axes[0,0].grid(True)
    
    # Weight change magnitudes
    layers = list(weight_changes.keys())
    magnitudes = [weight_changes[layer]['magnitude'] for layer in layers]
    axes[0,1].bar(layers, magnitudes)
    axes[0,1].set_title('Weight Change Magnitude')
    axes[0,1].set_ylabel('L2 Norm')
    axes[0,1].tick_params(axis='x', rotation=45)
    
    # Relative weight changes
    relative_changes = [weight_changes[layer]['relative'] for layer in layers]
    axes[0,2].bar(layers, relative_changes)
    axes[0,2].set_title('Relative Weight Changes')
    axes[0,2].set_ylabel('Relative Change')
    axes[0,2].tick_params(axis='x', rotation=45)
    
    # Initial correlations
    initial_corr, final_corr = correlations
    im1 = axes[1,0].imshow(initial_corr, cmap='coolwarm', vmin=-1, vmax=1)
    axes[1,0].set_title('Initial Neuron Correlations')
    plt.colorbar(im1, ax=axes[1,0])
    
    # Final correlations
    im2 = axes[1,1].imshow(final_corr, cmap='coolwarm', vmin=-1, vmax=1)
    axes[1,1].set_title('Final Neuron Correlations')
    plt.colorbar(im2, ax=axes[1,1])
    
    # Correlation difference
    corr_diff = final_corr - initial_corr
    vmax = max(abs(corr_diff.min()), abs(corr_diff.max()))
    im3 = axes[1,2].imshow(corr_diff, cmap='RdBu', vmin=-vmax, vmax=vmax)
    axes[1,2].set_title('Correlation Changes')
    plt.colorbar(im3, ax=axes[1,2])
    
    plt.tight_layout()
    os.makedirs('results', exist_ok=True)
    filename = f'results/simple_rnn_{mode.lower()}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Analysis plot saved: {filename}")

def run_experiment(mode='1D'):
    """Run a complete experiment for given mode"""
    print(f"\n{'='*50}")
    print(f"RUNNING {mode} EXPERIMENT")
    print(f"{'='*50}")
    
    # Create model
    if mode == '1D':
        model = SimpleRNN(input_size=2, hidden_size=8, output_size=1)
        X, y = create_1d_task_data(num_samples=500, seq_length=20)
    else:  # 3D
        model = SimpleRNN(input_size=5, hidden_size=8, output_size=1)
        X, y = create_3d_task_data(num_samples=500, seq_length=15)
    
    print(f"Created {mode} navigation task with {X.shape[0]} samples")
    print(f"Input shape: {X.shape}, Output shape: {y.shape}")
    
    # Extract initial weights (deep copy to avoid reference issues)
    initial_weights = extract_weights(model)
    
    # Train the model
    losses = train_model(model, X, y, epochs=150, lr=0.01)
    
    # Extract final weights
    final_weights = extract_weights(model)
    
    # Analysis
    weight_changes = analyze_weight_changes(initial_weights, final_weights)
    correlations = analyze_correlations(initial_weights, final_weights)
    
    # Create plots
    create_analysis_plots(mode, losses, weight_changes, correlations)
    
    return {
        'model': model,
        'losses': losses,
        'weight_changes': weight_changes,
        'correlations': correlations,
        'initial_weights': initial_weights,
        'final_weights': final_weights
    }

def main():
    """Main demonstration"""
    print("=== Simple RNN Weight Analysis Demo ===")
    print("Demonstrating clear weight and correlation changes")
    
    # Set seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    results = {}
    
    # Run experiments for both modes
    for mode in ['1D', '3D']:
        results[mode] = run_experiment(mode)
    
    # Summary comparison
    print(f"\n{'='*50}")
    print("SUMMARY COMPARISON")
    print(f"{'='*50}")
    
    for mode in ['1D', '3D']:
        result = results[mode]
        print(f"\n{mode} Results:")
        print(f"  Final loss: {result['losses'][-1]:.6f}")
        
        # Total weight change
        total_change = sum(result['weight_changes'][layer]['magnitude'] 
                          for layer in result['weight_changes'])
        print(f"  Total weight change: {total_change:.6f}")
        
        # Average relative change
        avg_relative = np.mean([result['weight_changes'][layer]['relative'] 
                               for layer in result['weight_changes']])
        print(f"  Average relative change: {avg_relative:.6f}")
    
    print(f"\nDemo complete! Generated analysis plots show clear weight and correlation changes.")

if __name__ == '__main__':
    main() 