#!/usr/bin/env python3
"""
Multi-Reward Seeking C. elegans Simulation

Complete standalone implementation featuring:
- 3D MuJoCo simulation with fixed API
- Multiple reward collection with chemotaxis navigation  
- Trajectory tracking and visualization
- Optimal path calculation and comparison
- Performance metrics and timing analysis
- Publication-quality plots

Usage:
    python multi_reward_elegans_demo.py [--num_rewards 4] [--episodes 3] [--headless]
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for macOS compatibility
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import mujoco
import mujoco.viewer
import time
import os
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from itertools import permutations


@dataclass
class RewardInfo:
    """Information about a reward target"""
    id: int
    position: np.ndarray
    collected: bool = False
    collection_time: float = 0.0
    collection_step: int = 0


@dataclass
class TrajectoryPoint:
    """Single point in the worm's trajectory"""
    time: float
    step: int
    position: np.ndarray
    velocity: np.ndarray
    rewards_collected: List[int]
    distance_to_nearest: float


class OptimalPathCalculator:
    """Calculate optimal paths for multi-reward collection"""
    
    def __init__(self, reward_positions: np.ndarray, start_position: np.ndarray):
        self.reward_positions = reward_positions
        self.start_position = start_position
        
    def calculate_tsp_optimal(self) -> Tuple[List[int], float, List[np.ndarray]]:
        """Calculate optimal order using TSP (brute force for small numbers)"""
        n_rewards = len(self.reward_positions)
        
        if n_rewards <= 1:
            return [0] if n_rewards == 1 else [], 0.0, []
        
        # Generate all possible orders
        min_distance = float('inf')
        best_order = None
        
        for order in permutations(range(n_rewards)):
            distance = self._calculate_path_distance(order)
            if distance < min_distance:
                min_distance = distance
                best_order = list(order)
        
        # Generate optimal path waypoints
        optimal_path = self._generate_path_waypoints(best_order)
        
        return best_order, min_distance, optimal_path
    
    def _calculate_path_distance(self, order: Tuple[int]) -> float:
        """Calculate total distance for a given order"""
        total_distance = 0.0
        current_pos = self.start_position
        
        for reward_idx in order:
            reward_pos = self.reward_positions[reward_idx]
            distance = np.linalg.norm(reward_pos - current_pos)
            total_distance += distance
            current_pos = reward_pos
            
        return total_distance
    
    def _generate_path_waypoints(self, order: List[int], points_per_segment: int = 20) -> List[np.ndarray]:
        """Generate smooth path waypoints following the optimal order"""
        waypoints = []
        current_pos = self.start_position.copy()
        
        for reward_idx in order:
            target_pos = self.reward_positions[reward_idx]
            
            # Generate interpolated points
            for i in range(points_per_segment):
                alpha = i / (points_per_segment - 1)
                point = current_pos + alpha * (target_pos - current_pos)
                waypoints.append(point)
                
            current_pos = target_pos
            
        return waypoints


class ChemotaxisController:
    """Chemotaxis-based navigation controller for C. elegans in 3D"""
    def __init__(self, n_segments: int = 14):
        self.n_segments = n_segments
        self.base_frequency = 2.0
        self.base_amplitude = 1.2
        self.chemotaxis_strength = 2.0
    def generate_action(self, time_step: float, worm_pos: np.ndarray, active_rewards: List[RewardInfo]) -> np.ndarray:
        t = time_step * 0.02
        omega = 2 * np.pi * self.base_frequency
        wavelength = 2.0
        actions = []
        for i in range(self.n_segments):
            phase = -2 * np.pi * wavelength * i / self.n_segments
            angle = self.base_amplitude * np.sin(omega * t + phase)
            actions.append(angle)
        # 3D bias
        if not active_rewards:
            return np.array(actions * 3)  # 3 actuators per segment
        nearest_reward = min(active_rewards, key=lambda r: np.linalg.norm(r.position - worm_pos))
        direction = (nearest_reward.position - worm_pos) / (np.linalg.norm(nearest_reward.position - worm_pos) + 1e-6)
        # Strong bias in all 3 axes
        bias_strength = 2.0
        bias = bias_strength * direction
        # Apply bias to each axis actuator
        actions_x = np.array(actions) + bias[0]
        actions_y = np.array(actions) + bias[1]
        actions_z = np.array(actions) + bias[2]
        # Stack for all actuators (x, y, z for each segment)
        return np.concatenate([actions_x, actions_y, actions_z])


class WormSimulation:
    """Main simulation class for multi-reward C. elegans"""
    
    def __init__(self, reward_positions: List[Tuple[float, float, float]] = None,
                 num_episodes: int = 3, render: bool = True, save_results: bool = True):
        
        self.num_episodes = num_episodes
        self.render = render
        self.save_results = save_results
        
        # Default reward configuration - MUCH closer for easy collection
        if reward_positions is None:
            self.reward_positions = [
                (1.5, 1.0, 0.5),   # T1 - Close
                (1.5, -1.0, 0.5),  # T2 - Close  
                (-1.0, 1.0, 0.5),  # T3 - Close
                (-1.0, -1.0, 0.5)  # T4 - Close
            ]
        else:
            self.reward_positions = reward_positions
            
        self.start_position = np.array([0.0, 0.0, 0.0])
        self.reward_radius = 0.8  # Bigger collection radius
        
        # Initialize components
        self.controller = ChemotaxisController()
        self.optimal_calculator = OptimalPathCalculator(
            np.array(self.reward_positions), self.start_position)
        
        # Results storage
        self.episode_results = []
        self.results_dir = "multi_reward_results"
        
        if self.save_results:
            os.makedirs(self.results_dir, exist_ok=True)
            
        # Create MuJoCo model
        self.model, self.data = self._create_mujoco_model()
        
    def _create_mujoco_model(self):
        """Create MuJoCo model with multiple rewards"""
        xml_content = self._generate_xml()
        model = mujoco.MjModel.from_xml_string(xml_content)
        data = mujoco.MjData(model)
        return model, data
        
    def _generate_xml(self) -> str:
        """Generate XML for MuJoCo simulation with multiple rewards and 3D movement"""
        xml_content = '''
        <mujoco model="multi_reward_swimmer_3d">
          <compiler angle="degree" coordinate="local" inertiafromgeom="true"/>
          <option density="4000" integrator="RK4" timestep="0.02" viscosity="0.1"/>
          <default>
            <geom conaffinity="1" condim="3" contype="1" material="geom"/>
            <joint armature='0.2' damping='0.1'/>
          </default>
          <asset>
            <texture builtin="gradient" height="100" rgb1="0.1 0.2 0.3" rgb2="0.05 0.1 0.15" type="skybox" width="100"/>
            <texture builtin="flat" height="1278" mark="cross" markrgb="0.3 0.3 0.3" name="texgeom" random="0.01" rgb1="0.15 0.25 0.35" rgb2="0.1 0.2 0.3" type="cube" width="127"/>
            <texture builtin="flat" height="100" name="texplane" rgb1="0.1 0.15 0.2" type="2d" width="100"/>
            <material name="MatPlane" reflectance="0.3" shininess="0.2" specular="0.2" texture="texplane"/>
            <material name="geom" texture="texgeom" texuniform="true"/>
            <material name="reward_active" rgba="1 0.3 0.3 0.9"/>
            <material name="reward_collected" rgba="0.3 1 0.3 0.7"/>
          </asset>
          <worldbody>
            <light cutoff="100" diffuse="0.8 0.8 0.8" dir="-0 0 -1.3" directional="true" exponent="1" pos="0 0 2" specular=".2 .2 .2"/>
            <!-- 3D box container -->
            <geom type="box" size="4 4 2" pos="0 0 1" rgba="0.1 0.2 0.3 0.1" contype="0" conaffinity="0"/>
            <geom conaffinity="0" condim="3" material="MatPlane" name="floor" pos="0 0 0" rgba="0.1 0.15 0.2 1" size="4 4 0.1" type="plane"/>
        '''
        # Add reward spheres in 3D
        for i, pos in enumerate(self.reward_positions):
            xml_content += f'''
            <geom name="reward_{i}" pos="{pos[0]} {pos[1]} {pos[2]}" size="0.25" type="sphere" 
                  material="reward_active" contype="0" conaffinity="0"/>
            '''
        xml_content += '''
            <!-- 3D C. elegans swimmer with 14 segments, each with ball joint for 3D movement -->
            <body name="head" pos="0 0 1">
              <camera name="track" mode="trackcom" pos="0 -8 4" xyaxes="1 0 0 0 1 1"/>
              <geom density="1000" fromto="0.3 0 0 0 0 0" size="0.08" type="capsule" 
                    rgba="0.8 0.9 1.0 1" contype="2" conaffinity="1"/>
              <joint axis="1 0 0" name="root_x" pos="0 0 0" type="slide"/>
              <joint axis="0 1 0" name="root_y" pos="0 0 0" type="slide"/>
              <joint axis="0 0 1" name="root_z" pos="0 0 0" type="slide"/>
              <joint axis="0 0 1" name="root_rot" pos="0 0 0" type="hinge"/>
        '''
        segment_length = 0.2
        for i in range(14):
            segment_name = f"segment_{i+1}"
            next_pos = f"{-segment_length} 0 0"
            joint_name = f"joint_{i+1}"
            joint_name_y = f"joint_y_{i+1}"
            joint_name_z = f"joint_z_{i+1}"
            joint_range = "-35 35"
            xml_content += f'''
              <body name="{segment_name}" pos="{next_pos}">
                <geom density="800" fromto="0 0 0 {-segment_length} 0 0" size="0.06" type="capsule" 
                      rgba="0.7 0.85 1.0 1" contype="2" conaffinity="1"/>
                <joint axis="0 0 1" limited="true" name="{joint_name}" pos="0 0 0" 
                       range="{joint_range}" type="hinge"/>
                <joint axis="0 1 0" limited="true" name="{joint_name_y}" pos="0 0 0" 
                       range="{joint_range}" type="hinge"/>
                <joint axis="1 0 0" limited="true" name="{joint_name_z}" pos="0 0 0" 
                       range="{joint_range}" type="hinge"/>
            '''
        for i in range(14):
            xml_content += "  </body>\n"
        xml_content += '''
            </body>
          </worldbody>
          <actuator>
        '''
        for i in range(14):
            joint_name = f"joint_{i+1}"
            joint_name_y = f"joint_y_{i+1}"
            joint_name_z = f"joint_z_{i+1}"
            xml_content += f'''    <motor ctrllimited="true" ctrlrange="-1 1" gear="80.0" joint="{joint_name}"/>\n'''
            xml_content += f'''    <motor ctrllimited="true" ctrlrange="-1 1" gear="80.0" joint="{joint_name_y}"/>\n'''
            xml_content += f'''    <motor ctrllimited="true" ctrlrange="-1 1" gear="80.0" joint="{joint_name_z}"/>\n'''
        xml_content += '''
          </actuator>
        </mujoco>
        '''
        return xml_content
        
    def run_simulation(self) -> List[Dict]:
        """Run complete multi-reward simulation"""
        print(f"Multi-Reward C. elegans Simulation")
        print(f"Rewards: {len(self.reward_positions)} targets")
        print(f"Episodes: {self.num_episodes}")
        print("=" * 50)
        
        # Calculate optimal path
        optimal_order, optimal_distance, optimal_path = self.optimal_calculator.calculate_tsp_optimal()
        print(f"✓ Optimal path: distance = {optimal_distance:.2f}")
        print(f"  Order: {[f'T{i+1}' for i in optimal_order]}")
        
        for episode in range(self.num_episodes):
            print(f"\n--- Episode {episode + 1}/{self.num_episodes} ---")
            result = self._run_episode(episode, optimal_path, optimal_order, optimal_distance)
            self.episode_results.append(result)
            self._print_episode_summary(result)
            
        # Generate analysis
        if self.save_results:
            self._generate_analysis()
            
        return self.episode_results
    
    def _run_episode(self, episode_num: int, optimal_path: List[np.ndarray], optimal_order: List[int], optimal_distance: float) -> Dict:
        """Run a single episode"""
        
        # Reset simulation
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[0] = 0  # x position
        self.data.qpos[1] = 0  # y position  
        self.data.qpos[2] = 0  # rotation
        
        # Set initial velocities to zero for stable start
        self.data.qvel[:] = 0
        
        # Forward dynamics to ensure proper initial state
        mujoco.mj_forward(self.model, self.data)
        
        # Initialize rewards
        rewards = [RewardInfo(i, np.array(pos)) for i, pos in enumerate(self.reward_positions)]
        active_rewards = rewards.copy()
        
        # Trajectory tracking
        trajectory = []
        episode_start_time = time.time()
        step_count = 0
        max_steps = 5000  # More time to collect rewards
        
        print(f"  Starting episode with {len(active_rewards)} rewards...")
        
        if self.render:
            with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
                # Set 3D perspective camera view
                viewer.cam.lookat[:] = [0, 0, 0.5]  # Look at worm height
                viewer.cam.distance = 8
                viewer.cam.elevation = -30  # Better 3D angle
                viewer.cam.azimuth = 135   # Side perspective
                
                print(f" 3D MuJoCo viewer opened!")
                
                while viewer.is_running() and step_count < max_steps and active_rewards:
                    step_start = time.time()
                    
                    # Get worm position
                    worm_pos = self.data.xpos[1]  # head position
                    
                    # Generate control action
                    action = self.controller.generate_action(step_count, worm_pos, active_rewards)
                    self.data.ctrl[:] = action
                    
                    # Step simulation
                    mujoco.mj_step(self.model, self.data)
                    
                    # Update camera to follow worm (dynamic tracking)
                    if step_count % 20 == 0:  # Update camera position every 20 steps
                        viewer.cam.lookat[:] = [worm_pos[0], worm_pos[1], 0.5]
                    
                    viewer.sync()
                    
                    # Check reward collection with debug output
                    collected_this_step = []
                    for reward in active_rewards[:]:  # Copy list to modify during iteration
                        distance = np.linalg.norm(worm_pos - reward.position)
                        
                        # Debug: Print distance every 100 steps for nearest reward
                        if step_count % 100 == 0 and reward == active_rewards[0]:
                            print(f"    Step {step_count}: Worm at {worm_pos[:2]}, Distance to T{reward.id+1}: {distance:.2f}")
                        
                        if distance <= self.reward_radius:
                            reward.collected = True
                            reward.collection_time = time.time() - episode_start_time
                            reward.collection_step = step_count
                            collected_this_step.append(reward.id)
                            active_rewards.remove(reward)
                            
                            # Update reward visual (change color)
                            geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"reward_{reward.id}")
                            if geom_id >= 0:
                                self.model.geom_rgba[geom_id] = [0.3, 1.0, 0.3, 0.7]  # Green when collected
                    
                    # Record trajectory point
                    nearest_distance = min([np.linalg.norm(worm_pos - r.position) for r in active_rewards]) if active_rewards else 0
                    
                    trajectory.append(TrajectoryPoint(
                        time=time.time() - episode_start_time,
                        step=step_count,
                        position=worm_pos.copy(),
                        velocity=self.data.qvel[:3].copy() if len(self.data.qvel) >= 3 else np.zeros(3),
                        rewards_collected=collected_this_step,
                        distance_to_nearest=nearest_distance
                    ))
                    
                    if collected_this_step:
                        print(f"    Step {step_count}: Collected T{collected_this_step[0]+1}! ({len(active_rewards)} remaining)")
                    
                    step_count += 1
                    
                    # Real-time control
                    time_until_next = self.model.opt.timestep - (time.time() - step_start)
                    if time_until_next > 0:
                        time.sleep(time_until_next)
        
        else:
            # Headless simulation
            while step_count < max_steps and active_rewards:
                worm_pos = self.data.xpos[1]
                action = self.controller.generate_action(step_count, worm_pos, active_rewards)
                self.data.ctrl[:] = action
                mujoco.mj_step(self.model, self.data)
                
                # Check collections and record trajectory (same logic as above)
                collected_this_step = []
                for reward in active_rewards[:]:
                    distance = np.linalg.norm(worm_pos - reward.position)
                    if distance <= self.reward_radius:
                        reward.collected = True
                        reward.collection_time = time.time() - episode_start_time
                        reward.collection_step = step_count
                        collected_this_step.append(reward.id)
                        active_rewards.remove(reward)
                
                nearest_distance = min([np.linalg.norm(worm_pos - r.position) for r in active_rewards]) if active_rewards else 0
                
                trajectory.append(TrajectoryPoint(
                    time=time.time() - episode_start_time,
                    step=step_count,
                    position=worm_pos.copy(),
                    velocity=self.data.qvel[:3].copy() if len(self.data.qvel) >= 3 else np.zeros(3),
                    rewards_collected=collected_this_step,
                    distance_to_nearest=nearest_distance
                ))
                
                step_count += 1
        
        total_time = time.time() - episode_start_time
        collected_rewards = [r for r in rewards if r.collected]
        
        return {
            'episode_num': episode_num,
            'total_time': total_time,
            'steps': step_count,
            'rewards_collected': len(collected_rewards),
            'total_rewards': len(rewards),
            'collection_efficiency': len(collected_rewards) / len(rewards),
            'all_collected': len(collected_rewards) == len(rewards),
            'rewards_info': rewards,
            'trajectory': trajectory,
            'optimal_path': optimal_path,
            'optimal_order': optimal_order,
            'optimal_distance': optimal_distance
        }
    
    def _print_episode_summary(self, result: Dict):
        """Print episode summary"""
        print(f"✓ Episode {result['episode_num'] + 1} completed!")
        print(f"  Time: {result['total_time']:.2f}s")
        print(f"  Steps: {result['steps']}")
        print(f"  Collected: {result['rewards_collected']}/{result['total_rewards']}")
        print(f"  Efficiency: {result['collection_efficiency']:.1%}")
        
        if result['all_collected']:
            print("ALL REWARDS COLLECTED!")
            
            # Print collection order and timing
            collected = [r for r in result['rewards_info'] if r.collected]
            collected.sort(key=lambda x: x.collection_time)
            print("    Collection order:", [f"T{r.id+1}({r.collection_time:.1f}s)" for r in collected])
    
    def _generate_analysis(self):
        """Generate comprehensive analysis and visualizations"""
        print(f"\nGenerating Analysis...")
        # Individual episode plots
        for result in self.episode_results:
            self._create_episode_plots(result)
        # Remove multi-episode comparison analysis
        self._create_summary_report()
        print(f"Analysis complete! Results in: {self.results_dir}/")
    
    def _create_episode_plots(self, result: Dict):
        """Create plots for a single episode"""
        episode_num = result['episode_num'] + 1
        
        # Extract trajectory data
        positions = np.array([p.position for p in result['trajectory']])
        times = np.array([p.time for p in result['trajectory']])
        
        # 3D Trajectory Plot
        fig = plt.figure(figsize=(15, 5))
        
        # 3D view
        ax1 = fig.add_subplot(131, projection='3d')
        
        # Plot trajectory
        ax1.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
                'b-', linewidth=2, alpha=0.8, label='Actual Path')
        
        # Plot optimal path if available
        if result['optimal_path']:
            opt_pos = np.array(result['optimal_path'])
            ax1.plot(opt_pos[:, 0], opt_pos[:, 1], opt_pos[:, 2], 
                    'r--', linewidth=2, alpha=0.6, label='Optimal Path')
        
        # Plot rewards
        for reward in result['rewards_info']:
            color = 'green' if reward.collected else 'red'
            ax1.scatter(*reward.position, s=200, c=color, alpha=0.8)
            ax1.text(reward.position[0], reward.position[1], reward.position[2] + 0.3, 
                    f'T{reward.id+1}', fontsize=10, ha='center')
        
        # Start position
        ax1.scatter(*positions[0], s=150, c='blue', marker='*', label='Start')
        
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Z (m)')
        ax1.set_title(f'Episode {episode_num}: 3D Trajectory')
        ax1.legend()
        
        # 2D Top view  
        ax2 = fig.add_subplot(132)
        ax2.plot(positions[:, 0], positions[:, 1], 'b-', linewidth=2, alpha=0.8, label='Actual')
        
        if result['optimal_path']:
            opt_pos = np.array(result['optimal_path'])
            ax2.plot(opt_pos[:, 0], opt_pos[:, 1], 'r--', linewidth=2, alpha=0.6, label='Optimal')
        
        for reward in result['rewards_info']:
            color = 'green' if reward.collected else 'red'
            ax2.scatter(reward.position[0], reward.position[1], s=200, c=color, alpha=0.8)
            ax2.text(reward.position[0] + 0.2, reward.position[1] + 0.2, 
                    f'T{reward.id+1}', fontsize=10)
        
        ax2.scatter(positions[0, 0], positions[0, 1], s=150, c='blue', marker='*', label='Start')
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Y (m)')
        ax2.set_title(f'Episode {episode_num}: Top View')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.axis('equal')
        
        # Performance metrics
        ax3 = fig.add_subplot(133)
        
        # Distance to nearest reward over time
        distances = [p.distance_to_nearest for p in result['trajectory']]
        ax3.plot(times, distances, 'g-', linewidth=2, label='Distance to Nearest')
        
        # Mark collection events
        for reward in result['rewards_info']:
            if reward.collected:
                ax3.axvline(reward.collection_time, color='red', linestyle='--', alpha=0.7)
                ax3.text(reward.collection_time, max(distances) * 0.8, f'T{reward.id+1}', 
                        rotation=90, ha='center', fontsize=8)
        
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Distance (m)')
        ax3.set_title(f'Episode {episode_num}: Collection Progress')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/episode_{episode_num}_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _format_rewards(self) -> str:
        """Format reward positions for display"""
        lines = []
        for i, pos in enumerate(self.reward_positions):
            lines.append(f"T{i+1}: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})")
        return "\n".join(lines)
    
    def _create_summary_report(self):
        """Create detailed text report"""
        report_path = f"{self.results_dir}/simulation_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("MULTI-REWARD C. ELEGANS SIMULATION REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("Simulation Configuration:\n")
            f.write(f"- Episodes: {self.num_episodes}\n")
            f.write(f"- Rewards: {len(self.reward_positions)}\n")
            f.write(f"- Start Position: {self.start_position}\n")
            f.write(f"- Reward Radius: {self.reward_radius}m\n")
            f.write(f"- Navigation: Chemotaxis-based\n\n")
            
            f.write("Reward Positions:\n")
            for i, pos in enumerate(self.reward_positions):
                f.write(f"  T{i+1}: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})\n")
            f.write("\n")
            
            # Optimal path analysis
            if self.episode_results:
                opt_order = self.episode_results[0]['optimal_order']
                opt_distance = self.episode_results[0]['optimal_distance']
                f.write(f"Optimal Path Analysis:\n")
                f.write(f"- Order: {[f'T{i+1}' for i in opt_order]}\n")
                f.write(f"- Distance: {opt_distance:.2f}m\n\n")
            
            # Episode results
            f.write("Episode Results:\n")
            f.write("-" * 30 + "\n")
            
            for result in self.episode_results:
                episode_num = result['episode_num'] + 1
                f.write(f"Episode {episode_num}:\n")
                f.write(f"  Time: {result['total_time']:.2f}s\n")
                f.write(f"  Steps: {result['steps']}\n")
                f.write(f"  Collected: {result['rewards_collected']}/{result['total_rewards']}\n")
                f.write(f"  Efficiency: {result['collection_efficiency']:.1%}\n")
                f.write(f"  Success: {result['all_collected']}\n")
                
                # Collection timeline
                collected_rewards = [r for r in result['rewards_info'] if r.collected]
                if collected_rewards:
                    collected_rewards.sort(key=lambda x: x.collection_time)
                    f.write(f"  Collection Order:\n")
                    for r in collected_rewards:
                        f.write(f"    T{r.id+1} at {r.collection_time:.1f}s (step {r.collection_step})\n")
                f.write(f"\n")
            
            # Summary statistics
            times = [r['total_time'] for r in self.episode_results]
            efficiencies = [r['collection_efficiency'] for r in self.episode_results]
            success_rate = sum(1 for r in self.episode_results if r['all_collected']) / len(self.episode_results)
            
            f.write("Summary Statistics:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Average Time: {np.mean(times):.2f} ± {np.std(times):.2f}s\n")
            f.write(f"Average Efficiency: {np.mean(efficiencies):.1%} ± {np.std(efficiencies):.1%}\n")
            f.write(f"Success Rate: {success_rate:.1%}\n")
            f.write(f"Best Episode: {np.argmax(efficiencies) + 1}\n")
            f.write(f"Fastest Episode: {np.argmin(times) + 1}\n")
        
        print(f"✓ Report saved: {report_path}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Multi-Reward C. elegans Simulation')
    parser.add_argument('--num_rewards', type=int, default=4, 
                        help='Number of rewards (default: 4)')
    parser.add_argument('--episodes', type=int, default=3,
                        help='Number of episodes (default: 3)')
    parser.add_argument('--headless', action='store_true',
                        help='Run without rendering')
    
    args = parser.parse_args()
    
    print("Multi-Reward C. elegans Digital Twin")
    print("Chemotaxis-Based Navigation with Trajectory Analysis")
    print()
    
    # Custom reward positions
    if args.num_rewards == 4:
        reward_positions = [
            (3.0, 2.5, 0.5),   # T1
            (3.0, -2.5, 0.5),  # T2
            (-2.5, 2.0, 0.5),  # T3
            (-2.5, -2.0, 0.5)  # T4
        ]
    elif args.num_rewards == 6:
        reward_positions = [
            (3.0, 2.5, 0.5), (3.0, -2.5, 0.5),
            (-2.5, 2.0, 0.5), (-2.5, -2.0, 0.5),
            (0.0, 3.0, 0.5), (0.0, -3.0, 0.5)
        ]
    else:
        # Generate random positions
        np.random.seed(42)
        reward_positions = [(np.random.uniform(-3, 3), np.random.uniform(-3, 3), 0.5) 
                          for _ in range(args.num_rewards)]
    
    # Create and run simulation
    simulation = WormSimulation(
        reward_positions=reward_positions,
        num_episodes=args.episodes,
        render=not args.headless,
        save_results=True
    )
    
    # Run simulation
    results = simulation.run_simulation()
    
    print("\nSimulation Complete!")
    return results


if __name__ == "__main__":
    main() 