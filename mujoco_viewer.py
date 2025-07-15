#!/usr/bin/env python3
"""
C. elegans MuJoCo Simulation Viewer
Shows the C. elegans swimmer in MuJoCo with reward-based behavior
"""

import mujoco
import mujoco.viewer
import numpy as np
import time
import os



def create_dynamic_swimmer_xml():
    """Create a stable swimmer XML with 15 segments to prevent tangling"""
    xml_content = '''
    <mujoco model="swimmer_15">
      <compiler angle="degree" coordinate="local" inertiafromgeom="true"/>
      <option density="4000" integrator="RK4" timestep="0.02" viscosity="0.1"/>
      <default>
        <geom conaffinity="1" condim="3" contype="1" material="geom"/>
        <joint armature='0.2' damping='0.1'/>
      </default>
      <asset>
        <texture builtin="gradient" height="100" rgb1="0.1 0.1 0.1" rgb2="0.05 0.05 0.05" type="skybox" width="100"/>
        <texture builtin="flat" height="1278" mark="cross" markrgb="0.3 0.3 0.3" name="texgeom" random="0.01" rgb1="0.15 0.15 0.15" rgb2="0.1 0.1 0.1" type="cube" width="127"/>
        <texture builtin="flat" height="100" name="texplane" rgb1="0.05 0.05 0.05" type="2d" width="100"/>
        <material name="MatPlane" reflectance="0.2" shininess="0.1" specular="0.1" texture="texplane"/>
        <material name="geom" texture="texgeom" texuniform="true"/>
        <material name="reward" rgba="1 0.2 0.2 0.8"/>
      </asset>
      <worldbody>
        <light cutoff="100" diffuse="0.7 0.7 0.7" dir="-0 0 -1.3" directional="true" exponent="1" pos="0 0 1.3" specular=".1 .1 .1"/>
        <geom conaffinity="0" condim="3" material="MatPlane" name="floor" pos="0 0 -0.2" rgba="0.05 0.05 0.05 1" size="40 40 0.1" type="plane"/>
        
        <!-- Reward source visualization -->
        <geom name="reward_source" pos="2 1.5 0.2" size="0.3" type="sphere" material="reward" contype="0" conaffinity="0"/>
        
        <!--  ================= C. ELEGANS SWIMMER (15 segments) ================= -->
        <body name="head" pos="0 0 0.1">
          <camera name="track" mode="trackcom" pos="0 -6 2.5" xyaxes="1 0 0 0 1 1"/>
          <geom density="1000" fromto="0.25 0 0 0 0 0" size="0.08" type="capsule" rgba="0.7 0.9 1.0 1" contype="2" conaffinity="1"/>
          <joint axis="1 0 0" name="root_x" pos="0 0 0" type="slide"/>
          <joint axis="0 1 0" name="root_y" pos="0 0 0" type="slide"/>
          <joint axis="0 0 1" name="root_rot" pos="0 0 0" type="hinge"/>
    '''
    
    # Generate 14 body segments with joints (reduced from 24 to prevent tangling)
    segment_length = 0.2
    for i in range(14):
        segment_name = f"segment_{i+1}"
        next_pos = f"{-segment_length} 0 0"
        joint_name = f"joint_{i+1}"
        joint_range = "-35 35"  # Much more restrictive range to prevent folding
        
        xml_content += f'''
          <body name="{segment_name}" pos="{next_pos}">
            <geom density="800" fromto="0 0 0 {-segment_length} 0 0" size="0.06" type="capsule" rgba="0.6 0.8 0.95 1" contype="2" conaffinity="1"/>
            <joint axis="0 0 1" limited="true" name="{joint_name}" pos="0 0 0" range="{joint_range}" type="hinge"/>
        '''
    
    # Close all the body tags
    for i in range(14):
        xml_content += "  </body>\n"
    
    xml_content += '''
        </body>
      </worldbody>
      <actuator>
    '''
    
    # Add actuators for all joints
    for i in range(14):
        joint_name = f"joint_{i+1}"
        xml_content += f'''    <motor ctrllimited="true" ctrlrange="-1 1" gear="80.0" joint="{joint_name}"/>\n'''
    
    xml_content += '''
      </actuator>
    </mujoco>
    '''
    
    return xml_content

def sinusoidal_controller(time_step, n_segments=14):
    """Generate sinusoidal swimming pattern similar to C. elegans"""
    t = time_step * 0.02  # 20ms timestep
    omega = 2 * np.pi * 1.5  # frequency for smooth swimming
    wavelength = 2.0  # longer wavelength for stable motion
    amplitude = 0.4  # moderate amplitude for good propulsion
    
    actions = []
    for i in range(n_segments):
        phase = -2 * np.pi * wavelength * i / n_segments
        angle = amplitude * np.sin(omega * t + phase)
        actions.append(angle)
    
    return np.array(actions)

def chemotaxis_controller(time_step, worm_pos, reward_pos, base_controller):
    """Modify swimming based on distance to reward (chemotaxis)"""
    actions = base_controller(time_step)
    
    # Calculate distance and direction to reward
    distance = np.linalg.norm(worm_pos[:2] - reward_pos[:2])
    direction = (reward_pos[:2] - worm_pos[:2]) / (distance + 1e-6)
    
    # Simulate concentration gradient (inversely proportional to distance)
    concentration = 1.0 / (1.0 + distance)
    
    # Bias turning based on concentration gradient - more subtle
    if distance > 0.8:  # Far from reward
        # Add gentle weathervane turning toward reward
        bias = 0.15 * np.sign(direction[1])  # Gentler turn toward reward
        # Apply bias gradually along the body
        for i in range(len(actions)):
            weight = np.exp(-i/8.0)  # Exponential decay from head to tail
            actions[i] += bias * weight
    
    return actions, concentration

def run_mujoco_simulation():
    """Run the C. elegans MuJoCo simulation with visualization"""
    print("=== C. elegans MuJoCo Simulation ===")
    print("Loading model...")
    
    # Create the XML model
    xml_content = create_dynamic_swimmer_xml()
    
    # Load the model
    model = mujoco.MjModel.from_xml_string(xml_content)
    data = mujoco.MjData(model)
    
    print(f"Model loaded successfully!")
    print(f"- Bodies: {model.nbody}")
    print(f"- Joints: {model.njnt}")
    print(f"- Actuators: {model.nu}")
    print(f"- DOF: {model.nv}")
    
    # Set initial position closer to camera view
    data.qpos[0] = 0   # x position (centered)
    data.qpos[1] = 0   # y position (centered)
    data.qpos[2] = 0   # rotation
    
    # Reward position closer for better visibility
    reward_pos = np.array([2, 1.5, 0])
    
    print("\nStarting simulation...")
    print("The worm will swim toward the red reward sphere!")
    print("Watch how it exhibits chemotaxis behavior!")
    
    step_count = 0
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Set camera to track the worm at good distance
        viewer.cam.lookat[:] = [0, 0, 0]
        viewer.cam.distance = 6  # Better viewing distance
        viewer.cam.elevation = -25  # Good angle to see the worm
        viewer.cam.azimuth = 45
        
        print("Simulation running! Close the viewer window to stop.")
        
        while viewer.is_running():
            step_start = time.time()
            
            # Get worm center of mass position
            worm_pos = data.xpos[1]  # head position
            
            # Generate control actions
            base_actions = sinusoidal_controller(step_count)
            actions, concentration = chemotaxis_controller(step_count, worm_pos, reward_pos, sinusoidal_controller)
            
            # Apply actions to actuators
            data.ctrl[:] = actions
            
            # Step the simulation
            mujoco.mj_step(model, data)
            
            # Update viewer
            viewer.sync()
            
            # Print status every 100 steps
            if step_count % 100 == 0:
                distance = np.linalg.norm(worm_pos[:2] - reward_pos[:2])
                print(f"Step {step_count}: Distance to reward: {distance:.2f}, Concentration: {concentration:.3f}")
                
                # Move reward occasionally to show dynamic behavior
                if step_count % 500 == 0 and step_count > 0:
                    reward_pos[0] = 5 * np.cos(step_count * 0.01)
                    reward_pos[1] = 5 * np.sin(step_count * 0.01)
                    # Update reward sphere position in simulation
                    model.geom('reward_source').pos[:] = reward_pos
            
            step_count += 1
            
            # Maintain real-time execution
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
    
    print("Simulation ended.")






if __name__ == '__main__':
    try:
        print("=== C. elegans Digital Twin Research Platform ===")
        print("Running stable 15-segment C. elegans simulation with chemotaxis...")
        
        run_mujoco_simulation()
            
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user.")
    except Exception as e:
        print(f"Error running simulation: {e}")
        print("Make sure you have mujoco and worm_assets installed properly.") 