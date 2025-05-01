import sys
import json
import matplotlib.pyplot as plt
import numpy as np
from embodied_llm import EmbodiedLLM, RobotState, EnvironmentState

def test_with_obstacles():
    """Test error handling with obstacles blocking the path."""
    # Initialize the embodied LLM
    llm = EmbodiedLLM()
    
    # Set up the environment with obstacles in the path to the shelf
    env_state = EnvironmentState(
        objects={
            "red_block": (0.5, 0.5, 0),
            "blue_block": (0.7, 0.3, 0)
        },
        obstacles=[
            (1.0, 0.8, 0.4, 0.4),  # Obstacle directly blocking path to shelf
        ]
    )
    
    robot_state = RobotState(
        position=(0, 0, 0),  # Start at home position
        gripper_open=False,
        holding_object=None
    )
    
    task = "pick up the red block and place it on the shelf"
    print(f"\n\nExecuting task with obstacle: {task}")
    print("=" * 50)
    
    # Generate behavior tree
    bt = llm.generate_behavior_tree(task, robot_state, env_state)
    print(f"Generated behavior tree: {bt.name}")
    
    # Record robot states for visualization
    robot_states = [robot_state]
    
    # Execute behavior tree - should encounter the obstacle and adapt
    success, new_robot_state, message = llm.execute_behavior_tree(bt, robot_state, env_state)
    robot_states.append(new_robot_state)
    
    print(f"Task execution {'succeeded' if success else 'failed'}: {message}")
    print(f"Execution history: {llm.execution_history}")
    
    # Visualize the execution
    visualize_error_handling(env_state, robot_states, 
                             "Obstacle Avoidance", 
                             "error_handling_obstacle.png")
    
    return {
        "task": task,
        "success": success,
        "message": message,
        "execution_history": llm.execution_history
    }

def test_object_not_found():
    """Test error handling when an object is not found."""
    # Initialize the embodied LLM
    llm = EmbodiedLLM()
    
    # Set up the environment without the target object
    env_state = EnvironmentState(
        objects={
            "blue_block": (0.7, 0.3, 0)
            # Red block is missing
        },
        obstacles=[]
    )
    
    robot_state = RobotState(
        position=(0, 0, 0),  # Start at home position
        gripper_open=False,
        holding_object=None
    )
    
    task = "pick up the red block and place it on the shelf"
    print(f"\n\nExecuting task with missing object: {task}")
    print("=" * 50)
    
    # Generate behavior tree
    bt = llm.generate_behavior_tree(task, robot_state, env_state)
    print(f"Generated behavior tree: {bt.name}")
    
    # Record robot states for visualization
    robot_states = [robot_state]
    
    # Execute behavior tree - should fail to find the object and adapt
    success, new_robot_state, message = llm.execute_behavior_tree(bt, robot_state, env_state)
    robot_states.append(new_robot_state)
    
    print(f"Task execution {'succeeded' if success else 'failed'}: {message}")
    print(f"Execution history: {llm.execution_history}")
    
    # Visualize the execution
    visualize_error_handling(env_state, robot_states, 
                             "Object Not Found", 
                             "error_handling_missing_object.png")
    
    return {
        "task": task,
        "success": success,
        "message": message,
        "execution_history": llm.execution_history
    }

def test_gripper_state_error():
    """Test error handling when the gripper is in wrong state."""
    # Initialize the embodied LLM
    llm = EmbodiedLLM()
    
    # Set up the environment
    env_state = EnvironmentState(
        objects={
            "red_block": (0.5, 0.5, 0),
            "blue_block": (0.7, 0.3, 0)
        },
        obstacles=[]
    )
    
    # Start with the gripper closed, which will cause a failure
    robot_state = RobotState(
        position=(0, 0, 0),
        gripper_open=False,  # Gripper already closed
        holding_object=None
    )
    
    task = "pick up the red block and place it on the shelf"
    print(f"\n\nExecuting task with gripper already closed: {task}")
    print("=" * 50)
    
    # Generate behavior tree
    bt = llm.generate_behavior_tree(task, robot_state, env_state)
    print(f"Generated behavior tree: {bt.name}")
    
    # Record robot states for visualization
    robot_states = [robot_state]
    
    # Execute behavior tree
    success, new_robot_state, message = llm.execute_behavior_tree(bt, robot_state, env_state)
    robot_states.append(new_robot_state)
    
    print(f"Task execution {'succeeded' if success else 'failed'}: {message}")
    print(f"Execution history: {llm.execution_history}")
    
    # Visualize the execution
    visualize_error_handling(env_state, robot_states, 
                             "Gripper State Error", 
                             "error_handling_gripper_state.png")
    
    return {
        "task": task,
        "success": success,
        "message": message,
        "execution_history": llm.execution_history
    }

def visualize_error_handling(env_state, robot_states, title, filename):
    """Visualize the execution of a task with error handling."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Set axis limits
    ax.set_xlim(-1, 3)
    ax.set_ylim(-1, 3)
    
    # Draw obstacles
    for obstacle in env_state.obstacles:
        x, y, width, height = obstacle
        rect = plt.Rectangle((x - width/2, y - height/2), width, height, color='red', alpha=0.3)
        ax.add_patch(rect)
    
    # Draw objects
    for obj_name, obj_pos in env_state.objects.items():
        color = 'red' if 'red' in obj_name else 'blue' if 'blue' in obj_name else 'green'
        ax.scatter(obj_pos[0], obj_pos[1], color=color, s=100, label=obj_name)
    
    # Draw locations
    ax.scatter(0, 0, color='black', marker='x', s=100, label='Home')
    ax.scatter(1, 1, color='black', marker='s', s=100, label='Shelf')
    ax.scatter(2, 0, color='black', marker='s', s=100, label='Table')
    
    # Draw robot path
    x_path = [state.position[0] for state in robot_states]
    y_path = [state.position[1] for state in robot_states]
    ax.plot(x_path, y_path, 'b-', label='Robot Path')
    
    # Draw robot at final position
    final_pos = robot_states[-1].position
    ax.scatter(final_pos[0], final_pos[1], color='purple', s=150, marker='o', label='Robot')
    
    ax.set_title(f'Error Handling: {title}')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.legend()
    
    # Save the figure
    plt.savefig(f'results/{filename}', dpi=300)
    plt.close(fig)

def main():
    """Run error handling tests."""
    results = []
    
    # Test 1: Obstacle in path
    results.append(test_with_obstacles())
    
    # Test 2: Object not found
    results.append(test_object_not_found())
    
    # Test 3: Gripper state error
    results.append(test_gripper_state_error())
    
    # Save all results
    with open('embodied_llm_experiment/results/error_handling_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return results

if __name__ == "__main__":
    main() 