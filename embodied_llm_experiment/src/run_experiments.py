import os
import sys
import json
import time
import matplotlib.pyplot as plt
import numpy as np
from embodied_llm import EmbodiedLLM, RobotState, EnvironmentState, visualize_execution
from test_error_handling import test_with_obstacles, test_object_not_found, test_gripper_state_error

def ensure_dirs_exist():
    """Ensure the necessary directories exist."""
    os.makedirs('results', exist_ok=True)
    os.makedirs('paper', exist_ok=True)

def run_basic_experiments():
    """Run the basic experiments without errors."""
    # Initialize the embodied LLM
    llm = EmbodiedLLM()
    
    # Set up the environment and robot state
    env_state = EnvironmentState(
        objects={
            "red_block": (0.5, 0.5, 0),
            "blue_block": (0.7, 0.3, 0)
        },
        obstacles=[]  # No obstacles for basic experiments
    )
    
    robot_state = RobotState(
        position=(0, 0, 0),  # Start at home position
        gripper_open=False,
        holding_object=None
    )
    
    # Generate and execute behavior trees for different tasks
    tasks = [
        "pick up the red block and place it on the shelf",
        "grab the blue block and put it on the table",
        "move the red block from the shelf to the table"
    ]
    
    all_results = []
    all_robot_states = [robot_state]
    initial_robot_state = robot_state
    
    for i, task in enumerate(tasks):
        print(f"\n\nExperiment {i+1}: {task}")
        print("=" * 50)
        
        # Reset robot state for each experiment
        if i > 0:
            robot_state = RobotState(
                position=(0, 0, 0),  # Start at home position
                gripper_open=False,
                holding_object=None
            )
        
        # Generate behavior tree
        bt = llm.generate_behavior_tree(task, robot_state, env_state)
        print(f"Generated behavior tree: {bt.name}")
        
        # Execute behavior tree
        start_time = time.time()
        success, new_robot_state, message = llm.execute_behavior_tree(bt, robot_state, env_state)
        execution_time = time.time() - start_time
        
        all_robot_states.append(new_robot_state)
        robot_state = new_robot_state  # Update for next task
        
        all_results.append({
            "task": task,
            "success": success,
            "message": message,
            "execution_time": execution_time,
            "execution_history": llm.execution_history
        })
        
        print(f"Task execution {'succeeded' if success else 'failed'}: {message}")
        print(f"Execution time: {execution_time:.2f} seconds")
    
    # Visualize the execution of all tasks
    visualize_multi_task_execution(env_state, all_robot_states, tasks, "basic_experiments.png")
    
    # Save results
    with open('results/basic_experiment_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    return all_results

def run_error_handling_experiments():
    """Run the error handling experiments."""
    results = []
    
    # Test 1: Obstacle in path
    results.append(test_with_obstacles())
    
    # Test 2: Object not found
    results.append(test_object_not_found())
    
    # Test 3: Gripper state error
    results.append(test_gripper_state_error())
    
    # Save all results
    with open('results/error_handling_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return results

def visualize_multi_task_execution(env_state, robot_states, tasks, filename):
    """Visualize the execution of multiple tasks."""
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
    
    ax.set_title(f'Robot Execution: Multiple Tasks')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.legend()
    
    # Save the figure
    plt.savefig(f'results/{filename}', dpi=300)
    plt.close(fig)

def generate_summary_statistics(basic_results, error_results):
    """Generate summary statistics from the experiment results."""
    summary = {
        "basic_experiments": {
            "total_tasks": len(basic_results),
            "successful_tasks": sum(1 for r in basic_results if r["success"]),
            "average_execution_time": sum(r["execution_time"] for r in basic_results) / len(basic_results)
        },
        "error_handling_experiments": {
            "total_tasks": len(error_results),
            "successful_tasks": sum(1 for r in error_results if r["success"]),
            "error_types": {
                "obstacles": sum(1 for r in error_results if "obstacle" in r["task"].lower()),
                "missing_objects": sum(1 for r in error_results if "missing" in r["task"].lower()),
                "gripper_state": sum(1 for r in error_results if "gripper" in r["task"].lower())
            },
            "recovery_success_rate": sum(1 for r in error_results if r["success"]) / len(error_results)
        }
    }
    
    with open('results/summary_statistics.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    return summary

def plot_success_rates(basic_results, error_results):
    """Plot success rates for the experiments."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate success rates
    basic_success_rate = sum(1 for r in basic_results if r["success"]) / len(basic_results)
    error_success_rate = sum(1 for r in error_results if r["success"]) / len(error_results)
    
    # Plot bar chart
    experiment_types = ['Basic Tasks', 'Error Recovery Tasks']
    success_rates = [basic_success_rate, error_success_rate]
    
    x = np.arange(len(experiment_types))
    bars = ax.bar(x, success_rates, width=0.6)
    
    # Add labels
    ax.set_ylabel('Success Rate')
    ax.set_title('Task Success Rates by Experiment Type')
    ax.set_xticks(x)
    ax.set_xticklabels(experiment_types)
    ax.set_ylim(0, 1.1)
    
    # Add value labels on top of bars
    for i, v in enumerate(success_rates):
        ax.text(i, v + 0.05, f'{v:.2f}', ha='center')
    
    plt.savefig('results/success_rates.png', dpi=300)
    plt.close(fig)

def main():
    """Run all experiments and collect results."""
    # Ensure directories exist
    ensure_dirs_exist()
    
    print("Running basic experiments...")
    basic_results = run_basic_experiments()
    
    print("\nRunning error handling experiments...")
    error_results = run_error_handling_experiments()
    
    print("\nGenerating summary statistics...")
    summary = generate_summary_statistics(basic_results, error_results)
    
    print("\nPlotting success rates...")
    plot_success_rates(basic_results, error_results)
    
    print("\nAll experiments completed successfully!")
    print(f"Results saved to 'results/'")
    
    return basic_results, error_results, summary

if __name__ == "__main__":
    main() 