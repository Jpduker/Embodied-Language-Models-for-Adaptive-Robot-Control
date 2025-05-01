# Embodied Language Models for Adaptive Robot Control

This repository contains a proof-of-concept implementation of an embodied language model for robot control. The system can generate behavior trees from natural language commands and adapt to errors during execution.

## Project Structure

- `src/`: Source code for the experiment
  - `embodied_llm.py`: Implementation of the embodied language model and behavior tree framework
  - `test_error_handling.py`: Tests for error recovery scenarios
  - `run_experiments.py`: Main script to run all experiments
- `results/`: Generated results from the experiments
  - Visualizations of robot paths
  - Task success rates
  - Execution logs
- `paper/`: Research paper in IEEE format describing the approach and results
  - `main.tex`: LaTeX source for the paper
  - `images/`: Figures for the paper

## Requirements

Python 3.7+ and the following packages:

```
numpy>=1.20.0
matplotlib>=3.4.0
openai>=0.27.0
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/embodied_llm_experiment.git
cd embodied_llm_experiment
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Running the Experiments

To run all experiments and generate results:

```bash
python src/run_experiments.py
```

This will:
1. Run basic pick-and-place tasks in an unobstructed environment
2. Test error recovery in three scenarios:
   - Obstacle avoidance
   - Object not found
   - Gripper state error
3. Generate visualizations and metrics in the `results/` directory

## Experiment Scenarios

### Basic Tasks
Standard pick-and-place operations in an unobstructed environment.

### Obstacle Avoidance
Tasks where the path to the target is blocked by an obstacle, testing the system's ability to find alternative routes.

### Object Not Found
Tasks where the specified object is not present in the environment, testing error detection and recovery for perception errors.

### Gripper State Error
Tasks where the gripper starts in an incorrect state, testing adaptation to state mismatches.

## Implementation Details

The system uses a behavior tree framework to represent and execute tasks. When a natural language command is received, the system generates a behavior tree with appropriate actions. During execution, if an error occurs, the system attempts to modify the tree to recover from the error.

The current implementation includes:
- A simulated 2D environment with objects, obstacles, and a robot
- A behavior tree framework with sequence, selector, action, and condition nodes
- Basic robot actions like move_to, pick_up, place, open_gripper, and close_gripper
- Error recovery mechanisms for common failures

## Limitations

The current implementation has several limitations:
- Simple path planning that cannot effectively navigate around obstacles
- Limited to three recovery attempts for any error
- Simplified 2D environment lacking real-world complexity
- Basic action primitives that limit task complexity

## Research Paper

A research paper describing the approach and results is included in the `paper/` directory. The paper is formatted according to IEEE conference style and can be compiled using LaTeX.

## License

[MIT License](LICENSE) 
