import openai
import numpy as np
import json
import time
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

@dataclass
class RobotState:
    """Class representing the state of a robot."""
    position: Tuple[float, float, float]  # x, y, z coordinates
    gripper_open: bool
    holding_object: Optional[str] = None
    battery_level: float = 100.0
    
    def to_dict(self):
        return {
            "position": self.position,
            "gripper_open": self.gripper_open,
            "holding_object": self.holding_object,
            "battery_level": self.battery_level
        }

@dataclass
class EnvironmentState:
    """Class representing the state of the environment."""
    objects: Dict[str, Tuple[float, float, float]]  # object_name -> position
    obstacles: List[Tuple[float, float, float, float]]  # x, y, width, height
    
    def to_dict(self):
        return {
            "objects": self.objects,
            "obstacles": self.obstacles
        }

class BTNode:
    """Base class for behavior tree nodes."""
    def __init__(self, name):
        self.name = name
        
    def execute(self, robot_state, env_state):
        raise NotImplementedError("Subclasses must implement execute")
    
    def to_dict(self):
        return {"name": self.name, "type": self.__class__.__name__}

class SequenceNode(BTNode):
    """Executes children in sequence until one fails."""
    def __init__(self, name, children=None):
        super().__init__(name)
        self.children = children or []
        
    def execute(self, robot_state, env_state):
        for child in self.children:
            success, robot_state, message = child.execute(robot_state, env_state)
            if not success:
                return False, robot_state, f"Failed at {child.name}: {message}"
        return True, robot_state, f"Sequence {self.name} completed successfully"
    
    def to_dict(self):
        result = super().to_dict()
        result["children"] = [child.to_dict() for child in self.children]
        return result

class SelectorNode(BTNode):
    """Executes children in sequence until one succeeds."""
    def __init__(self, name, children=None):
        super().__init__(name)
        self.children = children or []
        
    def execute(self, robot_state, env_state):
        for child in self.children:
            success, robot_state, message = child.execute(robot_state, env_state)
            if success:
                return True, robot_state, f"Selector {self.name} found successful child: {child.name}"
        return False, robot_state, f"All children of selector {self.name} failed"
    
    def to_dict(self):
        result = super().to_dict()
        result["children"] = [child.to_dict() for child in self.children]
        return result

class ActionNode(BTNode):
    """Leaf node that performs an action."""
    def __init__(self, name, action_func):
        super().__init__(name)
        self.action_func = action_func
        
    def execute(self, robot_state, env_state):
        return self.action_func(robot_state, env_state)
    
    def to_dict(self):
        result = super().to_dict()
        result["action"] = self.name
        return result

class ConditionNode(BTNode):
    """Leaf node that checks a condition."""
    def __init__(self, name, condition_func):
        super().__init__(name)
        self.condition_func = condition_func
        
    def execute(self, robot_state, env_state):
        return self.condition_func(robot_state, env_state)
    
    def to_dict(self):
        result = super().to_dict()
        result["condition"] = self.name
        return result

class EmbodiedLLM:
    """Class implementing an embodied language model for robot control."""
    
    def __init__(self, api_key=None):
        if api_key:
            openai.api_key = api_key
        
        # Action library - these would be the actual robot primitives
        self.actions = {
            "move_to": self._action_move_to,
            "pick_up": self._action_pick_up,
            "place": self._action_place,
            "open_gripper": self._action_open_gripper,
            "close_gripper": self._action_close_gripper,
        }
        
        # Condition library
        self.conditions = {
            "is_at": self._condition_is_at,
            "is_holding": self._condition_is_holding,
            "is_gripper_open": self._condition_is_gripper_open,
            "object_exists": self._condition_object_exists,
        }
        
        # Execution history for analysis
        self.execution_history = []
        
    def generate_behavior_tree(self, natural_language_command, robot_state, env_state):
        """Generate a behavior tree from a natural language command."""
        prompt = self._construct_bt_prompt(natural_language_command, robot_state, env_state)
        
        # In a real implementation, this would call the OpenAI API
        # response = openai.ChatCompletion.create(
        #     model="gpt-4",
        #     messages=[{"role": "user", "content": prompt}]
        # )
        # bt_json = response.choices[0].message.content
        
        # For the experiment, we'll use a hardcoded example based on the command
        bt_json = self._mock_llm_response(natural_language_command)
        
        # Parse the JSON and create BT nodes
        bt_dict = json.loads(bt_json)
        return self._parse_bt_node(bt_dict)
    
    def execute_behavior_tree(self, bt_root, robot_state, env_state, attempt=0, max_attempts=3):
        """Execute a behavior tree and handle errors."""
        self.execution_history = []
        
        # Prevent infinite recursion
        if attempt >= max_attempts:
            print(f"Reached maximum attempts ({max_attempts}). Could not complete the task.")
            return False, robot_state, "Maximum error recovery attempts reached"
        
        success, new_robot_state, message = bt_root.execute(robot_state, env_state)
        self.execution_history.append((success, message))
        
        # If execution fails, try to handle the error
        if not success:
            print(f"Execution failed: {message}")
            updated_bt = self._handle_error(bt_root, message, new_robot_state, env_state)
            if updated_bt:
                print("Attempting execution with updated behavior tree...")
                return self.execute_behavior_tree(updated_bt, new_robot_state, env_state, attempt + 1, max_attempts)
            else:
                print("Could not recover from error.")
                return False, new_robot_state, message
        
        return success, new_robot_state, message
    
    def _handle_error(self, bt_root, error_message, robot_state, env_state):
        """Use the LLM to modify the behavior tree based on the error."""
        prompt = self._construct_error_prompt(bt_root, error_message, robot_state, env_state)
        
        # In a real implementation, this would call the OpenAI API
        # response = openai.ChatCompletion.create(
        #     model="gpt-4",
        #     messages=[{"role": "user", "content": prompt}]
        # )
        # updated_bt_json = response.choices[0].message.content
        
        # For the experiment, we'll use a simple error handling strategy
        if "object not found" in error_message.lower():
            # Add a search behavior before attempting to pick up
            return self._mock_error_handling(bt_root, "object_not_found")
        elif "path blocked" in error_message.lower():
            # Add a path planning node
            return self._mock_error_handling(bt_root, "path_blocked")
        
        return None  # Could not handle this error
    
    def _construct_bt_prompt(self, command, robot_state, env_state):
        """Construct a prompt for generating a behavior tree."""
        return f"""
        Generate a behavior tree for a robot to execute the command: "{command}".
        
        Current robot state:
        {json.dumps(robot_state.to_dict(), indent=2)}
        
        Current environment state:
        {json.dumps(env_state.to_dict(), indent=2)}
        
        Available actions: {list(self.actions.keys())}
        Available conditions: {list(self.conditions.keys())}
        
        Return the behavior tree as a JSON object with the following structure:
        {{
            "type": "SequenceNode" | "SelectorNode" | "ActionNode" | "ConditionNode",
            "name": "descriptive name",
            "children": [...] (for SequenceNode and SelectorNode),
            "action": "action_name" (for ActionNode),
            "condition": "condition_name" (for ConditionNode),
            "parameters": {{...}} (optional parameters for actions or conditions)
        }}
        """
    
    def _construct_error_prompt(self, bt_root, error_message, robot_state, env_state):
        """Construct a prompt for handling an error during execution."""
        bt_json = json.dumps(bt_root.to_dict(), indent=2)
        
        return f"""
        The robot encountered an error while executing this behavior tree:
        {bt_json}
        
        Error message: {error_message}
        
        Current robot state:
        {json.dumps(robot_state.to_dict(), indent=2)}
        
        Current environment state:
        {json.dumps(env_state.to_dict(), indent=2)}
        
        Please provide an updated behavior tree that addresses this error.
        Return the updated behavior tree as a JSON object with the same structure as the original.
        """
    
    def _parse_bt_node(self, node_dict):
        """Parse a dictionary into a behavior tree node."""
        node_type = node_dict["type"]
        name = node_dict["name"]
        
        if node_type == "SequenceNode":
            node = SequenceNode(name)
            node.children = [self._parse_bt_node(child) for child in node_dict["children"]]
            return node
        elif node_type == "SelectorNode":
            node = SelectorNode(name)
            node.children = [self._parse_bt_node(child) for child in node_dict["children"]]
            return node
        elif node_type == "ActionNode":
            action_name = node_dict["action"]
            parameters = node_dict.get("parameters", {})
            action_func = lambda rs, es, name=action_name, params=parameters: self.actions[name](rs, es, params)
            return ActionNode(name, action_func)
        elif node_type == "ConditionNode":
            condition_name = node_dict["condition"]
            parameters = node_dict.get("parameters", {})
            condition_func = lambda rs, es, name=condition_name, params=parameters: self.conditions[name](rs, es, params)
            return ConditionNode(name, condition_func)
        else:
            raise ValueError(f"Unknown node type: {node_type}")
    
    def _mock_llm_response(self, command):
        """Mock the LLM response for the experiment."""
        if "pick" in command.lower() and "place" in command.lower():
            object_name = "red_block"
            if "red" in command.lower():
                object_name = "red_block"
            elif "blue" in command.lower():
                object_name = "blue_block"
            
            target = "shelf"
            if "shelf" in command.lower():
                target = "shelf"
            elif "table" in command.lower():
                target = "table"
            
            return json.dumps({
                "type": "SequenceNode",
                "name": "PickAndPlaceSequence",
                "children": [
                    {
                        "type": "ActionNode",
                        "name": "OpenGripperInit",
                        "action": "open_gripper",
                        "parameters": {}
                    },
                    {
                        "type": "ActionNode",
                        "name": f"MoveTo{object_name.capitalize()}",
                        "action": "move_to",
                        "parameters": {"object": object_name}
                    },
                    {
                        "type": "ActionNode",
                        "name": f"PickUp{object_name.capitalize()}",
                        "action": "pick_up",
                        "parameters": {"object": object_name}
                    },
                    {
                        "type": "ActionNode",
                        "name": f"MoveToTarget",
                        "action": "move_to",
                        "parameters": {"location": target}
                    },
                    {
                        "type": "ActionNode",
                        "name": "PlaceObject",
                        "action": "place",
                        "parameters": {"location": target}
                    }
                ]
            })
        else:
            # Default behavior tree for unknown commands
            return json.dumps({
                "type": "SequenceNode",
                "name": "DefaultBehavior",
                "children": [
                    {
                        "type": "ActionNode",
                        "name": "MoveToHome",
                        "action": "move_to",
                        "parameters": {"location": "home_position"}
                    }
                ]
            })
    
    def _mock_error_handling(self, bt_root, error_type):
        """Generate a modified behavior tree to handle a specific error."""
        if error_type == "object_not_found":
            # Add a search behavior
            new_bt = SequenceNode("SearchAndPickPlace")
            search_node = ActionNode("SearchForObject", 
                lambda rs, es: (True, rs, "Searching for object..."))
            
            # Add the search node and then copy the original tree
            new_bt.children = [search_node] + (bt_root.children if isinstance(bt_root, SequenceNode) else [bt_root])
            return new_bt
            
        elif error_type == "path_blocked":
            # Add path planning
            new_bt = SequenceNode("PathPlanAndExecute")
            plan_path_node = ActionNode("PlanAlternativePath", 
                lambda rs, es: (True, rs, "Planning alternative path..."))
            
            # Add the path planning node and then copy the original tree
            new_bt.children = [plan_path_node] + (bt_root.children if isinstance(bt_root, SequenceNode) else [bt_root])
            return new_bt
            
        return None
    
    # Action implementations
    def _action_move_to(self, robot_state, env_state, params):
        """Simulate moving the robot to a location or object."""
        target_pos = None
        
        if "object" in params:
            object_name = params["object"]
            if object_name in env_state.objects:
                target_pos = env_state.objects[object_name]
                print(f"Moving to object: {object_name} at {target_pos}")
            else:
                return False, robot_state, f"Object not found: {object_name}"
                
        elif "location" in params:
            location = params["location"]
            if location == "home_position":
                target_pos = (0, 0, 0)
                print(f"Moving to home position: {target_pos}")
            elif location == "shelf":
                target_pos = (1, 1, 1)
                print(f"Moving to shelf at {target_pos}")
            elif location == "table":
                target_pos = (2, 0, 0)
                print(f"Moving to table at {target_pos}")
            else:
                return False, robot_state, f"Unknown location: {location}"
        
        # Check for obstacles (simplified)
        for obstacle in env_state.obstacles:
            x, y, width, height = obstacle
            if (abs(target_pos[0] - x) < width/2 and 
                abs(target_pos[1] - y) < height/2):
                return False, robot_state, "Path blocked by obstacle"
        
        # Update robot position
        new_robot_state = RobotState(
            position=target_pos,
            gripper_open=robot_state.gripper_open,
            holding_object=robot_state.holding_object,
            battery_level=robot_state.battery_level - 1.0  # Decrease battery slightly
        )
        
        return True, new_robot_state, f"Moved to {target_pos}"
    
    def _action_pick_up(self, robot_state, env_state, params):
        """Simulate picking up an object."""
        if not robot_state.gripper_open:
            return False, robot_state, "Gripper is closed"
            
        if robot_state.holding_object:
            return False, robot_state, f"Already holding {robot_state.holding_object}"
        
        object_name = params["object"]
        if object_name not in env_state.objects:
            return False, robot_state, f"Object not found: {object_name}"
            
        object_pos = env_state.objects[object_name]
        if np.linalg.norm(np.array(robot_state.position) - np.array(object_pos)) > 0.1:
            return False, robot_state, f"Too far from {object_name} to pick up"
        
        # Update robot state
        new_robot_state = RobotState(
            position=robot_state.position,
            gripper_open=False,  # Close the gripper
            holding_object=object_name,
            battery_level=robot_state.battery_level - 0.5  # Decrease battery slightly
        )
        
        print(f"Picked up {object_name}")
        return True, new_robot_state, f"Picked up {object_name}"
    
    def _action_place(self, robot_state, env_state, params):
        """Simulate placing an object."""
        if robot_state.gripper_open:
            return False, robot_state, "Gripper is open"
            
        if not robot_state.holding_object:
            return False, robot_state, "Not holding any object"
        
        location = params.get("location", "current")
        print(f"Placing {robot_state.holding_object} at {location}")
        
        # Update robot state
        new_robot_state = RobotState(
            position=robot_state.position,
            gripper_open=True,  # Open the gripper
            holding_object=None,
            battery_level=robot_state.battery_level - 0.5  # Decrease battery slightly
        )
        
        # Update environment - the object is now at the robot's position
        env_state.objects[robot_state.holding_object] = robot_state.position
        
        return True, new_robot_state, f"Placed {robot_state.holding_object} at {location}"
    
    def _action_open_gripper(self, robot_state, env_state, params):
        """Simulate opening the gripper."""
        if robot_state.gripper_open:
            return True, robot_state, "Gripper already open"
            
        if robot_state.holding_object:
            # If holding an object, we drop it
            env_state.objects[robot_state.holding_object] = robot_state.position
            
        new_robot_state = RobotState(
            position=robot_state.position,
            gripper_open=True,
            holding_object=None,
            battery_level=robot_state.battery_level - 0.1
        )
        
        print("Opened gripper")
        return True, new_robot_state, "Opened gripper"
    
    def _action_close_gripper(self, robot_state, env_state, params):
        """Simulate closing the gripper."""
        if not robot_state.gripper_open:
            return True, robot_state, "Gripper already closed"
            
        # Check if any object is at the robot's position
        for obj_name, obj_pos in env_state.objects.items():
            if np.linalg.norm(np.array(robot_state.position) - np.array(obj_pos)) < 0.1:
                new_robot_state = RobotState(
                    position=robot_state.position,
                    gripper_open=False,
                    holding_object=obj_name,
                    battery_level=robot_state.battery_level - 0.1
                )
                print(f"Closed gripper and grasped {obj_name}")
                return True, new_robot_state, f"Closed gripper and grasped {obj_name}"
        
        new_robot_state = RobotState(
            position=robot_state.position,
            gripper_open=False,
            holding_object=None,
            battery_level=robot_state.battery_level - 0.1
        )
        
        print("Closed gripper")
        return True, new_robot_state, "Closed gripper"
    
    # Condition implementations
    def _condition_is_at(self, robot_state, env_state, params):
        """Check if the robot is at a specific location or object."""
        target_pos = None
        
        if "object" in params:
            object_name = params["object"]
            if object_name in env_state.objects:
                target_pos = env_state.objects[object_name]
            else:
                return False, robot_state, f"Object not found: {object_name}"
                
        elif "location" in params:
            location = params["location"]
            if location == "home_position":
                target_pos = (0, 0, 0)
            elif location == "shelf":
                target_pos = (1, 1, 1)
            elif location == "table":
                target_pos = (2, 0, 0)
            else:
                return False, robot_state, f"Unknown location: {location}"
        
        if np.linalg.norm(np.array(robot_state.position) - np.array(target_pos)) < 0.1:
            return True, robot_state, f"Robot is at the target position"
        else:
            return False, robot_state, f"Robot is not at the target position"
    
    def _condition_is_holding(self, robot_state, env_state, params):
        """Check if the robot is holding a specific object."""
        if "object" in params:
            object_name = params["object"]
            if robot_state.holding_object == object_name:
                return True, robot_state, f"Robot is holding {object_name}"
            else:
                return False, robot_state, f"Robot is not holding {object_name}"
        else:
            if robot_state.holding_object:
                return True, robot_state, f"Robot is holding {robot_state.holding_object}"
            else:
                return False, robot_state, "Robot is not holding any object"
    
    def _condition_is_gripper_open(self, robot_state, env_state, params):
        """Check if the gripper is open."""
        if robot_state.gripper_open:
            return True, robot_state, "Gripper is open"
        else:
            return False, robot_state, "Gripper is closed"
    
    def _condition_object_exists(self, robot_state, env_state, params):
        """Check if an object exists in the environment."""
        object_name = params["object"]
        if object_name in env_state.objects:
            return True, robot_state, f"Object {object_name} exists"
        else:
            return False, robot_state, f"Object {object_name} does not exist"

def visualize_execution(env_state, robot_states, task_description):
    """Visualize the execution of a task."""
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
    
    ax.set_title(f'Robot Execution: {task_description}')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.legend()
    
    # Save the figure
    plt.savefig('results/execution_visualization.png', dpi=300)
    plt.close(fig)

def main():
    """Run a simple experiment with the embodied LLM."""
    # Initialize the embodied LLM
    llm = EmbodiedLLM()
    
    # Set up the environment and robot state
    env_state = EnvironmentState(
        objects={
            "red_block": (0.5, 0.5, 0),
            "blue_block": (0.7, 0.3, 0)
        },
        obstacles=[
            (1.5, 0.5, 0.3, 0.3),  # Obstacle between objects and goal
        ]
    )
    
    robot_state = RobotState(
        position=(0, 0, 0),  # Start at home position
        gripper_open=False,
        holding_object=None
    )
    
    # Generate and execute behavior trees for different tasks
    tasks = [
        "pick up the red block and place it on the shelf",
        "grab the blue block and put it on the table"
    ]
    
    all_results = []
    all_robot_states = []
    
    for task in tasks:
        print(f"\n\nExecuting task: {task}")
        print("=" * 50)
        
        # Generate behavior tree
        bt = llm.generate_behavior_tree(task, robot_state, env_state)
        print(f"Generated behavior tree: {bt.name}")
        
        # Record robot states for visualization
        task_robot_states = [robot_state]
        
        # Execute behavior tree
        success, robot_state, message = llm.execute_behavior_tree(bt, robot_state, env_state)
        task_robot_states.append(robot_state)
        
        all_results.append({
            "task": task,
            "success": success,
            "message": message,
            "execution_history": llm.execution_history
        })
        all_robot_states.extend(task_robot_states)
        
        print(f"Task execution {'succeeded' if success else 'failed'}: {message}")
    
    # Visualize the execution
    visualize_execution(env_state, all_robot_states, "Multiple Tasks")
    
    # Save results
    with open('embodied_llm_experiment/results/execution_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    return all_results

if __name__ == "__main__":
    main() 