#thought -> evaluated value (0.4, This solution is invalid because x) -> thought prompt + this solution is invalid because + better eval
 
import json
import os
import time
import random
import io
from queue import Queue
 
DATA_PATH = './data'
import logging
 
import concurrent.futures
from queue import PriorityQueue
from typing import Any, Dict, Union
 
import numpy as np
from tree_of_thoughts.models.abstract_language_model import AbstractLanguageModel
 
from tree_of_thoughts.text_generation_web_ui import (
    build_text_generation_web_ui_client_llm,
    ui_default_parameters,
)
 
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
 
class TreeofThoughts:
    def __init__(self, model):
        self.model = model
        self.tree: Dict[str, Dict[str, Union[float, Dict[str, Any]]]] = {
            "nodes": {},
        }
        self.best_state = None
        self.best_value = float("-inf")
        self.history = [] #added line initalize history
 
 
    def save_tree_to_json(self, file_name):
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        with open(file_name, 'w') as json_file:
            json.dump(self.tree, json_file, indent=4)
 
    def logNewState(self, state, evaluation):
        if not (type(state) == str):
            state = " | ".join(state)
        if state in self.tree['nodes']:
            self.tree['nodes'][state]['thoughts'].append(evaluation)
        else:
            self.tree['nodes'][state] = {'thoughts': [evaluation]}
 
    def adjust_pruning_threshold_precentile(self, evaluated_thoughts, percentile):
        values = np.array(list(evaluated_thoughts.values()))
        if values.size == 0:
            return 0
        return max(np.percentile(values, percentile), 0.1)
    
 
    def adjust_pruning_threshold_moving_average(self, evaluated_thoughts, window_size):
        values = list(evaluated_thoughts.values())
        if len(values) < window_size:
            return np.mean(values) if values else 0
        else:
            return max(np.mean(values[-window_size:]), 0.1)
 
    def get_leaf_nodes(self):
        """Returns a list of leaf nodes in the tree."""
        leaf_nodes = []
        for node in self.tree['nodes']:
            if not self.tree['nodes'][node]['thoughts']:  # Leaf nodes have no thoughts
                leaf_nodes.append(node)
        return leaf_nodes
 
    def find_lca(self, node1, node2):
        """Finds the lowest common ancestor of two nodes."""
        ancestors1 = self.get_ancestors(node1)
        ancestors2 = self.get_ancestors(node2)
 
        # Iterate from the root to find the first common ancestor
        lca = None
        for ancestor in ancestors1:
            if ancestor in ancestors2:
                lca = ancestor
                break
        return lca
 
    def get_ancestors(self, node):
        """Returns a list of ancestors of a given node."""
        ancestors = []
        while node:
            ancestors.append(node)
            node = self.get_parent(node)
        return ancestors
 
    def get_parent(self, node):
        """Returns the parent of a given node."""
        for parent, child in self.tree['nodes'].items():
            if node in child['thoughts']:
                return parent
        return None
 
######################
class TreeofThoughtsBFS(TreeofThoughts):
    def solve(
        self,
        initial_prompt,
        num_thoughts,
        max_steps,
        max_states,
        pruning_threshold=0.5,
    ):
        current_states = [(initial_prompt, set())]
        state_values = {}
        dynamic_pruning_threshold = pruning_threshold
 
        try:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                for step in range(1, max_steps + 1):
                    selected_states = []
                    for state, rejected in current_states:
                        thoughts = self.model.generate_thoughts(
                            state, num_thoughts, initial_prompt, list(rejected)
                        )
                        futures = [
                            executor.submit(
                                self.model.evaluate_states,
                                {thought: 0},
                                initial_prompt,
                            )
                            for thought in thoughts
                        ]
                        concurrent.futures.wait(futures)
                        evaluated_thoughts = {}
                        for thought, fut in zip(thoughts, futures):
                            result = fut.result()
                            if isinstance(result, dict):  # Ensure the result is a dictionary
                                for state, value in result.items():
                                    if isinstance(value, (int, float)):  # Check if value is a number
                                        evaluated_thoughts[state] = value
 
                        print("Evaluated items length!")
                        print(len(evaluated_thoughts.items()))
 
                        if (
                            evaluated_thoughts
                        ):  # only adjust if you have evaluated thoughts
                            dynamic_pruning_threshold = (
                                self.adjust_pruning_threshold_moving_average(
                                    evaluated_thoughts, 5
                                )
                            )
                        print("Evaluated items length!")
                        print(len(evaluated_thoughts.items()))
                        for thought, value in evaluated_thoughts.items():
                            if value < dynamic_pruning_threshold:
                                print("Pruned!")
                                rejected.add(thought)
 
                            flattened_state = (
                                (state, thought)
                                if isinstance(state, str)
                                else (*state, thought)
                            )
                            print("Value found at!")
                            print(value)
                            selected_states.append((flattened_state, value))
 
                        selected_states.sort(key=lambda x: x[1], reverse=True)
                        selected_states = selected_states[
                            :max_states
                        ]  # Select only the top states
                        print("Length of rejected vector:")
                        print(len(rejected))
                        for state, value in selected_states:
                            if value >= dynamic_pruning_threshold:
                                state_values[state] = value
                                self.logNewState(state, value)
                                logger.debug(f"State Values: {state_values}")
 
            # if state_values:
            #     highest_rated_solution = max(state_values.items(), key=lambda x: x[1])
            #     print(f"highest rated solution: {highest_rated_solution}")
            #     highest_rated_state = highest_rated_solution[0]  # Use a different name to avoid confusion
            #     print(f'highest rated state: {highest_rated_state}')
            #     try:
            #         solution = self.model.generate_solution(initial_prompt, highest_rated_state)
            #     except Exception as e:
            #         logger.error(f"Error in generating solution: {e}")
            #         solution = None  # Set a fallback value for solution
 
            #     return solution if solution is not None else highest_rated_state  # Return highest rated state if solution is None
            if state_values:
                highest_rated_solution = max(
                    state_values.items(), key=lambda x: x[1]
                )
                highest_rated_state = highest_rated_solution[0]
                solution = self.model.generate_solution(
                    initial_prompt, highest_rated_solution[0][0]  # Extract state string from tuple
                )
                print(
                    "Highest_rated solution:"
                    f" {highest_rated_solution} highest_rated_solution:"
                    f" {highest_rated_solution} Solution: {solution}"
                )
 
                return solution if solution else highest_rated_state
 
            else:
                return None
 
        except Exception as e:
            logger.error(f"Error in tot_bfs: {e}")
            return None
 
######################
 
class TreeofThoughtsDFS(TreeofThoughts):
    def solve(self, initial_prompt, num_thoughts, max_steps, value_threshold, pruning_threshold=0.5):
        visited_states = set()
        output = []
 
        def dfs(state, step):
            nonlocal output
            if state in visited_states:
                return
            visited_states.add(state)
 
            if step > max_steps:
                thought = self.model.generate_thoughts(state, 1, initial_prompt)
                value = self.model.evaluate_states({state}, initial_prompt)[state]
                output.append((thought, value))
                return
 
            thoughts = self.model.generate_thoughts(state, num_thoughts, initial_prompt)
            evaluated_thoughts = self.model.evaluate_states({thought: 0 for thought in thoughts}, initial_prompt)
            filtered_thoughts = [thought for thought in thoughts if evaluated_thoughts[thought] >= pruning_threshold]
 
            for next_state in filtered_thoughts:
                state_value = self.model.evaluate_states({next_state: 0}, initial_prompt)[next_state]
 
                if state_value > value_threshold:
                    child = (state, next_state) if isinstance(state, str) else (*state, next_state)
                    dfs(child, step + 1)
 
        try:
            dfs(initial_prompt, 1)
            if output:
                best_state, _ = max(output, key=lambda x: x[1])
                solution = self.model.generate_solution(initial_prompt, best_state)
                return solution if solution else best_state
            else:
                return None
        except Exception as e:
            logger.error(f"Error in tot_dfs: {e}")
            return None
 
 
######################
class TreeofThoughtsBEST(TreeofThoughts):
    def __init__(self, model):
        super().__init__(model)
        self.visited_states = set()
 
    def solve(self, initial_prompt, num_thoughts, max_steps):
        state_queue = PriorityQueue()
        state_queue.put((0, initial_prompt))  # Priority, State
        self.visited_states.add(initial_prompt)
 
        for _ in range(max_steps):
            if state_queue.empty():
                return None  # No solution found within the given steps
 
            _, current_state = state_queue.get()
 
            if self.is_goal(current_state):
                return self.reconstruct_solution(current_state)
 
            thoughts = self.model.generate_thoughts(current_state, num_thoughts, initial_prompt)
            evaluated_thoughts = self.model.evaluate_states(thoughts, initial_prompt)
 
            for thought, value in evaluated_thoughts.items():
                next_state = (current_state, thought) if isinstance(current_state, str) else (*current_state, thought)
                if next_state not in self.visited_states:
                    self.visited_states.add(next_state)
                    heuristic_value = self.heuristic(next_state)
                    state_queue.put((-heuristic_value, next_state))  # Use negative because PriorityQueue is min-heap
 
        return None  # No solution found
 
    def heuristic(self, state):
        # Define the heuristic function
        # Example: return a negative value of the state's evaluation
        return -self.model.evaluate_state_heuristic(state)
 
    def is_goal(self, state):
        # Define goal criteria
        # Example: Check if state meets certain criteria
        return self.model.check_goal(state)
 
    def reconstruct_solution(self, state):
        # Reconstruct the path from the final state to the initial state
        solution_path = []
        while state:
            solution_path.append(state)
            state = self.get_parent(state)
        solution_path.reverse()
        return solution_path
 
    def get_parent(self, state):
        # Find parent of the given state
        for parent, child in self.tree['nodes'].items():
            if state in child['thoughts']:
                return parent
        return None
 
 
######################
class TreeofThoughtsASearch:
    def __init__(self, model):
        self.model = model
        self.logger = logging.getLogger(__name__)
        self.log_stream = io.StringIO()
        stream_handler = logging.StreamHandler(self.log_stream)
        self.logger.addHandler(stream_handler)
        self.logger.setLevel(logging.INFO)
 
    def solve(self, initial_prompt, num_thoughts=5, max_steps=30, pruning_threshold=0.4):
        # Initialize the open set with the initial prompt
        open_set = PriorityQueue()
        open_set.put((0, initial_prompt))
 
        # Sets of visited and expanded states
        visited_states = set()
        expanded_states = set()
 
        # g_scores, f_scores, and heuristic function
        g_scores = {initial_prompt: 0}
        f_scores = {initial_prompt: self.heuristic(initial_prompt)}
 
        # Parent tracking for path reconstruction
        came_from = {}
 
        while not open_set.empty() and len(visited_states) < max_steps:
            current_f_score, current_state = open_set.get()
 
            if current_state in expanded_states:
                continue
 
            expanded_states.add(current_state)
            visited_states.add(current_state)
 
            # Goal check (modify this as per the specific application's need)
            if self.is_goal(current_state):
                return self.reconstruct_path(came_from, current_state)
 
            thoughts = self.model.generate_thoughts(current_state, num_thoughts, initial_prompt)
            for thought in thoughts:
                next_state = (current_state, thought) if isinstance(current_state, str) else (*current_state, thought)
 
                if next_state in visited_states:
                    continue
 
                tentative_g_score = g_scores[current_state] + self.cost(current_state, next_state)
                if next_state not in g_scores or tentative_g_score < g_scores[next_state]:
                    came_from[next_state] = current_state
                    g_scores[next_state] = tentative_g_score
                    f_scores[next_state] = tentative_g_score + self.heuristic(next_state)
                    open_set.put((f_scores[next_state], next_state))
 
        # No solution found
        return None
 
    def heuristic(self, state):
        # Define the heuristic function here
        # For example, it might be based on the evaluation of the state
        # This function should never overestimate the cost to reach the goal
        return self.model.evaluate_state_heuristic(state)
 
    def cost(self, current_state, next_state):
        # Define the cost function between states
        # This could be a constant value or based on some criteria
        return 1  # Example: constant cost
 
    def is_goal(self, state):
        # Define the goal check here
        # The function should return True if the state meets the goal criteria
        return self.model.check_goal(state)
 
    def reconstruct_path(self, came_from, current_state):
        # Reconstruct the path from the start state to the goal state
        path = []
        while current_state in came_from:
            path.append(current_state)
            current_state = came_from[current_state]
        path.reverse()
        return path
 
######################
class MonteCarloTreeofThoughts(TreeofThoughts):
    def __init__(self, model, objective="balance"):
        super().__init__(model)
        self.objective = objective
        self.tree = {"nodes": {}}
 
    def solve(self, initial_prompt, num_thoughts, max_steps, max_states, exploration_constant):
        self.tree["nodes"][initial_prompt] = {"visits": 0, "value": 0, "children": []}
        for _ in range(max_steps):
            leaf = self.select(initial_prompt)
            self.expand(leaf, num_thoughts, initial_prompt)
            winner = self.simulate(leaf)
            self.backpropagate(leaf, winner)
 
        best_state = max(self.tree["nodes"][initial_prompt]["children"], key=lambda child: self.tree["nodes"][child]["value"])
        return best_state
 
    def select(self, state):
        while self.tree["nodes"][state]["children"]:
            state = max(self.tree["nodes"][state]["children"], key=lambda child: self.ucb1(child))
        return state
 
    def expand(self, state, num_thoughts, initial_prompt):
        if state not in self.tree["nodes"]:
            self.tree["nodes"][state] = {"visits": 0, "value": 0, "children": []}
            thoughts = self.model.generate_thoughts(state, num_thoughts, initial_prompt)
            for thought in thoughts:
                next_state = (state, thought) if isinstance(state, str) else (*state, thought)
                self.tree["nodes"][state]["children"].append(next_state)
                self.tree["nodes"][next_state] = {"visits": 0, "value": 0, "children": []}
 
    def simulate(self, state):
        # Implement the simulation logic specific to the problem domain
        # For example, randomly select a child and evaluate its value
        return np.random.choice(self.tree["nodes"][state]["children"])
 
    def backpropagate(self, state, winner):
        while state:
            self.tree["nodes"][state]["visits"] += 1
            if winner in self.tree["nodes"][state]["children"]:
                self.tree["nodes"][state]["value"] += 1
            state = self.get_parent(state)
 
    def ucb1(self, child):
        parent_visits = self.tree["nodes"][self.get_parent(child)]["visits"]
        child_visits = self.tree["nodes"][child]["visits"]
        if child_visits == 0:
            return float('inf')
        win_rate = self.tree["nodes"][child]["value"] / child_visits
        return win_rate + 2 * np.sqrt(np.log(parent_visits) / child_visits)
 
    def get_parent(self, node):
        for parent, data in self.tree["nodes"].items():
            if node in data["children"]:
                return parent
        return None
 
######################
 
class TextGenerationWebUILanguageModel(AbstractLanguageModel):
    def __init__(self, strategy="cot", evaluation_strategy="value"):
        thought_generator_params = ui_default_parameters()
        thought_generator_params["max_new_tokens"] = 50
        thought_generator_params["temperature"] = 0.5
        self.thought_generator = build_text_generation_web_ui_client_llm(parameters=thought_generator_params)
 
        state_voter_params = ui_default_parameters()
        state_voter_params["max_new_tokens"] = 10
        state_voter_params["temperature"] = 0.2
        self.state_voter = build_text_generation_web_ui_client_llm(parameters=state_voter_params)  
 
        value_evaluator_params = ui_default_parameters()
        value_evaluator_params["max_new_tokens"] = 10
        value_evaluator_params["temperature"] = 0.2
        self.value_evaluator = build_text_generation_web_ui_client_llm(parameters=value_evaluator_params)
 
 
        self.strategy = strategy
        self.evaluation_strategy = evaluation_strategy
 
    def generate_thoughts(self, state, k, initial_prompt, rejected_solutions=None):
        if self.strategy == "cot":
            if (type(state) == str):
                state_text = state
            else:
                state_text = '\n'.join(state)
            print("New state generating thought:", state, "\n\n")
            # prompt = f"Considering the thoughts you've had until now:\n\n{state_text}\n\nDevise the next coherent thought that will aid in advancing the reasoning process and achieving a solution to {initial_prompt}. Assess various scenarios, think unconventionally, anticipate potential challenges, and resolve any outstanding queries. Tap into your mind's full potential and make certain no open questions remain."
            prompt = f"""You're an TreeofThoughts, an superintelligent AI model devoted to helping Humans by any means necessary. You're purpose is to generate a series of solutions to comply with the user's instructions, you must generate solutions on the basis of determining the most reliable solution in the shortest amount of time, while taking rejected solutions into account and learning from them.
            Considering the reasoning provided:\n\n
            ###'{state_text}'\n\n###
            Devise the best possible solution for the task: {initial_prompt}, Here are evaluated solutions that were rejected:
            ###{rejected_solutions}###,
            complete the {initial_prompt} without making the same mistakes you did with the evaluated rejected solutions. Be simple. Be direct. Provide intuitive solutions as soon as you think of them."""
            
            prompt += self.ReAct_prompt
            # print(prompt)
            thoughts = self.generate_text(prompt, k)
            # print(thoughts)
            # print(f"Generated thoughts: {thoughts}")
            return thoughts
 
    def evaluate_states(self, states, initial_prompt):
        if not states:
            return {}
 
        if self.evaluation_strategy == 'value':
            state_values = {}
            for state in states:
                if (type(state) == str):
                    state_text = state
                else:
                    state_text = '\n'.join(state)
                print("We receive a state of type", type(state), "For state: ", state, "\n\n")
                # prompt = f"Given the current state of reasoning: '{state_text}', evaluate its value as a float between 0 and 1, become very pessimistic think of potential adverse risks on the probability of this state of reasoning achieveing {initial_prompt} and DO NOT RESPOND WITH ANYTHING ELSE: OTHER THAN AN FLOAT"
                prompt = f""" To achieve the following goal: '{initial_prompt}', pessimistically value the context of the past solutions and more importantly the latest generated solution you had AS A FLOAT BETWEEN 0 AND 1\n
                    Past solutions:\n\n
                    {state_text}\n       
                    If the solutions is not directly concretely making fast progress in achieving the goal, give it a lower score.
                    Evaluate all solutions AS A FLOAT BETWEEN 0 and 1:\n,  DO NOT RETURN ANYTHING ELSE
                """
                # and then inside backticks provide an simple and direct bulletpoint list as to why you evaluated this thought the way you did. Provide simple yet intuitive feedback.
                
                response = self.openai_api_call_handler(prompt, 10, 1)
                try:
                    value_text = self.openai_choice2text_handler(response.choices[0])
                    # print(f'state: {value_text}')
                    value = float(value_text)
                    print(f"Evaluated Thought Value: {value}")
                except ValueError:
                    value = 0  # Assign a default value if the conversion fails
                state_values[state] = value
            return state_values
 
        elif self.evaluation_strategy == 'vote':
            states_text = '\n'.join([' '.join(state) for state in states])
 
            prompt = f"Given the following states of reasoning, vote for the best state utilizing an scalar value 1-10:\n{states_text}\n\nVote, on the probability of this state of reasoning achieveing {initial_prompt} and become very pessimistic very NOTHING ELSE"
 
            response = self.openai_api_call_handler(prompt, 50, 1)
 
            print(f'state response: {response}')
 
            best_state_text = self.openai_choice2text_handler(response.choices[0])
 
            print(f"Best state text: {best_state_text}")
 
            best_state = tuple(best_state_text.split())
 
            print(f'best_state: {best_state}')
 
            return {state: 1 if state == best_state else 0 for state in states}
 
        else:
            raise ValueError("Invalid evaluation strategy. Choose 'value' or 'vote'.")
 
