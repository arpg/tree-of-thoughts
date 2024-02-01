import os
from tree_of_thoughts.models.llamacpp_models import LlamacppLanguageModel
from tree_of_thoughts.treeofthoughts import TreeofThoughtsBFS

# Server URL and Model Name for Llamacpp Language Model
server_url = "http://172.206.254.216:8040/completion"  # Replace with your server URL

# Initialize the LlamacppLanguageModel class with the server URL and model name
model = LlamacppLanguageModel(model_name="", server_url=server_url)

# Initialize the MonteCarloTreeofThoughts class with the Llamacpp model
tree_of_thoughts = TreeofThoughtsBFS(model)

# Craft an initial prompt for your task
initial_prompt = "Five philosophers dine together at the same table. Each philosopher has his own plate at the table. There is a fork between each plate. The dish served is a kind of spaghetti which has to be eaten with two forks. Each philosopher can only alternately think and eat. Moreover, a philosopher can only eat his spaghetti when he has both a left and right fork. Thus two forks will only be available when his two nearest neighbors are thinking, not eating. After an individual philosopher finishes eating, he will put down both forks. The problem is how to design a regimen (a concurrent algorithm) such that any philosopher will not starve; i.e., each can forever continue to alternate between eating and thinking, assuming that no philosopher can know when others may want to eat or think (an issue of incomplete information)."

"""
Input: 2 8 8 14
Possible next steps:
2 + 8 = 10 (left: 8 10 14)
8 / 2 = 4 (left: 4 8 14)
14 + 2 = 16 (left: 8 8 16)
2 * 8 = 16 (left: 8 14 16)
8 - 2 = 6 (left: 6 8 14)
14 - 8 = 6 (left: 2 6 8)
14 /  2 = 7 (left: 7 8 8)
14 - 2 = 12 (left: 8 8 12)
Input: use 4 numbers and basic arithmetic operations (+-*/) to obtain 24 in 1 equation
Possible next steps:
"""

# Set the parameters for solving the task
num_thoughts = 5
max_steps = 5
max_states = 10
pruning_threshold = 0.5

# Solve the task using the Tree of Thoughts method
solution = tree_of_thoughts.solve(
    initial_prompt=initial_prompt,
    num_thoughts=num_thoughts, 
    max_steps=max_steps, 
    max_states=max_states, 
    pruning_threshold=pruning_threshold,
)

print(f"Solution: {solution}")
