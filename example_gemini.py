import os
from tree_of_thoughts.models.gemini_model import GeminiLanguageModel
from tree_of_thoughts.treeofthoughts import TreeofThoughtsBFS

# Server URL and Model Name for Llamacpp Language Model
server_url = "http://172.206.254.216:8040/completion"  # Replace with your server URL

# Initialize the LlamacppLanguageModel class with the server URL and model name
model = GeminiLanguageModel()

# Initialize the MonteCarloTreeofThoughts class with the Llamacpp model
tree_of_thoughts = TreeofThoughtsBFS(model)

# Craft an initial prompt for your task
initial_prompt = "Write code for calculating the taylor series."

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
num_thoughts = 4
max_steps = 4
max_states = 50
pruning_threshold = 0.3

# Solve the task using the Tree of Thoughts method
solution = tree_of_thoughts.solve(
    initial_prompt=initial_prompt,
    num_thoughts=num_thoughts,
    max_steps=max_steps, 
    max_states=max_states, 
    pruning_threshold=pruning_threshold,
)

print(f"Solution: {solution}")
