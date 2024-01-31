import os
import time
import requests
import logging
from tree_of_thoughts.models.abstract_language_model import AbstractLanguageModel
import concurrent.futures

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LlamacppLanguageModel(AbstractLanguageModel):
    def __init__(self, server_url, model_name, strategy="tot", evaluation_strategy="value", enable_ReAct_prompting=False):
        self.server_url = server_url
        self.model_name = model_name
        self.strategy = strategy
        self.evaluation_strategy = evaluation_strategy
        self.ReAct_prompt = ''
        if enable_ReAct_prompting:
            self.ReAct_prompt = "Write down your observations in format 'Observation:xxxx', then write down your thoughts in format 'Thoughts:xxxx'."

    def server_api_call_handler(self, prompt, n_predict, temperature, top_k, top_p):
        headers = {"Content-Type": "application/json"}
        data = {
            "prompt": prompt,
            "n_predict": n_predict,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "echo": True,
        }
        try:
            response = requests.post(self.server_url, json=data, headers=headers)
            if response.status_code == 200:
                response_data = response.json()
                if "content" in response_data:
                    return response_data["content"]
                else:
                    logger.error("Response does not contain 'content' key")
                    return None
            else:
                logger.error(f"Error generating text: {response}")
                return None
        except Exception as e:
            logger.error(f"Exception occurred while generating text: {e}")
            return None

    def generate_text(self, prompt, n_predict=2048, temperature=0.1, top_k=200, top_p=0.99):
        enhanced_prompt = prompt + self.ReAct_prompt
        response_content = self.server_api_call_handler(enhanced_prompt, n_predict, temperature, top_k, top_p)
        return response_content

    def generate_thoughts(self, state, k, initial_prompt, rejected_solutions=None, n_predict=2048, temperature=0.1, top_k=200, top_p=0.99):
        if isinstance(state, str):
            state_text = state
        else:
            state_text = '\n'.join(state)
        print("New state generating thought:", state, "\n\n")

        prompt = f"""
        As TreeofThoughts, a superintelligent AI model, your primary goal is to assist humans by all necessary means. Your task involves generating a series of solutions that adhere to the user's instructions. Focus on identifying the most reliable solution in the shortest possible time. Importantly, consider previously rejected solutions, learning from them to avoid repeating the same errors.

        Given the context:\n\n
        ### '{state_text}'\n\n###
        Your objective is to formulate the optimal solution for the task: {initial_prompt}. Take into account these evaluated but rejected solutions: 
        ###{rejected_solutions}###. 
        Ensure to complete the {initial_prompt} task by not repeating the mistakes associated with the rejected solutions."""

        prompt += self.ReAct_prompt

        thoughts = []
        for _ in range(k):
            response_content = self.generate_text(prompt, n_predict, temperature, top_k, top_p)
            thoughts.append(response_content)
        return thoughts

    def generate_solution(self, initial_prompt, state, rejected_solutions=None, n_predict=2048, temperature=0.1, top_k=200, top_p=0.99):
        try:
            if isinstance(state, list):
                state_text = '\n'.join(state)
            else:
                state_text = state

            prompt = f"""
            As TreeofThoughts, a superintelligent AI model, your primary goal is to assist humans by all necessary means. Your task involves generating a series of solutions that adhere to the user's instructions. Focus on identifying the most reliable solution in the shortest possible time. Importantly, consider previously rejected solutions, learning from them to avoid repeating the same errors.

            Given the context:\n\n
            ### '{state_text}'\n\n###
            Your objective is to formulate the optimal solution for the task: {initial_prompt}. Take into account these evaluated but rejected solutions: 
            ###{rejected_solutions}###. 
            Ensure to complete the {initial_prompt} task by not repeating the mistakes associated with the rejected solutions."""

            prompt += self.ReAct_prompt

            response_content = self.generate_text(prompt, n_predict, temperature, top_k, top_p)
            return response_content
        except Exception as e:
            logger.error(f"Error in generate_solution: {e}")
            return None

    def evaluate_states(self, states, initial_prompt, n_predict=2048, temperature=0.1, top_k=200, top_p=0.99):
        if not states:
            return {}

        state_values = {}
        for state in states:
            if isinstance(state, str):
                state_text = state
            else:
                state_text = '\n'.join(state)
            print("We receive a state of type", type(state), "For state: ", state, "\n\n")

            prompt = f"""
            To achieve the goal: '{initial_prompt}', consider the context of previous solutions and, more critically, the most recently generated solution. Assign a value to each solution as a float ranging between 0 and 1.\n\n
            Previous solutions:\n\n
            {state_text}\n
            Evaluate all solutions, assigning each a float value between 0 and 1. Do not return any other information."""

            response_content = self.generate_text(prompt, n_predict, temperature, top_k, top_p)
            try:
                value = float(response_content)
                state_values[state] = value
            except ValueError:
                state_values[state] = 0  # Assign a default value if the conversion fails

        return state_values

class OptimizedLlamacppLanguageModel(LlamacppLanguageModel):
    def __init__(self, server_url, model_name, strategy="cot", evaluation_strategy="value", cache_enabled=True, enable_ReAct_prompting=False):
        super().__init__(server_url, model_name, strategy, evaluation_strategy, enable_ReAct_prompting)
        self.cache_enabled = cache_enabled
        self.thought_cache = {}
        self.state_evaluation_cache = {}

    def parallel_generate_thoughts(self, states, k):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            thoughts = list(executor.map(lambda state: self.generate_thoughts(state, k), states))
            print(f"Parallel generated thoughts: {thoughts}")
        return thoughts

    def parallel_evaluate_states(self, states, initial_prompt):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            state_values = list(executor.map(self.evaluate_states, states, initial_prompt))
            print(f"Parallel evaluated state values: {state_values}")
        return state_values
