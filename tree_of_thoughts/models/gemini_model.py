import logging
import re
import toml
from google.generativeai import GenerativeModel, configure
from tree_of_thoughts.models.abstract_language_model import AbstractLanguageModel

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Load Google API key from the TOML config file
config = toml.load('config.toml')  # Adjust the path if your config.toml is located elsewhere
GOOGLE_API_KEY = config['google']['api_key']

# Configure the gemini model with the API key
configure(api_key=GOOGLE_API_KEY)
gemini_model = GenerativeModel('gemini-pro')

class GeminiLanguageModel(AbstractLanguageModel):
    def __init__(self, model_name='gemini-pro', strategy="tot", evaluation_strategy="value", enable_ReAct_prompting=False):
        self.model_name = model_name
        self.strategy = strategy
        self.evaluation_strategy = evaluation_strategy
        self.ReAct_prompt = ""
        if enable_ReAct_prompting:
            self.ReAct_prompt = "Write down your observations in format 'Observation:xxxx', then write down your thoughts in format 'Thoughts:xxxx'."

    def generate_text(self, prompt, n_predict=2048, temperature=0.1, top_k=200, top_p=0.99):
        enhanced_prompt = prompt + self.ReAct_prompt
        try:
            # Use the gemini model for text generation
            response = gemini_model.generate_content(enhanced_prompt)
            return response.text
        except Exception as e:
            logger.error(f"Exception occurred while generating text: {e}")
            return None

    def generate_thoughts(
        self,
        state,
        k,
        initial_prompt,
        rejected_solutions=None,
        n_predict=2048,
        temperature=0.1,
        top_k=200,
        top_p=0.99,
    ):
        if isinstance(state, str):
            state_text = state
        else:
            state_text = "\n".join(state)
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
            response_content = self.generate_text(
                prompt, n_predict, temperature, top_k, top_p
            )
            thoughts.append(response_content)
        return thoughts

    def generate_solution(
        self,
        initial_prompt,
        state,
        rejected_solutions=None,
        n_predict=2048,
        temperature=0.1,
        top_k=200,
        top_p=0.99,
    ):
        print("Generate solution called!")
        try:
            if isinstance(state, list):
                state_text = "\n".join(state)
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

            response_content = self.generate_text(
                prompt, n_predict, temperature, top_k, top_p
            )
            return response_content
        except Exception as e:
            logger.error(f"Error in generate_solution: {e}")
            return None

    def evaluate_states(self, states, initial_prompt):
        if not states:
            return {}

        if self.evaluation_strategy == "value":
            state_values = {}
            for state in states:
                if type(state) == str:
                    state_text = state
                else:
                    state_text = "\n".join(state)
                print(
                    "We receive a state of type",
                    type(state),
                    "For state: ",
                    state,
                    "\n\n",
                )
                prompt = f"""To achieve the following goal: '{initial_prompt}', pessimistically value the context of the past solutions and more importantly the latest generated solution you had AS A FLOAT BETWEEN 0 AND 1\n
                    Past solutions:\n\n
                    {state_text}\n       
                    If the solution is not directly concretely making fast progress in achieving the goal, give it a lower score.
                    Evaluate all solutions AS A FLOAT BETWEEN 0 and 1:\n, DO NOT RETURN ANYTHING ELSE
                """

                response = self.generate_text(prompt)
                # Use regular expressions to find floats in the response
                match = re.search(r"[-+]?[0-9]*\.?[0-9]+", response)
                if match:
                    try:
                        value = float(match.group())
                        print(f"Evaluated Thought Value: {value}")
                    except ValueError:
                        value = 0  # Assign a default value if the conversion fails
                else:
                    value = 0  # Assign a default value if no float is found
                state_values[state] = value
            print("Length of state values")
            print(len(state_values))
            return state_values

        elif self.evaluation_strategy == "vote":
            states_text = "\n".join([" ".join(state) for state in states])
            prompt = (
                "Given the following states of reasoning, vote for the best"
                " state utilizing an scalar value"
                f" 1-10:\n{states_text}\n\nVote, on the probability of this"
                f" state of reasoning achieveing {initial_prompt} and become"
                " very pessimistic very NOTHING ELSE"
            )
            response = self.openai_api_call_handler(prompt, 50, 1)
            print(f"state response: {response}")
            best_state_text = self.openai_choice2text_handler(response.choices[0])
            print(f"Best state text: {best_state_text}")
            best_state = tuple(best_state_text.split())
            print(f"best_state: {best_state}")

            return {state: 1 if state == best_state else 0 for state in states}

        else:
            raise ValueError("Invalid evaluation strategy. Choose 'value' or 'vote'.")
