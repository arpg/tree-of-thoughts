import os
import openai
import time
from  tree_of_thoughts.models.abstract_language_model import AbstractLanguageModel
import concurrent.futures
import logging 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OpenAILanguageModel(AbstractLanguageModel):
    def __init__(self, api_key, strategy="cot", evaluation_strategy="value", api_base="", api_model="", enable_ReAct_prompting=True):
        if api_key == "" or api_key is None:
            api_key = os.environ.get("OPENAI_API_KEY", "")
        if api_key != "":
            openai.api_key = api_key
        else:
            raise Exception("Please provide OpenAI API key")

        if api_base == ""or api_base is None:
            api_base = os.environ.get("OPENAI_API_BASE", "")
        if api_base != "":
            openai.api_base = api_base
            logger.info(f'Using custom api_base {api_base}')
            
        if api_model == "" or api_model is None:
            api_model = os.environ.get("OPENAI_API_MODEL", "")
        if api_model != "":
            self.api_model = api_model
        else:
            self.api_model = "text-davinci-003"
        logger.info(f'Using api_model {self.api_model}')

        self.use_chat_api = 'gpt' in self.api_model
        self.ReAct_prompt = ''
        if enable_ReAct_prompting:
            self.ReAct_prompt = "Write down your observations in format 'Observation:xxxx', then write down your thoughts in format 'Thoughts:xxxx'."
        
        self.strategy = strategy
        self.evaluation_strategy = evaluation_strategy

    def openai_api_call_handler(self, prompt, max_tokens, temperature, k=1, stop=None):
        while True:
            try:
                if self.use_chat_api:
                    messages = [{"role": "user", "content": prompt}]
                    response = openai.ChatCompletion.create(model=self.api_model, messages=messages, max_tokens=max_tokens, temperature=temperature)
                else:
                    response = openai.Completion.create(engine=self.api_model, prompt=prompt, n=k, max_tokens=max_tokens, stop=stop, temperature=temperature)
                logger.info(f"API Response: {response}")
                return response
            except openai.error.RateLimitError as e:
                sleep_duration = int(os.environ.get("OPENAI_RATE_TIMEOUT", 30))
                logger.warning(f'Rate limit error: {e}. Sleeping for {sleep_duration} seconds.')
                time.sleep(sleep_duration)
            except Exception as e:
                logger.error(f"API call exception: {e}")
                break

    def openai_choice2text_handler(self, choice):
        if self.use_chat_api:
            text = choice['message']['content']
        else:
            text = choice.text.strip()
        return text
    
    def generate_text(self, prompt, k):
        thoughts = []
        try:
            if self.use_chat_api:
                for _ in range(k):
                    response = self.openai_api_call_handler(prompt, 400, 0.5, k)
                    text = self.openai_choice2text_handler(response.choices[0])
                    thoughts += [text]
            else:
                response = self.openai_api_call_handler(prompt, 300, 0.5, k)
                thoughts = [self.openai_choice2text_handler(choice) for choice in response.choices]
            logger.info(f"Generated thoughts: {thoughts}")
        except Exception as e:
            logger.error(f"Error in generate_text: {e}")
        return thoughts

    def generate_thoughts(self, state, k, initial_prompt, rejected_solutions=None):
        state_text = state if isinstance(state, str) else '\n'.join(state)
        prompt = f"Considering the thoughts you've had until now:\n\n{state_text}\n\nDevise the next coherent thought that will aid in advancing the reasoning process and achieving a solution to {initial_prompt}."
        prompt += self.ReAct_prompt
        thoughts = self.generate_text(prompt, k)
        logger.info(f"Generated thoughts for state: {state_text}")
        return thoughts

    def generate_solution(self, initial_prompt, state, rejected_solutions=None):
        try:
            state_text = state if isinstance(state, str) else '\n'.join(state)
            prompt = f"Given the current state '{state_text}', generate a solution for the task: {initial_prompt}."
            answer = self.generate_text(prompt, 1)[0]  # Assuming generate_text returns a list
            logger.info(f"Generated solution: {answer}")
            return answer
        except Exception as e:
            logger.error(f"Error in generate_solution: {e}")
            return None

    def evaluate_states(self, states, initial_prompt):
        state_values = {}
        try:
            for state in states:
                state_text = state if isinstance(state, str) else '\n'.join(state)
                prompt = f"Given the current state '{state_text}', evaluate its value for achieving {initial_prompt}."
                response = self.openai_api_call_handler(prompt, 10, 1)
                value_text = self.openai_choice2text_handler(response.choices[0])
                try:
                    value = float(value_text)
                except ValueError:
                    value = 0  # default value in case of conversion failure
                state_values[state] = value
                logger.info(f"Evaluated state value for '{state_text}': {value}")
        except Exception as e:
            logger.error(f"Error in evaluate_states: {e}")
        return state_values

class OptimizedOpenAILanguageModel(OpenAILanguageModel):
    def __init__(self, api_key, strategy="cot", evaluation_strategy="value", cache_enabled=True, api_base="", api_model="", enable_ReAct_prompting=False):
        super().__init__(api_key, strategy, evaluation_strategy, api_base, api_model, enable_ReAct_prompting)
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
    
