from llama_cpp import Llama
from config.config_Meta_Llama_3_1_8B_Instruct_Q4_K_M import config
import config.paths as paths

class LLMManager:
    def __init__(self):
        self.llm = None
        self.load_model()

    def load_model(self):
        print("Loading Llama 3.1")
        self.llm = Llama(
            model_path=paths.llm_model,
            **config["load_params"],
        )
        print("Llama 3.1 loaded")

    def generate_response(self, messages, stream=False):
        if stream:
            return self._generate_stream_response(messages)
        return self._generate_single_response(messages)

    def _generate_single_response(self, messages):
        response = self.llm.create_chat_completion(
            messages=messages,
            max_tokens=500,
            temperature=config["inference_params"]["temp"]
        )
        yield "", response["choices"][0]["message"]["content"]

    def _generate_stream_response(self, messages):
        response_text = ""
        for token in self.llm.create_chat_completion(
            messages=messages,
            max_tokens=500,
            temperature=config["inference_params"]["temp"],
            stream=True
        ):
            token_text = token["choices"][0]["delta"].get("content", "")
            response_text += token_text
            yield token_text, response_text