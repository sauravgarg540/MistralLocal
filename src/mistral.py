import requests
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from .config import MAX_TOKENS, HF_API_KEY


class Mistral:
    def __init__(self, model_id="mistralai/Mistral-7B-Instruct-v0.2", mode='serverless'):
        self.mode = mode
        self.model_id = model_id
        self.historical_messages = []
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        if self.mode == 'serverless':
            self.api_url = f"https://api-inference.huggingface.co/models/{self.model_id}"
            self.headers = {"Authorization": f"Bearer {HF_API_KEY}"}
        else:
            self.bnb_config = BitsAndBytesConfig(load_in_4bit=True,
                                                 bnb_4bit_quant_type='nf4',
                                                 bnb_4bit_use_double_quant=True,
                                                 bnb_4bit_compute_dtype=torch.bfloat16)
            self.device = "cuda:0"
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                trust_remote_code=True,
                quantization_config=self.bnb_config,
                torch_dtype=torch.float16,
                device_map=self.device,
            )

    def generate_response_serverless(self, message: str) -> str:
        """Generate response using the serverless model.
        Args:
            message (str): The message to generate a response for.
        Returns:
            str: The generated response.
        """
        self.historical_messages.append({"role": "user", "content": message})
        text_input = self.tokenizer.apply_chat_template(self.historical_messages,
                                                    add_generation_prompt=True,
                                                    return_tensors="pt",
                                                   tokenize=False)
        payload = {
            "inputs": text_input,
            "parameters": {
                "return_full_text": False,
                "max_new_tokens": MAX_TOKENS,
            }
        }
        response = requests.post(self.api_url, headers=self.headers, json=payload)
        response = response.json()
        output = response[0]["generated_text"]
        self.historical_messages.append({"role": "assistant", "content": output})
        return output

    def generate_response_local(self, message: str):
        """Generate response using the local model.
        Args:
            message (str): The message to generate a response for.
        Returns:
            str: The generated response.
        """
        self.historical_messages.append({"role": "user", "content": message})
        text_input = self.tokenizer.apply_chat_template(self.historical_messages,
                                                   add_generation_prompt=True,
                                                   return_tensors="pt").to(self.device)

        outputs = self.model.generate(text_input,
                                      max_new_tokens=2048,
                                      pad_token_id=self.tokenizer.eos_token_id,

                                      )
        output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        while '[INST]' in output:
            end_index = output.rfind('[/INST]') + len('[/INST]')
            output = output[end_index:]
        self.historical_messages.append({"role": "assistant", "content": output})
        return output

    def generate_response(self, message: str):
        """
        Generate response using the model.
        Args:
            message (str): The message to generate a response for.
        Returns:
            str: The generated response.
        """
        if self.mode == 'serverless':
            return self.generate_response_serverless(message)
        else:
            return self.generate_response_local(message)
