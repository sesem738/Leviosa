import os
from typing import List, Tuple
from abc import ABC, abstractmethod
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI, ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.llms import LlamaCpp
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from dotenv import load_dotenv
from PIL import Image
import base64

# Set environment variables for API keys
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
os.environ["ANTHROPIC_API_KEY"] = os.getenv('ANTHROPIC_API_KEY')
os.environ["GOOGLE_API_KEY"] = os.getenv('GEMINI_API_KEY')

class LLMInterface(ABC):
    """Abstract base class for LLM interfaces."""
    
    @abstractmethod
    def process_input(self, text: str, image_path: str) -> str:
        """Process text and image input and return a response."""
        pass


class GPT4Interface(LLMInterface):
    """Interface for GPT-4 model."""
    
    def __init__(self):
        self.model = ChatOpenAI(model_name="gpt-4o", max_tokens=300)
    
    def process_input(self, text: str, image_path: str) -> str:
        with open(image_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
        
        messages = [
            HumanMessage(
                content=[
                    {"type": "text", "text": text},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{encoded_image}"
                        }
                    }
                ]
            )
        ]
        response = self.model.invoke(messages)
        return response.content

class ClaudeInterface(LLMInterface):
    """Interface for Claude model."""
    
    def __init__(self):
        self.model = ChatAnthropic(model_name="claude-3-opus-20240229")
    
    def process_input(self, text: str, image_path: str) -> str:
        image = Image.open(image_path)
        messages = [
            HumanMessage(
                content=[
                    {"type": "text", "text": text},
                    {"type": "image_url", "image_url": image_path}
                ]
            )
        ]
        response = self.model.invoke(messages)
        return response.content

# class LlamaInterface(LLMInterface):
#     """Interface for Llama model."""
    
#     def __init__(self):
#         self.model = LlamaCpp(model_path="path/to/llama/model.bin")
    
#     def process_input(self, text: str, image_path: str) -> str:
#         # Note: LlamaCpp doesn't support image input directly
#         response = self.model(f"Image description: [Describe image here]\n\nText: {text}")
#         return response

class GeminiInterface(LLMInterface):
    """Interface for Gemini model."""
    
    def __init__(self):
        self.model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
    
    def process_input(self, text: str, image_path: str) -> str:
        image = Image.open(image_path)
        messages = [
            HumanMessage(
                content=[
                    {"type": "text", "text": text},
                    {"type": "image_url", "image_url": image_path}
                ]
            )
        ]
        response = self.model.invoke(messages)
        return response.content

def run_experiment(models: List[LLMInterface], prompts: List[Tuple[str, str]]) -> List[dict]:
    """
    Run the experiment with given models and prompts.
    
    Args:
    models (List[LLMInterface]): List of LLM interfaces to test
    prompts (List[Tuple[str, str]]): List of (text, image_path) tuples
    
    Returns:
    List[dict]: List of dictionaries containing results for each model and prompt
    """
    results = []
    
    for model in models:
        model_name = model.__class__.__name__
        for text, image_path in prompts:
            try:
                response = model.process_input(text, image_path)
                results.append({
                    "model": model_name,
                    "text_prompt": text,
                    "image_path": image_path,
                    "response": response
                })
            except Exception as e:
                results.append({
                    "model": model_name,
                    "text_prompt": text,
                    "image_path": image_path,
                    "error": str(e)
                })
    
    return results

def main():
    # Initialize models
    models = [
        GPT4Interface(),
        ClaudeInterface(),
        # LlamaInterface(),
        GeminiInterface()
    ]
    
    # Define prompts (text and image pairs)
    prompts = [
        ("Describe the contents of this image.", "/Users/pascaldao/Dev/Leviosa/sources/data/plots/waypoints_20240621_165925.png"),
    ]
    
    # Run the experiment
    results = run_experiment(models, prompts)
    
    # Print or process results
    for result in results:
        print(f"Model: {result['model']}")
        print(f"Text Prompt: {result['text_prompt']}")
        print(f"Image Path: {result['image_path']}")
        if 'response' in result:
            print(f"Response: {result['response']}")
        else:
            print(f"Error: {result['error']}")
        print("---")

if __name__ == "__main__":
    main()
