from openai import OpenAI
import google.generativeai as genai
import pandas as pd



API_FPATH = "/home/shivin/repos/tensor-wrap/api_keys.csv"

class LLMClient:
    """Generates optimization ideas for kernels using LLMs."""
    
    def __init__(self, mock_mode=False):
        """Initialize the idea generator.
        
        Args:
            mock_mode: If True, use mock responses for dry runs instead of actual LLM calls
        """
        self.api_keys = pd.read_csv(API_FPATH).set_index("Name").to_dict()["API_Key"]
        self.mock_mode = mock_mode
        
        # Setup LLM providers if available and not in mock mode
        if not mock_mode:
            if "OpenAI" in self.api_keys and not pd.isna(self.api_keys["OpenAI"]):
                self.provider = "OpenAI"
                self.client = OpenAI(api_key=self.api_keys["OpenAI"])
                self.model = "o3-2025-04-16"
            elif "Google" in self.api_keys and not pd.isna(self.api_keys["Google"]):
                self.provider = "Google"
                genai.configure(api_key=self.api_keys["Google"])
                # self.model = "gemini-2.0-flash"
                self.model = "gemini-2.5-flash-lite-preview-06-17"
                # self.model = "gemini-2.5-flash-preview-05-20"
                self.client = genai.GenerativeModel(self.model)
            elif "Deepseek-V3-Free" in self.api_keys and not pd.isna(self.api_keys["Deepseek-V3-Free"]):
                self.provider = "OpenAI"
                self.client = OpenAI(
                    # base_url="https://api.deepseek.com/v1", 
                    base_url="https://openrouter.ai/api/v1", 
                    api_key=self.api_keys["Deepseek-V3-Free"]
                )
                self.model = "deepseek/deepseek-chat-v3-0324:free"
            else:
                raise RuntimeError("No LLM provider available.")
        else:
            self.provider = "mock"
