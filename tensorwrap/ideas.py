import os
import jinja2
from typing import List

# Try to import different LLM providers, with fallbacks
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import google.generativeai as genai
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False

class IdeaGenerator:
    """Generates optimization ideas for kernels using LLMs."""
    
    def __init__(self, mock_mode=False):
        """Initialize the idea generator.
        
        Args:
            mock_mode: If True, use mock responses for dry runs instead of actual LLM calls
        """
        self.template_loader = jinja2.FileSystemLoader("tensorwrap/templates")
        self.template_env = jinja2.Environment(loader=self.template_loader)
        self.brainstorm_template = self.template_env.get_template("brainstorm.j2")
        self.mock_mode = mock_mode
        
        # Setup LLM providers if available and not in mock mode
        if not mock_mode:
            if OPENAI_AVAILABLE and os.environ.get("OPENAI_API_KEY"):
                self.provider = "openai"
                self.openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            elif GOOGLE_AVAILABLE and os.environ.get("GOOGLE_API_KEY"):
                self.provider = "google"
                genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
            else:
                raise RuntimeError("No LLM provider available. Set OPENAI_API_KEY or GOOGLE_API_KEY.")
        else:
            self.provider = "mock"
    
    def generate_ideas(self, baseline_code: str, num_ideas: int = 4) -> List[str]:
        """Generate optimization ideas for a baseline kernel code.
        
        Args:
            baseline_code: The baseline kernel code to optimize
            num_ideas: Number of ideas to generate
            
        Returns:
            List of optimization ideas as strings
        """
        prompt = self.brainstorm_template.render(code=baseline_code)
        
        if self.mock_mode:
            return self._generate_mock_ideas(num_ideas)
        elif self.provider == "openai":
            return self._generate_with_openai(prompt, num_ideas)
        else:  # google
            return self._generate_with_google(prompt, num_ideas)
    
    def _generate_with_openai(self, prompt: str, num_ideas: int) -> List[str]:
        """Generate ideas using OpenAI API.
        
        Args:
            prompt: The prompt to send to the API
            num_ideas: Number of ideas to generate
            
        Returns:
            List of generated ideas
        """
        # Using the latest o3 model
        response = self.openai_client.chat.completions.create(
            model="o3-2025-04-16",  # latest o3 model
            messages=[
                {"role": "system", "content": "You are an expert CUDA kernel optimizer."},
                {"role": "user", "content": prompt}
            ]
            # o3-2025-04-16 only supports the default temperature (1.0)
        )
        
        # Parse response and extract ideas
        text = response.choices[0].message.content
        ideas = self._parse_ideas(text, num_ideas)
        return ideas
    
    def _generate_with_google(self, prompt: str, num_ideas: int) -> List[str]:
        """Generate ideas using Google Generative AI API.
        
        Args:
            prompt: The prompt to send to the API
            num_ideas: Number of ideas to generate
            
        Returns:
            List of generated ideas
        """
        # TODO: Implement actual Google Generative AI API call
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(prompt)
        
        # Parse response and extract ideas
        text = response.text
        ideas = self._parse_ideas(text, num_ideas)
        return ideas
    
    def _generate_mock_ideas(self, num_ideas: int) -> List[str]:
        """Generate mock ideas for testing without LLM API calls.
        
        Args:
            num_ideas: Number of ideas to generate
            
        Returns:
            List of mock optimization ideas
        """
        mock_ideas = [
            "Use shared memory to cache frequently accessed data and reduce global memory accesses",
            "Implement loop tiling/blocking to improve cache locality", 
            "Unroll inner loops to reduce loop overhead and increase instruction-level parallelism",
            "Vectorize operations using SIMD instructions for better throughput",
            "Optimize memory access patterns to ensure coalesced memory access",
            "Reduce thread synchronization points to minimize overhead"
        ]
        return mock_ideas[:num_ideas]
        
    def _parse_ideas(self, text: str, num_ideas: int) -> List[str]:
        """Parse ideas from LLM response.
        
        Args:
            text: Text response from LLM
            num_ideas: Expected number of ideas
            
        Returns:
            List of parsed ideas
        """
        # This is a simple parsing logic; may need to be more robust in practice
        lines = text.split('\n')
        ideas = []
        
        for line in lines:
            line = line.strip()
            # Look for numbered lists like "1." or "1)"
            if (line.startswith("- ") or 
                any(line.startswith(f"{i}. ") or line.startswith(f"{i}) ") for i in range(1, num_ideas + 1))):
                ideas.append(line.split(" ", 1)[1] if " " in line else line)
        
        # Ensure we have the correct number of ideas
        return ideas[:num_ideas]