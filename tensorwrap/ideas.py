import os
import jinja2
from typing import List

from openai import OpenAI
import google.generativeai as genai
import pandas as pd
from .llm_api import LLMClient


class IdeaGenerator(LLMClient):
    """Generates optimization ideas for kernels using LLMs."""
    
    def __init__(self, mock_mode=False):
        """Initialize the idea generator.
        
        Args:
            mock_mode: If True, use mock responses for dry runs instead of actual LLM calls
        """
        super().__init__(mock_mode)

        self.template_loader = jinja2.FileSystemLoader("tensorwrap/templates")
        self.template_env = jinja2.Environment(loader=self.template_loader)
        self.brainstorm_template = self.template_env.get_template("brainstorm.j2")

    
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
        elif self.provider == "OpenAI":
            return self._generate_with_openai(prompt, num_ideas)
        elif self.provider == "Google":
            return self._generate_with_google(prompt, num_ideas)
    
    def _generate_with_openai(self, prompt: str, num_ideas: int) -> List[str]:
        """Generate ideas using OpenAI API.
        
        Args:
            prompt: The prompt to send to the API
            num_ideas: Number of ideas to generate
            
        Returns:
            List of generated ideas
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an expert CUDA kernel optimizer."},
                {"role": "user", "content": prompt}
            ]
        )
        
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
        response = self.client.generate_content(prompt)
        
        text = response.text
        # print("." * 80)
        # print(text)
        # print("." * 80)
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
        lines = text.split('\n')
        ideas = []
        
        cur_idea = ""
        for line in lines:
            line = line.strip()
            if "StartIdea" in line:
                cur_idea = ""
            elif "EndIdea" in line:
                ideas.append(cur_idea.strip())
            else:
                cur_idea += line + "\n"
        
        return ideas[:num_ideas]