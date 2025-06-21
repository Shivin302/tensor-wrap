import os
import re
import jinja2
from typing import List, Optional, Dict, Any, Union

from .llm_api import LLMClient
from .problems.matmul import example_kernels



class CodeGenerator(LLMClient):
    """Generates optimized kernel code from ideas."""
    
    def __init__(self, mock_mode=False, problem_path=None, max_iterations=3):
        """Initialize the code generator.
        
        Args:
            mock_mode: If True, use mock responses for testing without API calls
            problem_path: Path to the problem specification directory
            max_iterations: Maximum number of optimization iterations for self-healing
        """
        super().__init__(mock_mode)

        self.template_loader = jinja2.FileSystemLoader("tensorwrap/templates")
        self.template_env = jinja2.Environment(loader=self.template_loader)
        self.implement_template = self.template_env.get_template("implement.j2")
        self.problem_path = problem_path
        self.max_iterations = max_iterations

    
    def generate_code(self, baseline_code: str, idea: str, problem_spec: Any = None) -> Union[str, Dict[str, Any]]:
        """Generate optimized kernel code from an idea.
        
        Args:
            baseline_code: The baseline kernel code to optimize
            idea: The optimization idea to implement
            problem_spec: The problem specification (for self-healing optimizer)
            
        Returns:
            Generated optimized code or a dict with code and metadata if self-healing is used
        """
        prompt = self.implement_template.render(baseline_code=baseline_code, idea=idea, problem_spec=problem_spec)
        
        if self.mock_mode:
            return self._generate_mock_code(baseline_code, idea)
        elif self.provider == "OpenAI":
            return self._generate_with_openai(prompt)
        else:  # google
            return self._generate_with_google(prompt)
    
    def _generate_with_openai(self, prompt: str) -> str:
        """Generate code using OpenAI API.
        
        Args:
            prompt: The prompt to send to the API
            
        Returns:
            The generated code
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an expert CUDA engineer."},
                {"role": "user", "content": prompt}
            ]
        )
        
        text = response.choices[0].message.content
        return self._extract_code(text)
    

    def _generate_with_google(self, prompt: str) -> str:
        """Generate code using Google Generative AI API.
        
        Args:
            prompt: The prompt to send to the API
            
        Returns:
            The generated code
        """
        response = self.client.generate_content(prompt)
        
        text = response.text
        return self._extract_code(text)
    

    def _generate_mock_code(self, baseline_code: str, idea: str) -> str:
        """Generate mock optimized kernel code for testing without LLM API calls.
        
        Args:
            baseline_code: The baseline kernel code to optimize
            idea: The optimization idea to implement
            
        Returns:
            Mock optimized kernel code
        """
        # Create a simple optimization based on the idea
        if "shared memory" in idea.lower():
            return example_kernels.shared_memory
        elif "tiling" in idea.lower() or "blocking" in idea.lower():
            return example_kernels.tiling
        elif "unroll" in idea.lower():
            return example_kernels.unroll
        else:
            # Default optimization with vectorization
            return example_kernels.vectorize
    

    def _extract_code(self, text: str) -> Optional[str]:
        """Extract code from LLM response.
        
        Args:
            text: The LLM response text
            
        Returns:
            Extracted code or None if no code found
        """
        # Look for code blocks wrapped in triple backticks
        code_regex = r'```(?:cpp|c\+\+)?\s*\n([\s\S]*?)\n```'
        match = re.search(code_regex, text)
        
        if match:
            return match.group(1).strip()
        
        # Fallback: if no code blocks found, return entire response
        # In practice, you'd want to make this more robust
        return text.strip()