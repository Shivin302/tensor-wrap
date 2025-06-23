import os
import re
import jinja2
from typing import List, Optional, Dict, Any, Union

from .llm_api import LLMClient
from .problems.matmul import kernels as matmul_kernels

from dataclasses import dataclass
from pathlib import Path

from .schemas import ProblemSpec, KernelCandidate



@dataclass
class CompileResult:
    """Result of a compilation attempt."""
    success: bool
    error: Optional[str] = None
    output_path: Optional[str] = None


@dataclass
class ValidationResult:
    """Result of a validation attempt."""
    success: bool
    error: Optional[str] = None
    latency_ms: Optional[float] = None
    correctness: Optional[bool] = None


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
                {"role": "system", "content": "You are an expert performance engineer."},
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
            return matmul_kernels.shared_memory
        elif "tiling" in idea.lower() or "blocking" in idea.lower():
            return matmul_kernels.tiling
        elif "unroll" in idea.lower():
            return matmul_kernels.unroll
        else:
            # Default optimization with vectorization
            return matmul_kernels.vectorize
    

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



class SelfHealingCodeGenerator(CodeGenerator):
    """Uses LLM to iteratively debug and fix kernels until a viable candidate is found."""
    
    def __init__(self, problem_path, evaluator, max_iterations=3):
        """Initialize the optimizer.
        
        Args:
            problem_path: Path to problem specification directory
            max_iterations: Maximum number of optimization iterations
        """
        super().__init__(False, problem_path, max_iterations)
        self.evaluator = evaluator
        self.debug_template = self.template_env.get_template("debug_kernel.j2")
        

    def generate_code(self, idea, baseline_code, problem_spec):
        """Generate a viable kernel candidate through LLM-based iterative debugging.
        
        Args:
            idea: The optimization idea
            baseline_code: The baseline kernel code
            problem_spec: Problem specification object
            
        Returns:
            Dictionary with optimized code and metadata, or None if optimization failed
        """
        
        attempt_history = []
        current_iteration = 0
        
        while current_iteration < self.max_iterations:
            current_iteration += 1
            print(f"Iteration {current_iteration}/{self.max_iterations}")
            
            if current_iteration == 1:
                prompt = self.implement_template.render(baseline_code=baseline_code, idea=idea, problem_spec=problem_spec)
            else:
                prompt = self.debug_template.render(baseline_code=baseline_code, idea=idea, problem_spec=problem_spec, last_attempt=attempt_history[-1])
                
            if self.provider == "OpenAI":
                candidate = self._generate_with_openai(prompt)
            elif self.provider == "Google":
                candidate = self._generate_with_google(prompt)
            else:
                raise ValueError(f"Unknown provider: {self.provider}")
            
            if not candidate:
                print("Failed to generate candidate code")
                break
                
            # Try to compile and validate
            compile_result, validation_result = self._test_candidate(candidate)
            
            attempt_history.append({
                "candidate": candidate,
                "compile_result": compile_result,
                "validation_result": validation_result
            })
            
            print()
            if compile_result.success and validation_result.success:
                print(f"Successfully compiled kernel in {current_iteration} iterations!")
                return {
                    "code": candidate,
                    "idea": idea,
                    "iterations": current_iteration,
                    "latency_ms": validation_result.latency_ms
                }
            else:
                if not compile_result.success:
                    print(f"Compilation failed: {compile_result.error}")
                elif not validation_result.success:
                    print(f"Validation failed: {validation_result.error}")
        
        print(f"Failed to compile kernel after {self.max_iterations} iterations")
        return candidate


    def _test_candidate(self, candidate_code):
        """Test if the candidate compiles and produces correct results."""
        # 1. Try compiling the code
        compile_result = self._compile_candidate(candidate_code)
        
        # 2. If compilation succeeds, validate correctness and performance
        validation_result = ValidationResult(
            success=False, 
            error="Not validated - compilation failed"
        )
        
        if compile_result.success:
            validation_result = self._validate_candidate(candidate_code)
            
        return compile_result, validation_result
    

    def _compile_candidate(self, candidate_code):
        """Compile the candidate code and return result."""
        try:
            # Use the evaluator's _compile method
            output_path = self.evaluator._compile(candidate_code)
            if output_path:
                return CompileResult(success=True, error=None, output_path=output_path)
            else:
                return CompileResult(success=False, error="Compilation failed")
        except Exception as e:
            print(f"Exception during compilation: {e}")
            return CompileResult(success=False, error=str(e))
    

    def _validate_candidate(self, candidate_code):
        """Validate candidate correctness and performance."""
        try:
            # Create a temporary candidate to validate using our stored problem_name
            candidate = KernelCandidate(
                problem=self.evaluator.problem_spec.name,
                round=0,
                code=candidate_code,
                idea="Optimization candidate"
            )
            
            # Check if code contains GIL management issues
            gil_keywords = ["PyGILState_", "Py_BEGIN_ALLOW_THREADS", "Py_END_ALLOW_THREADS"]
            uses_advanced_gil = any(keyword in candidate_code for keyword in gil_keywords)
            if uses_advanced_gil:
                print("WARNING: Detected advanced GIL handling in candidate. This may cause issues with pybind11.")
                return ValidationResult(success=False, error="Advanced GIL handling detected. This requires careful implementation with pybind11.", latency_ms=None)
            
            # Use the evaluator to validate
            is_correct, latency_ms = self.evaluator.evaluate(candidate)
            
            if not is_correct:
                return ValidationResult(success=False, error="Output is not correct", latency_ms=None)
            
            return ValidationResult(success=True, error=None, latency_ms=latency_ms)
        
        except Exception as e:
            print(f"Exception during validation: {e}")
            return ValidationResult(success=False, error=str(e), latency_ms=None)
    
    def _extract_code(self, llm_response):
        """Extract code from LLM response."""
        # Simple extraction of code between triple backticks
        code_matches = re.findall(r"```(?:cpp)?\n(.*?)```", llm_response, re.DOTALL)
        if code_matches:
            return code_matches[0].strip()
        return llm_response.strip()
