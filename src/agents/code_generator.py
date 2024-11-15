import os
import datetime
from typing import TypedDict, Sequence, Any, Literal
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langsmith import traceable
from src.agents.llama_client import LlamaClient

# Load environment variables
load_dotenv()

NodeTypes = Literal["generate", "review", "await_human_input", "prepare_execution", "execute", "complete", "error"]

class WorkflowState(TypedDict):
    """Type definition for the workflow state"""
    messages: Sequence[dict]
    current_code: str | None
    next_step: str
    attempts: int
    status: dict
    errors: list[str]
    review_feedback: dict | None
    execution_result: dict | None
    human_feedback: dict | None
    metadata: dict

class CodeGenerationRequest(BaseModel):
    """Model for code generation request"""
    prompt: str = Field(..., description="A description of the code to be generated")

class CodeGenerationResult(TypedDict):
    """Type definition for generation result"""
    success: bool
    code: str | None
    error: str | None
    requires_human_review: bool
    confidence_score: float
    generation_metadata: dict

class CodeGenerator:
    """Agent responsible for generating code based on user prompts"""
    
    def __init__(self, model_path: str = "src/models/codellama-7b-instruct.gguf", max_attempts: int = 3):
        """Initialize the code generator."""
        self.client = LlamaClient(
            model_path=model_path,
            temperature=0.7,
            max_tokens=2000
        )
        self.max_attempts = max_attempts

    @traceable
    def generate(self, state: WorkflowState) -> WorkflowState:
        """Generate code based on the latest prompt in the message history."""
        try:
            # Check attempt limit
            if state["attempts"] >= self.max_attempts:
                return self._handle_max_attempts_reached(state)

            # Get the last human message
            prompt = self._get_latest_prompt(state)
            if not prompt:
                return self._handle_error(state, "No prompt found in message history")

            # Generate the code
            result = self._generate_code(prompt)
            
            # Update state based on generation result
            return self._update_state(state, result)

        except Exception as e:
            return self._handle_error(state, f"Generation error: {str(e)}")

    def _generate_code(self, prompt: str) -> CodeGenerationResult:
        """Core code generation logic."""
        messages = [
            HumanMessage(content=f"""Generate Python code for the following request: {prompt}

Requirements:
1. Include proper error handling with try/except blocks
2. Add docstrings and inline comments where necessary
3. Follow PEP 8 style guidelines
4. Make the code modular and reusable
5. Include type hints where appropriate
6. Add input validation where necessary

Provide only the code without any explanations or markdown formatting.

After generating the code, also provide a confidence score between 0 and 1 indicating how well the generated code meets the requirements.
Format: 
<code>
[generated code here]
</code>
<confidence>0.XX</confidence>""")
        ]

        response = self.client.invoke(messages)
        
        # Parse the response
        code_content = self._clean_code(response.content)
        confidence_score = self._extract_confidence(response.content)
        
        # Determine if human review is needed based on confidence score
        requires_review = confidence_score < 0.8

        return CodeGenerationResult(
            success=True,
            code=code_content,
            error=None,
            requires_human_review=requires_review,
            confidence_score=confidence_score,
            generation_metadata={
                "model_used": self.client.model_name,
                "temperature": self.client.temperature,
                "timestamp": str(datetime.datetime.now()),
                "prompt_length": len(prompt)
            }
        )

    def _update_state(self, state: WorkflowState, result: CodeGenerationResult) -> WorkflowState:
        """Update the workflow state with generation results."""
        state["attempts"] += 1
        
        # Handle generated code content
        if isinstance(result["code"], dict) and "content" in result["code"]:
            code_content = result["code"]["content"]
        elif isinstance(result["code"], dict) and "role" in result["code"]:
            # Handle assistant message format
            code_content = result["code"].get("content", "")
        else:
            code_content = str(result["code"]) if result["code"] else None
        
        # Update state
        state["current_code"] = code_content
        state["metadata"]["generation"] = result["generation_metadata"]
        state["status"]["generation"] = result["success"]
        
        if result["error"]:
            state["errors"].append(result["error"])
        
        # Add message to history
        state["messages"].append({
            "role": "assistant",
            "content": code_content,
            "type": "code_generation",
            "metadata": {
                "confidence_score": result["confidence_score"],
                "requires_review": result["requires_human_review"]
            }
        })
        
        return state
    
    def _handle_error(self, state: WorkflowState, error_msg: str) -> WorkflowState:
        """Handle errors in the generation process."""
        state["errors"].append(error_msg)
        state["status"]["generation"] = False
        state["next_step"] = "error"
        return state

    def _handle_max_attempts_reached(self, state: WorkflowState) -> WorkflowState:
        """Handle the case when maximum attempts are reached."""
        return self._handle_error(
            state,
            f"Max generation attempts ({self.max_attempts}) reached"
        )

    def _get_latest_prompt(self, state: WorkflowState) -> str | None:
        """Get the latest prompt from the message history."""
        for message in reversed(state["messages"]):
            if message.get("role") == "human":
                return message.get("content")
        return None

    @staticmethod
    def _clean_code(content: str) -> str:
        """Clean and validate the code content."""
        # First try to extract code from tags if present
        import re
        code_match = re.search(r'<code>(.*?)</code>', content, re.DOTALL)
        if code_match:
            code = code_match.group(1).strip()
        else:
            # If no tags, treat the whole content as code
            code = content.strip()
        
        # Remove markdown code blocks if present
        code = re.sub(r'^```python\n', '', code)
        code = re.sub(r'^```\n', '', code)
        code = re.sub(r'\n```$', '', code)
        
        # Basic validation
        try:
            import ast
            ast.parse(code)
            return code
        except SyntaxError:
            # If parsing fails, try to fix common issues
            # Replace smart quotes
            code = code.replace('"', '"').replace('"', '"')
            code = code.replace(''', "'").replace(''', "'")
            
            try:
                ast.parse(code)
                return code
            except SyntaxError:
                # If still invalid, return original cleaned code
                return code

    @staticmethod
    def _extract_confidence(content: str) -> float:
        """Extract confidence score from the response content."""
        import re
        confidence_match = re.search(r'<confidence>(0\.\d+)</confidence>', content)
        return float(confidence_match.group(1)) if confidence_match else 0.0