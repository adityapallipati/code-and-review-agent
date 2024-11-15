import logging
import os
import ast
import datetime
from typing import TypedDict, Sequence, Any
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langsmith import traceable

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

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

class SecurityCheck(TypedDict):
    """Results of security analysis"""
    has_issues: bool
    issues: list[str]
    risk_level: str
    recommendations: list[str]

class PerformanceCheck(TypedDict):
    """Results of performance analysis"""
    has_issues: bool
    issues: list[str]
    recommendations: list[str]
    complexity_analysis: dict

class CodeQualityCheck(TypedDict):
    """Results of code quality analysis"""
    has_issues: bool
    issues: list[str]
    recommendations: list[str]
    metrics: dict

class CodeReviewResult(TypedDict):
    """Complete code review results"""
    success: bool
    approved: bool
    security: SecurityCheck
    performance: PerformanceCheck
    quality: CodeQualityCheck
    requires_human_review: bool
    confidence_score: float
    review_metadata: dict

class CodeReviewer:
    """Agent responsible for reviewing and analyzing generated code"""
    
    def __init__(self, model_name: str = "gpt-4-turbo-preview"):
        """Initialize the code reviewer."""
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not found")
        
        self.client = ChatOpenAI(
            model=model_name,
            temperature=0.1,  # Lower temperature for more consistent analysis
            api_key=self.api_key
        )

    def review(self, state: WorkflowState) -> WorkflowState:
        """Review the generated code and provide comprehensive feedback."""
        try:
            if not state.get("current_code"):
                logger.error("No code available for review")
                return self._handle_error(state, "No code available for review")

            logger.info(f"Reviewing code:\n{state['current_code']}")
            
            # Perform static analysis
            static_analysis = self._perform_static_analysis(state["current_code"])
            
            try:
                # Get LLM-based review
                review_result = self._perform_llm_review(state["current_code"], static_analysis)
                
                # Update state with review results
                updated_state = {
                    **state,
                    "review_feedback": {
                        "approved": review_result["approved"],
                        "security": review_result["security"],
                        "performance": review_result["performance"],
                        "quality": review_result["quality"],
                        "requires_human_review": review_result["requires_human_review"],
                        "confidence_score": review_result["confidence_score"],
                        "metadata": review_result["review_metadata"]
                    },
                    "status": {**state.get("status", {}), "review": True},
                    "messages": state.get("messages", []) + [{
                        "role": "assistant",
                        "content": "Code Review Complete",
                        "type": "review",
                        "review_data": review_result
                    }]
                }
                
                # Set next step based on review results
                if review_result["requires_human_review"]:
                    updated_state["next_step"] = "await_human_input"
                elif review_result["approved"]:
                    updated_state["next_step"] = "prepare_execution"
                else:
                    updated_state["next_step"] = "generate"
                    
                logger.info(f"Review complete. Next step: {updated_state['next_step']}")
                return updated_state
                
            except Exception as e:
                logger.error(f"Error in review process: {e}")
                return self._handle_error(state, f"Review error: {str(e)}")
                
        except Exception as e:
            logger.error(f"Unexpected error in review: {e}")
            return self._handle_error(state, f"Review error: {str(e)}")

    def _perform_static_analysis(self, code: str) -> dict:
        """Perform static code analysis."""
        try:
            tree = ast.parse(code)
            
            # Collect metrics
            metrics = {
                "num_functions": len([n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]),
                "num_classes": len([n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]),
                "num_imports": len([n for n in ast.walk(tree) if isinstance(n, ast.Import) or isinstance(n, ast.ImportFrom)]),
                "has_type_hints": bool(len([n for n in ast.walk(tree) if isinstance(n, ast.AnnAssign)])),
                "has_docstrings": bool(len([n for n in ast.walk(tree) if isinstance(n, ast.Expr) and isinstance(n.value, ast.Str)])),
                "complexity": self._calculate_complexity(tree)
            }
            
            # Security checks
            security_issues = self._check_security_issues(tree)
            
            return {
                "metrics": metrics,
                "security_issues": security_issues,
                "syntax_valid": True
            }
            
        except SyntaxError:
            return {
                "metrics": {},
                "security_issues": [],
                "syntax_valid": False
            }

    def _calculate_complexity(self, tree: ast.AST) -> dict:
        """Calculate code complexity metrics."""
        complexity = {
            "cyclomatic": 1,
            "cognitive": 0
        }
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity["cyclomatic"] += 1
                complexity["cognitive"] += 1
            elif isinstance(node, ast.BoolOp):
                complexity["cyclomatic"] += len(node.values) - 1
            
        return complexity

    def _check_security_issues(self, tree: ast.AST) -> list[str]:
        """Check for common security issues."""
        issues = []
        
        for node in ast.walk(tree):
            # Check for dangerous functions
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in ['exec', 'eval', 'input']:
                    issues.append(f"Potentially dangerous function used: {node.func.id}")
            
            # Check for file operations
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id == 'open':
                    if not any(isinstance(parent, ast.With) for parent in ast.walk(tree)):
                        issues.append("File operation without context manager ('with' statement)")
            
            # Check for hardcoded credentials
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        if any(cred in target.id.lower() for cred in ['password', 'secret', 'token', 'key']):
                            issues.append(f"Possible hardcoded credential in variable: {target.id}")
        
        return issues

    def _perform_llm_review(self, code: str, static_analysis: dict) -> CodeReviewResult:
        """Get LLM-based code review."""
        messages = [
            SystemMessage(content="""You are an expert code reviewer. Review the provided Python code and return a JSON object with the following structure:
    {
        "security": {
            "has_issues": boolean,
            "issues": [],
            "risk_level": "low",
            "recommendations": []
        },
        "performance": {
            "has_issues": boolean,
            "issues": [],
            "recommendations": [],
            "complexity_analysis": {
                "time_complexity": "O(1)",
                "space_complexity": "O(1)"
            }
        },
        "quality": {
            "has_issues": boolean,
            "issues": [],
            "recommendations": [],
            "metrics": {
                "maintainability": 10,
                "readability": 10,
                "testability": 10
            }
        },
        "approved": boolean,
        "requires_human_review": boolean,
        "confidence_score": number
    }

    Return ONLY the JSON object, no other text."""),
            HumanMessage(content=f"""Review this Python code:

    {code}

    Static analysis results:
    {static_analysis}""")
        ]

        try:
            response = self.client.invoke(messages)
            
            # Parse the JSON response
            import json
            review_data = json.loads(response.content)
            
            return CodeReviewResult(
                success=True,
                approved=review_data["approved"],
                security=review_data["security"],
                performance=review_data["performance"],
                quality=review_data["quality"],
                requires_human_review=review_data["requires_human_review"],
                confidence_score=review_data["confidence_score"],
                review_metadata={
                    "static_analysis": static_analysis,
                    "timestamp": str(datetime.datetime.now())
                }
            )
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response: {e}")
            logger.error(f"Response content: {response.content}")
            raise ValueError(f"Failed to parse review response: {e}")
        except Exception as e:
            logger.error(f"Error during LLM review: {e}")
            raise

    def _update_state(self, state: WorkflowState, result: CodeReviewResult) -> WorkflowState:
        """Update workflow state with review results."""
        state["review_feedback"] = {
            "approved": result["approved"],
            "security": result["security"],
            "performance": result["performance"],
            "quality": result["quality"],
            "requires_human_review": result["requires_human_review"],
            "confidence_score": result["confidence_score"]
        }
        
        state["status"]["review"] = result["success"]
        
        # Determine next step based on review results
        if result["requires_human_review"]:
            state["next_step"] = "await_human_input"
        elif not result["approved"]:
            state["next_step"] = "generate"
        else:
            state["next_step"] = "prepare_execution"
        
        state["messages"].append({
            "role": "assistant",
            "content": "Code Review Complete",
            "type": "review",
            "review_data": state["review_feedback"]
        })
        
        return state

    def _handle_error(self, state: WorkflowState, error_msg: str) -> WorkflowState:
        """Handle errors in the review process."""
        logger.error(f"Handling review error: {error_msg}")
        return {
            **state,
            "errors": state.get("errors", []) + [error_msg],
            "status": {**state.get("status", {}), "review": False},
            "next_step": "error"
        }

# Conditional routing functions
def should_fix_code(state: WorkflowState) -> bool:
    """Determine if code needs fixing based on review."""
    review_feedback = state.get("review_feedback", {})
    return (
        state["status"].get("review") is True 
        and review_feedback is not None
        and not review_feedback.get("approved", False)
        and not review_feedback.get("requires_human_review", False)
    )

def should_proceed_to_execution(state: WorkflowState) -> bool:
    """Determine if code can proceed to execution."""
    review_feedback = state.get("review_feedback", {})
    return (
        state["status"].get("review") is True 
        and review_feedback is not None
        and review_feedback.get("approved", False)
        and not review_feedback.get("requires_human_review", False)
    )

def needs_human_review(state: WorkflowState) -> bool:
    """Determine if human review is needed."""
    review_feedback = state.get("review_feedback", {})
    return (
        state["status"].get("review") is True 
        and review_feedback is not None
        and review_feedback.get("requires_human_review", True)
    )