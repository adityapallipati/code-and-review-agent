import os
import ast
import datetime
from typing import TypedDict, Sequence, Any
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langsmith import traceable

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

    @traceable
    def review(self, state: WorkflowState) -> WorkflowState:
        """Review the generated code and provide comprehensive feedback."""
        try:
            if not state["current_code"]:
                return self._handle_error(state, "No code to review")

            # Perform static analysis
            static_analysis = self._perform_static_analysis(state["current_code"])
            
            # Get LLM-based review
            review_result = self._perform_llm_review(state["current_code"], static_analysis)
            
            # Update state with review results
            return self._update_state(state, review_result)

        except Exception as e:
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
            SystemMessage(content="""You are an expert code reviewer. Analyze the provided Python code and provide a detailed review focusing on:
1. Security vulnerabilities
2. Performance optimization opportunities
3. Code quality and best practices
4. Potential bugs and edge cases
5. Input validation and error handling

Format your response as a JSON object with the following structure:
{
    "security": {
        "has_issues": boolean,
        "issues": [string],
        "risk_level": "low"|"medium"|"high",
        "recommendations": [string]
    },
    "performance": {
        "has_issues": boolean,
        "issues": [string],
        "recommendations": [string],
        "complexity_analysis": {
            "time_complexity": string,
            "space_complexity": string
        }
    },
    "quality": {
        "has_issues": boolean,
        "issues": [string],
        "recommendations": [string],
        "metrics": {
            "maintainability": number,
            "readability": number,
            "testability": number
        }
    },
    "approved": boolean,
    "confidence_score": number,
    "requires_human_review": boolean
}"""),
            HumanMessage(content=f"""Review this Python code:

{code}

Static analysis results:
{static_analysis}""")
        ]

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
        state["errors"].append(error_msg)
        state["status"]["review"] = False
        state["next_step"] = "error"
        return state

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