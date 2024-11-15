import operator
import os
import sys
import yaml
import logging
import tempfile
import subprocess
from typing import Annotated, TypedDict, Sequence, Any, Dict, Callable
from typing_extensions import TypeAlias
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langsmith import traceable
from langgraph.graph import StateGraph
from langchain_core.messages import HumanMessage, SystemMessage
import datetime

# Local imports
from src.agents.code_generator import CodeGenerator
from src.agents.code_reviewer import CodeReviewer

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Type definitions
NodeType: TypeAlias = str

class TaskRequest(BaseModel):
    """Input model for task execution"""
    prompt: str = Field(..., description="Description of the code to generate")
    max_attempts: int = Field(default=3, description="Maximum generation attempts")
    timeout_seconds: int = Field(default=300, description="Timeout for entire workflow")
    review_requirements: dict = Field(
        default_factory=lambda: {
            "security_check": True,
            "performance_check": True,
            "quality_check": True,
            "require_tests": True
        }
    )
    execution_requirements: dict = Field(
        default_factory=lambda: {
            "allow_file_operations": False,
            "allow_network_access": False,
            "max_runtime_seconds": 30
        }
    )
def last_value(x: Any, y: Any) -> Any:
    """Take the last value in case of concurrent updates."""
    return y

class WorkflowState(TypedDict):
    """Complete workflow state definition"""
    messages: Annotated[list[dict], operator.add]
    current_code: Annotated[str | None, last_value]  # Changed from list[dict] to str | None
    next_step: Annotated[str, last_value]
    attempts: Annotated[int, last_value]
    status: Annotated[dict, last_value]
    errors: Annotated[list[str], operator.add]
    review_feedback: Annotated[dict | None, last_value]
    execution_result: Annotated[dict | None, last_value]
    human_feedback: Annotated[dict | None, last_value]
    metadata: Annotated[dict, last_value]
    route: Annotated[str | None, last_value]

class TaskManager:
    """Manages the code generation, review, and execution workflow"""
    
    def __init__(self):
        """Initialize the task manager."""
        self.config = self._load_config()
        self.workflow_graph = self._build_workflow_graph()
        logger.info("TaskManager initialized with workflow graph")

    def _load_config(self) -> dict:
        """Load configuration from yaml file."""
        try:
            with open("src/config/config.yaml", "r") as file:
                config = yaml.safe_load(file)
                logger.info("Configuration loaded successfully")
                return config
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            raise

    def _build_workflow_graph(self) -> StateGraph:
        """Build simplified workflow graph with clear state transitions."""
        from langgraph.graph import START, END
        
        workflow = StateGraph(WorkflowState)
        
        # Define the nodes with simpler responsibilities
        workflow.add_node("generate", self._generation_node)
        workflow.add_node("check_code", self._review_node)
        workflow.add_node("human_review", self._human_input_node)
        workflow.add_node("execute", self._execution_node)
        
        # Define the decision function
        def decide_next_step(state: WorkflowState) -> str:
            """Determine next step based on state."""
            logger.info(f"Deciding next step. Current state: {state.get('next_step')}")
            
            if state.get("errors"):
                logger.info("Errors found, ending workflow")
                return "end"
                
            review_feedback = state.get("review_feedback", {})
            if review_feedback.get("requires_human_review"):
                return "human_review"
            elif review_feedback.get("approved"):
                return "execute"
            else:
                return "generate"
        
        # Add edges with clear flow
        workflow.add_edge(START, "generate")
        workflow.add_edge("generate", "check_code")
        workflow.add_conditional_edges(
            "check_code",
            decide_next_step,
            {
                "human_review": "human_review",
                "execute": "execute",
                "generate": "generate",
                "end": END
            }
        )
        workflow.add_edge("human_review", "execute")
        workflow.add_edge("execute", END)
        
        # Set entry point
        workflow.set_entry_point("generate")
        
        logger.info("Workflow graph built successfully")
        return workflow.compile()
    
    def _generation_node(self, state: WorkflowState) -> WorkflowState:
        """Generate code with simpler success/failure states."""
        logger.info("Starting code generation...")
        try:
            updated_state = {
                **state,
                "status": {"generation": True},
                "next_step": "check_code",
                "errors": []
            }
            
            # Generate code
            generator = CodeGenerator()
            updated_state = generator.generate(updated_state)
            
            if not updated_state.get("current_code"):
                return {**updated_state, "errors": ["Code generation failed"]}
                
            return updated_state
            
        except Exception as e:
            logger.error(f"Generation error: {e}")
            return {**state, "errors": [str(e)]}
        
    def _review_node(self, state: WorkflowState) -> WorkflowState:
        """Review code with clear pass/fail paths."""
        logger.info("Starting code review...")
        logger.info(f"Code to review: {state.get('current_code')}")
        
        try:
            reviewer = CodeReviewer()
            
            # Get review feedback with explicit flag for human review
            review_feedback = reviewer._perform_llm_review(
                state["current_code"],
                {"syntax_valid": True}  # Simplified static analysis
            )
            
            # Update state with review results
            return {
                **state,
                "review_feedback": review_feedback,
                "status": {**state.get("status", {}), "review": True},
                "next_step": (
                    "human_review" if review_feedback.get("requires_human_review")
                    else "execute" if review_feedback.get("approved")
                    else "generate"
                )
            }
            
        except Exception as e:
            logger.error(f"Review error: {e}")
            return {**state, "errors": [f"Review error: {str(e)}"]}

    def _human_input_node(self, state: WorkflowState) -> WorkflowState:
        """Get human input with proper console interaction."""
        logger.info("Awaiting human input...")
        
        try:
            from rich.console import Console
            from rich.prompt import Confirm
            
            console = Console()
            console.print("\n[bold blue]Code Review Required[/bold blue]")
            console.print(f"\nCode to review:\n{state['current_code']}")
            console.print("\nReview feedback:", state["review_feedback"])
            
            approved = Confirm.ask("\nWould you like to approve this code?", default=False)
            
            return {
                **state,
                "human_feedback": {
                    "approved": approved,
                    "timestamp": str(datetime.datetime.now())
                },
                "next_step": "execute" if approved else "generate"
            }
            
        except Exception as e:
            logger.error(f"Human input error: {e}")
            return {**state, "errors": [f"Human input error: {str(e)}"]}

    def _preparation_node(self, state: WorkflowState) -> WorkflowState:
        """Prepare code for execution."""
        logger.info("Preparing for execution...")
        state["status"]["execution_prepared"] = True
        state["next_step"] = "prepare_execution"  # Will be updated by router
        return state

    def _prepare_execution(self, state: WorkflowState) -> WorkflowState:
        """Prepare code for execution with proper requirements."""
        logger.info("Preparing for execution...")
        try:
            execution_requirements = state.get("metadata", {}).get("request_config", {}).get("execution_requirements", {})
            
            # Validate the code meets requirements
            code_content = state["current_code"]
            
            # Check for file operations if not allowed
            if not execution_requirements.get("allow_file_operations", False):
                if "open(" in code_content or "write" in code_content or "read" in code_content:
                    return {
                        **state,
                        "errors": ["File operations not allowed"],
                        "status": {**state.get("status", {}), "execution_prepared": False},
                        "next_step": "error"
                    }
            
            # Check for network access if not allowed
            if not execution_requirements.get("allow_network_access", False):
                if "requests" in code_content or "urllib" in code_content or "socket" in code_content:
                    return {
                        **state,
                        "errors": ["Network access not allowed"],
                        "status": {**state.get("status", {}), "execution_prepared": False},
                        "next_step": "error"
                    }
            
            return {
                **state,
                "status": {**state.get("status", {}), "execution_prepared": True},
                "next_step": "execute",
                "execution_config": {
                    "timeout": execution_requirements.get("max_runtime_seconds", 5),
                    "allow_file_ops": execution_requirements.get("allow_file_operations", False),
                    "allow_network": execution_requirements.get("allow_network_access", False)
                }
            }
            
        except Exception as e:
            logger.error(f"Preparation error: {e}")
            return {
                **state,
                "errors": [f"Preparation error: {str(e)}"],
                "status": {**state.get("status", {}), "execution_prepared": False},
                "next_step": "error"
            }
    
    def _execution_node(self, state: WorkflowState) -> WorkflowState:
        """Execute the approved code."""
        logger.info("Starting code execution...")
        try:
            code_content = state["current_code"]
            logger.info(f"Executing code:\n{code_content}")
            
            # Check if code contains input() function
            if "input(" in code_content:
                # Modify the code to use a default value instead of waiting for input
                modified_code = code_content.replace(
                    'input("Enter your name: ")',
                    '"User"  # Modified for automated execution'
                )
                
                with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tmp:
                    tmp.write(modified_code)
                    tmp.flush()
                    
                    try:
                        start_time = datetime.datetime.now()
                        result = subprocess.run(
                            ['python', tmp.name],
                            capture_output=True,
                            text=True,
                            timeout=5
                        )
                        end_time = datetime.datetime.now()
                        runtime = (end_time - start_time).total_seconds()
                        
                        # Create execution result
                        execution_result = {
                            "success": result.returncode == 0,
                            "output": result.stdout,
                            "error": result.stderr if result.returncode != 0 else None,
                            "runtime_seconds": runtime,
                            "note": "Code was modified to run automatically with default input 'User'"
                        }
                        
                        logger.info(f"Execution completed. Output: {result.stdout}")
                        
                        # Update state
                        return {
                            **state,
                            "execution_result": execution_result,
                            "status": {**state.get("status", {}), "execution": True},
                            "next_step": "complete",
                            "messages": state.get("messages", []) + [{
                                "role": "assistant",
                                "content": "Code execution complete (with automated input)",
                                "type": "execution",
                                "execution_result": execution_result
                            }]
                        }
                        
                    except subprocess.TimeoutExpired as te:
                        logger.error("Execution timed out")
                        return {
                            **state,
                            "execution_result": {
                                "success": False,
                                "output": None,
                                "error": f"Execution timed out after 5 seconds",
                                "runtime_seconds": 5
                            },
                            "status": {**state.get("status", {}), "execution": False},
                            "next_step": "error"
                        }
                        
                    finally:
                        # Clean up the temp file
                        try:
                            os.unlink(tmp.name)
                        except:
                            pass
                            
            else:
                # Execute non-interactive code normally
                with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tmp:
                    tmp.write(code_content)
                    tmp.flush()
                    
                    try:
                        start_time = datetime.datetime.now()
                        result = subprocess.run(
                            ['python', tmp.name],
                            capture_output=True,
                            text=True,
                            timeout=5
                        )
                        end_time = datetime.datetime.now()
                        runtime = (end_time - start_time).total_seconds()
                        
                        execution_result = {
                            "success": result.returncode == 0,
                            "output": result.stdout,
                            "error": result.stderr if result.returncode != 0 else None,
                            "runtime_seconds": runtime
                        }
                        
                        logger.info(f"Execution completed. Output: {result.stdout}")
                        
                        return {
                            **state,
                            "execution_result": execution_result,
                            "status": {**state.get("status", {}), "execution": True},
                            "next_step": "complete",
                            "messages": state.get("messages", []) + [{
                                "role": "assistant",
                                "content": "Code execution complete",
                                "type": "execution",
                                "execution_result": execution_result
                            }]
                        }
                        
                    except subprocess.TimeoutExpired as te:
                        logger.error("Execution timed out")
                        return {
                            **state,
                            "execution_result": {
                                "success": False,
                                "output": None,
                                "error": f"Execution timed out after 5 seconds",
                                "runtime_seconds": 5
                            },
                            "status": {**state.get("status", {}), "execution": False},
                            "next_step": "error"
                        }
                        
                    finally:
                        # Clean up the temp file
                        try:
                            os.unlink(tmp.name)
                        except:
                            pass
                            
        except Exception as e:
            logger.error(f"Execution error: {e}")
            return {
                **state,
                "execution_result": {
                    "success": False,
                    "output": None,
                    "error": str(e),
                    "runtime_seconds": 0
                },
                "status": {**state.get("status", {}), "execution": False},
                "next_step": "error"
            }

    def route_after_generation(self, state: WorkflowState) -> str:
        """Route after code generation."""
        logger.info("Routing after generation...")
        logger.info(f"Current state: {state['status']}")
        
        if state["errors"]:
            logger.info("Routing to error due to errors")
            return "error"
            
        if state["attempts"] >= self.config["code_generator"].get("max_attempts", 3):
            logger.info("Routing to error due to max attempts")
            return "error"
            
        if state["messages"][-1].get("metadata", {}).get("requires_review", False):
            logger.info("Routing to human review")
            return "await_human_input"
            
        if state["status"].get("generation") is True:
            logger.info("Routing to review")
            return "review"
            
        logger.info("Routing back to generate")
        return "generate"

    def route_after_review(self, state: WorkflowState) -> str:
        """Route after code review."""
        logger.info("Routing after review...")
        if state["errors"]:
            return "error"
        review_feedback = state.get("review_feedback", {})
        if review_feedback.get("requires_human_review"):
            return "await_human_input"
        if not review_feedback.get("approved"):
            return "generate"
        return "prepare_execution"

    def route_after_human_input(self, state: WorkflowState) -> str:
        """Route after human input."""
        logger.info("Routing after human input...")
        if state["errors"]:
            return "error"
        if not state["human_feedback"].get("approved", False):
            return "generate"
        return "prepare_execution"

    def route_after_preparation(self, state: WorkflowState) -> str:
        """Route after preparation."""
        logger.info("Routing after preparation...")
        if state["errors"]:
            return "error"
        if state["status"].get("execution_prepared"):
            return "execute"
        return "error"

    def route_after_execution(self, state: WorkflowState) -> str:
        """Route after execution."""
        logger.info("Routing after execution...")
        if state["errors"]:
            return "error"
        if state["execution_result"].get("success"):
            return "complete"
        return "error"

    def _complete_workflow(self, state: WorkflowState) -> WorkflowState:
        """Handle workflow completion."""
        logger.info("Completing workflow...")
        state["metadata"]["end_time"] = str(datetime.datetime.now())
        state["next_step"] = "complete"
        return state

    def _handle_error(self, state: WorkflowState) -> WorkflowState:
        """Handle workflow errors."""
        # Get the most recent error if there are multiple
        errors = state.get("errors", [])
        error_message = errors[-1] if errors else "Unknown error"
        logger.error(f"Handling error: {error_message}")
        
        return {
            **state,
            "status": {**state.get("status", {}), "error": True},
            "next_step": "complete",  # End the workflow instead of looping
            "metadata": {
                **state.get("metadata", {}), 
                "end_time": str(datetime.datetime.now()),
                "error": error_message
            }
        }

    @traceable
    def execute_workflow(self, request: TaskRequest) -> dict:
        """Execute the full workflow based on the request."""
        try:
            logger.info(f"Starting workflow with prompt: {request.prompt}")
            
            # Initialize workflow state
            initial_state: WorkflowState = {
                "messages": [{"role": "human", "content": request.prompt}],
                "current_code": None,
                "next_step": "generate",
                "attempts": 0,
                "status": {"initialized": True},
                "errors": [],
                "review_feedback": None,
                "execution_result": None,
                "human_feedback": None,
                "metadata": {
                    "start_time": str(datetime.datetime.now()),
                    "request_config": request.dict(),
                    "workflow_id": os.urandom(16).hex()
                },
                "route": None
            }
            
            try:
                # Execute workflow
                final_state = self.workflow_graph.invoke(initial_state)
                logger.info("Workflow completed")
                
                # Check for errors
                if final_state.get("errors"):
                    return {
                        "status": "error",
                        "errors": final_state["errors"],
                        "generated_code": final_state.get("current_code"),
                        "metadata": final_state.get("metadata", {})
                    }
                
                # Return successful result
                return {
                    "status": "success",
                    "generated_code": final_state.get("current_code"),
                    "review_feedback": final_state.get("review_feedback"),
                    "execution_result": final_state.get("execution_result"),
                    "metadata": final_state.get("metadata", {})
                }
                
            except Exception as e:
                logger.error(f"Workflow execution failed: {e}")
                return {
                    "status": "error",
                    "error": str(e),
                    "metadata": initial_state["metadata"]
                }
                
        except Exception as e:
            logger.error(f"Workflow initialization failed: {e}")
            return {
                "status": "error",
                "error": str(e)
            }