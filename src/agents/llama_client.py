import logging
from typing import List, Optional, Any, Dict
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.outputs import ChatResult, ChatGeneration
import platform
import sys
import os
from llama_cpp import Llama

logger = logging.getLogger(__name__)

class LlamaClient(BaseChatModel):
    def __init__(
        self,
        model_path: str = "src/models/codellama-7b-instruct.gguf",
        temperature: float = 0.7,
        max_tokens: int = 2000,
        top_p: float = 0.95,
        n_ctx: int = 2048,
        n_gpu_layers: Optional[int] = None,
        verbose: bool = True
    ):
        """Initialize Llama client."""
        # Initialize base class first
        model_kwargs: Dict[str, Any] = {
            "model_path": model_path,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "n_ctx": n_ctx,
        }
        super().__init__(**model_kwargs)
        
        # Store parameters
        self._temperature = float(temperature)
        self._max_tokens = int(max_tokens)
        self._top_p = float(top_p)
        self._n_ctx = int(n_ctx)
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        logger.info(f"Python version: {sys.version}")
        logger.info(f"Platform: {platform.platform()}")
        logger.info(f"Model path: {model_path}")
        
        try:
            # Configure specifically for M1 Mac
            if platform.system() == "Darwin" and platform.processor() == "arm":
                n_gpu_layers = 35  # Good balance for 7B model on M1
                logger.info("Configuring for Apple Silicon with Metal support")
            else:
                n_gpu_layers = 0  # Fallback for non-Metal systems
            
            # Initialize the model with explicit types for all numeric parameters
            self._model = Llama(
                model_path=str(model_path),
                n_ctx=int(n_ctx),
                n_threads=8,
                n_gpu_layers=int(n_gpu_layers),
                verbose=bool(verbose),
                n_batch=512,
                use_mmap=True,
                use_mlock=False,
                embedding=False,
                last_n_tokens_size=64,    # Explicit context window for token tracking
                n_parts=-1,               # Auto-detect number of parts
                seed=-1,                  # Random seed
                f16_kv=True,             # Use half precision for key/value cache
                logits_all=False,         # Only compute logits for the last token
                vocab_only=False,         # Load the full model
                use_mmap_force=False      # Don't force mmap usage
            )
            logger.info("Successfully initialized Llama model")
            
        except Exception as e:
            logger.error(f"Failed to initialize model: {str(e)}")
            raise
    @property
    def model_name(self) -> str:
        """Get the model name."""
        return "codellama-7b-instruct"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_id: Optional[str] = None,
        **kwargs
    ) -> ChatResult:
        """Generate response using Llama model."""
        try:
            formatted_prompt = self._format_messages(messages)
            logger.info(f"Sending prompt to model: {formatted_prompt[:100]}...")
            
            # Generate completion
            response = self._model.create_completion(
                prompt=formatted_prompt,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
                top_p=self._top_p,
                stop=stop or ["[/INST]", "Human:", "Assistant:"],
                echo=False
            )
            
            # Extract the generated text
            if response and "choices" in response and len(response["choices"]) > 0:
                response_text = response["choices"][0]["text"]
            else:
                response_text = ""
                
            logger.info(f"Received response from model: {response_text[:100]}...")
            
            generation = ChatGeneration(
                message=HumanMessage(content=response_text),
                generation_info={"finish_reason": "stop"}
            )
            return ChatResult(generations=[generation])
            
        except Exception as e:
            logger.error(f"Error during generation: {str(e)}")
            raise

    def _format_messages(self, messages: List[BaseMessage]) -> str:
        """Format messages specifically for CodeLlama."""
        formatted = "[INST] You are an expert coding assistant. "
        
        for message in messages:
            if isinstance(message, SystemMessage):
                formatted += f"{message.content}\n"
            elif isinstance(message, HumanMessage):
                formatted += f"\nHuman: {message.content}\n"
            else:
                formatted += f"\nAssistant: {message.content}\n"
        
        formatted += "\nAssistant: [/INST]"
        return formatted.strip()

    @property
    def _llm_type(self) -> str:
        return "llama-chat"

    def invoke(self, input_str: str, stop: Optional[List[str]] = None) -> str:
        """Direct invocation method for compatibility."""
        try:
            # Add explicit instruction for JSON response if needed
            if "JSON" in input_str:
                input_str = f"""[INST] You are an expert code reviewer. 
Your response must be a valid JSON object.
Do not include any additional text or explanations.
Only output the JSON object itself.

{input_str} [/INST]"""
            else:
                input_str = f"[INST] {input_str} [/INST]"
            
            response = self._model.create_completion(
                prompt=input_str,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
                top_p=self._top_p,
                stop=stop or ["[/INST]", "Human:", "Assistant:"],
                echo=False
            )
            
            if response and "choices" in response and len(response["choices"]) > 0:
                text = response["choices"][0]["text"].strip()
                logger.info(f"Raw model response: {text[:200]}...")  # Log first 200 chars
                return text
            return ""
            
        except Exception as e:
            logger.error(f"Error in invoke: {str(e)}")
            return ""