import json
import re
import random
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
import openai
import anthropic
import os
from dotenv import load_dotenv

# Try to import Google Generative AI, but make it optional
GOOGLE_AVAILABLE = False
try:
    import google.generativeai as genai
    GOOGLE_AVAILABLE = True
except ImportError:
    pass

# Load environment variables from .env file
load_dotenv()

class BaseModel(ABC):
    """Abstract base class for all models"""
    
    @abstractmethod
    def generate_candidates(self, clue: str, length: str, k: int = 5) -> str:
        """Generate k candidate answers with probabilities. Returns raw response string."""
        pass
    
    def get_name(self) -> str:
        """Return model name for identification"""
        return self.__class__.__name__

class LLMModel(BaseModel):
    """Single configurable wrapper for different LLM providers"""
    
    def __init__(self, provider: str = "stub", **config):
        self.provider = provider
        self.config = config
        
        # Configure based on provider
        if provider == "openai":
            api_key = config.get('api_key') or os.environ.get('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("OpenAI API key required. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
            self.client = openai.OpenAI(api_key=api_key)
            self.model_name = config.get('model_name', 'gpt-4')
            
        elif provider == "anthropic":
            api_key = config.get('api_key') or os.environ.get('ANTHROPIC_API_KEY')
            if not api_key:
                raise ValueError("Anthropic API key required. Set ANTHROPIC_API_KEY environment variable or pass api_key parameter.")
            self.client = anthropic.Anthropic(api_key=api_key)
            self.model_name = config.get('model_name', 'claude-3-sonnet')
            
        elif provider == "gemini":
            if not GOOGLE_AVAILABLE:
                raise ValueError("Google Generative AI package not available. Install with: pip install google-generativeai")
            api_key = config.get('api_key') or os.environ.get('GOOGLE_API_KEY')
            if not api_key:
                raise ValueError("Google API key required. Set GOOGLE_API_KEY environment variable or pass api_key parameter.")
            genai.configure(api_key=api_key)
            self.model_name = config.get('model_name', 'gemini-1.5-pro')
            self.client = genai.GenerativeModel(self.model_name)
            
        elif provider == "stub":
            self.client = None
            self.model_name = "stub"
            
        else:
            raise ValueError(f"Unknown provider: {provider}. Available: stub, openai, anthropic, gemini")
    
    def generate_candidates(self, clue: str, length: str, k: int = 5) -> str:
        """Generate candidates using the configured provider"""
        if self.provider == "stub":
            return self._generate_stub_candidates(clue, length, k)
        else:
            return self._generate_llm_candidates(clue, length, k)
    
    def _generate_stub_candidates(self, clue: str, length: str, k: int = 5) -> str:
        """Generate random candidates for testing"""
        target_len = self._parse_length(length)
        
        candidates_data = []
        for i in range(k):
            if target_len:
                answer = ''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ', k=target_len))
            else:
                answer = ''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ', k=random.randint(3, 8)))
            
            prob = max(0.1, 1.0 - (i * 0.15) + random.uniform(-0.05, 0.05))
            prob = min(1.0, prob)
            
            candidates_data.append({
                "answer": answer,
                "confidence": prob,
                "reasoning": f"Random candidate {i+1}"
            })
        
        candidates_data.sort(key=lambda x: x['confidence'], reverse=True)
        return json.dumps({"candidates": candidates_data}, indent=2)
    
    def _generate_llm_candidates(self, clue: str, length: str, k: int = 5) -> str:
        """Generate candidates using the configured LLM provider"""
        target_len = self._parse_length(length)
        prompt = self._build_prompt(clue, length, target_len, k)
        
        try:
            if self.provider == "openai":
                # GPT-5 doesn't support custom temperature, only default (1)
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are an expert cryptic crossword solver. Provide accurate answers in the exact format requested."},
                        {"role": "user", "content": prompt}
                    ],
                    max_completion_tokens=1000
                )
                content = response.choices[0].message.content
                
            elif self.provider == "anthropic":
                response = self.client.messages.create(
                    model=self.model_name,
                    max_tokens=500,
                    messages=[{"role": "user", "content": prompt}]
                )
                content = response.content[0].text
                
            elif self.provider == "gemini":
                response = self.client.generate_content(prompt)
                content = response.text
            
            return content
            
        except Exception as e:
            # Don't fall back to stub - let the error bubble up
            raise Exception(f"{self.provider} API error: {e}")
    
    def _build_prompt(self, clue: str, length: str, target_len: Optional[int], k: int) -> str:
        """Build the prompt for LLM providers"""
        # Build the correct number of example candidates
        example_candidates = []
        for i in range(k):
            confidence = 0.95 - (i * 0.1)
            example_candidates.append(f'    {{"answer": "ANSWER{i+1}", "confidence": {confidence:.2f}, "reasoning": "brief explanation"}}')
        
        examples_text = ',\n'.join(example_candidates)
        
        return f"""Solve this cryptic crossword clue and provide {k} candidate answer{'s' if k > 1 else ''} with confidence scores.

Clue: "{clue}"
Target length: {target_len if target_len else "any length"}

Please respond with exactly {k} candidate{'s' if k > 1 else ''} in this JSON format:
{{
  "candidates": [
{examples_text}
  ]
}}

Ensure all answers are uppercase and the confidence scores are between 0 and 1. Only return valid JSON."""

    def _parse_length(self, length: str) -> Optional[int]:
        """Parse length string to get target length for cryptic crossword answers
        
        Handles formats like:
        - 9 -> 9 (clean integer from target_length)
        - (9) -> 9 (single word with parentheses)
        - (5,6) -> 11 (two words: 5 + 6 = 11, but answer will have a space)
        - (3,4,2) -> 9 (three words: 3 + 4 + 2 = 9, but answer will have spaces)
        """
        if not length:
            return None
        
        # First try to parse as clean integer (from target_length)
        try:
            return int(length)
        except ValueError:
            pass
        
        # If that fails, try to parse as enumeration format with parentheses
        clean_length = length.strip('()')
        if ',' in clean_length:
            # Multi-word answer: sum the lengths
            try:
                parts = [int(part.strip()) for part in clean_length.split(',')]
                return sum(parts)
            except ValueError:
                return None
        else:
            # Single word answer
            try:
                return int(clean_length)
            except ValueError:
                return None
    
    def get_name(self) -> str:
        """Return model name for identification"""
        return f"{self.provider}-{self.model_name}"

def get_model(model_type: str = "stub", **kwargs) -> BaseModel:
    """Factory function to get model instance - now just returns LLMModel"""
    return LLMModel(model_type, **kwargs)

# Legacy aliases for backward compatibility
StubModel = lambda **kwargs: LLMModel("stub", **kwargs)
OpenAIModel = lambda **kwargs: LLMModel("openai", **kwargs)
AnthropicModel = lambda **kwargs: LLMModel("anthropic", **kwargs)
GoogleGeminiModel = lambda **kwargs: LLMModel("gemini", **kwargs)
