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
    def generate_candidates(self, clue: str, k: int = 5) -> str:
        """Generate k candidate answers with probabilities. Returns raw response string."""
        pass
    
    def get_name(self) -> str:
        """Return model name for identification"""
        return self.__class__.__name__

class LLMModel(BaseModel):
    """Single configurable wrapper for different LLM providers"""
    
    def __init__(self, provider: str = "stub", efficient_mode: bool = False, extended_prompt: bool = False, **config):
        self.provider = provider
        self.efficient_mode = efficient_mode  # New flag for fast processing mode
        self.extended_prompt = extended_prompt  # New flag for enhanced prompts
        self.config = config
        
        # Configure based on provider
        if provider == "openai":
            api_key = config.get('api_key') or os.environ.get('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("OpenAI API key required. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
            self.client = openai.OpenAI(api_key=api_key)
            self.model_name = config.get('model_name', 'gpt-5')
            
        elif provider == "anthropic":
            api_key = config.get('api_key') or os.environ.get('ANTHROPIC_API_KEY')
            if not api_key:
                raise ValueError("Anthropic API key required. Set ANTHROPIC_API_KEY environment variable or pass api_key parameter.")
            self.client = anthropic.Anthropic(api_key=api_key)
            self.model_name = config.get('model_name', 'claude-3-5-sonnet-20241022')
            
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
    
    def generate_candidates(self, clue: str, k: int = 5) -> str:
        """Generate candidates using the configured provider"""
        if self.provider == "stub":
            return self._generate_stub_candidates(clue, k)
        else:
            if self.efficient_mode:
                return self._generate_efficient_answer(clue)
            else:
                return self._generate_llm_candidates(clue, k)
    
    def generate_batch_answers(self, clues: List[Dict[str, str]], batch_size: int = 10) -> List[str]:
        """Generate answers for multiple clues in batches for maximum efficiency"""
        if self.provider == "stub":
            return [self._generate_stub_candidates(c['clue'], c['length'], 1) for c in clues]
        
        if self.efficient_mode:
            return self._generate_batch_efficient(clues, batch_size)
        else:
            return [self._generate_llm_candidates(c['clue'], c['length'], 1) for c in clues]
    
    def _generate_stub_candidates(self, clue: str, length: str, k: int = 5) -> str:
        """Generate random candidates for testing"""
        if self.efficient_mode:
            # In efficient mode, just return a simple answer
            target_len = self._parse_length(length)
            if target_len:
                answer = ''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ', k=target_len))
            else:
                answer = ''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ', k=random.randint(3, 8)))
            return answer
        
        # Original behavior for detailed mode
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
    
    def _generate_efficient_answer(self, clue: str) -> str:
        """Generate a single answer efficiently without probabilities for fast processing"""
        
        if self.extended_prompt:
            prompt = self._build_extended_prompt(clue)
        else:
            prompt = f"""Solve this cryptic crossword clue and provide just the answer.

Clue: "{clue}"

Respond with only the answer in uppercase letters, nothing else."""
        
        try:
            if self.provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are an expert cryptic crossword solver. Provide only the answer in uppercase letters."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=50,
                    temperature=0.1
                )
                content = response.choices[0].message.content.strip().upper()
                
            elif self.provider == "anthropic":
                response = self.client.messages.create(
                    model=self.model_name,
                    max_tokens=50,
                    messages=[{"role": "user", "content": prompt}]
                )
                content = response.content[0].text.strip().upper()
                
            elif self.provider == "gemini":
                response = self.client.generate_content(prompt)
                content = response.text.strip().upper()
            
            # Clean up the response to just get the answer
            content = re.sub(r'[^A-Z\s]', '', content).strip()
            return content
            
        except Exception as e:
            raise Exception(f"{self.provider} API error: {e}")

    def _build_extended_prompt(self, clue: str) -> str:
        """Build an extended prompt with cryptic crossword solving guidance"""
        return f"""You are an expert cryptic crossword clue solver. Cryptic crossword clue contains three elements:

1. Definition (of the answer, often found at the start or end of the clue)
2. Indicator (indicates the type of wordplay)
3. Fodder (for the wordplay)

Common indicators:
-anagram indicators: "confused", "broken", "upset", etc.
-container indicators: "possessing", "hugging", "holding", etc.
-hidden word indicators: "held in", "a bit of", "part of", etc.
-reversal indicators: "backwards", "flipped", "held up", etc.
-deletion indicators: "missing", "cut", "absent", etc.
-letter position indicators: "first", "head", "middle", etc.
-homophone indicators: "said", "heard", "reportedly", etc.

Note: double definition and charade clues don't have indicator words, 
but they generally give a sense of one thing following another.

EXAMPLES:
"Bird seen in the museum (3)" -> "seen in" indicates containment, and length is 3 -> "EMU"
"Honestly crazy, in secret (2,3,3)" -> "crazy" indicates an anagram, and length is 2+3+3=8 -> "ONTHESLY"
"Stringed instrument untruthful person heard (4)" -> "heard" indicates homophone and the length is 4 -> "LYRE"
"Wear out an important part of a car (4)" -> no clear indicator words and legnth is 4 -> "TIRE"
"Returned beer of kings (5)" -> "returned" indicates reversal and length is 5 -> "LAGER"

POSSIBLE ANALYSIS STEPS:
1. Identify the definition, indicator, and fodder
2. Apply the wordplay to get the answer
3. Verify the answer fits the length requirement

Clue: "{clue}"

Provide ONLY the answer in uppercase letters, nothing else."""

    def _generate_batch_efficient(self, clues: List[Dict[str, str]], batch_size: int) -> List[str]:
        """Generate answers for multiple clues efficiently using batching"""
        results = []
        
        for i in range(0, len(clues), batch_size):
            batch = clues[i:i + batch_size]
            batch_prompt = self._build_batch_prompt(batch)
            
            try:
                if self.provider == "openai":
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[
                            {"role": "system", "content": "You are an expert cryptic crossword solver. Provide only the answers in uppercase letters, one per line."},
                            {"role": "user", "content": batch_prompt}
                        ],
                        max_tokens=batch_size * 20,
                        temperature=0.1
                    )
                    content = response.choices[0].message.content.strip()
                    
                elif self.provider == "anthropic":
                    response = self.client.messages.create(
                        model=self.model_name,
                        max_tokens=batch_size * 20,
                        messages=[{"role": "user", "content": batch_prompt}]
                    )
                    content = response.content[0].text.strip()
                    
                elif self.provider == "gemini":
                    response = self.client.generate_content(batch_prompt)
                    content = response.text.strip()
                
                # Parse batch response into individual answers
                batch_answers = self._parse_batch_response(content, len(batch))
                results.extend(batch_answers)
                
            except Exception as e:
                # Fall back to individual processing for this batch
                print(f"Batch processing failed, falling back to individual: {e}")
                for clue in batch:
                    try:
                        answer = self._generate_efficient_answer(clue['clue'])
                        results.append(answer)
                    except Exception as e2:
                        print(f"Failed to process clue: {e2}")
                        results.append("")
        
        return results
    
    def _build_batch_prompt(self, clues: List[Dict[str, str]]) -> str:
        """Build a prompt for processing multiple clues at once"""
        if self.extended_prompt:
            return self._build_extended_batch_prompt(clues)
        clues_block = "\n".join([f'{i}. Clue: "{clue["clue"]}"' for i, clue in enumerate(clues, 1)])
        return f"""Solve these cryptic crossword clues and provide just the answers.

{clues_block}

Respond with only the answers in uppercase letters, one per line, in order."""
    
    def _build_extended_batch_prompt(self, clues: List[Dict[str, str]]) -> str:
        """Build an extended batch prompt with cryptic crossword solving guidance"""
        clues_block = "\n".join([f'{i}. Clue: "{clue["clue"]}"' for i, clue in enumerate(clues, 1)])
        return f"""You are an expert cryptic crossword solver. Cryptic crossword clue contains three elements:

1. Definition (of the answer, often found at the start or end of the clue)
2. Indicator (indicates the type of wordplay)
3. Fodder (for the wordplay)

Common indicators:
-anagram indicators: "confused", "broken", "upset", etc.
-container indicators: "possessing", "hugging", "holding", etc.
-hidden word indicators: "held in", "a bit of", "part of", etc.
-reversal indicators: "backwards", "flipped", "held up", etc.
-deletion indicators: "missing", "cut", "absent", etc.
-letter position indicators: "first", "head", "middle", etc.
-homophone indicators: "said", "heard", "reportedly", etc.

Note: double definition and charade clues don't have indicator words, 
but they generally give a sense of one thing following another.

EXAMPLES:
"Bird seen in the museum (3)" -> "seen in" indicates containment, and length is 3 -> "EMU"
"Honestly crazy, in secret (2,3,3)" -> "crazy" indicates an anagram, and length is 2+3+3=8 -> "ONTHESLY"
"Stringed instrument untruthful person heard (4)" -> "heard" indicates homophone and the length is 4 -> "LYRE"
"Wear out an important part of a car (4)" -> no clear indicator words and legnth is 4 -> "TIRE"
"Returned beer of kings (5)" -> "returned" indicates reversal and length is 5 -> "LAGER"

POSSIBLE ANALYSIS STEPS:
1. Identify the definition, indicator, and fodder
2. Apply the wordplay to get the answer
3. Verify the answer fits the length requirement

CLUES TO SOLVE:
{clues_block}

Respond with only the answers in uppercase letters, one per line, in order."""
    
    def _parse_batch_response(self, content: str, expected_count: int) -> List[str]:
        """Parse a batch response into individual answers"""
        lines = [line.strip().upper() for line in content.split('\n') if line.strip()]
        
        # Clean up each line to just get letters and spaces
        cleaned_lines = []
        for line in lines:
            cleaned = re.sub(r'[^A-Z\s]', '', line).strip()
            if cleaned:
                cleaned_lines.append(cleaned)
        
        # Ensure we have the right number of answers
        while len(cleaned_lines) < expected_count:
            cleaned_lines.append("")
        
        return cleaned_lines[:expected_count]

    def _generate_llm_candidates(self, clue: str, k: int = 5) -> str:
        """Generate candidates using the configured LLM provider"""
        prompt = self._build_prompt(clue, k)
        
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
    
    def _build_prompt(self, clue: str, k: int) -> str:
        """Build the prompt for LLM providers"""
        if self.extended_prompt:
            return self._build_extended_json_prompt(clue, k)
        
        # Build the correct number of example candidates
        example_candidates = []
        for i in range(k):
            confidence = 0.95 - (i * 0.1)
            example_candidates.append(f'    {{"answer": "ANSWER{i+1}", "confidence": {confidence:.2f}, "reasoning": "brief explanation"}}')
        
        examples_text = ',\n'.join(example_candidates)
        
        return f"""Solve this cryptic crossword clue and provide {k} candidate answer{'s' if k > 1 else ''} with confidence scores.

Clue: "{clue}"

Please respond with exactly {k} candidate{'s' if k > 1 else ''} in this JSON format:
{{
  "candidates": [
{examples_text}
  ]
}}

Ensure all answers are uppercase and the confidence scores are between 0 and 1. Only return valid JSON."""

    def _build_extended_json_prompt(self, clue: str, k: int) -> str:
        """Build an extended JSON prompt with cryptic crossword solving guidance"""
        # Build the correct number of example candidates
        example_candidates = []
        for i in range(k):
            confidence = 0.95 - (i * 0.1)
            example_candidates.append(f'    {{"answer": "ANSWER{i+1}", "confidence": {confidence:.2f}, "reasoning": "brief explanation"}}')
        
        examples_text = ',\n'.join(example_candidates)
        
        return f"""You are an expert cryptic crossword clue solver. Cryptic crossword clue contains three elements:

1. Definition (of the answer, often found at the start or end of the clue)
2. Indicator (indicates the type of wordplay)
3. Fodder (for the wordplay)

Common indicators:
-anagram indicators: "confused", "broken", "upset", etc.
-container indicators: "possessing", "hugging", "holding", etc.
-hidden word indicators: "held in", "a bit of", "part of", etc.
-reversal indicators: "backwards", "flipped", "held up", etc.
-deletion indicators: "missing", "cut", "absent", etc.
-letter position indicators: "first", "head", "middle", etc.
-homophone indicators: "said", "heard", "reportedly", etc.

Note: double definition and charade clues don't have indicator words, 
but they generally give a sense of one thing following another.

EXAMPLES:
"Bird seen in the museum (3)" -> "seen in" indicates containment, and length is 3 -> "EMU"
"Honestly crazy, in secret (2,3,3)" -> "crazy" indicates an anagram, and length is 2+3+3=8 -> "ONTHESLY"
"Stringed instrument untruthful person heard (4)" -> "heard" indicates homophone and the length is 4 -> "LYRE"
"Wear out an important part of a car (4)" -> no clear indicator words and legnth is 4 -> "TIRE"
"Returned beer of kings (5)" -> "returned" indicates reversal and length is 5 -> "LAGER"

POSSIBLE ANALYSIS STEPS:
1. Identify the definition, indicator, and fodder
2. Apply the wordplay to get the answer
3. Verify the answer fits the length requirement

Clue: "{clue}"

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
