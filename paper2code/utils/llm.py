"""
Lightweight LLM client abstraction with Ollama support (and stubs for OpenAI/Anthropic).
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional

import aiohttp


@dataclass
class LLMConfig:
    provider: str
    model: str
    base_url: Optional[str] = None
    temperature: float = 0.3
    max_tokens: int = 2048
    api_key: Optional[str] = None
    context_window: int = 32768  # Default context window size


class LLMClient:
    def __init__(self, config: LLMConfig):
        self.config = config

    async def generate(self, prompt: str, system: Optional[str] = None) -> str:
        provider = (self.config.provider or 'ollama').lower()
        if provider == 'ollama':
            return await self._generate_ollama(prompt, system)
        # Stubs for other providers can be added here
        return ""
    
    async def generate_json(self, prompt: str, system: Optional[str] = None) -> Dict[str, Any]:
        """Generate JSON response with robust parsing and fallbacks."""
        response = await self.generate(prompt, system)
        return self._parse_json_response(response)
    
    async def analyze_full_document(self, content: str, system: str, merge_strategy: str = "comprehensive") -> Dict[str, Any]:
        """
        Analyze full document content respecting context window limits.
        
        Args:
            content: Full document content
            system: System prompt for analysis
            merge_strategy: How to merge chunk results ("comprehensive", "summary", "first_chunk")
            
        Returns:
            Merged analysis results
        """
        # Estimate tokens (rough: 1 token ≈ 4 characters)
        estimated_tokens = len(content) // 4
        system_tokens = len(system) // 4 if system else 0
        available_tokens = self.config.context_window - system_tokens - self.config.max_tokens - 500  # Safety buffer
        
        if estimated_tokens <= available_tokens:
            # Content fits in one request
            return await self.generate_json(content, system)
        
        # Need to chunk the content
        chunk_size = available_tokens * 4  # Convert back to characters
        chunks = self._create_intelligent_chunks(content, chunk_size)
        
        chunk_results = []
        for i, chunk in enumerate(chunks):
            chunk_prompt = f"Document chunk {i+1}/{len(chunks)}:\n\n{chunk}"
            result = await self.generate_json(chunk_prompt, system)
            chunk_results.append(result)
        
        # Merge results based on strategy
        return self._merge_chunk_results(chunk_results, merge_strategy)
    
    def _create_intelligent_chunks(self, content: str, max_chunk_size: int) -> list[str]:
        """Create intelligent chunks that respect document structure."""
        if len(content) <= max_chunk_size:
            return [content]
        
        chunks = []
        
        # Split by sections first (markdown headers)
        sections = re.split(r'\n(?=#{1,6}\s)', content)
        
        current_chunk = ""
        overlap_size = min(1000, max_chunk_size // 10)  # 10% overlap
        
        for section in sections:
            # If section alone is too big, split it further
            if len(section) > max_chunk_size:
                # Save current chunk if it has content
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = ""
                
                # Split large section by paragraphs
                paragraphs = section.split('\n\n')
                section_chunk = ""
                
                for para in paragraphs:
                    if len(section_chunk + para) > max_chunk_size:
                        if section_chunk:
                            chunks.append(section_chunk)
                            # Add overlap from previous chunk
                            section_chunk = section_chunk[-overlap_size:] + para
                        else:
                            # Single paragraph too large, split by sentences
                            sentences = re.split(r'[.!?]+\s+', para)
                            sentence_chunk = ""
                            for sentence in sentences:
                                if len(sentence_chunk + sentence) > max_chunk_size:
                                    if sentence_chunk:
                                        chunks.append(sentence_chunk)
                                    sentence_chunk = sentence
                                else:
                                    sentence_chunk += sentence + ". "
                            if sentence_chunk:
                                section_chunk = sentence_chunk
                    else:
                        section_chunk += para + "\n\n"
                
                if section_chunk:
                    current_chunk = section_chunk
            else:
                # Check if adding this section exceeds chunk size
                if len(current_chunk + section) > max_chunk_size:
                    if current_chunk:
                        chunks.append(current_chunk)
                        # Start new chunk with overlap
                        current_chunk = current_chunk[-overlap_size:] + section
                    else:
                        current_chunk = section
                else:
                    current_chunk += section
        
        # Add final chunk
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def _merge_chunk_results(self, chunk_results: list[Dict[str, Any]], strategy: str) -> Dict[str, Any]:
        """Merge results from multiple chunks."""
        if not chunk_results:
            return {'algorithms': [], 'formulas': [], 'components': []}
        
        if strategy == "first_chunk":
            return chunk_results[0]
        
        # Merge all results
        merged = {
            'algorithms': [],
            'formulas': [],
            'components': [],
            'concepts': []
        }
        
        seen_algorithms = set()
        seen_formulas = set()
        seen_components = set()
        seen_concepts = set()
        
        for result in chunk_results:
            if isinstance(result, dict):
                # Merge algorithms
                for alg in result.get('algorithms', []):
                    if isinstance(alg, dict):
                        alg_name = alg.get('name', '')
                        if alg_name and alg_name not in seen_algorithms:
                            merged['algorithms'].append(alg)
                            seen_algorithms.add(alg_name)
                    elif isinstance(alg, str) and alg not in seen_algorithms:
                        merged['algorithms'].append({'name': alg, 'content': ''})
                        seen_algorithms.add(alg)
                
                # Merge formulas
                for formula in result.get('formulas', []):
                    if isinstance(formula, dict):
                        formula_text = formula.get('formula', '')
                        if formula_text and formula_text not in seen_formulas:
                            merged['formulas'].append(formula)
                            seen_formulas.add(formula_text)
                    elif isinstance(formula, str) and formula not in seen_formulas:
                        merged['formulas'].append({'formula': formula, 'type': 'extracted'})
                        seen_formulas.add(formula)
                
                # Merge components
                for comp in result.get('components', []):
                    if isinstance(comp, dict):
                        comp_name = comp.get('name', '')
                        if comp_name and comp_name not in seen_components:
                            merged['components'].append(comp)
                            seen_components.add(comp_name)
                    elif isinstance(comp, str) and comp not in seen_components:
                        merged['components'].append({'name': comp})
                        seen_components.add(comp)
                
                # Merge concepts
                for concept in result.get('concepts', []):
                    if isinstance(concept, str) and concept not in seen_concepts:
                        merged['concepts'].append(concept)
                        seen_concepts.add(concept)
        
        return merged
    
    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON response with multiple fallback strategies."""
        if not response or not response.strip():
            return {}
        
        # Strategy 1: Direct JSON parsing
        try:
            return json.loads(response.strip())
        except json.JSONDecodeError:
            pass
        
        # Strategy 2: Extract JSON from markdown code blocks
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        
        # Strategy 3: Find JSON-like content between braces
        brace_match = re.search(r'\{.*\}', response, re.DOTALL)
        if brace_match:
            try:
                json_text = brace_match.group(0)
                # Clean common issues
                json_text = re.sub(r',\s*}', '}', json_text)  # Remove trailing commas
                json_text = re.sub(r',\s*]', ']', json_text)  # Remove trailing commas in arrays
                return json.loads(json_text)
            except json.JSONDecodeError:
                pass
        
        # Strategy 4: Try to extract key-value pairs manually
        try:
            return self._extract_key_values(response)
        except:
            pass
        
        # Strategy 5: Return structured fallback
        return {
            'error': 'Could not parse JSON response',
            'raw_response': response[:500] + ('...' if len(response) > 500 else ''),
            'algorithms': [],
            'formulas': [],
            'components': []
        }
    
    def _extract_key_values(self, text: str) -> Dict[str, Any]:
        """Extract key-value pairs from unstructured text as fallback."""
        result = {
            'algorithms': [],
            'formulas': [],
            'components': []
        }
        
        # Look for algorithm mentions
        alg_matches = re.findall(r'algorithm[s]?\s*:?\s*([^\n.]+)', text, re.IGNORECASE)
        for match in alg_matches[:5]:  # Limit to 5
            result['algorithms'].append({'name': match.strip(), 'content': ''})
        
        # Look for formula mentions
        formula_matches = re.findall(r'formula[s]?\s*:?\s*([^\n.]+)', text, re.IGNORECASE)
        for match in formula_matches[:5]:
            result['formulas'].append({'formula': match.strip(), 'type': 'extracted'})
        
        # Look for component mentions
        comp_matches = re.findall(r'component[s]?\s*:?\s*([^\n.]+)', text, re.IGNORECASE)
        for match in comp_matches[:5]:
            result['components'].append({'name': match.strip()})
        
        return result

    async def _generate_ollama(self, prompt: str, system: Optional[str]) -> str:
        url = (self.config.base_url or 'http://127.0.0.1:11434').rstrip('/') + '/api/generate'
        payload = {
            'model': self.config.model or 'llama3.2:latest',
            'prompt': (f"{system}\n\n{prompt}" if system else prompt),
            'stream': False,
            'options': {
                'temperature': self.config.temperature,
                'num_predict': self.config.max_tokens,
            },
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=600)) as resp:
                resp.raise_for_status()
                data = await resp.json()
                response = data.get('response', '')
                
                # Log response for debugging
                if len(response) < 100:
                    print(f"DEBUG: Short LLM response: {response}")
                
                return response


def build_llm_client(config_manager) -> LLMClient:
    llm_cfg = config_manager.get('llm', {})
    provider = llm_cfg.get('preferred_provider', 'ollama')
    temperature = llm_cfg.get('temperature', 0.3)
    max_tokens = llm_cfg.get('max_tokens', 2048)
    
    # Model-specific context windows
    context_windows = {
        'deepseek-r1:8b': 32768,
        'qwen3:4b': 32768,
        'aya:8b': 8192,
        'llama3.2:latest': 8192,
        'gpt-4': 8192,
        'gpt-4-turbo': 128000,
        'gpt-3.5-turbo': 16384,
        'claude-3-5-sonnet-20241022': 200000,
        'claude-3-haiku-20240307': 200000,
    }

    if provider == 'ollama':
        model = llm_cfg.get('ollama', {}).get('default_model', 'llama3.2:latest')
        base_url = config_manager.get_secret('ollama.base_url', llm_cfg.get('ollama', {}).get('base_url'))
        context_window = context_windows.get(model, 8192)
        return LLMClient(LLMConfig(
            provider='ollama', 
            model=model, 
            base_url=base_url, 
            temperature=temperature, 
            max_tokens=max_tokens,
            context_window=context_window
        ))
    
    elif provider == 'openai':
        model = llm_cfg.get('openai', {}).get('default_model', 'gpt-4')
        api_key = config_manager.get_secret('openai.api_key')
        context_window = context_windows.get(model, 8192)
        return LLMClient(LLMConfig(
            provider='openai',
            model=model,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            context_window=context_window
        ))
    
    elif provider == 'anthropic':
        model = llm_cfg.get('anthropic', {}).get('default_model', 'claude-3-5-sonnet-20241022')
        api_key = config_manager.get_secret('anthropic.api_key')
        context_window = context_windows.get(model, 200000)
        return LLMClient(LLMConfig(
            provider='anthropic',
            model=model,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            context_window=context_window
        ))

    # Fallback
    return LLMClient(LLMConfig(provider='ollama', model='llama3.2:latest', context_window=8192))


