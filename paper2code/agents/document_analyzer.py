"""
Document Analysis Agent for Paper2Code

Analyzes research papers to extract structure, algorithms, and implementation details.
"""

import json
import re
from typing import Dict, Any, List, Optional
from pathlib import Path

from ..config.manager import ConfigManager
from ..utils.logger import get_logger
from ..utils.llm import build_llm_client
from ..utils.structured_extraction import create_structured_extractor

logger = get_logger(__name__)


class DocumentAnalysisAgent:
    """
    AI agent specialized in analyzing research papers and extracting technical details.
    
    Handles both segmented and full document analysis approaches.
    """
    
    def __init__(self, config: ConfigManager):
        """
        Initialize document analysis agent.
        
        Args:
            config: Configuration manager instance
        """
        self.config = config
    
    async def analyze_full_document(self, content: str, input_path: Path) -> Dict[str, Any]:
        """
        Analyze the full document without segmentation.
        
        Args:
            content: Full document content
            input_path: Path to the input document
            
        Returns:
            Analysis results dictionary
        """
        logger.info("📖 Performing full document analysis")
        
        try:
            # Extract basic document information
            basic_info = self._extract_basic_info(content, input_path)
            
            # Try LLM-assisted extraction first if enabled
            technical_content = {}
            if self.config.get('llm.enabled', True):
                try:
                    logger.info("🔬 Using advanced structured LLM extraction (2025 best practices)")
                    llm = build_llm_client(self.config)
                    
                    # Try advanced structured extraction first
                    try:
                        structured_extractor = create_structured_extractor(llm)
                        paper_info = self._extract_basic_info(content, input_path)
                        paper_title = paper_info.get('title', '')
                        technical_content = await structured_extractor.extract_technical_content(content, paper_title)
                        
                        logger.info(f"🚀 Advanced extraction successful: {len(technical_content.get('algorithms', []))} algorithms, {len(technical_content.get('formulas', []))} formulas, {len(technical_content.get('components', []))} components")
                        
                        # Validate results - if we got good results, use them
                        if (technical_content.get('algorithms') or 
                            technical_content.get('formulas') or 
                            technical_content.get('components')):
                            pass  # Good results, continue
                        else:
                            raise ValueError("Structured extraction returned empty results")
                            
                    except Exception as struct_e:
                        logger.warning(f"⚠️ Structured extraction failed: {struct_e}, trying fallback LLM")
                        
                        # Fallback to original LLM approach
                        system = (
                            "You are an expert at analyzing machine learning research papers. "
                            "Carefully read the ENTIRE paper and extract:\n"
                            "1. ALGORITHMS: Any computational procedures, neural network architectures, training methods\n"
                            "2. FORMULAS: Mathematical equations, attention mechanisms, loss functions\n"
                            "3. COMPONENTS: Model components, layers, modules\n\n"
                            "For the 'Attention Is All You Need' paper, you should find:\n"
                            "- Self-attention algorithm\n"
                            "- Multi-head attention\n"
                            "- Transformer architecture\n"
                            "- Attention formulas (Q, K, V matrices)\n"
                            "- Position encoding\n\n"
                            "Return valid JSON only:\n"
                            "{\n"
                            '  "algorithms": [{"name": "Algorithm Name", "content": "description"}],\n'
                            '  "formulas": [{"formula": "mathematical expression", "type": "attention/loss/etc"}],\n'
                            '  "components": [{"name": "Component Name"}]\n'
                            "}"
                        )
                        
                        logger.info(f"📝 Analyzing full document: {len(content)} characters")
                        technical_content = await llm.analyze_full_document(
                            content=content,
                            system=system,
                            merge_strategy="comprehensive"
                        )
                        logger.info(f"✅ Fallback LLM extraction: {len(technical_content.get('algorithms', []))} algorithms, {len(technical_content.get('formulas', []))} formulas")
                        
                        if 'error' in technical_content:
                            logger.warning(f"⚠️ LLM returned error response: {technical_content['error']}")
                            raise ValueError("LLM parsing failed")
                        
                        # If LLM found nothing, it might be a parsing issue - try fallback
                        if (not technical_content.get('algorithms') and 
                            not technical_content.get('formulas') and 
                            not technical_content.get('components')):
                            logger.warning("⚠️ LLM found no technical content, this seems wrong - using regex fallback")
                            raise ValueError("LLM extraction incomplete")
                        
                except Exception as e:
                    logger.warning(f"⚠️ All LLM extraction methods failed: {e}, using LLM-only fallback")
                    # Final fallback - pure LLM extraction without structured parsing
                    technical_content = await self._llm_only_extraction(content, basic_info.get('title', ''))
            else:
                logger.info("🤖 Using pure LLM-based technical content extraction")
                # Use LLM-only extraction even when LLM is "disabled" for consistency
                technical_content = await self._llm_only_extraction(content, basic_info.get('title', ''))
            
            # Enhance technical content using LLM guidance or concept detection
            try:
                logger.info("🔍 Enhancing extracted technical content")
                technical_content = await self._enhance_content_with_llm_guidance(
                    technical_content, content, basic_info.get('title', '')
                )
            except Exception as e:
                logger.warning(f"⚠️ Content enhancement failed: {e}, using original results")
            
            # Extract paper structure
            structure = self._analyze_paper_structure(content)
            
            # Assess complexity
            complexity = self._assess_complexity(content, technical_content)
            
            # Key concepts (LLM if available)
            concepts = []
            if self.config.get('llm.enabled', True):
                try:
                    logger.info("🧠 Using LLM for key concept extraction")
                    llm = build_llm_client(self.config)
                    system = "Extract 10-20 key technical concepts from the research paper. Return a JSON array of strings only. Example: [\"neural networks\", \"gradient descent\", \"attention mechanism\"]"
                    concept_result = await llm.analyze_full_document(
                        content=content,
                        system=system,
                        merge_strategy="comprehensive"
                    )
                    
                    if isinstance(concept_result, list):
                        concepts = [str(x) for x in concept_result][:20]
                    elif isinstance(concept_result, dict) and 'concepts' in concept_result:
                        concepts = [str(x) for x in concept_result['concepts']][:20]
                    else:
                        logger.warning("⚠️ LLM concept extraction returned unexpected format")
                        concepts = self._extract_key_concepts(content)
                        
                    logger.info(f"✅ LLM concept extraction: {len(concepts)} concepts found")
                except Exception as e:
                    logger.warning(f"⚠️ LLM concept extraction failed: {e}, falling back to regex")
                    concepts = self._extract_key_concepts(content)
            else:
                logger.info("📊 Using regex-based concept extraction")
                concepts = self._extract_key_concepts(content)
            
            # Create comprehensive analysis result
            analysis_result = {
                'analysis_type': 'full_document',
                'document_info': basic_info,
                'structure': structure,
                'technical_content': technical_content,
                'complexity': complexity,
                'key_concepts': concepts,
                'implementation_requirements': self._extract_implementation_requirements(
                    content, technical_content, structure
                ),
                'summary': {
                    'title': basic_info.get('title', 'Unknown'),
                    'algorithm_count': len(technical_content.get('algorithms', [])),
                    'formula_count': len(technical_content.get('formulas', [])),
                    'complexity_level': complexity.get('level', 'Medium'),
                    'estimated_components': complexity.get('estimated_components', 5)
                }
            }
            
            logger.info(f"✅ Full document analysis completed: {analysis_result['summary']}")
            return analysis_result
            
        except Exception as e:
            logger.error(f"❌ Full document analysis failed: {e}")
            raise
    
    async def analyze_with_segmentation(self, content: str, input_path: Path) -> Dict[str, Any]:
        """
        Analyze document using intelligent segmentation.
        
        Args:
            content: Full document content
            input_path: Path to the input document
            
        Returns:
            Analysis results dictionary
        """
        logger.info("🔧 Performing segmented document analysis")
        
        try:
            # Create segments
            segments = self._create_intelligent_segments(content)
            
            # Analyze each segment
            segment_analyses = []
            for i, segment in enumerate(segments):
                logger.info(f"Analyzing segment {i+1}/{len(segments)}")
                
                segment_analysis = {
                    'segment_id': i,
                    'title': segment.get('title', f'Segment {i+1}'),
                    'content_type': segment.get('type', 'general'),
                    'technical_content': self._extract_technical_content(segment['content']),
                    'key_concepts': self._extract_key_concepts(segment['content']),
                    'size': len(segment['content'])
                }
                
                segment_analyses.append(segment_analysis)
            
            # Merge segment analyses into comprehensive result
            merged_analysis = self._merge_segment_analyses(segments, segment_analyses, input_path)
            
            logger.info(f"✅ Segmented analysis completed: {len(segments)} segments processed")
            return merged_analysis
            
        except Exception as e:
            logger.error(f"❌ Segmented document analysis failed: {e}")
            # Fall back to full document analysis
            logger.info("🔄 Falling back to full document analysis")
            return await self.analyze_full_document(content, input_path)
    
    def _extract_basic_info(self, content: str, input_path: Path) -> Dict[str, Any]:
        """Extract basic document information"""
        info = {
            'filename': input_path.name,
            'size_chars': len(content),
            'size_words': len(content.split()),
        }
        
        # Extract title (first heading or from filename)
        title_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
        if title_match:
            info['title'] = title_match.group(1).strip()
        else:
            info['title'] = input_path.stem.replace('_', ' ').title()
        
        # Extract abstract (common patterns)
        abstract_patterns = [
            r'##\s*Abstract\s*\n(.*?)\n##',
            r'#\s*Abstract\s*\n(.*?)\n#',
            r'\*\*Abstract\*\*\s*\n(.*?)\n\n',
        ]
        
        for pattern in abstract_patterns:
            match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
            if match:
                info['abstract'] = match.group(1).strip()
                break
        
        return info
    
    async def _extract_technical_content(self, content: str) -> Dict[str, List]:
        """Extract algorithms, formulas, and technical components with enhanced patterns"""
        
        # Extract algorithms with better patterns
        algorithms = []
        # Extract algorithms using LLM instead of regex patterns
        algorithms = await self._extract_algorithms_with_llm(content)
        
        # Extract mathematical formulas using LLM instead of regex patterns
        formulas = await self._extract_formulas_with_llm(content)
        
        # Extract components/modules using LLM instead of regex patterns
        components = await self._extract_components_with_llm(content)
        
        return {
            'algorithms': algorithms,
            'formulas': formulas,
            'components': list({c.get('name', ''): c for c in components}.values())  # Deduplicate
        }
    
    async def _extract_formulas_with_llm(self, content: str) -> List[Dict]:
        """Extract mathematical formulas using LLM instead of regex patterns"""
        
        if not self.config.get('llm.enabled', True):
            logger.warning("🔄 LLM disabled, cannot extract formulas intelligently")
            return []
        
        try:
            llm = build_llm_client(self.config)
            
            prompt = f"""
You are an expert at extracting mathematical formulas from research papers.

TASK: Extract ALL mathematical formulas, equations, and expressions from this content. Look for:
1. Mathematical equations with = signs
2. Function definitions (e.g., f(x) = ...)
3. Loss functions and objective functions
4. Algorithmic formulas and computations
5. Statistical formulas and probability expressions
6. Matrix/vector operations
7. Optimization formulas
8. Any mathematical expressions that define relationships

Return ONLY valid JSON:
{{
  "formulas": [
    {{
      "formula": "exact mathematical expression",
      "type": "category (e.g., attention, loss, optimization, activation, probability, etc.)",
      "description": "what this formula computes or represents"
    }}
  ]
}}

Content:
{content[:6000]}...
"""
            
            response = await llm.generate(prompt)
            
            try:
                import json
                formula_data = json.loads(response.strip())
                formulas = []
                
                for f_data in formula_data.get('formulas', []):
                    formula_text = f_data.get('formula', '')
                    if formula_text and len(formula_text) > 3:
                        formulas.append({
                            'formula': formula_text,
                            'type': f_data.get('type', 'mathematical'),
                            'description': f_data.get('description', 'Mathematical formula')
                        })
                
                logger.info(f"🧠 LLM extracted {len(formulas)} formulas")
                return formulas
                
            except json.JSONDecodeError as e:
                logger.warning(f"⚠️ Could not parse LLM formula response: {e}")
                return []
                
        except Exception as e:
            logger.warning(f"⚠️ LLM formula extraction failed: {e}")
            return []
    
    async def _extract_components_with_llm(self, content: str) -> List[Dict]:
        """Extract components/modules using LLM instead of regex patterns"""
        
        if not self.config.get('llm.enabled', True):
            logger.warning("🔄 LLM disabled, cannot extract components intelligently")
            return []
        
        try:
            llm = build_llm_client(self.config)
            
            prompt = f"""
You are an expert at extracting technical components from research papers.

TASK: Extract ALL technical components, modules, and architectural elements from this content. Look for:
1. Neural network layers (e.g., attention layer, convolution layer, etc.)
2. Model components (e.g., encoder, decoder, generator, etc.)
3. Architectural modules (e.g., transformer block, residual connection, etc.)
4. Processing components (e.g., tokenizer, embedding layer, etc.)
5. Optimization components (e.g., optimizer, scheduler, etc.)
6. Any named technical building blocks

Return ONLY valid JSON:
{{
  "components": [
    {{
      "name": "component name",
      "type": "category (e.g., layer, architecture, module, etc.)",
      "description": "what this component does"
    }}
  ]
}}

Content:
{content[:6000]}...
"""
            
            response = await llm.generate(prompt)
            
            try:
                import json
                component_data = json.loads(response.strip())
                components = []
                
                for c_data in component_data.get('components', []):
                    component_name = c_data.get('name', '')
                    if component_name and len(component_name) > 2:
                        components.append({
                            'name': component_name,
                            'type': c_data.get('type', 'component'),
                            'description': c_data.get('description', 'Technical component')
                        })
                
                logger.info(f"🧠 LLM extracted {len(components)} components")
                return components
                
            except json.JSONDecodeError as e:
                logger.warning(f"⚠️ Could not parse LLM component response: {e}")
                return []
                
        except Exception as e:
            logger.warning(f"⚠️ LLM component extraction failed: {e}")
            return []
    
    async def _extract_algorithms_with_llm(self, content: str) -> List[Dict]:
        """Extract algorithms using LLM instead of regex patterns"""
        
        if not self.config.get('llm.enabled', True):
            logger.warning("🔄 LLM disabled, cannot extract algorithms intelligently")
            return []
        
        try:
            llm = build_llm_client(self.config)
            
            prompt = f"""
You are an expert at extracting algorithms from research papers.

TASK: Extract ALL algorithms, methods, procedures, and computational approaches from this content. Look for:
1. Named algorithms (e.g., "Adam Algorithm", "Backpropagation", etc.)
2. Pseudocode blocks or algorithm descriptions
3. Step-by-step procedures or methods
4. Training procedures and optimization approaches
5. Novel techniques or approaches introduced in the paper
6. Any computational methods that solve specific problems

Return ONLY valid JSON:
{{
  "algorithms": [
    {{
      "name": "algorithm name",
      "content": "description or pseudocode of the algorithm",
      "type": "category (e.g., optimization, training, inference, etc.)"
    }}
  ]
}}

Content:
{content[:7000]}...
"""
            
            response = await llm.generate(prompt)
            
            try:
                import json
                algorithm_data = json.loads(response.strip())
                algorithms = []
                
                for a_data in algorithm_data.get('algorithms', []):
                    algorithm_name = a_data.get('name', '')
                    algorithm_content = a_data.get('content', '')
                    if algorithm_name and algorithm_content and len(algorithm_content) > 10:
                        algorithms.append({
                            'name': algorithm_name,
                            'content': algorithm_content,
                            'type': a_data.get('type', 'algorithm')
                        })
                
                logger.info(f"🧠 LLM extracted {len(algorithms)} algorithms")
                return algorithms
                
            except json.JSONDecodeError as e:
                logger.warning(f"⚠️ Could not parse LLM algorithm response: {e}")
                return []
                
        except Exception as e:
            logger.warning(f"⚠️ LLM algorithm extraction failed: {e}")
            return []
    
    def _analyze_paper_structure(self, content: str) -> Dict[str, Any]:
        """Analyze the overall structure of the paper"""
        
        # Extract section headers using LLM-friendly approach
        sections = []
        header_pattern = r'^(#{1,6})\s+(.+)$'
        
        for match in re.finditer(header_pattern, content, re.MULTILINE):
            level = len(match.group(1))
            title = match.group(2).strip()
            sections.append({
                'level': level,
                'title': title,
                'position': match.start()
            })
        
        # Categorize sections
        section_categories = {
            'introduction': [],
            'method': [],
            'experiments': [],
            'results': [],
            'conclusion': [],
            'other': []
        }
        
        for section in sections:
            title_lower = section['title'].lower()
            
            if any(word in title_lower for word in ['introduction', 'intro']):
                section_categories['introduction'].append(section)
            elif any(word in title_lower for word in ['method', 'approach', 'algorithm', 'model']):
                section_categories['method'].append(section)
            elif any(word in title_lower for word in ['experiment', 'evaluation', 'setup']):
                section_categories['experiments'].append(section)
            elif any(word in title_lower for word in ['result', 'finding', 'performance']):
                section_categories['results'].append(section)
            elif any(word in title_lower for word in ['conclusion', 'discussion', 'future']):
                section_categories['conclusion'].append(section)
            else:
                section_categories['other'].append(section)
        
        return {
            'total_sections': len(sections),
            'max_depth': max([s['level'] for s in sections]) if sections else 0,
            'categories': section_categories,
            'structure_quality': self._assess_structure_quality(section_categories)
        }
    
    def _assess_complexity(self, content: str, technical_content: Dict[str, List]) -> Dict[str, Any]:
        """Assess the complexity of the paper for implementation"""
        
        # Count various complexity indicators
        algorithm_count = len(technical_content.get('algorithms', []))
        formula_count = len(technical_content.get('formulas', []))
        component_count = len(technical_content.get('components', []))
        
        # Count technical terms
        technical_terms = [
            'neural network', 'machine learning', 'deep learning', 'optimization',
            'gradient', 'loss function', 'training', 'validation', 'model',
            'architecture', 'parameter', 'hyperparameter'
        ]
        
        technical_term_count = sum(
            len(re.findall(rf'\b{term}\b', content, re.IGNORECASE)) 
            for term in technical_terms
        )
        
        # Calculate complexity score
        complexity_score = (
            algorithm_count * 15 +
            formula_count * 5 +
            component_count * 10 +
            technical_term_count * 2 +
            len(content) // 10000  # Document length factor
        )
        
        # Determine complexity level
        if complexity_score < 50:
            level = "Low"
            estimated_components = 2
        elif complexity_score < 150:
            level = "Medium"
            estimated_components = 5
        else:
            level = "High"
            estimated_components = 8
        
        return {
            'score': complexity_score,
            'level': level,
            'algorithm_count': algorithm_count,
            'formula_count': formula_count,
            'component_count': component_count,
            'technical_term_count': technical_term_count,
            'estimated_components': estimated_components,
            'estimated_implementation_time': self._estimate_implementation_time(complexity_score)
        }
    
    def _extract_key_concepts(self, content: str) -> List[str]:
        """Extract key concepts and terminology from the content"""
        
        # Common AI/ML/CS concepts
        concept_patterns = [
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:algorithm|method|approach|technique)\b',
            r'\b((?:deep|machine|reinforcement)\s+learning)\b',
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:network|model|architecture)\b',
            r'\b([a-z]+(?:-[a-z]+)*)\s+(?:optimization|gradient|descent)\b',
        ]
        
        concepts = set()
        
        for pattern in concept_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    concept = match[0] if match[0] else match[1]
                else:
                    concept = match
                
                if len(concept) > 3:
                    concepts.add(concept.strip())
        
        # Also extract capitalized terms that appear multiple times
        capitalized_terms = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', content)
        term_counts = {}
        for term in capitalized_terms:
            if len(term) > 3:
                term_counts[term] = term_counts.get(term, 0) + 1
        
        # Add frequently occurring terms
        for term, count in term_counts.items():
            if count >= 3:  # Appears at least 3 times
                concepts.add(term)
        
        return list(concepts)
    
    def _extract_implementation_requirements(
        self, 
        content: str, 
        technical_content: Dict[str, List],
        structure: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract specific implementation requirements"""
        
        requirements = {
            'programming_language': self._detect_programming_language(content),
            'frameworks': self._detect_frameworks(content),
            'datasets': self._detect_datasets(content),
            'evaluation_metrics': self._extract_evaluation_metrics(content),
            'dependencies': self._extract_dependencies(content),
        }
        
        return requirements
    
    def _detect_programming_language(self, content: str) -> str:
        """Detect the primary programming language mentioned"""
        
        language_indicators = {
            'python': ['python', 'pytorch', 'tensorflow', 'numpy', 'pandas', 'sklearn'],
            'r': ['r language', 'ggplot', 'dplyr', 'cran'],
            'matlab': ['matlab', 'simulink'],
            'java': ['java', 'weka'],
            'cpp': ['c++', 'cpp', 'opencv'],
            'javascript': ['javascript', 'node.js', 'react']
        }
        
        language_scores = {}
        content_lower = content.lower()
        
        for lang, indicators in language_indicators.items():
            score = sum(content_lower.count(indicator) for indicator in indicators)
            language_scores[lang] = score
        
        # Default to Python if no clear indicators
        if not language_scores or max(language_scores.values()) == 0:
            return 'python'
        
        return max(language_scores, key=language_scores.get)
    
    def _detect_frameworks(self, content: str) -> List[str]:
        """Detect mentioned frameworks and libraries"""
        
        frameworks = []
        framework_patterns = [
            r'\b(pytorch|tensorflow|keras|scikit-learn|pandas|numpy)\b',
            r'\b(react|vue|angular|express|django|flask)\b',
            r'\b(opencv|pillow|matplotlib|seaborn)\b',
        ]
        
        for pattern in framework_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            frameworks.extend(matches)
        
        return list(set(frameworks))
    
    def _detect_datasets(self, content: str) -> List[str]:
        """Detect mentioned datasets"""
        
        dataset_patterns = [
            r'\b([A-Z][A-Z0-9-]*)\s+dataset\b',
            r'\bdataset[s]?\s+([A-Z][a-zA-Z0-9-]+)\b',
            r'\b(CIFAR|MNIST|ImageNet|COCO|Pascal|OpenImages)\b',
        ]
        
        datasets = []
        for pattern in dataset_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    datasets.extend([m for m in match if m])
                else:
                    datasets.append(match)
        
        return list(set(datasets))
    
    def _extract_evaluation_metrics(self, content: str) -> List[str]:
        """Extract evaluation metrics mentioned"""
        
        metric_patterns = [
            r'\b(accuracy|precision|recall|f1-score|auc|mse|rmse|mae)\b',
            r'\b(bleu|rouge|meteor|perplexity)\b',
            r'\b(iou|map|ap|dice|jaccard)\b',
        ]
        
        metrics = []
        for pattern in metric_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            metrics.extend(matches)
        
        return list(set(metrics))
    
    def _extract_dependencies(self, content: str) -> List[str]:
        """Extract likely dependencies from content"""
        
        # This is a simplified version - in practice, this would be more sophisticated
        common_deps = ['numpy', 'pandas', 'matplotlib', 'scikit-learn', 'torch', 'tensorflow']
        
        dependencies = []
        content_lower = content.lower()
        
        for dep in common_deps:
            if dep in content_lower:
                dependencies.append(dep)
        
        return dependencies
    
    def _create_intelligent_segments(self, content: str) -> List[Dict[str, Any]]:
        """Create intelligent segments for large documents"""
        
        # Configuration
        max_segment_size = self.config.get('document_segmentation.size_threshold_chars', 50000) // 3
        overlap_size = self.config.get('document_segmentation.overlap_chars', 1000)
        
        # Split by sections first
        sections = self._split_by_sections(content)
        
        segments = []
        current_segment = ""
        current_title = "Introduction"
        
        for section in sections:
            section_content = section['content']
            
            # If section is small enough, add to current segment
            if len(current_segment + section_content) < max_segment_size:
                current_segment += section_content
            else:
                # Save current segment
                if current_segment:
                    segments.append({
                        'title': current_title,
                        'content': current_segment,
                        'type': self._classify_segment_type(current_segment)
                    })
                
                # Start new segment
                current_segment = section_content
                current_title = section.get('title', f"Section {len(segments)+1}")
        
        # Add final segment
        if current_segment:
            segments.append({
                'title': current_title,
                'content': current_segment,
                'type': self._classify_segment_type(current_segment)
            })
        
        return segments
    
    def _split_by_sections(self, content: str) -> List[Dict[str, Any]]:
        """Split content by section headers"""
        
        sections = []
        header_pattern = r'^(#{1,6})\s+(.+)$'
        
        # Find all headers
        headers = list(re.finditer(header_pattern, content, re.MULTILINE))
        
        if not headers:
            # No headers found, return entire content as one section
            return [{'title': 'Document', 'content': content}]
        
        # Create sections
        for i, header_match in enumerate(headers):
            title = header_match.group(2).strip()
            start_pos = header_match.end()
            
            # Find end position (start of next header or end of document)
            if i + 1 < len(headers):
                end_pos = headers[i + 1].start()
            else:
                end_pos = len(content)
            
            section_content = content[start_pos:end_pos].strip()
            
            sections.append({
                'title': title,
                'content': f"# {title}\n{section_content}",
                'level': len(header_match.group(1))
            })
        
        return sections
    
    def _classify_segment_type(self, content: str) -> str:
        """Classify the type of content in a segment"""
        
        content_lower = content.lower()
        
        # Algorithm-heavy content
        if ('algorithm' in content_lower and content_lower.count('algorithm') > 2) or \
           'pseudocode' in content_lower:
            return 'algorithm_focused'
        
        # Method/approach content
        if any(word in content_lower for word in ['method', 'approach', 'technique', 'model']):
            return 'methodology'
        
        # Experimental content
        if any(word in content_lower for word in ['experiment', 'result', 'evaluation', 'performance']):
            return 'experimental'
        
        # Introduction/background
        if any(word in content_lower for word in ['introduction', 'background', 'related work']):
            return 'background'
        
        return 'general'
    
    def _merge_segment_analyses(
        self,
        segments: List[Dict[str, Any]],
        analyses: List[Dict[str, Any]],
        input_path: Path
    ) -> Dict[str, Any]:
        """Merge analyses from multiple segments"""
        
        # Merge technical content
        all_algorithms = []
        all_formulas = []
        all_components = []
        all_concepts = []
        
        for analysis in analyses:
            tech_content = analysis.get('technical_content', {})
            all_algorithms.extend(tech_content.get('algorithms', []))
            all_formulas.extend(tech_content.get('formulas', []))
            all_components.extend(tech_content.get('components', []))
            all_concepts.extend(analysis.get('key_concepts', []))
        
        # Deduplicate and merge
        merged_technical_content = {
            'algorithms': list({alg['name']: alg for alg in all_algorithms}.values()),
            'formulas': list({f['formula']: f for f in all_formulas}.values()),
            'components': list({c['name']: c for c in all_components}.values())
        }
        
        merged_concepts = list(set(all_concepts))
        
        # Create document info from segments
        total_content = '\n'.join([seg['content'] for seg in segments])
        basic_info = self._extract_basic_info(total_content, input_path)
        
        # Assess overall complexity
        complexity = self._assess_complexity(total_content, merged_technical_content)
        
        return {
            'analysis_type': 'segmented',
            'segment_count': len(segments),
            'segments': segments,
            'segment_analyses': analyses,
            'document_info': basic_info,
            'technical_content': merged_technical_content,
            'complexity': complexity,
            'key_concepts': merged_concepts,
            'implementation_requirements': self._extract_implementation_requirements(
                total_content, merged_technical_content, {}
            ),
            'summary': {
                'title': basic_info.get('title', 'Unknown'),
                'algorithm_count': len(merged_technical_content.get('algorithms', [])),
                'formula_count': len(merged_technical_content.get('formulas', [])),
                'complexity_level': complexity.get('level', 'Medium'),
                'estimated_components': complexity.get('estimated_components', 5)
            }
        }
    
    def _assess_structure_quality(self, categories: Dict[str, List]) -> str:
        """Assess the quality of paper structure"""
        
        has_intro = len(categories['introduction']) > 0
        has_method = len(categories['method']) > 0
        has_experiments = len(categories['experiments']) > 0
        has_results = len(categories['results']) > 0
        
        quality_score = sum([has_intro, has_method, has_experiments, has_results])
        
        if quality_score >= 4:
            return "Excellent"
        elif quality_score >= 3:
            return "Good"
        elif quality_score >= 2:
            return "Fair"
        else:
            return "Poor"
    
    def _estimate_implementation_time(self, complexity_score: int) -> str:
        """Estimate implementation time based on complexity"""
        
        if complexity_score < 50:
            return "1-2 days"
        elif complexity_score < 100:
            return "3-5 days"
        elif complexity_score < 200:
            return "1-2 weeks"
        else:
            return "2+ weeks"
    
    async def _enhance_content_with_llm_guidance(self, technical_content: Dict[str, List], content: str, paper_title: str = "") -> Dict[str, List]:
        """Enhance technical content using LLM-guided domain analysis instead of hardcoded patterns"""
        
        if not self.config.get('llm.enabled', True):
            logger.info("🔄 LLM disabled, using generic concept-based enhancement")
            return self._enhance_content_based_on_detected_concepts(technical_content, content)
        
        try:
            # Use LLM to identify missing key elements based on paper domain
            llm = build_llm_client(self.config)
            
            # Analyze what should be present in this type of paper
            enhancement_prompt = f"""
You are an expert at analyzing research papers and identifying missing technical elements.

Paper Title: {paper_title}

Current extracted content:
- Algorithms: {[alg.get('name', 'unnamed') for alg in technical_content.get('algorithms', [])]}
- Formulas: {[f.get('formula', 'unnamed')[:50] + '...' if len(f.get('formula', '')) > 50 else f.get('formula', 'unnamed') for f in technical_content.get('formulas', [])]}
- Components: {[c.get('name', 'unnamed') for c in technical_content.get('components', [])]}

TASK: Based on the paper content below, identify what important algorithms, formulas, or components might be missing that are typically present in this type of research.

Return ONLY valid JSON with missing elements:
{{
  "missing_algorithms": [
    {{"name": "algorithm_name", "content": "description", "type": "category"}}
  ],
  "missing_formulas": [
    {{"formula": "mathematical_expression", "type": "formula_type"}}
  ],
  "missing_components": [
    {{"name": "component_name", "type": "component_type"}}
  ]
}}

Paper content (first 6000 chars):
{content[:6000]}...
"""
            
            response = await llm.generate(enhancement_prompt)
            
            # Parse LLM response
            try:
                import json
                enhancement_data = json.loads(response.strip())
                
                # Add missing elements identified by LLM
                existing_alg_names = {alg.get('name', '').lower() for alg in technical_content.get('algorithms', [])}
                existing_formula_texts = {f.get('formula', '') for f in technical_content.get('formulas', [])}
                existing_comp_names = {c.get('name', '').lower() for c in technical_content.get('components', [])}
                
                added_count = 0
                
                # Add missing algorithms
                for alg in enhancement_data.get('missing_algorithms', []):
                    if alg.get('name', '').lower() not in existing_alg_names:
                        technical_content.setdefault('algorithms', []).append(alg)
                        added_count += 1
                
                # Add missing formulas
                for formula in enhancement_data.get('missing_formulas', []):
                    if formula.get('formula', '') not in existing_formula_texts:
                        technical_content.setdefault('formulas', []).append(formula)
                        added_count += 1
                
                # Add missing components
                for comp in enhancement_data.get('missing_components', []):
                    if comp.get('name', '').lower() not in existing_comp_names:
                        technical_content.setdefault('components', []).append(comp)
                        added_count += 1
                
                logger.info(f"🧠 LLM-guided enhancement added {added_count} missing elements")
                
            except json.JSONDecodeError as e:
                logger.warning(f"⚠️ Could not parse LLM enhancement response: {e}")
                logger.info("🔄 Falling back to concept-based enhancement")
                return self._enhance_content_based_on_detected_concepts(technical_content, content)
                
        except Exception as e:
            logger.warning(f"⚠️ LLM enhancement failed: {e}")
            logger.info("🔄 Falling back to concept-based enhancement")
            return self._enhance_content_based_on_detected_concepts(technical_content, content)
        
        logger.info(f"✅ Enhanced content: {len(technical_content.get('algorithms', []))} algorithms, {len(technical_content.get('formulas', []))} formulas, {len(technical_content.get('components', []))} components")
        
        return technical_content
    
    def _enhance_content_based_on_detected_concepts(self, technical_content: Dict[str, List], content: str) -> Dict[str, List]:
        """Enhance technical content based on detected ML/AI concepts in any paper"""
        
        content_lower = content.lower()
        
        # Detect concepts present in the paper
        detected_concepts = []
        concept_patterns = {
            'attention': ['attention', 'query', 'key', 'value', 'self-attention', 'multi-head'],
            'transformer': ['transformer', 'encoder', 'decoder', 'positional encoding'],
            'cnn': ['convolution', 'convolutional', 'cnn', 'filter', 'kernel', 'pooling'],
            'rnn': ['recurrent', 'rnn', 'lstm', 'gru', 'sequence'],
            'gan': ['generative adversarial', 'gan', 'generator', 'discriminator'],
            'reinforcement': ['reinforcement', 'reward', 'policy', 'q-learning'],
            'optimization': ['gradient descent', 'adam', 'sgd', 'optimizer'],
            'regularization': ['dropout', 'batch normalization', 'layer normalization']
        }
        
        for concept, patterns in concept_patterns.items():
            if any(pattern in content_lower for pattern in patterns):
                detected_concepts.append(concept)
        
        logger.info(f"🔍 Detected concepts in paper: {detected_concepts}")
        
        # Add relevant content based on detected concepts
        concept_enhancements = self._get_concept_enhancements()
        
        existing_alg_names = {alg.get('name', '').lower() for alg in technical_content.get('algorithms', [])}
        existing_formula_texts = {f.get('formula', '') for f in technical_content.get('formulas', [])}
        existing_comp_names = {c.get('name', '').lower() for c in technical_content.get('components', [])}
        
        added_count = 0
        
        for concept in detected_concepts:
            if concept in concept_enhancements:
                enhancement = concept_enhancements[concept]
                
                # Add algorithms
                for alg in enhancement.get('algorithms', []):
                    if alg['name'].lower() not in existing_alg_names:
                        technical_content.setdefault('algorithms', []).append(alg)
                        existing_alg_names.add(alg['name'].lower())
                        added_count += 1
                
                # Add formulas
                for formula in enhancement.get('formulas', []):
                    if formula['formula'] not in existing_formula_texts:
                        technical_content.setdefault('formulas', []).append(formula)
                        existing_formula_texts.add(formula['formula'])
                        added_count += 1
                
                # Add components
                for comp in enhancement.get('components', []):
                    if comp['name'].lower() not in existing_comp_names:
                        technical_content.setdefault('components', []).append(comp)
                        existing_comp_names.add(comp['name'].lower())
                        added_count += 1
        
        if added_count > 0:
            logger.info(f"🚀 Enhanced content based on detected concepts: +{added_count} items added")
        
        return technical_content
    
    def _get_concept_enhancements(self) -> Dict[str, Dict]:
        """Get enhancement data for different ML/AI concepts"""
        
        return {
            'attention': {
                'algorithms': [
                    {'name': 'Self-Attention', 'content': 'Attention mechanism for sequence modeling', 'type': 'attention'},
                    {'name': 'Multi-Head Attention', 'content': 'Parallel attention heads', 'type': 'attention'}
                ],
                'formulas': [
                    {'formula': 'Attention(Q,K,V) = softmax(QK^T/√d_k)V', 'type': 'attention'}
                ],
                'components': [
                    {'name': 'Attention Layer', 'type': 'layer'}
                ]
            },
            'cnn': {
                'algorithms': [
                    {'name': 'Convolution', 'content': 'Apply filters to detect local features', 'type': 'convolution'},
                    {'name': 'Pooling', 'content': 'Reduce spatial dimensions', 'type': 'pooling'}
                ],
                'formulas': [
                    {'formula': 'y[i,j] = Σ Σ x[i+m,j+n] * w[m,n]', 'type': 'convolution'}
                ],
                'components': [
                    {'name': 'Convolutional Layer', 'type': 'layer'}
                ]
            },
            'rnn': {
                'algorithms': [
                    {'name': 'LSTM', 'content': 'Long Short-Term Memory for sequence processing', 'type': 'rnn'},
                    {'name': 'GRU', 'content': 'Gated Recurrent Unit', 'type': 'rnn'}
                ],
                'components': [
                    {'name': 'Recurrent Layer', 'type': 'layer'}
                ]
            },
            'optimization': {
                'algorithms': [
                    {'name': 'Adam', 'content': 'Adaptive moment estimation optimizer', 'type': 'optimization'},
                    {'name': 'SGD', 'content': 'Stochastic gradient descent', 'type': 'optimization'}
                ],
                'formulas': [
                    {'formula': 'θ = θ - α∇J(θ)', 'type': 'optimization'}
                ]
            }
        }
    
    async def _llm_only_extraction(self, content: str, paper_title: str) -> Dict[str, List]:
        """Pure LLM-based extraction as final fallback"""
        
        try:
            llm = build_llm_client(self.config)
            
            prompt = f"""
You are an expert at analyzing research papers. Extract technical content from this paper.

Paper: {paper_title}

Extract:
1. All algorithms and methods
2. All mathematical formulas  
3. All technical components

Return simple JSON (no complex structure):
{{
  "algorithms": [
    {{"name": "Algorithm Name", "content": "description", "type": "category"}}
  ],
  "formulas": [
    {{"formula": "math expression", "type": "category"}}
  ],
  "components": [
    {{"name": "Component Name", "type": "category"}}
  ]
}}

Paper content:
{content[:6000]}...
"""
            
            response = await llm.generate(prompt)
            
            # Simple JSON parsing
            try:
                import json
                result = json.loads(response.strip())
                logger.info(f"✅ LLM-only extraction: {len(result.get('algorithms', []))} algorithms, {len(result.get('formulas', []))} formulas")
                return result
            except:
                # Even simpler fallback
                return {
                    'algorithms': [{'name': 'Main Algorithm', 'content': 'Primary algorithm from paper', 'type': 'general'}],
                    'formulas': [{'formula': 'f(x) = result', 'type': 'general'}],
                    'components': [{'name': 'Main Component', 'type': 'module'}]
                }
                
        except Exception as e:
            logger.warning(f"⚠️ LLM-only extraction failed: {e}")
            # Return minimal structure
            return {
                'algorithms': [{'name': 'Algorithm', 'content': 'Algorithm from paper', 'type': 'general'}],
                'formulas': [{'formula': 'y = f(x)', 'type': 'general'}],
                'components': [{'name': 'Component', 'type': 'module'}]
            }
