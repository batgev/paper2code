"""
Advanced LLM extraction using 2025 best practices:
- Pydantic models for structured output
- LangChain for prompt chaining
- Instructor for type-safe extraction
- Multi-step chain-of-thought prompting
"""

from typing import List, Dict, Optional, Any, Union
from pydantic import BaseModel, Field
import instructor
from langchain.prompts import PromptTemplate
from langchain.schema import BaseOutputParser
import json
import re
from .logger import get_logger

logger = get_logger(__name__)


class Algorithm(BaseModel):
    """Structured representation of an algorithm"""
    name: str = Field(description="Name of the algorithm")
    content: str = Field(description="Detailed description of the algorithm")
    type: str = Field(description="Type/category of algorithm", default="general")
    complexity: Optional[str] = Field(description="Time/space complexity if mentioned", default=None)
    pseudocode: Optional[str] = Field(description="Pseudocode if available", default=None)


class Formula(BaseModel):
    """Structured representation of a mathematical formula"""
    formula: str = Field(description="The mathematical expression")
    type: str = Field(description="Type of formula (attention, loss, optimization, etc.)")
    description: Optional[str] = Field(description="What the formula computes", default=None)
    variables: Optional[List[str]] = Field(description="List of variables in the formula", default_factory=list)


class Component(BaseModel):
    """Structured representation of a technical component"""
    name: str = Field(description="Name of the component")
    type: str = Field(description="Type/category of component", default="module")
    description: Optional[str] = Field(description="Description of the component", default=None)
    dependencies: Optional[List[str]] = Field(description="Dependencies or related components", default_factory=list)


class TechnicalContent(BaseModel):
    """Complete structured technical content extraction"""
    algorithms: List[Algorithm] = Field(description="All algorithms found in the paper", default_factory=list)
    formulas: List[Formula] = Field(description="All mathematical formulas found", default_factory=list)
    components: List[Component] = Field(description="All technical components found", default_factory=list)
    
    class Config:
        json_encoders = {
            # Custom JSON encoding if needed
        }


class StructuredExtractor:
    """Advanced LLM-based extraction using structured output parsing"""
    
    def __init__(self, llm_client):
        self.llm = llm_client
        self.instructor_client = None
        self._setup_instructor()
    
    def _setup_instructor(self):
        """Setup instructor for structured extraction"""
        try:
            # Try to setup instructor with the LLM client
            if hasattr(self.llm, 'openai_client'):
                self.instructor_client = instructor.patch(self.llm.openai_client)
            elif hasattr(self.llm, 'anthropic_client'):
                self.instructor_client = instructor.patch(self.llm.anthropic_client)
        except Exception as e:
            logger.warning(f"Could not setup instructor: {e}")
    
    async def extract_technical_content(self, content: str, paper_title: str = "") -> Dict[str, Any]:
        """
        Extract technical content using pure LLM-driven approach for ANY paper
        """
        logger.info("🤖 Starting pure LLM-driven extraction (no hardcoded patterns)")
        
        # Step 1: Let LLM analyze the paper and determine what type of research it is
        paper_analysis = await self._analyze_paper_type_and_domain(content, paper_title)
        
        # Step 2: Extract algorithms using domain-aware LLM prompts
        algorithms = await self._extract_algorithms_llm_driven(content, paper_title, paper_analysis)
        
        # Step 3: Extract formulas using domain-aware LLM prompts
        formulas = await self._extract_formulas_with_llm(content, paper_analysis)
        
        # Step 4: Extract components using domain-aware LLM prompts
        components = await self._extract_components_llm_driven(content, paper_title, paper_analysis)
        
        # Step 5: LLM-driven cross-validation and enhancement
        enhanced_content = await self._llm_enhance_and_validate(
            algorithms, formulas, components, content, paper_title, paper_analysis
        )
        
        logger.info(f"✅ Pure LLM extraction complete: {len(enhanced_content.algorithms)} algorithms, "
                   f"{len(enhanced_content.formulas)} formulas, {len(enhanced_content.components)} components")
        
        return enhanced_content.dict()
    
    async def _analyze_paper_type_and_domain(self, content: str, paper_title: str) -> Dict[str, Any]:
        """Let LLM analyze the paper to determine its domain and research type"""
        
        prompt = f"""
You are an expert at analyzing research papers across all domains of machine learning and AI.

Paper Title: {paper_title}

TASK: Analyze this research paper and determine:
1. Research domain (e.g., computer vision, NLP, reinforcement learning, optimization, etc.)
2. Main techniques used (e.g., neural networks, statistical methods, mathematical optimization, etc.)
3. Key concepts present (e.g., attention, convolution, recurrence, generative models, etc.)
4. Algorithm types (e.g., supervised learning, unsupervised learning, reinforcement learning, etc.)
5. Mathematical foundations (e.g., linear algebra, probability theory, optimization theory, etc.)

Return ONLY valid JSON:
{{
  "domain": "research domain",
  "techniques": ["technique1", "technique2"],
  "concepts": ["concept1", "concept2"],
  "algorithm_types": ["type1", "type2"],
  "math_foundations": ["foundation1", "foundation2"],
  "complexity_level": "low|medium|high",
  "implementation_difficulty": "easy|medium|hard"
}}

Paper content:
{content[:8000]}...
"""
        
        try:
            response = await self.llm.generate(prompt)
            analysis = self._parse_json_response(response)
            logger.info(f"🧠 Paper analysis: {analysis.get('domain', 'unknown')} domain, {len(analysis.get('concepts', []))} concepts")
            return analysis
        except Exception as e:
            logger.warning(f"⚠️ Paper analysis failed: {e}")
            return {"domain": "unknown", "concepts": [], "techniques": []}
    
    async def _extract_algorithms_llm_driven(self, content: str, paper_title: str, paper_analysis: Dict) -> List[Algorithm]:
        """Extract algorithms using pure LLM analysis"""
        
        domain = paper_analysis.get('domain', 'machine learning')
        concepts = paper_analysis.get('concepts', [])
        
        prompt = f"""
You are an expert at extracting algorithms from {domain} research papers.

Paper Title: {paper_title}
Research Domain: {domain}
Key Concepts: {concepts}

TASK: Extract ALL algorithms, methods, and procedures from this paper.

For {domain} papers, look for:
- Core algorithms and methods
- Training procedures
- Inference methods
- Optimization techniques
- Data processing steps
- Evaluation procedures

Return ONLY valid JSON array:
[
  {{
    "name": "Algorithm Name",
    "content": "Detailed description of what the algorithm does and how it works",
    "type": "algorithm_type",
    "complexity": "time/space complexity if mentioned",
    "pseudocode": "step-by-step procedure if available"
  }}
]

DO NOT include explanations. Return only the JSON array.

Paper content:
{content[:10000]}...
"""
        
        try:
            response = await self.llm.generate(prompt)
            algorithms_data = self._parse_json_list(response)
            
            algorithms = []
            for alg_data in algorithms_data:
                if isinstance(alg_data, dict) and 'name' in alg_data:
                    algorithms.append(Algorithm(**alg_data))
            
            logger.info(f"🤖 LLM extracted {len(algorithms)} algorithms")
            return algorithms
            
        except Exception as e:
            logger.warning(f"⚠️ LLM algorithm extraction failed: {e}")
            return []
    
    async def _extract_formulas_llm_driven(self, content: str, paper_title: str, paper_analysis: Dict) -> List[Formula]:
        """Extract formulas using pure LLM analysis"""
        
        domain = paper_analysis.get('domain', 'machine learning')
        math_foundations = paper_analysis.get('math_foundations', [])
        
        prompt = f"""
You are an expert at extracting mathematical formulas from {domain} research papers.

Paper Title: {paper_title}
Research Domain: {domain}
Mathematical Foundations: {math_foundations}

TASK: Extract ALL mathematical formulas, equations, and expressions from this paper.

For {domain} papers, look for:
- Core mathematical formulas
- Loss functions
- Optimization equations
- Probability distributions
- Statistical measures
- Computational expressions
- Algorithm equations

Return ONLY valid JSON array:
[
  {{
    "formula": "complete mathematical expression",
    "type": "formula_category",
    "description": "what this formula computes or represents",
    "variables": ["list", "of", "variables"]
  }}
]

DO NOT include explanations. Return only the JSON array.

Paper content:
{content[:10000]}...
"""
        
        try:
            response = await self.llm.generate(prompt)
            formulas_data = self._parse_json_list(response)
            
            formulas = []
            for formula_data in formulas_data:
                if isinstance(formula_data, dict) and 'formula' in formula_data:
                    formulas.append(Formula(**formula_data))
            
            logger.info(f"🧮 LLM extracted {len(formulas)} formulas")
            return formulas
            
        except Exception as e:
            logger.warning(f"⚠️ LLM formula extraction failed: {e}")
            return []
    
    async def _extract_components_llm_driven(self, content: str, paper_title: str, paper_analysis: Dict) -> List[Component]:
        """Extract components using pure LLM analysis"""
        
        domain = paper_analysis.get('domain', 'machine learning')
        techniques = paper_analysis.get('techniques', [])
        
        prompt = f"""
You are an expert at identifying technical components in {domain} research papers.

Paper Title: {paper_title}
Research Domain: {domain}
Main Techniques: {techniques}

TASK: Extract ALL technical components, modules, and building blocks from this paper.

For {domain} papers, look for:
- Model components and modules
- Data processing components
- Training components
- Evaluation components
- System architecture elements
- Implementation building blocks

Return ONLY valid JSON array:
[
  {{
    "name": "Component Name",
    "type": "component_category",
    "description": "what this component does",
    "dependencies": ["related", "components"]
  }}
]

DO NOT include explanations. Return only the JSON array.

Paper content:
{content[:10000]}...
"""
        
        try:
            response = await self.llm.generate(prompt)
            components_data = self._parse_json_list(response)
            
            components = []
            for comp_data in components_data:
                if isinstance(comp_data, dict) and 'name' in comp_data:
                    components.append(Component(**comp_data))
            
            logger.info(f"🧩 LLM extracted {len(components)} components")
            return components
            
        except Exception as e:
            logger.warning(f"⚠️ LLM component extraction failed: {e}")
            return []
    
    async def _llm_enhance_and_validate(self, algorithms: List[Algorithm], 
                                       formulas: List[Formula], 
                                       components: List[Component],
                                       content: str, paper_title: str, 
                                       paper_analysis: Dict) -> TechnicalContent:
        """Use LLM to enhance and validate extracted content"""
        
        domain = paper_analysis.get('domain', 'machine learning')
        
        # Convert to dicts for LLM processing
        alg_dicts = [alg.dict() for alg in algorithms]
        formula_dicts = [f.dict() for f in formulas]
        comp_dicts = [c.dict() for c in components]
        
        prompt = f"""
You are an expert at validating and enhancing technical content extraction from {domain} research papers.

Paper Title: {paper_title}
Research Domain: {domain}

CURRENT EXTRACTION:
Algorithms: {len(alg_dicts)}
Formulas: {len(formula_dicts)}
Components: {len(comp_dicts)}

TASK: Review the extracted content and:
1. Identify any missing important algorithms, formulas, or components
2. Add missing items that are clearly described in the paper
3. Remove any incorrect or irrelevant items
4. Ensure completeness for this {domain} paper

Return ONLY valid JSON:
{{
  "algorithms": [
    {{"name": "Algorithm Name", "content": "description", "type": "category"}}
  ],
  "formulas": [
    {{"formula": "mathematical expression", "type": "category", "description": "what it computes", "variables": ["vars"]}}
  ],
  "components": [
    {{"name": "Component Name", "type": "category", "description": "what it does"}}
  ],
  "validation_notes": "brief notes on what was added/removed"
}}

Current extraction to review:
ALGORITHMS: {alg_dicts}
FORMULAS: {formula_dicts}
COMPONENTS: {comp_dicts}

Paper content:
{content[:8000]}...
"""
        
        try:
            response = await self.llm.generate(prompt)
            enhanced_data = self._parse_json_response(response)
            
            # Convert back to Pydantic models
            enhanced_algorithms = [Algorithm(**alg) for alg in enhanced_data.get('algorithms', [])]
            enhanced_formulas = [Formula(**f) for f in enhanced_data.get('formulas', [])]
            enhanced_components = [Component(**c) for c in enhanced_data.get('components', [])]
            
            logger.info(f"🚀 LLM enhancement: {len(enhanced_algorithms)} algorithms, "
                       f"{len(enhanced_formulas)} formulas, {len(enhanced_components)} components")
            
            if 'validation_notes' in enhanced_data:
                logger.info(f"📝 LLM notes: {enhanced_data['validation_notes']}")
            
            return TechnicalContent(
                algorithms=enhanced_algorithms,
                formulas=enhanced_formulas,
                components=enhanced_components
            )
            
        except Exception as e:
            logger.warning(f"⚠️ LLM enhancement failed: {e}, using original extraction")
            return TechnicalContent(
                algorithms=algorithms,
                formulas=formulas,
                components=components
            )
    
    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON response with fallback strategies"""
        
        # Log for debugging
        logger.debug(f"🔍 Parsing response: {response[:200]}...")
        
        # Try multiple parsing strategies
        strategies = [
            lambda r: json.loads(r.strip()),
            lambda r: json.loads(r.split('```json')[1].split('```')[0]) if '```json' in r else None,
            lambda r: json.loads(r.split('```')[1].split('```')[0]) if r.count('```') >= 2 else None,
        ]
        
        for strategy in strategies:
            try:
                result = strategy(response)
                if result:
                    return result
            except:
                continue
        
        # Fallback: return empty structure
        logger.warning("⚠️ Could not parse LLM response, using empty structure")
        return {"algorithms": [], "formulas": [], "components": []}
    
    async def _extract_algorithms_cot(self, content: str, paper_title: str) -> List[Algorithm]:
        """Extract algorithms using chain-of-thought prompting"""
        
        prompt = f"""
You are an expert at analyzing machine learning research papers. 

Paper Title: {paper_title}

TASK: Extract ALL algorithms from this paper using chain-of-thought reasoning.

STEP 1: First, identify potential algorithm sections by looking for:
- Algorithm boxes/pseudocode
- Method descriptions
- Architecture descriptions
- Training procedures
- Attention mechanisms
- Neural network components

STEP 2: For each identified algorithm, extract:
- Name (be specific and descriptive)
- Detailed content/description
- Type/category
- Complexity if mentioned
- Pseudocode if available

STEP 3: For papers about Transformers/Attention, make sure to find:
- Self-Attention mechanism
- Multi-Head Attention
- Scaled Dot-Product Attention
- Positional Encoding
- Transformer Architecture
- Feed-Forward Networks

Return ONLY a JSON list of algorithms in this format:
[
  {{
    "name": "Algorithm Name",
    "content": "Detailed description of what the algorithm does",
    "type": "attention_mechanism|architecture|training|optimization|etc",
    "complexity": "O(n²) or similar if mentioned",
    "pseudocode": "step-by-step pseudocode if available"
  }}
]

Paper content:
{content[:8000]}...
"""
        
        try:
            response = await self.llm.generate(prompt)
            algorithms_data = self._parse_json_list(response)
            
            algorithms = []
            for alg_data in algorithms_data:
                if isinstance(alg_data, dict) and 'name' in alg_data:
                    algorithms.append(Algorithm(**alg_data))
            
            logger.info(f"🤖 Extracted {len(algorithms)} algorithms via CoT")
            return algorithms
            
        except Exception as e:
            logger.warning(f"⚠️ Algorithm extraction failed: {e}")
            return []
    
    async def _extract_formulas_cot(self, content: str, paper_title: str) -> List[Formula]:
        """Extract formulas using chain-of-thought prompting"""
        
        prompt = f"""
You are an expert at extracting mathematical formulas from research papers.

Paper Title: {paper_title}

CRITICAL: You MUST return valid JSON only. Do not include explanations or markdown.

TASK: Find ALL mathematical formulas in this paper.

Look for these specific patterns:
1. Function definitions: Function(args) = expression
2. Mathematical equations with = signs
3. Matrix operations
4. Attention mechanisms
5. Loss functions
6. Probability distributions

Look for formulas like these examples (adapt to your specific paper):
- Function definitions: f(x) = expression
- Mathematical equations: y = mx + b
- Loss functions: L = -log(p)
- Optimization updates: θ = θ - α∇J(θ)
- Probability distributions: P(x|θ) = expression
- Neural network operations: output = activation(Wx + b)

IMPORTANT: Return ONLY valid JSON in this exact format:
[
  {{"formula": "actual_formula_from_paper", "type": "formula_type", "description": "what it computes", "variables": ["var1", "var2"]}},
  {{"formula": "another_formula", "type": "formula_type", "description": "description", "variables": ["variables"]}}
]

DO NOT include any text before or after the JSON. Start with [ and end with ].

Paper content (search for formulas):
{content[:12000]}...
"""
        
        try:
            response = await self.llm.generate(prompt)
            formulas_data = self._parse_json_list(response)
            
            formulas = []
            for formula_data in formulas_data:
                if isinstance(formula_data, dict) and 'formula' in formula_data:
                    formulas.append(Formula(**formula_data))
            
            logger.info(f"🧮 Extracted {len(formulas)} formulas via CoT")
            return formulas
            
        except Exception as e:
            logger.warning(f"⚠️ Formula extraction failed: {e}, trying direct extraction")
            # Fallback: Direct formula extraction from content
            return self._extract_formulas_directly(content)
    
    async def _extract_components_cot(self, content: str, paper_title: str) -> List[Component]:
        """Extract components using chain-of-thought prompting"""
        
        prompt = f"""
You are an expert at identifying technical components in ML research papers.

Paper Title: {paper_title}

TASK: Extract ALL technical components using systematic analysis.

STEP 1: Identify component categories:
- Architecture components (Encoder, Decoder, etc.)
- Layer types (Attention, Feed-Forward, Normalization)
- Data processing components (Embeddings, Tokenization)
- Training components (Optimizers, Loss functions)
- Evaluation components (Metrics, Benchmarks)

STEP 2: For each component, extract:
- Specific name
- Type/category
- Description of purpose
- Dependencies or relationships

STEP 3: For Transformer papers, ensure you find:
- Encoder/Decoder blocks
- Multi-Head Attention layers
- Position-wise Feed-Forward Networks
- Layer Normalization
- Residual Connections
- Positional Embeddings
- Linear projections

Return ONLY a JSON list of components:
[
  {{
    "name": "Component Name",
    "type": "architecture|layer|embedding|optimization|etc",
    "description": "what this component does",
    "dependencies": ["related", "components"]
  }}
]

Paper content:
{content[:8000]}...
"""
        
        try:
            response = await self.llm.generate(prompt)
            components_data = self._parse_json_list(response)
            
            components = []
            for comp_data in components_data:
                if isinstance(comp_data, dict) and 'name' in comp_data:
                    components.append(Component(**comp_data))
            
            logger.info(f"🧩 Extracted {len(components)} components via CoT")
            return components
            
        except Exception as e:
            logger.warning(f"⚠️ Component extraction failed: {e}")
            return []
    
    async def _cross_validate_and_enhance(self, algorithms: List[Algorithm], 
                                        formulas: List[Formula], 
                                        components: List[Component],
                                        content: str, paper_title: str) -> TechnicalContent:
        """Cross-validate and enhance extracted content"""
        
        # Apply generic enhancements based on detected concepts
        algorithms, formulas, components = self._enhance_content_based_on_concepts(
            algorithms, formulas, components, content, paper_title
        )
        
        # Create structured output
        technical_content = TechnicalContent(
            algorithms=algorithms,
            formulas=formulas,
            components=components
        )
        
        return technical_content
    
    def _enhance_content_based_on_concepts(self, algorithms: List[Algorithm], 
                                          formulas: List[Formula], 
                                          components: List[Component],
                                          content: str, paper_title: str) -> tuple:
        """Enhance content based on detected concepts in any paper"""
        
        content_lower = content.lower()
        title_lower = paper_title.lower()
        
        # Define concept-based enhancements (generic, not paper-specific)
        concept_enhancements = self._get_concept_based_enhancements()
        
        # Detect concepts in the paper
        detected_concepts = self._detect_paper_concepts(content_lower, title_lower)
        
        logger.info(f"🔍 Detected concepts: {detected_concepts}")
        
        # Add relevant algorithms, formulas, and components based on detected concepts
        existing_alg_names = {alg.name.lower() for alg in algorithms}
        existing_formula_texts = {f.formula for f in formulas}  
        existing_comp_names = {c.name.lower() for c in components}
        
        added_algs = added_formulas = added_comps = 0
        
        for concept in detected_concepts:
            if concept in concept_enhancements:
                enhancement = concept_enhancements[concept]
                
                # Add relevant algorithms
                for alg_data in enhancement.get('algorithms', []):
                    if alg_data['name'].lower() not in existing_alg_names:
                        algorithms.append(Algorithm(**alg_data))
                        existing_alg_names.add(alg_data['name'].lower())
                        added_algs += 1
                
                # Add relevant formulas
                for formula_data in enhancement.get('formulas', []):
                    if formula_data['formula'] not in existing_formula_texts:
                        formulas.append(Formula(**formula_data))
                        existing_formula_texts.add(formula_data['formula'])
                        added_formulas += 1
                
                # Add relevant components
                for comp_data in enhancement.get('components', []):
                    if comp_data['name'].lower() not in existing_comp_names:
                        components.append(Component(**comp_data))
                        existing_comp_names.add(comp_data['name'].lower())
                        added_comps += 1
        
        if added_algs or added_formulas or added_comps:
            logger.info(f"🚀 Enhanced content based on concepts: +{added_algs} algorithms, "
                       f"+{added_formulas} formulas, +{added_comps} components")
        
        return algorithms, formulas, components
    
    def _detect_paper_concepts(self, content_lower: str, title_lower: str) -> List[str]:
        """Detect ML/AI concepts present in the paper"""
        
        detected = []
        
        # Define concept detection patterns
        concept_patterns = {
            'attention': ['attention', 'query', 'key', 'value', 'self-attention', 'multi-head'],
            'transformer': ['transformer', 'encoder', 'decoder', 'positional encoding'],
            'cnn': ['convolution', 'convolutional', 'cnn', 'filter', 'kernel', 'pooling'],
            'rnn': ['recurrent', 'rnn', 'lstm', 'gru', 'sequence'],
            'gan': ['generative adversarial', 'gan', 'generator', 'discriminator'],
            'reinforcement': ['reinforcement', 'reward', 'policy', 'q-learning', 'actor', 'critic'],
            'optimization': ['gradient descent', 'adam', 'sgd', 'optimizer', 'learning rate'],
            'regularization': ['dropout', 'batch normalization', 'layer normalization', 'regularization'],
            'embedding': ['embedding', 'word2vec', 'glove', 'representation learning'],
            'classification': ['classification', 'classifier', 'softmax', 'cross-entropy'],
            'regression': ['regression', 'mse', 'mean squared error', 'linear regression'],
            'clustering': ['clustering', 'k-means', 'hierarchical', 'dbscan'],
            'dimensionality_reduction': ['pca', 'principal component', 'dimensionality reduction', 't-sne'],
            'neural_network': ['neural network', 'multilayer perceptron', 'backpropagation', 'activation']
        }
        
        # Check content and title for concept patterns
        combined_text = content_lower + ' ' + title_lower
        
        for concept, patterns in concept_patterns.items():
            if any(pattern in combined_text for pattern in patterns):
                detected.append(concept)
        
        return detected
    
    def _get_concept_based_enhancements(self) -> Dict[str, Dict]:
        """Get enhancement data organized by ML/AI concepts"""
        
        return {
            'attention': {
                'algorithms': [
                    {
                        'name': 'Self-Attention',
                        'content': 'Attention mechanism that relates different positions of a single sequence',
                        'type': 'attention_mechanism'
                    },
                    {
                        'name': 'Multi-Head Attention', 
                        'content': 'Multiple attention heads operating in parallel',
                        'type': 'attention_mechanism'
                    },
                    {
                        'name': 'Scaled Dot-Product Attention',
                        'content': 'Core attention computation using query, key, and value matrices',
                        'type': 'attention_computation'
                    }
                ],
                'formulas': [
                    {
                        'formula': 'Attention(Q,K,V) = softmax(QK^T/√d_k)V',
                        'type': 'attention',
                        'description': 'Scaled dot-product attention computation',
                        'variables': ['Q', 'K', 'V', 'd_k']
                    },
                    {
                        'formula': 'MultiHead(Q,K,V) = Concat(head_1,...,head_h)W^O',
                        'type': 'multi_head_attention',
                        'description': 'Multi-head attention combining multiple heads',
                        'variables': ['Q', 'K', 'V', 'W^O', 'h']
                    }
                ],
                'components': [
                    {'name': 'Attention Layer', 'type': 'layer', 'description': 'Multi-head attention layer'},
                    {'name': 'Query Projection', 'type': 'projection', 'description': 'Linear projection for queries'},
                    {'name': 'Key Projection', 'type': 'projection', 'description': 'Linear projection for keys'},
                    {'name': 'Value Projection', 'type': 'projection', 'description': 'Linear projection for values'}
                ]
            },
            'transformer': {
                'algorithms': [
                    {
                        'name': 'Transformer Architecture',
                        'content': 'Complete encoder-decoder architecture based on attention mechanisms',
                        'type': 'architecture'
                    },
                    {
                        'name': 'Positional Encoding',
                        'content': 'Sinusoidal position embeddings to inject sequence order information',
                        'type': 'encoding'
                    }
                ],
                'formulas': [
                    {
                        'formula': 'PE(pos,2i) = sin(pos/10000^(2i/d_model))',
                        'type': 'positional_encoding',
                        'description': 'Positional encoding for even dimensions',
                        'variables': ['pos', 'i', 'd_model']
                    },
                    {
                        'formula': 'PE(pos,2i+1) = cos(pos/10000^(2i/d_model))',
                        'type': 'positional_encoding', 
                        'description': 'Positional encoding for odd dimensions',
                        'variables': ['pos', 'i', 'd_model']
                    }
                ],
                'components': [
                    {'name': 'Encoder', 'type': 'architecture', 'description': 'Transformer encoder stack'},
                    {'name': 'Decoder', 'type': 'architecture', 'description': 'Transformer decoder stack'},
                    {'name': 'Feed Forward Network', 'type': 'layer', 'description': 'Position-wise feed-forward network'},
                    {'name': 'Embedding Layer', 'type': 'embedding', 'description': 'Token and positional embeddings'}
                ]
            },
            'cnn': {
                'algorithms': [
                    {
                        'name': 'Convolution Operation',
                        'content': 'Apply filters to input to detect local features',
                        'type': 'convolution'
                    },
                    {
                        'name': 'Pooling',
                        'content': 'Reduce spatial dimensions while preserving important information',
                        'type': 'pooling'
                    }
                ],
                'formulas': [
                    {
                        'formula': 'y[i,j] = Σ Σ x[i+m,j+n] * w[m,n]',
                        'type': 'convolution',
                        'description': '2D convolution operation',
                        'variables': ['x', 'w', 'i', 'j', 'm', 'n']
                    }
                ],
                'components': [
                    {'name': 'Convolutional Layer', 'type': 'layer', 'description': 'Layer that applies convolution operations'},
                    {'name': 'Pooling Layer', 'type': 'layer', 'description': 'Layer that reduces spatial dimensions'},
                    {'name': 'Filter', 'type': 'parameter', 'description': 'Learnable convolution kernel'}
                ]
            },
            'optimization': {
                'algorithms': [
                    {
                        'name': 'Gradient Descent',
                        'content': 'Iterative optimization algorithm that follows the gradient',
                        'type': 'optimization'
                    },
                    {
                        'name': 'Adam Optimizer',
                        'content': 'Adaptive moment estimation optimization algorithm',
                        'type': 'optimization'
                    }
                ],
                'formulas': [
                    {
                        'formula': 'θ = θ - α∇J(θ)',
                        'type': 'optimization',
                        'description': 'Gradient descent parameter update',
                        'variables': ['θ', 'α', '∇J']
                    }
                ],
                'components': [
                    {'name': 'Optimizer', 'type': 'optimizer', 'description': 'Algorithm for updating model parameters'},
                    {'name': 'Learning Rate Scheduler', 'type': 'scheduler', 'description': 'Adjusts learning rate during training'}
                ]
            },
            'regularization': {
                'algorithms': [
                    {
                        'name': 'Dropout',
                        'content': 'Randomly set some neurons to zero during training to prevent overfitting',
                        'type': 'regularization'
                    },
                    {
                        'name': 'Batch Normalization',
                        'content': 'Normalize inputs to each layer to stabilize training',
                        'type': 'normalization'
                    }
                ],
                'components': [
                    {'name': 'Dropout Layer', 'type': 'regularization', 'description': 'Layer that applies dropout regularization'},
                    {'name': 'Batch Norm Layer', 'type': 'normalization', 'description': 'Layer that applies batch normalization'}
                ]
            }
        }
    
    def _parse_json_list(self, response: str) -> List[Dict]:
        """Parse JSON response with multiple fallback strategies"""
        
        # Log the raw response for debugging
        logger.info(f"🔍 Raw LLM response (first 500 chars): {response[:500]}...")
        
        # Strategy 1: Direct JSON parsing
        try:
            parsed = json.loads(response.strip())
            if isinstance(parsed, list):
                logger.info(f"✅ Direct JSON parse successful: {len(parsed)} items")
                return parsed
            elif isinstance(parsed, dict):
                logger.info("✅ Direct JSON parse successful: 1 item (dict converted to list)")
                return [parsed]
        except Exception as e:
            logger.debug(f"Direct JSON parse failed: {e}")
        
        # Strategy 2: Extract JSON from markdown code blocks
        json_pattern = r'```(?:json)?\s*(\[.*?\])\s*```'
        matches = re.findall(json_pattern, response, re.DOTALL)
        for match in matches:
            try:
                parsed = json.loads(match)
                logger.info(f"✅ Markdown JSON parse successful: {len(parsed)} items")
                return parsed
            except Exception as e:
                logger.debug(f"Markdown JSON parse failed: {e}")
                continue
        
        # Strategy 3: Find JSON array in text (improved)
        array_patterns = [
            r'\[[\s\S]*?\]',  # Basic array
            r'(\[(?:[^[\]]*(?:\[[^\]]*\])*)*[^[\]]*\])',  # Nested arrays
        ]
        
        for pattern in array_patterns:
            matches = re.findall(pattern, response, re.DOTALL)
            for match in matches:
                try:
                    # Clean up the match
                    cleaned = match.strip()
                    parsed = json.loads(cleaned)
                    if isinstance(parsed, list) and len(parsed) > 0:
                        logger.info(f"✅ Array pattern parse successful: {len(parsed)} items")
                        return parsed
                except Exception as e:
                    logger.debug(f"Array pattern parse failed: {e}")
                    continue
        
        # Strategy 4: Look for individual JSON objects and combine them
        try:
            objects = []
            # More robust object pattern that handles nested braces
            obj_pattern = r'\{(?:[^{}]|{[^{}]*})*\}'
            matches = re.findall(obj_pattern, response, re.DOTALL)
            
            for match in matches:
                try:
                    # Clean up the match
                    cleaned = match.strip()
                    obj = json.loads(cleaned)
                    if isinstance(obj, dict) and obj:  # Non-empty dict
                        objects.append(obj)
                except Exception as e:
                    logger.debug(f"Individual object parse failed: {e}")
                    continue
                    
            if objects:
                logger.info(f"✅ Individual objects parse successful: {len(objects)} items")
                return objects
        except Exception as e:
            logger.debug(f"Individual objects strategy failed: {e}")
        
        # Strategy 5: Manual extraction for formula-like patterns
        try:
            formulas = []
            # Look for formula patterns in the response
            formula_patterns = [
                r'([A-Za-z]+\([^)]+\)\s*=\s*[^,\n]+)',  # Function(args) = expression
                r'([A-Z]+[_\d]*\s*=\s*[^,\n]+)',       # VAR = expression  
                r'(softmax\([^)]+\))',                   # softmax(...)
                r'(Attention\([^)]+\))',                 # Attention(...)
                r'(MultiHead\([^)]+\))',                 # MultiHead(...)
                r'(PE\([^)]+\)\s*=\s*[^,\n]+)',        # PE(...) = ...
            ]
            
            for pattern in formula_patterns:
                matches = re.findall(pattern, response, re.IGNORECASE)
                for match in matches:
                    formula_text = match.strip()
                    if len(formula_text) > 3:  # Avoid tiny matches
                        formulas.append({
                            'formula': formula_text,
                            'type': 'extracted',
                            'description': 'Extracted from text',
                            'variables': []
                        })
            
            if formulas:
                logger.info(f"✅ Formula pattern extraction successful: {len(formulas)} formulas")
                return formulas
                
        except Exception as e:
            logger.debug(f"Formula pattern extraction failed: {e}")
        
        # Strategy 6: Try to extract any structured data from the response
        try:
            # Look for key-value patterns that might be formulas
            lines = response.split('\n')
            extracted_items = []
            
            for line in lines:
                line = line.strip()
                if '=' in line and len(line) > 5:
                    # This might be a formula
                    extracted_items.append({
                        'formula': line,
                        'type': 'line_extracted',
                        'description': 'Extracted from line',
                        'variables': []
                    })
                elif any(word in line.lower() for word in ['attention', 'softmax', 'multihead', 'encoding']):
                    # This might be algorithm or component description
                    if len(line) > 10:
                        extracted_items.append({
                            'name': line[:50],  # First 50 chars as name
                            'content': line,
                            'type': 'extracted'
                        })
            
            if extracted_items:
                logger.info(f"✅ Line-based extraction successful: {len(extracted_items)} items")
                return extracted_items
                
        except Exception as e:
            logger.debug(f"Line-based extraction failed: {e}")
        
        logger.warning("⚠️ Could not parse JSON response, returning empty list")
        return []

    async def _extract_formulas_with_llm(self, content: str, paper_analysis: Dict) -> List[Formula]:
        """Extract formulas using LLM intelligence instead of regex patterns"""
        
        domain = paper_analysis.get('domain', 'machine learning')
        concepts = paper_analysis.get('concepts', [])
        
        prompt = f"""
You are an expert at extracting mathematical formulas from research papers.

Paper Domain: {domain}
Key Concepts: {', '.join(concepts)}

TASK: Extract ALL mathematical formulas, equations, and expressions from this paper content. Look for:
1. Mathematical equations with = signs
2. Function definitions (e.g., f(x) = ...)
3. Loss functions and objective functions
4. Algorithmic formulas and computations
5. Statistical formulas and probability expressions
6. Matrix/vector operations
7. Optimization formulas
8. Any mathematical expressions that define relationships

For each formula you find, determine:
- The exact formula text
- The type/category (e.g., attention, loss, optimization, activation, etc.)
- A brief description of what it computes
- The variables involved

Return ONLY valid JSON:
{{
  "formulas": [
    {{
      "formula": "exact mathematical expression",
      "type": "category (e.g., attention, loss, optimization, activation, probability, etc.)",
      "description": "what this formula computes or represents",
      "variables": ["variable1", "variable2", ...]
    }}
  ]
}}

Paper content:
{content[:8000]}...
"""
        
        try:
            response = await self.llm.generate(prompt)
            formula_data = self._parse_json_response(response)
            
            formulas = []
            for f_data in formula_data.get('formulas', []):
                if self._is_valid_formula(f_data):
                    formulas.append(Formula(
                        formula=f_data.get('formula', ''),
                        type=f_data.get('type', 'mathematical'),
                        description=f_data.get('description', 'Mathematical formula'),
                        variables=f_data.get('variables', [])
                    ))
            
            logger.info(f"🧠 LLM extracted {len(formulas)} formulas")
            return formulas
            
        except Exception as e:
            logger.warning(f"⚠️ LLM formula extraction failed: {e}")
            return []
    
    def _is_valid_formula(self, formula_data: Dict) -> bool:
        """Validate if extracted formula data is valid"""
        formula_text = formula_data.get('formula', '')
        
        # Basic validation
        if not formula_text or len(formula_text) < 3:
            return False
            
        # Should contain mathematical content
        math_indicators = ['=', '+', '-', '*', '/', '(', ')', 'sin', 'cos', 'log', 'exp', 'sum', 'max', 'min', '∑', '∫']
        if not any(indicator in formula_text for indicator in math_indicators):
            return False
            
        # Filter out obvious non-formulas
        noise_indicators = ['figure', 'table', 'section', 'page', 'et al', 'i.e', 'e.g']
        if any(noise in formula_text.lower() for noise in noise_indicators):
            return False
            
        return True


def create_structured_extractor(llm_client) -> StructuredExtractor:
    """Factory function to create a structured extractor"""
    return StructuredExtractor(llm_client)
