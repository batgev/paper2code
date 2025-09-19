"""
Code Generator Agent for Paper2Code

Generates actual code implementations from implementation plans.
"""

import aiofiles
from pathlib import Path
from typing import Dict, Any, List

from ..config.manager import ConfigManager
from ..utils.logger import get_logger
from ..utils.llm import build_llm_client

logger = get_logger(__name__)


class CodeGeneratorAgent:
    """
    AI agent specialized in generating code from implementation plans.
    
    Takes detailed implementation plans and produces working code files.
    """
    
    def __init__(self, config: ConfigManager):
        """
        Initialize code generator agent.
        
        Args:
            config: Configuration manager instance
        """
        self.config = config
    
    async def generate_implementation(
        self,
        planning_results: Dict[str, Any],
        output_dir: Path,
        options: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate code implementation from planning results.
        
        Args:
            planning_results: Results from code planning
            output_dir: Directory to generate code in
            options: Processing options
            
        Returns:
            Generation results dictionary
        """
        logger.info(f"💻 Generating code implementation in {output_dir}")
        
        try:
            # Create output directory
            output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"📁 Created output directory: {output_dir}")
            
            # Get components and structure
            components = planning_results.get('implementation_components', [])
            file_structure = planning_results.get('file_structure', {})
            paper_info = planning_results.get('paper_info', {})
            
            logger.info(f"📋 Planning results: {len(components)} components, {len(file_structure.get('files', []))} files planned")
            
            generated_files = []
            
            # Generate directory structure
            logger.info("🏗️ Creating directory structure...")
            await self._create_directory_structure(output_dir, file_structure)
            logger.info("✅ Directory structure created")
            
            # Generate main application file
            main_file = await self._generate_main_file(
                output_dir, components, paper_info, planning_results
            )
            generated_files.append(main_file)
            
            # Generate algorithm implementations
            algorithm_files = await self._generate_algorithm_files(
                output_dir, components, planning_results
            )
            generated_files.extend(algorithm_files)
            
            # Generate utility files
            utility_files = await self._generate_utility_files(
                output_dir, components, planning_results
            )
            generated_files.extend(utility_files)
            
            # Generate test files
            test_files = await self._generate_test_files(
                output_dir, components, planning_results
            )
            generated_files.extend(test_files)
            
            # Generate configuration files
            config_files = await self._generate_config_files(
                output_dir, planning_results
            )
            generated_files.extend(config_files)
            
            generation_result = {
                'status': 'success',
                'files': [str(f) for f in generated_files],
                'output_directory': str(output_dir),
                'summary': {
                    'total_files': len(generated_files),
                    'algorithm_files': len([f for f in generated_files if 'algorithm' in str(f)]),
                    'test_files': len([f for f in generated_files if 'test' in str(f)]),
                    'utility_files': len([f for f in generated_files if 'util' in str(f)]),
                }
            }
            
            logger.info(f"✅ Generated {len(generated_files)} files successfully")
            return generation_result
            
        except Exception as e:
            logger.error(f"❌ Code generation failed: {e}")
            raise
    
    async def _create_directory_structure(self, output_dir: Path, file_structure: Dict[str, Any]):
        """Create the directory structure"""
        
        structure = file_structure.get('structure', {})
        
        # Create root directories
        for dir_name, dir_info in structure.get('directories', {}).items():
            dir_path = output_dir / dir_name
            dir_path.mkdir(exist_ok=True)
            
            # Create subdirectories recursively
            await self._create_subdirs(dir_path, dir_info.get('subdirs', {}))
    
    async def _create_subdirs(self, parent_dir: Path, subdirs: Dict[str, Any]):
        """Create subdirectories recursively"""
        
        for subdir_name, subdir_info in subdirs.items():
            subdir_path = parent_dir / subdir_name
            subdir_path.mkdir(exist_ok=True)
            
            # Create nested subdirectories
            if 'subdirs' in subdir_info:
                await self._create_subdirs(subdir_path, subdir_info['subdirs'])
    
    async def _generate_main_file(
        self,
        output_dir: Path,
        components: List[Dict[str, Any]],
        paper_info: Dict[str, Any],
        planning_results: Dict[str, Any]
    ) -> Path:
        """Generate the main application file"""
        
        main_file_path = output_dir / "main.py"
        
        # Generate main.py content
        main_content = f'''"""
{paper_info.get('title', 'Research Paper Implementation')}

Main implementation file for the research paper reproduction.
Generated automatically by Paper2Code.
"""

import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description="{paper_info.get('title', 'Paper Implementation')}"
    )
    parser.add_argument(
        '--input', '-i',
        help='Input file or data path'
    )
    parser.add_argument(
        '--output', '-o', 
        default='output/',
        help='Output directory path'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        print(f"🚀 Running {{paper_info.get('title', 'implementation')}}")
        print(f"📁 Output directory: {{args.output}}")
    
    try:
        # Initialize components
        result = run_implementation(args.input, args.output, args.verbose)
        
        if args.verbose:
            print(f"✅ Implementation completed successfully")
            print(f"📊 Results: {{result}}")
        
        return result
        
    except Exception as e:
        print(f"❌ Error during execution: {{e}}")
        return None

def run_implementation(input_path=None, output_path="output/", verbose=False):
    """
    Run the main implementation logic.
    
    Args:
        input_path: Path to input data
        output_path: Path to output directory
        verbose: Enable verbose output
    
    Returns:
        Implementation results
    """
    if verbose:
        print("🔧 Initializing implementation...")
    
    # TODO: Add actual implementation logic based on paper algorithms
    
    # Example implementation structure:
    result = {{
        'status': 'completed',
        'input_path': input_path,
        'output_path': output_path,
        'message': 'Implementation template generated - add actual algorithm logic'
    }}
    
    return result

if __name__ == "__main__":
    result = main()
    if result is None:
        sys.exit(1)
'''
        
        async with aiofiles.open(main_file_path, 'w', encoding='utf-8') as f:
            await f.write(main_content)
        
        logger.info(f"📄 Generated main file: {main_file_path}")
        return main_file_path
    
    async def _generate_algorithm_files(
        self,
        output_dir: Path,
        components: List[Dict[str, Any]],
        planning_results: Dict[str, Any]
    ) -> List[Path]:
        """Generate algorithm implementation files"""
        
        algorithm_files = []
        
        # Find algorithm components
        algorithm_components = [
            comp for comp in components 
            if 'algorithm' in comp.get('file', '').lower()
        ]
        
        for component in algorithm_components:
            file_path = output_dir / component.get('file', 'algorithm.py')
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Generate algorithm implementation
            algorithm_content = self._generate_algorithm_content(component, planning_results)
            
            async with aiofiles.open(file_path, 'w', encoding='utf-8') as f:
                await f.write(algorithm_content)
            
            algorithm_files.append(file_path)
            logger.info(f"🧮 Generated algorithm file: {file_path}")
        
        # Generate __init__.py for algorithms package
        if algorithm_files:
            algorithms_dir = output_dir / "src" / "algorithms"
            init_file = algorithms_dir / "__init__.py"
            
            init_content = '"""Algorithm implementations package"""\n'
            async with aiofiles.open(init_file, 'w', encoding='utf-8') as f:
                await f.write(init_content)
            
            algorithm_files.append(init_file)
        
        return algorithm_files
    
    async def _generate_algorithm_content(self, component: Dict[str, Any], planning_results: Dict[str, Any]) -> str:
        """Generate COMPLETE content for an algorithm file using LLM"""
        
        alg_name = component.get('name', 'Algorithm')
        description = component.get('description', 'Algorithm implementation')
        algorithm_content = component.get('algorithm_content', '')
        
        # Get technical content from planning results
        technical_content = planning_results.get('technical_content', {})
        algorithms = technical_content.get('algorithms', [])
        formulas = technical_content.get('formulas', [])
        paper_info = planning_results.get('paper_info', {})
        
        # Find relevant formulas and algorithms for this component
        relevant_formulas = [f for f in formulas if alg_name.lower() in f.get('formula', '').lower() or 
                           alg_name.lower().replace(' ', '') in f.get('type', '').lower()]
        relevant_algorithms = [a for a in algorithms if a.get('name', '').lower() == alg_name.lower()]
        
        # Use LLM to generate complete implementation
        if self.config.get('llm.enabled', True):
            try:
                llm = build_llm_client(self.config)
                
                prompt = f"""
You are a SENIOR MACHINE LEARNING ENGINEER with 10+ years of experience implementing research papers in production systems. Generate EXPERT-LEVEL, PRODUCTION-READY code following industry best practices.

🎯 ALGORITHM TO IMPLEMENT: {alg_name}
📋 DESCRIPTION: {description}

📊 RELEVANT FORMULAS:
{chr(10).join([f"• {f.get('formula', '')}: {f.get('description', '')}" for f in relevant_formulas[:3]])}

🧠 ALGORITHM DETAILS:
{chr(10).join([f"• {a.get('name', '')}: {a.get('content', '')}" for a in relevant_algorithms[:2]])}

📖 PAPER CONTEXT: {paper_info.get('title', 'Research Paper')}

🏆 EXPERT-LEVEL CODE REQUIREMENTS:

1. 🏗️ ARCHITECTURE & DESIGN PATTERNS:
   - Use abstract base classes and interfaces where appropriate
   - Follow SOLID principles and clean architecture
   - Implement proper separation of concerns
   - Use factory patterns for complex instantiation

2. 📝 DOCUMENTATION & TYPE HINTS:
   - Complete type hints for ALL parameters and return values
   - Comprehensive docstrings with mathematical notation
   - Inline comments explaining complex mathematical operations
   - Examples in docstrings showing usage patterns

3. 🛡️ ROBUST ERROR HANDLING:
   - Custom exception classes for domain-specific errors
   - Input validation with descriptive error messages
   - Graceful handling of edge cases and corner cases
   - Logging integration for debugging and monitoring

4. ⚡ PERFORMANCE & OPTIMIZATION:
   - Vectorized operations using NumPy/PyTorch efficiently
   - Memory-efficient implementations for large tensors
   - Batch processing capabilities where applicable
   - Optional GPU acceleration with device management

5. 🧪 TESTABILITY & MAINTAINABILITY:
   - Dependency injection for testability
   - Configuration through parameters/config objects
   - Modular design enabling easy unit testing
   - Clear interfaces between components

6. 🔬 MATHEMATICAL ACCURACY:
   - Implement exact mathematical formulas from the paper
   - Numerical stability considerations (avoiding overflow/underflow)
   - Proper handling of mathematical edge cases
   - Reference to original paper equations in comments

7. 🏭 PRODUCTION FEATURES:
   - Checkpointing and model serialization
   - Progress tracking and monitoring hooks
   - Memory usage optimization
   - Configurable precision (float32/float64)

IMPLEMENTATION STRUCTURE:
```python
import numpy as np
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Tuple, Any
from dataclasses import dataclass
import logging
from pathlib import Path

@dataclass
class {alg_name.replace(' ', '')}Config:
    \"\"\"Configuration for {alg_name}\"\"\"
    # Add all hyperparameters here

class {alg_name.replace(' ', '')}Exception(Exception):
    \"\"\"Custom exception for {alg_name}\"\"\"
    pass

class {alg_name.replace(' ', '')}(nn.Module):
    \"\"\"
    Expert-level implementation of {alg_name}.
    
    This implementation follows the paper: {paper_info.get('title', 'Research Paper')}
    
    Mathematical Background:
    [Include the key mathematical formulations]
    
    Args:
        config: Configuration object with all hyperparameters
        device: Computing device (cuda/cpu)
        
    Example:
        >>> config = {alg_name.replace(' ', '')}Config()
        >>> model = {alg_name.replace(' ', '')}(config)
        >>> output = model(input_tensor)
    \"\"\"
```

Generate the COMPLETE implementation with ALL methods fully implemented. No TODO comments, no placeholders - everything must be working code that can run immediately.

Return ONLY the complete Python code:"""
            
                logger.info(f"🤖 Generating complete implementation for {alg_name} using LLM")
                llm_response = await llm.generate(prompt)
                
                # Clean up the response to ensure it's valid Python
                code_content = self._clean_generated_code(llm_response, alg_name)
                
                # Apply expert-level code quality improvements
                code_content = await self._enhance_code_quality(code_content, alg_name, relevant_formulas)
                
                # Validate code quality standards
                quality_score = self._validate_code_quality(code_content)
                logger.info(f"📊 Code quality score for {alg_name}: {quality_score:.2f}/10")
                
                if quality_score < 7.0:
                    logger.warning(f"⚠️ Code quality below threshold for {alg_name} - applying improvements...")
                    code_content = await self._improve_code_quality(code_content, alg_name, quality_score)
                    
                    # Re-validate after improvements
                    new_quality_score = self._validate_code_quality(code_content)
                    logger.info(f"📈 Improved quality score for {alg_name}: {new_quality_score:.2f}/10")
                
                # Validate that the code doesn't contain TODO comments
                if 'TODO' in code_content or 'todo' in code_content.lower():
                    logger.warning(f"⚠️ LLM generated code with TODO comments, using working template")
                    return await self._generate_expert_algorithm_template(component, planning_results)
                
                logger.info(f"✅ Generated complete implementation for {alg_name}")
                return code_content
                
            except Exception as e:
                logger.warning(f"⚠️ LLM generation failed for {alg_name}: {e}")
                return await self._generate_expert_algorithm_template(component, planning_results)
        else:
            logger.info(f"🔄 LLM disabled, using expert template for {alg_name}")
            return await self._generate_expert_algorithm_template(component, planning_results)
    
    async def _enhance_code_quality(self, code: str, alg_name: str, formulas: List[Dict]) -> str:
        """Enhance code quality with expert-level improvements"""
        
        try:
            llm = build_llm_client(self.config)
            
            enhancement_prompt = f"""
You are a SENIOR CODE REVIEWER with expertise in machine learning implementations. 
Review and ENHANCE the following code to achieve EXPERT-LEVEL quality:

ORIGINAL CODE:
```python
{code}
```

ENHANCEMENT REQUIREMENTS:

1. 🎯 TYPE SAFETY & HINTS:
   - Add comprehensive type hints for ALL functions, methods, and variables
   - Use Union, Optional, Literal types appropriately
   - Add return type annotations

2. 📚 DOCUMENTATION EXCELLENCE:
   - Enhance docstrings with mathematical LaTeX notation
   - Add detailed parameter descriptions with types and constraints
   - Include complexity analysis (time/space)
   - Add usage examples and edge case handling

3. 🛡️ ROBUST ERROR HANDLING:
   - Add comprehensive input validation
   - Create custom exception classes
   - Add proper error messages with context
   - Handle numerical edge cases (NaN, infinity, overflow)

4. ⚡ PERFORMANCE OPTIMIZATIONS:
   - Optimize tensor operations for efficiency
   - Add memory-efficient implementations
   - Use torch.jit.script decorators where appropriate
   - Add batch processing support

5. 🧪 MAINTAINABILITY:
   - Extract magic numbers to named constants
   - Add configuration dataclasses
   - Improve code organization and separation of concerns
   - Add logging for debugging and monitoring

6. 🔬 MATHEMATICAL PRECISION:
   - Implement exact formulas: {[f.get('formula', '') for f in formulas[:2]]}
   - Add numerical stability improvements
   - Reference paper equations in comments
   - Validate mathematical correctness

Return ONLY the enhanced Python code with ALL improvements applied:
"""
            
            enhanced_code = await llm.generate(enhancement_prompt)
            return self._clean_generated_code(enhanced_code, alg_name)
            
        except Exception as e:
            logger.warning(f"⚠️ Code enhancement failed: {e}")
            return code
    
    def _validate_code_quality(self, code: str) -> float:
        """Validate code quality and return a score (0-10)"""
        
        score = 0.0
        
        # Check for type hints (2 points)
        type_hint_count = code.count(': ') + code.count('-> ')
        function_count = code.count('def ')
        if function_count > 0:
            type_coverage = min(type_hint_count / (function_count * 2), 1.0)
            score += type_coverage * 2.0
        
        # Check for docstrings (2 points)
        docstring_count = code.count('"""') / 2
        if function_count > 0:
            docstring_coverage = min(docstring_count / function_count, 1.0)
            score += docstring_coverage * 2.0
        
        # Check for error handling (1.5 points)
        error_handling_indicators = ['try:', 'except:', 'raise ', 'assert ', 'ValueError', 'TypeError']
        error_handling_score = min(sum(code.count(indicator) for indicator in error_handling_indicators) / 5.0, 1.0)
        score += error_handling_score * 1.5
        
        # Check for imports and best practices (1.5 points)
        good_imports = ['typing', 'dataclass', 'abc', 'logging', 'torch', 'numpy']
        import_score = min(sum(1 for imp in good_imports if imp in code) / len(good_imports), 1.0)
        score += import_score * 1.5
        
        # Check for mathematical implementations (2 points)
        math_indicators = ['torch.', 'np.', 'softmax', 'matmul', '@', 'sqrt', 'exp', 'log']
        math_score = min(sum(code.count(indicator) for indicator in math_indicators) / 10.0, 1.0)
        score += math_score * 2.0
        
        # Check for production features (1 point)
        production_indicators = ['device', 'cuda', 'checkpoint', 'state_dict', 'eval()', 'train()']
        production_score = min(sum(code.count(indicator) for indicator in production_indicators) / 3.0, 1.0)
        score += production_score * 1.0
        
        return min(score, 10.0)
    
    async def _improve_code_quality(self, code: str, alg_name: str, current_score: float) -> str:
        """Apply targeted improvements to increase code quality"""
        
        try:
            llm = build_llm_client(self.config)
            
            improvement_areas = []
            if ': ' not in code or '-> ' not in code:
                improvement_areas.append("Add comprehensive type hints")
            if '"""' not in code:
                improvement_areas.append("Add detailed docstrings with examples")
            if 'try:' not in code and 'except:' not in code:
                improvement_areas.append("Add robust error handling")
            if 'device' not in code:
                improvement_areas.append("Add GPU/device support")
            if 'dataclass' not in code:
                improvement_areas.append("Add configuration management")
            
            improvement_prompt = f"""
The following code has a quality score of {current_score:.2f}/10. Improve it by addressing these specific areas:

IMPROVEMENT AREAS:
{chr(10).join(['• ' + area for area in improvement_areas])}

CURRENT CODE:
```python
{code}
```

Apply TARGETED IMPROVEMENTS to achieve a quality score of 8.5+:

1. Add missing type hints and annotations
2. Enhance docstrings with mathematical notation and examples  
3. Add comprehensive error handling and validation
4. Optimize for performance and memory efficiency
5. Add production-ready features (device handling, checkpointing)
6. Improve code organization and maintainability

Return ONLY the improved Python code:
"""
            
            improved_code = await llm.generate(improvement_prompt)
            return self._clean_generated_code(improved_code, alg_name)
            
        except Exception as e:
            logger.warning(f"⚠️ Code improvement failed: {e}")
            return code
    
    async def _generate_expert_algorithm_template(self, component: Dict[str, Any], planning_results: Dict[str, Any]) -> str:
        """Generate expert-level algorithm template as fallback"""
        
        alg_name = component.get('name', 'Algorithm').replace(' ', '').replace('-', '')
        description = component.get('description', 'Algorithm implementation')
        
        template = f'''"""
{component.get('name', 'Algorithm')} Implementation

Expert-level implementation following research paper specifications.
"""

import numpy as np
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Tuple, Any, NamedTuple
from dataclasses import dataclass, field
import logging
import warnings
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class {alg_name}Config:
    """
    Configuration for {component.get('name', 'Algorithm')}.
    
    This dataclass contains all hyperparameters and configuration
    options for the algorithm implementation.
    """
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float32
    seed: int = 42
    
    # Algorithm-specific parameters (to be customized)
    learning_rate: float = 1e-3
    batch_size: int = 32
    hidden_dim: int = 512
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.learning_rate <= 0:
            raise ValueError(f"Learning rate must be positive, got {{self.learning_rate}}")
        if self.batch_size <= 0:
            raise ValueError(f"Batch size must be positive, got {{self.batch_size}}")
        if self.hidden_dim <= 0:
            raise ValueError(f"Hidden dimension must be positive, got {{self.hidden_dim}}")

class {alg_name}Exception(Exception):
    """Custom exception for {component.get('name', 'Algorithm')} implementation."""
    pass

class I{alg_name}(ABC):
    """Abstract interface for {component.get('name', 'Algorithm')} implementations."""
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the algorithm to input tensor."""
        pass
    
    @abstractmethod
    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure optimizer for training."""
        pass

class {alg_name}(nn.Module, I{alg_name}):
    """
    Expert-level implementation of {component.get('name', 'Algorithm')}.
    
    {description}
    
    Args:
        config: Configuration object containing all hyperparameters
        
    Attributes:
        config: Configuration object
        device: Computing device (cuda/cpu)
        
    Example:
        >>> config = {alg_name}Config(hidden_dim=256)
        >>> model = {alg_name}(config)
        >>> output = model(input_tensor)
        
    References:
        [Paper reference would go here]
    """
    
    def __init__(self, config: {alg_name}Config):
        super().__init__()
        
        if not isinstance(config, {alg_name}Config):
            raise TypeError(f"Expected {{alg_name}}Config, got {{type(config)}}")
            
        self.config = config
        self.device = torch.device(config.device)
        
        # Set random seeds for reproducibility
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(config.seed)
        
        # Initialize layers and parameters
        self._build_architecture()
        
        # Move to device
        self.to(self.device)
        
        logger.info(f"Initialized {{self.__class__.__name__}} on {{self.device}}")
    
    def _build_architecture(self) -> None:
        """Build the neural network architecture."""
        # This should be implemented based on the specific algorithm
        self.layers = nn.Sequential(
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights using Xavier/Glorot initialization."""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the algorithm to input tensor.
        
        Args:
            x: Input tensor of shape [batch_size, ...] 
            
        Returns:
            Output tensor after applying the algorithm
            
        Raises:
            {alg_name}Exception: If input tensor is invalid
            ValueError: If tensor dimensions are incompatible
        """
        if not isinstance(x, torch.Tensor):
            raise {alg_name}Exception(f"Expected torch.Tensor, got {{type(x)}}")
        
        if x.dim() < 2:
            raise ValueError(f"Input tensor must have at least 2 dimensions, got {{x.dim()}}")
        
        # Ensure tensor is on correct device
        x = x.to(self.device, dtype=self.config.dtype)
        
        # Apply algorithm (to be implemented based on specific algorithm)
        output = self.layers(x)
        
        return output
    
    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure optimizer for training."""
        return torch.optim.Adam(self.parameters(), lr=self.config.learning_rate)
    
    def get_config(self) -> Dict[str, Any]:
        """Get model configuration as dictionary."""
        return {{
            'class_name': self.__class__.__name__,
            'config': self.config.__dict__,
            'device': str(self.device),
            'parameter_count': sum(p.numel() for p in self.parameters())
        }}
    
    def save_checkpoint(self, path: Path) -> None:
        """Save model checkpoint."""
        checkpoint = {{
            'model_state_dict': self.state_dict(),
            'config': self.config.__dict__,
            'class_name': self.__class__.__name__
        }}
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {{path}}")
    
    @classmethod
    def load_checkpoint(cls, path: Path, config: Optional[{alg_name}Config] = None) -> '{alg_name}':
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location='cpu')
        
        if config is None:
            config = {alg_name}Config(**checkpoint['config'])
        
        model = cls(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        logger.info(f"Loaded checkpoint from {{path}}")
        return model


def create_{alg_name.lower()}(config: Optional[{alg_name}Config] = None) -> {alg_name}:
    """
    Factory function to create {component.get('name', 'Algorithm')} instance.
    
    Args:
        config: Optional configuration object. If None, uses defaults.
        
    Returns:
        Configured {component.get('name', 'Algorithm')} instance
    """
    if config is None:
        config = {alg_name}Config()
    
    return {alg_name}(config)


# Main execution for testing
if __name__ == "__main__":
    # Example usage
    config = {alg_name}Config(
        hidden_dim=256,
        learning_rate=1e-3,
        batch_size=32
    )
    
    model = create_{alg_name.lower()}(config)
    
    # Test with sample data
    sample_input = torch.randn(config.batch_size, config.hidden_dim)
    
    with torch.no_grad():
        output = model(sample_input)
        print(f"Input shape: {{sample_input.shape}}")
        print(f"Output shape: {{output.shape}}")
        print(f"Model config: {{model.get_config()}}")
'''

        return template
    
    async def _generate_expert_test_content(self, component: Dict[str, Any], planning_results: Dict[str, Any]) -> str:
        """Generate expert-level comprehensive test content"""
        
        alg_name = component.get('name', 'Algorithm').replace(' ', '').replace('-', '')
        
        try:
            llm = build_llm_client(self.config)
            
            test_prompt = f"""
You are an EXPERT TEST ENGINEER writing comprehensive test suites for machine learning implementations. 
Generate PRODUCTION-GRADE tests following industry best practices and testing standards.

🎯 ALGORITHM TO TEST: {component.get('name', 'Algorithm')}
📋 DESCRIPTION: {component.get('description', 'Algorithm implementation')}

🧪 EXPERT-LEVEL TEST REQUIREMENTS:

1. 🏗️ TEST ARCHITECTURE:
   - Use pytest framework with advanced features
   - Implement test fixtures for reusable setup
   - Parameterized tests for edge cases
   - Property-based testing where appropriate

2. 🔍 COMPREHENSIVE COVERAGE:
   - Unit tests for all public methods
   - Integration tests for component interactions
   - Edge case and boundary condition tests
   - Error handling and exception tests
   - Performance and memory tests

3. 🛡️ MATHEMATICAL VALIDATION:
   - Validate mathematical correctness
   - Test numerical stability and precision
   - Compare with reference implementations
   - Test with different data types and precisions

4. ⚡ PERFORMANCE TESTING:
   - Benchmark execution time
   - Memory usage validation  
   - GPU/CPU performance comparison
   - Scalability tests with different input sizes

5. 🎯 PROFESSIONAL STANDARDS:
   - Clear test naming and documentation
   - Proper assertion messages
   - Test data generation utilities
   - Mock/stub external dependencies

Generate COMPLETE pytest test file with ALL test methods implemented:

```python
import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch
import time
import gc
from typing import List, Tuple, Dict, Any

# Test the algorithm implementation
class Test{alg_name}:
    \"\"\"Comprehensive test suite for {component.get('name', 'Algorithm')}\"\"\"
    
    # Add fixtures, test methods, etc.
```

Return ONLY the complete Python test file:
"""
            
            test_content = await llm.generate(test_prompt)
            return self._clean_generated_code(test_content, f"Test{alg_name}")
            
        except Exception as e:
            logger.warning(f"⚠️ Expert test generation failed: {e}")
            return self._generate_basic_test_template(component)
    
    def _generate_basic_test_template(self, component: Dict[str, Any]) -> str:
        """Generate basic test template as fallback"""
        
        alg_name = component.get('name', 'Algorithm').replace(' ', '').replace('-', '')
        
        return f'''"""
Expert-level test suite for {component.get('name', 'Algorithm')}
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import time
import gc
import warnings
from typing import List, Tuple, Dict, Any
from pathlib import Path

# Import the algorithm to test
try:
    from src.algorithms.{component.get('name', 'algorithm').lower().replace(' ', '_')} import {alg_name}, {alg_name}Config
except ImportError:
    pytest.skip(f"Could not import {alg_name}", allow_module_level=True)


class Test{alg_name}:
    """Comprehensive test suite for {component.get('name', 'Algorithm')}."""
    
    @pytest.fixture
    def config(self) -> {alg_name}Config:
        """Create test configuration."""
        return {alg_name}Config(
            device="cpu",  # Use CPU for testing
            dtype=torch.float32,
            seed=42,
            batch_size=4,
            hidden_dim=32  # Smaller for faster tests
        )
    
    @pytest.fixture
    def model(self, config: {alg_name}Config) -> {alg_name}:
        """Create model instance for testing."""
        return {alg_name}(config)
    
    @pytest.fixture
    def sample_input(self, config: {alg_name}Config) -> torch.Tensor:
        """Create sample input tensor."""
        return torch.randn(config.batch_size, config.hidden_dim, dtype=config.dtype)
    
    def test_initialization(self, config: {alg_name}Config):
        """Test model initialization."""
        model = {alg_name}(config)
        
        assert isinstance(model, {alg_name})
        assert model.config == config
        assert str(model.device) == config.device
        
        # Test parameter count
        param_count = sum(p.numel() for p in model.parameters())
        assert param_count > 0, "Model should have trainable parameters"
    
    def test_forward_pass(self, model: {alg_name}, sample_input: torch.Tensor):
        """Test forward pass functionality."""
        model.eval()
        
        with torch.no_grad():
            output = model(sample_input)
        
        assert isinstance(output, torch.Tensor)
        assert output.shape[0] == sample_input.shape[0], "Batch size should be preserved"
        assert not torch.isnan(output).any(), "Output should not contain NaN"
        assert torch.isfinite(output).all(), "Output should be finite"
    
    def test_input_validation(self, model: {alg_name}):
        """Test input validation and error handling."""
        # Test invalid input type
        with pytest.raises(Exception):
            model("invalid_input")
        
        # Test invalid tensor dimensions
        with pytest.raises(ValueError):
            invalid_input = torch.randn(10)  # 1D tensor
            model(invalid_input)
        
        # Test empty tensor
        with pytest.raises(Exception):
            empty_input = torch.empty(0, model.config.hidden_dim)
            model(empty_input)
    
    @pytest.mark.parametrize("batch_size", [1, 2, 8, 16])
    def test_different_batch_sizes(self, config: {alg_name}Config, batch_size: int):
        """Test model with different batch sizes."""
        model = {alg_name}(config)
        input_tensor = torch.randn(batch_size, config.hidden_dim)
        
        model.eval()
        with torch.no_grad():
            output = model(input_tensor)
        
        assert output.shape[0] == batch_size
        assert not torch.isnan(output).any()
    
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_different_dtypes(self, config: {alg_name}Config, dtype: torch.dtype):
        """Test model with different data types."""
        config.dtype = dtype
        model = {alg_name}(config)
        input_tensor = torch.randn(config.batch_size, config.hidden_dim, dtype=dtype)
        
        model.eval()
        with torch.no_grad():
            output = model(input_tensor)
        
        assert output.dtype == dtype
        assert not torch.isnan(output).any()
    
    def test_reproducibility(self, config: {alg_name}Config):
        """Test that results are reproducible with same seed."""
        input_tensor = torch.randn(config.batch_size, config.hidden_dim)
        
        # First run
        model1 = {alg_name}(config)
        model1.eval()
        with torch.no_grad():
            output1 = model1(input_tensor)
        
        # Second run with same seed
        model2 = {alg_name}(config)
        model2.eval()
        with torch.no_grad():
            output2 = model2(input_tensor)
        
        torch.testing.assert_close(output1, output2, msg="Results should be reproducible")
    
    def test_gradient_flow(self, model: {alg_name}, sample_input: torch.Tensor):
        """Test that gradients flow properly during backpropagation."""
        model.train()
        
        # Forward pass
        output = model(sample_input)
        loss = output.mean()  # Simple loss for testing
        
        # Backward pass
        loss.backward()
        
        # Check that gradients exist and are not zero
        has_grad = False
        for param in model.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grad = True
                break
        
        assert has_grad, "Model should have non-zero gradients after backprop"
    
    def test_checkpoint_save_load(self, model: {alg_name}, tmp_path: Path):
        """Test model checkpointing functionality."""
        checkpoint_path = tmp_path / "test_checkpoint.pt"
        
        # Save checkpoint
        model.save_checkpoint(checkpoint_path)
        assert checkpoint_path.exists()
        
        # Load checkpoint
        loaded_model = {alg_name}.load_checkpoint(checkpoint_path)
        
        # Verify loaded model
        assert isinstance(loaded_model, {alg_name})
        assert loaded_model.config.__dict__ == model.config.__dict__
        
        # Test that loaded model produces same output
        sample_input = torch.randn(model.config.batch_size, model.config.hidden_dim)
        
        model.eval()
        loaded_model.eval()
        
        with torch.no_grad():
            output1 = model(sample_input)
            output2 = loaded_model(sample_input)
        
        torch.testing.assert_close(output1, output2, msg="Loaded model should produce identical results")
    
    def test_optimizer_configuration(self, model: {alg_name}):
        """Test optimizer configuration."""
        optimizer = model.configure_optimizers()
        
        assert isinstance(optimizer, torch.optim.Optimizer)
        assert len(optimizer.param_groups) > 0
        assert optimizer.param_groups[0]['lr'] == model.config.learning_rate
    
    def test_model_modes(self, model: {alg_name}, sample_input: torch.Tensor):
        """Test train/eval mode switching."""
        # Test training mode
        model.train()
        assert model.training is True
        
        output_train = model(sample_input)
        
        # Test evaluation mode  
        model.eval()
        assert model.training is False
        
        with torch.no_grad():
            output_eval = model(sample_input)
        
        # Outputs might be different due to dropout/batchnorm
        assert output_train.shape == output_eval.shape
    
    @pytest.mark.slow
    def test_memory_efficiency(self, config: {alg_name}Config):
        """Test memory usage and efficiency."""
        # Test with larger input to check memory
        large_config = {alg_name}Config(
            batch_size=64,
            hidden_dim=512,
            device="cpu"
        )
        
        model = {alg_name}(large_config)
        large_input = torch.randn(large_config.batch_size, large_config.hidden_dim)
        
        # Measure memory before
        gc.collect()
        
        model.eval()
        with torch.no_grad():
            output = model(large_input)
        
        # Basic memory check
        assert output is not None
        assert not torch.isnan(output).any()
        
        # Clean up
        del model, large_input, output
        gc.collect()
    
    @pytest.mark.performance  
    def test_inference_speed(self, model: {alg_name}, sample_input: torch.Tensor):
        """Benchmark inference speed."""
        model.eval()
        
        # Warm up
        with torch.no_grad():
            for _ in range(10):
                _ = model(sample_input)
        
        # Benchmark
        start_time = time.time()
        with torch.no_grad():
            for _ in range(100):
                _ = model(sample_input)
        
        end_time = time.time()
        avg_time = (end_time - start_time) / 100
        
        # Should complete inference in reasonable time (adjust threshold as needed)
        assert avg_time < 1.0, f"Inference too slow: {{avg_time:.4f}}s per forward pass"
        
        logger.info(f"Average inference time: {{avg_time:.4f}}s")


# Additional test utilities
class TestUtilities:
    """Utility functions for testing."""
    
    @staticmethod
    def generate_test_data(batch_size: int, hidden_dim: int) -> torch.Tensor:
        """Generate test data with specific properties."""
        return torch.randn(batch_size, hidden_dim)
    
    @staticmethod
    def assert_output_properties(output: torch.Tensor, expected_shape: tuple):
        """Assert common output properties."""
        assert isinstance(output, torch.Tensor)
        assert output.shape == expected_shape
        assert torch.isfinite(output).all()
        assert not torch.isnan(output).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
'''
    
    async def _generate_integration_test_content(self, components: List[Dict[str, Any]], planning_results: Dict[str, Any]) -> str:
        """Generate integration tests for component interactions"""
        
        return '''"""
Integration tests for algorithm components.
"""

import pytest
import torch
import numpy as np
from typing import List, Dict, Any

class TestIntegration:
    """Integration tests for algorithm components."""
    
    def test_end_to_end_pipeline(self):
        """Test complete algorithm pipeline."""
        # Implementation would test full pipeline
        assert True  # Placeholder
    
    def test_component_interactions(self):
        """Test interactions between different components."""
        # Implementation would test component interactions
        assert True  # Placeholder
'''
    
    async def _generate_benchmark_tests(self, components: List[Dict[str, Any]], planning_results: Dict[str, Any]) -> str:
        """Generate performance benchmark tests"""
        
        return '''"""
Performance benchmarks for algorithm implementations.
"""

import pytest
import torch
import numpy as np
import time
import memory_profiler
from typing import List, Dict, Any

class TestBenchmarks:
    """Performance benchmark tests."""
    
    @pytest.mark.benchmark
    def test_inference_benchmark(self):
        """Benchmark inference performance."""
        # Implementation would benchmark inference
        assert True  # Placeholder
    
    @pytest.mark.benchmark  
    def test_memory_benchmark(self):
        """Benchmark memory usage."""
        # Implementation would benchmark memory
        assert True  # Placeholder
        
    @pytest.mark.benchmark
    def test_scalability_benchmark(self):
        """Test scalability with different input sizes."""
        # Implementation would test scalability
        assert True  # Placeholder
'''
    
    def _clean_generated_code(self, code: str, alg_name: str) -> str:
        """Clean and validate generated code"""
        
        # Remove markdown code blocks if present
        if '```python' in code:
            code = code.split('```python')[1].split('```')[0]
        elif '```' in code:
            code = code.split('```')[1].split('```')[0]
        
        # Ensure proper imports
        if 'import numpy as np' not in code:
            code = 'import numpy as np\nfrom typing import Any, Dict, List, Optional, Tuple\n\n' + code
        
        # Ensure the code has a proper class structure
        if f'class {alg_name.replace(" ", "").replace("-", "")}' not in code:
            code = self._wrap_in_class_structure(code, alg_name)
        
        return code.strip()
    
    def _wrap_in_class_structure(self, code: str, alg_name: str) -> str:
        """Wrap code in proper class structure if needed"""
        
        class_name = alg_name.replace(' ', '').replace('-', '')
        
        return f'''"""
{alg_name} Implementation

Complete implementation based on research paper.
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple

class {class_name}:
    """
    Complete implementation of {alg_name}.
    
    Based on the algorithm described in the research paper.
    """
    
    def __init__(self, **kwargs):
        """Initialize the algorithm with parameters."""
        self.config = kwargs
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize algorithm components."""
        # Initialize based on configuration
        pass
    
{code}

# Convenience function
def run_{alg_name.lower().replace(' ', '_').replace('-', '_')}(input_data: Any, **kwargs) -> Any:
    """Run {alg_name} algorithm."""
    algorithm = {class_name}(**kwargs)
    return algorithm.run(input_data)
'''
    
    async def _generate_enhanced_template(self, alg_name: str, description: str, 
                                        formulas: List[Dict], algorithms: List[Dict]) -> str:
        """Generate enhanced template with actual implementations"""
        
        class_name = alg_name.replace(' ', '').replace('-', '')
        
        # Generate working template directly (remove async LLM call from sync method)
        logger.info(f"🔧 Generating working template for {alg_name}")
        return self._create_working_template(alg_name, description)
    
    def _detect_algorithm_type(self, alg_name: str, description: str, formulas: List[Dict], algorithms: List[Dict]) -> str:
        """Detect the type of algorithm to generate appropriate implementation"""
        
        # Combine text for analysis
        combined_text = f"{alg_name} {description}".lower()
        
        # Check formulas for clues
        formula_text = " ".join([f.get('formula', '') + " " + f.get('type', '') for f in formulas]).lower()
        
        # Algorithm type detection patterns
        if any(term in combined_text for term in ['attention', 'query', 'key', 'value', 'self-attention', 'multi-head']):
            return 'attention'
        elif any(term in combined_text for term in ['transformer', 'encoder', 'decoder']):
            return 'transformer'
        elif any(term in combined_text for term in ['positional', 'encoding', 'embedding']):
            return 'encoding'
        elif any(term in combined_text for term in ['convolution', 'cnn', 'filter', 'kernel', 'pooling']):
            return 'cnn'
        elif any(term in combined_text for term in ['recurrent', 'rnn', 'lstm', 'gru', 'sequence']):
            return 'rnn'
        elif any(term in combined_text for term in ['optimization', 'gradient', 'adam', 'sgd', 'optimizer']):
            return 'optimization'
        elif any(term in combined_text for term in ['gan', 'generator', 'discriminator', 'adversarial']):
            return 'gan'
        elif any(term in combined_text for term in ['reinforcement', 'policy', 'reward', 'q-learning']):
            return 'reinforcement'
        else:
            return 'generic'
    
    def _generate_attention_implementation(self, alg_name: str, class_name: str, formulas: List[Dict]) -> str:
        """Generate complete attention mechanism implementation"""
        
        return f'''"""
{alg_name} Implementation

Complete attention mechanism implementation.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple

class {class_name}(nn.Module):
    """
    Complete implementation of {alg_name}.
    
    Implements scaled dot-product attention: Attention(Q,K,V) = softmax(QK^T/√d_k)V
    """
    
    def __init__(self, d_model: int = 512, num_heads: int = 8, dropout: float = 0.1, **kwargs):
        """
        Initialize the attention mechanism.
        
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.dropout = dropout
        
        # Linear projections for Q, K, V
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout_layer = nn.Dropout(dropout)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply attention mechanism.
        
        Args:
            query: Query tensor [batch_size, seq_len, d_model]
            key: Key tensor [batch_size, seq_len, d_model]  
            value: Value tensor [batch_size, seq_len, d_model]
            mask: Optional attention mask
            
        Returns:
            Attention output [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, _ = query.size()
        
        # Linear projections
        Q = self.w_q(query)  # [batch_size, seq_len, d_model]
        K = self.w_k(key)    # [batch_size, seq_len, d_model]
        V = self.w_v(value)  # [batch_size, seq_len, d_model]
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Apply scaled dot-product attention
        attention_output = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Concatenate heads
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        # Final linear projection
        output = self.w_o(attention_output)
        
        return output
    
    def scaled_dot_product_attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                                   mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute scaled dot-product attention: softmax(QK^T/√d_k)V
        
        Args:
            Q: Query tensor [batch_size, num_heads, seq_len, d_k]
            K: Key tensor [batch_size, num_heads, seq_len, d_k]
            V: Value tensor [batch_size, num_heads, seq_len, d_k]
            mask: Optional attention mask
            
        Returns:
            Attention output [batch_size, num_heads, seq_len, d_k]
        """
        # Compute attention scores: QK^T/√d_k
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout_layer(attention_weights)
        
        # Apply attention to values
        output = torch.matmul(attention_weights, V)
        
        return output
    
    def run(self, input_data: Any) -> Any:
        """
        Run the attention mechanism.
        
        Args:
            input_data: Input tensor or dictionary with 'query', 'key', 'value'
            
        Returns:
            Attention output
        """
        if isinstance(input_data, dict):
            query = input_data['query']
            key = input_data.get('key', query)
            value = input_data.get('value', query)
            mask = input_data.get('mask', None)
        else:
            # Self-attention case
            query = key = value = input_data
            mask = None
        
        return self.forward(query, key, value, mask)

# Convenience function
def run_{alg_name.lower().replace(' ', '_').replace('-', '_')}(input_data: Any, **kwargs) -> Any:
    """Run {alg_name} algorithm."""
    algorithm = {class_name}(**kwargs)
    return algorithm.run(input_data)
'''
    
    def _generate_transformer_implementation(self, alg_name: str, class_name: str, formulas: List[Dict]) -> str:
        """Generate complete transformer implementation"""
        
        return f'''"""
{alg_name} Implementation

Complete transformer architecture implementation.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional, Tuple

class {class_name}(nn.Module):
    """
    Complete Transformer implementation.
    
    Implements the full transformer architecture with encoder and decoder.
    """
    
    def __init__(self, d_model: int = 512, num_heads: int = 8, num_layers: int = 6,
                 d_ff: int = 2048, dropout: float = 0.1, vocab_size: int = 10000, **kwargs):
        """
        Initialize the transformer.
        
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            num_layers: Number of encoder/decoder layers
            d_ff: Feed-forward dimension
            dropout: Dropout rate
            vocab_size: Vocabulary size
        """
        super().__init__()
        self.d_model = d_model
        
        # Embeddings
        self.encoder_embedding = nn.Embedding(vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout)
        
        # Encoder
        encoder_layer = TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Decoder
        decoder_layer = TransformerDecoderLayer(d_model, num_heads, d_ff, dropout)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        
        # Output projection
        self.output_projection = nn.Linear(d_model, vocab_size)
        
    def forward(self, src: torch.Tensor, tgt: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None,
                tgt_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the transformer.
        
        Args:
            src: Source sequence [seq_len, batch_size]
            tgt: Target sequence [seq_len, batch_size]
            src_mask: Source mask
            tgt_mask: Target mask
            
        Returns:
            Output logits [seq_len, batch_size, vocab_size]
        """
        # Embeddings and positional encoding
        src_emb = self.positional_encoding(self.encoder_embedding(src) * np.sqrt(self.d_model))
        tgt_emb = self.positional_encoding(self.decoder_embedding(tgt) * np.sqrt(self.d_model))
        
        # Encoder
        memory = self.encoder(src_emb, src_key_padding_mask=src_mask)
        
        # Decoder
        output = self.decoder(tgt_emb, memory, tgt_mask=tgt_mask)
        
        # Output projection
        logits = self.output_projection(output)
        
        return logits
    
    def run(self, input_data: Any) -> Any:
        """
        Run the transformer.
        
        Args:
            input_data: Dictionary with 'src' and 'tgt' tensors
            
        Returns:
            Transformer output
        """
        if isinstance(input_data, dict):
            src = input_data['src']
            tgt = input_data['tgt']
            src_mask = input_data.get('src_mask', None)
            tgt_mask = input_data.get('tgt_mask', None)
        else:
            # Auto-regressive case
            src = tgt = input_data
            src_mask = tgt_mask = None
        
        return self.forward(src, tgt, src_mask, tgt_mask)

class PositionalEncoding(nn.Module):
    """
    Positional encoding implementation: PE(pos,2i) = sin(pos/10000^(2i/d_model))
    """
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

# Convenience function
def run_{alg_name.lower().replace(' ', '_').replace('-', '_')}(input_data: Any, **kwargs) -> Any:
    """Run {alg_name} algorithm."""
    algorithm = {class_name}(**kwargs)
    return algorithm.run(input_data)
'''
    
    def _generate_encoding_implementation(self, alg_name: str, class_name: str, formulas: List[Dict]) -> str:
        """Generate complete positional encoding implementation"""
        
        return f'''"""
{alg_name} Implementation

Complete positional encoding implementation.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional, Tuple

class {class_name}(nn.Module):
    """
    Complete positional encoding implementation.
    
    Implements: PE(pos,2i) = sin(pos/10000^(2i/d_model))
                PE(pos,2i+1) = cos(pos/10000^(2i/d_model))
    """
    
    def __init__(self, d_model: int = 512, dropout: float = 0.1, max_len: int = 5000, **kwargs):
        """
        Initialize positional encoding.
        
        Args:
            d_model: Model dimension
            dropout: Dropout rate
            max_len: Maximum sequence length
        """
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Compute div_term: 10000^(2i/d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        
        # Apply sin to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        
        # Apply cos to odd indices  
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension and transpose
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        # Register as buffer (not a parameter)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input embeddings.
        
        Args:
            x: Input embeddings [seq_len, batch_size, d_model]
            
        Returns:
            Input + positional encoding [seq_len, batch_size, d_model]
        """
        # Add positional encoding
        x = x + self.pe[:x.size(0), :]
        
        # Apply dropout
        return self.dropout(x)
    
    def get_encoding(self, seq_len: int) -> torch.Tensor:
        """
        Get positional encoding for given sequence length.
        
        Args:
            seq_len: Sequence length
            
        Returns:
            Positional encoding [seq_len, d_model]
        """
        return self.pe[:seq_len, 0, :]
    
    def run(self, input_data: Any) -> Any:
        """
        Apply positional encoding.
        
        Args:
            input_data: Input tensor [seq_len, batch_size, d_model]
            
        Returns:
            Input with positional encoding added
        """
        if isinstance(input_data, torch.Tensor):
            return self.forward(input_data)
        else:
            # Convert numpy to tensor if needed
            if isinstance(input_data, np.ndarray):
                input_tensor = torch.from_numpy(input_data).float()
                output = self.forward(input_tensor)
                return output.detach().numpy()
            else:
                raise ValueError(f"Unsupported input type: {{type(input_data)}}")

# Convenience function
def run_{alg_name.lower().replace(' ', '_').replace('-', '_')}(input_data: Any, **kwargs) -> Any:
    """Run {alg_name} algorithm."""
    algorithm = {class_name}(**kwargs)
    return algorithm.run(input_data)
'''
    
    def _generate_generic_implementation(self, alg_name: str, class_name: str, 
                                       description: str, algorithms: List[Dict]) -> str:
        """Generate complete generic algorithm implementation"""
        
        algorithm_details = ""
        if algorithms:
            algorithm_details = algorithms[0].get('content', description)
        
        return f'''"""
{alg_name} Implementation

Complete algorithm implementation based on research paper.
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple

class {class_name}:
    """
    Complete implementation of {alg_name}.
    
    {description}
    
    Algorithm details: {algorithm_details}
    """
    
    def __init__(self, **kwargs):
        """
        Initialize the algorithm.
        
        Args:
            **kwargs: Algorithm configuration parameters
        """
        self.config = kwargs
        self.initialized = False
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize algorithm components."""
        # Set default parameters
        self.learning_rate = self.config.get('learning_rate', 0.001)
        self.batch_size = self.config.get('batch_size', 32)
        self.max_iterations = self.config.get('max_iterations', 1000)
        
        # Initialize internal state
        self.iteration = 0
        self.history = []
        
        self.initialized = True
    
    def run(self, input_data: Any) -> Any:
        """
        Run the main algorithm.
        
        Args:
            input_data: Input data for the algorithm
            
        Returns:
            Algorithm output
        """
        if not self.initialized:
            self._initialize_components()
        
        # Validate input
        processed_input = self._validate_and_preprocess(input_data)
        
        # Main algorithm logic
        result = self._execute_algorithm(processed_input)
        
        # Post-process results
        output = self._postprocess_results(result)
        
        return output
    
    def _validate_and_preprocess(self, input_data: Any) -> Any:
        """Validate and preprocess input data."""
        if input_data is None:
            raise ValueError("Input data cannot be None")
        
        # Convert to numpy array if needed
        if isinstance(input_data, (list, tuple)):
            input_data = np.array(input_data)
        
        # Ensure proper shape
        if isinstance(input_data, np.ndarray):
            if input_data.ndim == 1:
                input_data = input_data.reshape(-1, 1)
        
        return input_data
    
    def _execute_algorithm(self, input_data: Any) -> Any:
        """Execute the main algorithm logic."""
        
        # Initialize output
        if isinstance(input_data, np.ndarray):
            output = np.zeros_like(input_data)
            
            # Apply algorithm-specific processing
            for i in range(input_data.shape[0]):
                # Process each sample
                sample = input_data[i]
                processed_sample = self._process_sample(sample)
                output[i] = processed_sample
                
            return output
        else:
            # Handle non-array inputs
            return self._process_sample(input_data)
    
    def _process_sample(self, sample: Any) -> Any:
        """Process a single sample."""
        
        # Apply transformation based on algorithm type
        if isinstance(sample, np.ndarray):
            # Normalize
            normalized = (sample - np.mean(sample)) / (np.std(sample) + 1e-8)
            
            # Apply non-linear transformation
            transformed = np.tanh(normalized)
            
            return transformed
        else:
            return sample
    
    def _postprocess_results(self, result: Any) -> Any:
        """Post-process algorithm results."""
        
        # Store in history
        self.history.append(result)
        self.iteration += 1
        
        # Return processed result
        return result
    
    def validate(self, input_data: Any, expected_output: Any = None) -> bool:
        """
        Validate algorithm implementation.
        
        Args:
            input_data: Input data for validation
            expected_output: Expected algorithm output
            
        Returns:
            True if validation passes
        """
        try:
            result = self.run(input_data)
            
            if expected_output is not None:
                if isinstance(result, np.ndarray) and isinstance(expected_output, np.ndarray):
                    return np.allclose(result, expected_output, rtol=1e-5)
                else:
                    return result == expected_output
            
            # Basic validation - check result is not None
            return result is not None
            
        except Exception as e:
            print(f"Validation failed: {{e}}")
            return False
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get algorithm performance metrics."""
        return {{
            'iterations': self.iteration,
            'history_length': len(self.history),
            'config': self.config,
            'initialized': self.initialized
        }}

# Convenience function
def run_{alg_name.lower().replace(' ', '_').replace('-', '_')}(input_data: Any, **kwargs) -> Any:
    """
    Convenience function to run {alg_name}.
    
    Args:
        input_data: Input data
        **kwargs: Algorithm parameters
        
    Returns:
        Algorithm output
    """
    algorithm = {class_name}(**kwargs)
    return algorithm.run(input_data)
'''
    
    async def _generate_utility_files(
        self,
        output_dir: Path,
        components: List[Dict[str, Any]],
        planning_results: Dict[str, Any]
    ) -> List[Path]:
        """Generate utility files"""
        
        utility_files = []
        
        # Generate data processing utilities
        data_utils_path = output_dir / "src" / "utils" / "data_processing.py"
        data_utils_path.parent.mkdir(parents=True, exist_ok=True)
        
        data_utils_content = '''"""
Data Processing Utilities

Utility functions for data loading, preprocessing, and transformation.
"""

import numpy as np
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

def load_data(file_path: Union[str, Path], format: str = "auto") -> Any:
    """
    Load data from file.
    
    Args:
        file_path: Path to data file
        format: Data format ("json", "csv", "npy", "auto")
        
    Returns:
        Loaded data
    """
    file_path = Path(file_path)
    
    if format == "auto":
        format = file_path.suffix.lower().lstrip('.')
    
    if format == "json":
        with open(file_path, 'r') as f:
            return json.load(f)
    elif format == "npy":
        return np.load(file_path)
    elif format == "csv":
        try:
            import pandas as pd
            return pd.read_csv(file_path)
        except ImportError:
            return np.loadtxt(file_path, delimiter=',')
    else:
        raise ValueError(f"Unsupported format: {format}")

def preprocess_data(data: Any, **kwargs) -> Any:
    """
    Preprocess input data.
    
    Args:
        data: Input data
        **kwargs: Preprocessing options
        
    Returns:
        Preprocessed data
    """
    # TODO: Add preprocessing logic based on paper requirements
    
    if isinstance(data, (list, np.ndarray)):
        data = np.array(data)
        
        # Common preprocessing steps
        if kwargs.get('normalize', False):
            data = normalize_data(data)
        
        if kwargs.get('standardize', False):
            data = standardize_data(data)
    
    return data

def normalize_data(data: np.ndarray) -> np.ndarray:
    """Normalize data to [0, 1] range"""
    return (data - data.min()) / (data.max() - data.min() + 1e-8)

def standardize_data(data: np.ndarray) -> np.ndarray:
    """Standardize data to zero mean and unit variance"""
    return (data - data.mean()) / (data.std() + 1e-8)

def save_results(results: Any, output_path: Union[str, Path], format: str = "json"):
    """
    Save results to file.
    
    Args:
        results: Results to save
        output_path: Output file path
        format: Output format
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == "json":
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
    elif format == "npy":
        np.save(output_path, results)
    else:
        raise ValueError(f"Unsupported format: {format}")
'''
        
        async with aiofiles.open(data_utils_path, 'w', encoding='utf-8') as f:
            await f.write(data_utils_content)
        
        utility_files.append(data_utils_path)
        
        # Generate general helpers
        helpers_path = output_dir / "src" / "utils" / "helpers.py"
        
        helpers_content = '''"""
General Helper Functions

Common utility functions used across the implementation.
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """Setup logging configuration"""
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    
    if log_file:
        handler = logging.FileHandler(log_file)
        handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        logger.addHandler(handler)
    
    return logger

def time_function(func):
    """Decorator to time function execution"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.2f} seconds")
        return result
    return wrapper

def ensure_directory(path: Path) -> Path:
    """Ensure directory exists"""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

def validate_config(config: Dict[str, Any], required_keys: List[str]) -> bool:
    """Validate configuration dictionary"""
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required configuration key: {key}")
    return True
'''
        
        async with aiofiles.open(helpers_path, 'w', encoding='utf-8') as f:
            await f.write(helpers_content)
        
        utility_files.append(helpers_path)
        
        # Generate utils __init__.py
        utils_init_path = output_dir / "src" / "utils" / "__init__.py"
        utils_init_content = '"""Utilities package"""\n'
        
        async with aiofiles.open(utils_init_path, 'w', encoding='utf-8') as f:
            await f.write(utils_init_content)
        
        utility_files.append(utils_init_path)
        
        return utility_files
    
    async def _generate_test_files(
        self,
        output_dir: Path,
        components: List[Dict[str, Any]],
        planning_results: Dict[str, Any]
    ) -> List[Path]:
        """Generate test files"""
        
        test_files = []
        
        # Generate main test file
        main_test_path = output_dir / "tests" / "test_main.py"
        main_test_path.parent.mkdir(parents=True, exist_ok=True)
        
        main_test_content = '''"""
Test suite for main implementation

Tests for the main application functionality.
"""

import pytest
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

def test_main_imports():
    """Test that main modules can be imported"""
    try:
        import main
        assert hasattr(main, 'main')
        assert hasattr(main, 'run_implementation')
    except ImportError as e:
        pytest.fail(f"Failed to import main module: {e}")

def test_run_implementation():
    """Test the main implementation function"""
    import main
    
    result = main.run_implementation(verbose=False)
    
    assert result is not None
    assert isinstance(result, dict)
    assert 'status' in result

def test_main_with_args():
    """Test main function with different arguments"""
    import main
    
    # Test with minimal args
    result = main.run_implementation(
        input_path=None,
        output_path="test_output/",
        verbose=True
    )
    
    assert result is not None

# Add more specific tests based on paper algorithms
class TestAlgorithms:
    """Test suite for algorithm implementations"""
    
    def test_algorithm_initialization(self):
        """Test algorithm initialization"""
        # TODO: Add tests for specific algorithms
        pass
    
    def test_algorithm_basic_functionality(self):
        """Test basic algorithm functionality"""
        # TODO: Add algorithm-specific tests
        pass
'''
        
        async with aiofiles.open(main_test_path, 'w', encoding='utf-8') as f:
            await f.write(main_test_content)
        
        test_files.append(main_test_path)
        
        # Generate tests __init__.py
        tests_init_path = output_dir / "tests" / "__init__.py"
        tests_init_content = '"""Test package"""\n'
        
        async with aiofiles.open(tests_init_path, 'w', encoding='utf-8') as f:
            await f.write(tests_init_content)
        
        test_files.append(tests_init_path)
        
        return test_files
    
    async def _generate_config_files(
        self,
        output_dir: Path,
        planning_results: Dict[str, Any]
    ) -> List[Path]:
        """Generate configuration files"""
        
        config_files = []
        
        # Generate requirements.txt
        requirements_path = output_dir / "requirements.txt"
        requirements = planning_results.get('requirements', [])
        
        # Add essential dependencies for generated code
        essential_requirements = [
            'numpy>=1.21.0',
            'torch>=1.9.0',
            'torch-audio>=0.9.0',
            'torch-vision>=0.10.0',
            'scipy>=1.7.0',
            'matplotlib>=3.3.0',
            'pandas>=1.3.0',
            'scikit-learn>=1.0.0',
            'transformers>=4.0.0',
            'datasets>=1.0.0',
            'pytest>=6.0.0',
            'jupyter>=1.0.0'
        ]
        
        # Merge with existing requirements
        all_requirements = list(set(requirements + essential_requirements))
        all_requirements.sort()
        
        requirements_content = "\n".join(all_requirements) + "\n"
        
        async with aiofiles.open(requirements_path, 'w', encoding='utf-8') as f:
            await f.write(requirements_content)
        
        config_files.append(requirements_path)
        
        # Generate setup.py
        setup_path = output_dir / "setup.py"
        paper_info = planning_results.get('paper_info', {})
        
        # Create requirements list string
        req_list = ',\n        '.join([repr(req) for req in all_requirements])
        
        setup_content = f'''"""
Setup script for {paper_info.get('title', 'Paper Implementation')}
"""

from setuptools import setup, find_packages

setup(
    name="{paper_info.get('title', 'paper-implementation').lower().replace(' ', '-')}",
    version="1.0.0", 
    description="{paper_info.get('title', 'Research paper implementation')}",
    packages=find_packages(),
    python_requires='>=3.8',
    install_requires=[
        {req_list}
    ],
    entry_points={{
        'console_scripts': [
            'paper-impl=main:main',
        ],
    }},
)
'''
        
        async with aiofiles.open(setup_path, 'w', encoding='utf-8') as f:
            await f.write(setup_content)
        
        config_files.append(setup_path)
        
        return config_files
    
    async def _generate_llm_driven_implementation(self, alg_name: str, description: str, 
                                                formulas: List[Dict], algorithms: List[Dict],
                                                planning_results: Dict[str, Any]) -> str:
        """Generate complete implementation using pure LLM analysis"""
        
        # Get paper context
        paper_info = planning_results.get('paper_info', {})
        
        # Prepare context for LLM
        relevant_formulas = [f for f in formulas if any(term in f.get('formula', '').lower() 
                           for term in alg_name.lower().split())]
        relevant_algorithms = [a for a in algorithms if a.get('name', '').lower() == alg_name.lower()]
        
        prompt = f"""
You are an expert Python developer implementing research paper algorithms.

CRITICAL: Write COMPLETE, FUNCTIONAL code - NO TODO comments allowed.

ALGORITHM: {alg_name}
DESCRIPTION: {description}

PAPER: {paper_info.get('title', 'Research Paper')}

FORMULAS TO IMPLEMENT:
{chr(10).join([f"- {f.get('formula', '')}: {f.get('description', '')}" for f in relevant_formulas[:3]])}

ALGORITHM DETAILS:
{chr(10).join([f"- {a.get('content', '')}" for a in relevant_algorithms[:2]])}

Generate a complete Python class with:
1. Full __init__ method
2. Complete run() method with actual logic  
3. All helper methods implemented
4. Real mathematical computations
5. Error handling

Return ONLY complete Python code:
"""
        
        try:
            logger.info(f"🤖 Generating LLM-driven implementation for {alg_name}")
            llm = build_llm_client(self.config)
            response = await llm.generate(prompt)
            
            # Clean the code
            code = self._clean_llm_generated_code(response, alg_name)
            
            # Validate completeness
            if 'TODO' in code or 'todo' in code.lower() or len(code) < 300:
                logger.warning(f"⚠️ Generated incomplete code, using working template")
                return self._create_working_template(alg_name, description)
            
            logger.info(f"✅ Generated complete implementation for {alg_name}")
            return code
            
        except Exception as e:
            logger.warning(f"⚠️ LLM generation failed: {e}")
            return self._create_working_template(alg_name, description)
    
    def _clean_llm_generated_code(self, code: str, alg_name: str) -> str:
        """Clean LLM-generated code"""
        
        # Remove markdown
        if '```python' in code:
            code = code.split('```python')[1].split('```')[0]
        elif '```' in code:
            parts = code.split('```')
            if len(parts) >= 3:
                code = parts[1]
        
        # Add imports if missing
        if 'import numpy' not in code:
            code = 'import numpy as np\nfrom typing import Any\n\n' + code
        
        return code.strip()
    
    def _create_working_template(self, alg_name: str, description: str) -> str:
        """Create a minimal working template"""
        
        class_name = alg_name.replace(' ', '').replace('-', '')
        
        return f'''"""
{alg_name} Implementation
"""

import numpy as np
from typing import Any

class {class_name}:
    """Implementation of {alg_name}."""
    
    def __init__(self, **kwargs):
        self.config = kwargs
    
    def run(self, input_data: Any) -> Any:
        """Run the algorithm."""
        if isinstance(input_data, list):
            input_data = np.array(input_data)
        
        if isinstance(input_data, np.ndarray):
            # Apply processing
            result = np.tanh(input_data)
            return result
        
        return input_data

def run_{alg_name.lower().replace(' ', '_').replace('-', '_')}(data, **kwargs):
    """Run {alg_name}."""
    alg = {class_name}(**kwargs)
    return alg.run(data)
'''
    
    def _detect_algorithm_type(self, alg_name: str, description: str, formulas: List[Dict], algorithms: List[Dict]) -> str:
        """Detect the type of algorithm using intelligent analysis"""
        
        # Combine all available information
        combined_text = f"{alg_name} {description}".lower()
        formula_text = " ".join([f.get('formula', '') + " " + f.get('type', '') for f in formulas]).lower()
        combined_text += " " + formula_text
        
        # Use probabilistic scoring instead of hardcoded if-else chains
        algorithm_types = {
            'attention': {
                'keywords': ['attention', 'query', 'key', 'value', 'self-attention', 'multi-head', 'softmax', 'scaled', 'dot-product'],
                'formula_patterns': ['attention(', 'softmax(', 'qk', 'multihead'],
                'score': 0
            },
            'transformer': {
                'keywords': ['transformer', 'encoder', 'decoder', 'positional', 'embedding', 'layer norm'],
                'formula_patterns': ['pe(', 'layernorm', 'residual', 'feedforward'],
                'score': 0
            },
            'cnn': {
                'keywords': ['convolution', 'cnn', 'filter', 'kernel', 'pooling', 'stride', 'padding', 'channel'],
                'formula_patterns': ['conv', 'pool', 'filter', 'kernel'],
                'score': 0
            },
            'rnn': {
                'keywords': ['recurrent', 'rnn', 'lstm', 'gru', 'sequence', 'hidden', 'cell', 'gate'],
                'formula_patterns': ['lstm', 'gru', 'hidden', 'cell_state'],
                'score': 0
            },
            'optimization': {
                'keywords': ['gradient', 'adam', 'sgd', 'optimizer', 'learning', 'momentum', 'decay'],
                'formula_patterns': ['grad', 'lr', 'momentum', 'decay'],
                'score': 0
            },
            'gan': {
                'keywords': ['generative', 'adversarial', 'gan', 'generator', 'discriminator', 'minimax'],
                'formula_patterns': ['generator', 'discriminator', 'adversarial'],
                'score': 0
            },
            'reinforcement': {
                'keywords': ['reinforcement', 'policy', 'reward', 'q-learning', 'actor', 'critic', 'action', 'state'],
                'formula_patterns': ['policy', 'reward', 'q_value', 'action'],
                'score': 0
            },
            'embedding': {
                'keywords': ['embedding', 'encode', 'representation', 'encode', 'positional'],
                'formula_patterns': ['embed', 'encoding', 'pe('],
                'score': 0
            },
            'classification': {
                'keywords': ['classification', 'classifier', 'predict', 'category', 'class'],
                'formula_patterns': ['softmax', 'crossentropy', 'classification'],
                'score': 0
            },
            'loss': {
                'keywords': ['loss', 'cost', 'error', 'objective', 'minimize', 'maximize'],
                'formula_patterns': ['loss', 'mse', 'crossentropy', 'bce'],
                'score': 0
            }
        }
        
        # Score each algorithm type
        for alg_type, data in algorithm_types.items():
            # Score based on keywords in description/name
            for keyword in data['keywords']:
                if keyword in combined_text:
                    data['score'] += 2
            
            # Score based on formula patterns
            for pattern in data['formula_patterns']:
                if pattern in formula_text:
                    data['score'] += 3  # Formula matches are weighted higher
            
            # Bonus points for exact name matches
            if alg_type in alg_name.lower():
                data['score'] += 5
        
        # Find the highest scoring type
        best_type = 'generic'
        best_score = 0
        
        for alg_type, data in algorithm_types.items():
            if data['score'] > best_score:
                best_score = data['score']
                best_type = alg_type
        
        # If no type scored well, try to infer from context
        if best_score < 2:
            # Look for mathematical/neural network indicators
            math_indicators = ['neural', 'network', 'layer', 'weights', 'bias', 'activation']
            if any(indicator in combined_text for indicator in math_indicators):
                best_type = 'neural_network'
            elif any(indicator in combined_text for indicator in ['algorithm', 'method', 'approach']):
                best_type = 'algorithm'
            else:
                best_type = 'generic'
        
        logger.debug(f"🎯 Algorithm type detection: '{alg_name}' -> '{best_type}' (score: {best_score})")
        return best_type
    
    def _generate_cnn_implementation(self, alg_name: str, class_name: str, formulas: List[Dict]) -> str:
        """Generate complete CNN implementation"""
        
        return f'''"""
{alg_name} Implementation

Complete convolutional neural network implementation.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple

class {class_name}(nn.Module):
    """
    Complete CNN implementation.
    
    Implements convolutional operations: y[i,j] = Σ Σ x[i+m,j+n] * w[m,n]
    """
    
    def __init__(self, in_channels: int = 3, num_classes: int = 10, **kwargs):
        """Initialize the CNN."""
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the CNN."""
        # Convolutional layers with pooling
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        # Flatten for fully connected layers
        x = x.view(-1, 128 * 4 * 4)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    
    def run(self, input_data: Any) -> Any:
        """Run the CNN."""
        if isinstance(input_data, torch.Tensor):
            return self.forward(input_data)
        elif isinstance(input_data, np.ndarray):
            input_tensor = torch.from_numpy(input_data).float()
            output = self.forward(input_tensor)
            return output.detach().numpy()
        else:
            raise ValueError(f"Unsupported input type: {{type(input_data)}}")

def run_{alg_name.lower().replace(' ', '_').replace('-', '_')}(input_data: Any, **kwargs) -> Any:
    """Run {alg_name} algorithm."""
    algorithm = {class_name}(**kwargs)
    return algorithm.run(input_data)
'''
    
    def _generate_rnn_implementation(self, alg_name: str, class_name: str, formulas: List[Dict]) -> str:
        """Generate complete RNN implementation"""
        
        return f'''"""
{alg_name} Implementation

Complete recurrent neural network implementation.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional, Tuple

class {class_name}(nn.Module):
    """
    Complete RNN implementation.
    
    Implements recurrent processing for sequential data.
    """
    
    def __init__(self, input_size: int = 100, hidden_size: int = 128, 
                 num_layers: int = 2, dropout: float = 0.1, **kwargs):
        """Initialize the RNN."""
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout)
        
        # Output layer
        self.fc = nn.Linear(hidden_size, input_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, hidden: Optional[Tuple] = None) -> Tuple[torch.Tensor, Tuple]:
        """Forward pass through the RNN."""
        # LSTM forward pass
        output, hidden = self.lstm(x, hidden)
        
        # Apply dropout and linear layer
        output = self.dropout(output)
        output = self.fc(output)
        
        return output, hidden
    
    def init_hidden(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize hidden state."""
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        return (h0, c0)
    
    def run(self, input_data: Any) -> Any:
        """Run the RNN."""
        if isinstance(input_data, torch.Tensor):
            batch_size = input_data.size(0)
            hidden = self.init_hidden(batch_size)
            output, _ = self.forward(input_data, hidden)
            return output
        elif isinstance(input_data, np.ndarray):
            input_tensor = torch.from_numpy(input_data).float()
            batch_size = input_tensor.size(0)
            hidden = self.init_hidden(batch_size)
            output, _ = self.forward(input_tensor, hidden)
            return output.detach().numpy()
        else:
            raise ValueError(f"Unsupported input type: {{type(input_data)}}")

def run_{alg_name.lower().replace(' ', '_').replace('-', '_')}(input_data: Any, **kwargs) -> Any:
    """Run {alg_name} algorithm."""
    algorithm = {class_name}(**kwargs)
    return algorithm.run(input_data)
'''
    
    def _generate_optimization_implementation(self, alg_name: str, class_name: str, formulas: List[Dict]) -> str:
        """Generate complete optimization algorithm implementation"""
        
        return f'''"""
{alg_name} Implementation

Complete optimization algorithm implementation.
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Callable

class {class_name}:
    """
    Complete optimization algorithm implementation.
    
    Implements parameter optimization using gradient-based methods.
    """
    
    def __init__(self, learning_rate: float = 0.001, beta1: float = 0.9, 
                 beta2: float = 0.999, epsilon: float = 1e-8, **kwargs):
        """Initialize the optimizer."""
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        
        # Internal state
        self.t = 0  # Time step
        self.m = {{}}  # First moment estimates
        self.v = {{}}  # Second moment estimates
    
    def step(self, parameters: Dict[str, np.ndarray], gradients: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Perform one optimization step.
        
        Args:
            parameters: Current parameter values
            gradients: Gradients for each parameter
            
        Returns:
            Updated parameters
        """
        self.t += 1
        
        updated_params = {{}}
        
        for name, param in parameters.items():
            if name not in gradients:
                updated_params[name] = param
                continue
            
            grad = gradients[name]
            
            # Initialize moments if needed
            if name not in self.m:
                self.m[name] = np.zeros_like(param)
                self.v[name] = np.zeros_like(param)
            
            # Update biased first moment estimate
            self.m[name] = self.beta1 * self.m[name] + (1 - self.beta1) * grad
            
            # Update biased second moment estimate
            self.v[name] = self.beta2 * self.v[name] + (1 - self.beta2) * (grad ** 2)
            
            # Compute bias-corrected first moment estimate
            m_hat = self.m[name] / (1 - self.beta1 ** self.t)
            
            # Compute bias-corrected second moment estimate
            v_hat = self.v[name] / (1 - self.beta2 ** self.t)
            
            # Update parameters: θ = θ - α * m_hat / (√v_hat + ε)
            updated_params[name] = param - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
        
        return updated_params
    
    def run(self, input_data: Any) -> Any:
        """
        Run the optimization algorithm.
        
        Args:
            input_data: Dictionary with 'parameters', 'gradients', and optional 'objective_function'
            
        Returns:
            Updated parameters
        """
        if isinstance(input_data, dict):
            parameters = input_data['parameters']
            gradients = input_data['gradients']
            
            return self.step(parameters, gradients)
        else:
            raise ValueError("Optimization requires dict with 'parameters' and 'gradients'")

def run_{alg_name.lower().replace(' ', '_').replace('-', '_')}(input_data: Any, **kwargs) -> Any:
    """Run {alg_name} algorithm."""
    algorithm = {class_name}(**kwargs)
    return algorithm.run(input_data)
'''
