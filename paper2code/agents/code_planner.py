"""
Code Planning Agent for Paper2Code

Creates comprehensive implementation plans from paper analysis results.
"""

import json
import yaml
from typing import Dict, Any, List, Optional

from ..config.manager import ConfigManager  
from ..utils.logger import get_logger
from ..utils.llm import build_llm_client

logger = get_logger(__name__)


class CodePlanningAgent:
    """
    AI agent specialized in creating detailed implementation plans from paper analysis.
    
    Transforms research paper analysis into structured code implementation roadmaps.
    """
    
    def __init__(self, config: ConfigManager):
        """
        Initialize code planning agent.
        
        Args:
            config: Configuration manager instance
        """
        self.config = config
    
    async def create_implementation_plan(
        self,
        analysis_results: Dict[str, Any],
        repository_results: Dict[str, Any],
        options: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create comprehensive implementation plan from analysis results.
        
        Args:
            analysis_results: Results from document analysis
            repository_results: Results from repository discovery  
            options: Processing options
            
        Returns:
            Detailed implementation plan
        """
        logger.info("🏗️ Creating implementation plan")
        
        try:
            # Extract key information
            technical_content = analysis_results.get('technical_content', {})
            complexity = analysis_results.get('complexity', {})
            requirements = analysis_results.get('implementation_requirements', {})
            
            # Create file structure (LLM-assisted if available)
            file_structure = self._design_file_structure(
                technical_content, complexity, requirements
            )
            
            # Plan implementation components
            implementation_components = self._plan_implementation_components(
                technical_content, analysis_results, repository_results
            )
            
            # Create validation approach
            validation_plan = self._create_validation_plan(
                technical_content, analysis_results
            )
            
            # Plan environment setup
            environment_setup = self._plan_environment_setup(
                requirements, repository_results
            )
            
            # Create implementation strategy
            implementation_strategy = self._create_implementation_strategy(
                implementation_components, complexity
            )
            
            # Generate requirements list
            dependencies = self._generate_requirements_list(
                requirements, repository_results
            )
            
            planning_result = {
                'paper_info': {
                    'title': analysis_results.get('document_info', {}).get('title', 'Unknown'),
                    'complexity_level': complexity.get('level', 'Medium'),
                    'estimated_components': complexity.get('estimated_components', 5)
                },
                'file_structure': file_structure,
                'implementation_components': implementation_components,
                'validation_approach': validation_plan,
                'environment_setup': environment_setup,
                'implementation_strategy': implementation_strategy,
                'requirements': dependencies,
                'primary_language': requirements.get('programming_language', 'python'),
                'mode': options.get('mode', 'comprehensive'),
                'summary': {
                    'total_files': len(file_structure.get('files', [])),
                    'main_algorithms': len(technical_content.get('algorithms', [])),
                    'estimated_time': complexity.get('estimated_implementation_time', '1 week'),
                    'primary_language': requirements.get('programming_language', 'python')
                }
            }
            
            logger.info(f"✅ Implementation plan created: {planning_result['summary']}")
            return planning_result
            
        except Exception as e:
            logger.error(f"❌ Implementation planning failed: {e}")
            raise
    
    def _design_file_structure(
        self,
        technical_content: Dict[str, Any],
        complexity: Dict[str, Any],
        requirements: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Design the file and directory structure for implementation"""
        
        language = requirements.get('programming_language', 'python')
        algorithms = technical_content.get('algorithms', [])
        components = technical_content.get('components', [])
        
        # Base structure for Python projects
        if language == 'python':
            structure = {
                'root_files': [
                    'main.py',
                    'requirements.txt', 
                    'README.md',
                    'setup.py'
                ],
                'directories': {
                    'src': {
                        'files': ['__init__.py'],
                        'subdirs': {}
                    },
                    'tests': {
                        'files': ['__init__.py', 'test_main.py'],
                        'subdirs': {}
                    },
                    'docs': {
                        'files': ['documentation.md'],
                        'subdirs': {}
                    },
                    'data': {
                        'files': ['sample_data.py'],
                        'subdirs': {}
                    }
                }
            }
            
            # Add algorithm-specific files
            if algorithms:
                algorithm_files = []
                for i, alg in enumerate(algorithms):
                    alg_name = self._sanitize_filename(alg.get('name', f'algorithm_{i+1}'))
                    algorithm_files.append(f'{alg_name}.py')
                    
                structure['directories']['src']['subdirs']['algorithms'] = {
                    'files': ['__init__.py'] + algorithm_files,
                    'subdirs': {}
                }
            
            # Add model/component files
            if components:
                model_files = []
                for comp in components:
                    # Handle both string and dict components
                    if isinstance(comp, dict):
                        comp_name = self._sanitize_filename(comp.get('name', 'component'))
                    else:
                        comp_name = self._sanitize_filename(str(comp))
                    model_files.append(f'{comp_name}.py')
                
                structure['directories']['src']['subdirs']['models'] = {
                    'files': ['__init__.py'] + model_files,
                    'subdirs': {}
                }
            
            # Add utilities
            structure['directories']['src']['subdirs']['utils'] = {
                'files': ['__init__.py', 'helpers.py', 'data_processing.py'],
                'subdirs': {}
            }
            
        else:
            # Generic structure for other languages
            structure = {
                'root_files': ['main.' + self._get_file_extension(language), 'README.md'],
                'directories': {
                    'src': {'files': [], 'subdirs': {}},
                    'tests': {'files': [], 'subdirs': {}},
                    'docs': {'files': ['documentation.md'], 'subdirs': {}}
                }
            }
        
        # Flatten structure for easier access
        all_files = []
        all_files.extend(structure['root_files'])
        
        def collect_files(dir_info, path_prefix=''):
            for filename in dir_info.get('files', []):
                all_files.append(f"{path_prefix}{filename}")
            
            for dirname, subdir_info in dir_info.get('subdirs', {}).items():
                collect_files(subdir_info, f"{path_prefix}{dirname}/")
        
        for dirname, dir_info in structure.get('directories', {}).items():
            collect_files(dir_info, f"{dirname}/")
        
        return {
            'structure': structure,
            'files': all_files,
            'language': language
        }
    
    def _plan_implementation_components(
        self,
        technical_content: Dict[str, Any],
        analysis_results: Dict[str, Any],
        repository_results: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Plan specific implementation components"""
        
        components = []
        algorithms = technical_content.get('algorithms', [])
        formulas = technical_content.get('formulas', [])
        tech_components = technical_content.get('components', [])
        
        # Main implementation component
        components.append({
            'name': 'Main Application',
            'file': 'main.py',
            'priority': 1,
            'description': 'Entry point and main execution logic',
            'dependencies': [],
            'implementation_notes': [
                'Setup argument parsing',
                'Initialize core components',
                'Implement main execution flow',
                'Handle errors and logging'
            ]
        })
        
        # Algorithm implementations
        for i, alg in enumerate(algorithms):
            alg_name = self._sanitize_filename(alg.get('name', f'algorithm_{i+1}'))
            components.append({
                'name': alg.get('name', f'Algorithm {i+1}'),
                'file': f'src/algorithms/{alg_name}.py',
                'priority': 2,
                'description': f'Implementation of {alg.get("name", "algorithm")}',
                'algorithm_content': alg.get('content', ''),
                'dependencies': ['numpy'] if 'numpy' in str(alg.get('content', '')).lower() else [],
                'implementation_notes': [
                    'Implement core algorithm logic',
                    'Add input validation',
                    'Optimize for performance',
                    'Add comprehensive tests'
                ]
            })
        
        # Model/component implementations
        for comp in tech_components:
            # Handle both string and dict components
            if isinstance(comp, dict):
                comp_name = self._sanitize_filename(comp.get('name', 'component'))
                comp_display_name = comp.get('name', 'Component')
            else:
                comp_name = self._sanitize_filename(str(comp))
                comp_display_name = str(comp)
                
            components.append({
                'name': comp_display_name,
                'file': f'src/models/{comp_name}.py',
                'priority': 2,
                'description': f'Implementation of {comp_display_name}',
                'dependencies': self._infer_dependencies(comp if isinstance(comp, dict) else {'name': str(comp)}),
                'implementation_notes': [
                    'Implement component architecture',
                    'Add configuration options',
                    'Implement forward/backward methods',
                    'Add serialization support'
                ]
            })
        
        # Utility components
        components.append({
            'name': 'Data Processing',
            'file': 'src/utils/data_processing.py',
            'priority': 3,
            'description': 'Data loading, preprocessing, and utility functions',
            'dependencies': ['numpy', 'pandas'],
            'implementation_notes': [
                'Implement data loading functions',
                'Add preprocessing pipelines',
                'Create data validation utilities',
                'Add visualization helpers'
            ]
        })
        
        components.append({
            'name': 'Helper Functions', 
            'file': 'src/utils/helpers.py',
            'priority': 3,
            'description': 'General utility and helper functions',
            'dependencies': [],
            'implementation_notes': [
                'Implement common utilities',
                'Add logging configuration',
                'Create file I/O helpers',
                'Add mathematical utilities'
            ]
        })
        
        # Test components
        components.append({
            'name': 'Test Suite',
            'file': 'tests/test_main.py',
            'priority': 4,
            'description': 'Comprehensive test suite for all components',
            'dependencies': ['pytest'],
            'implementation_notes': [
                'Implement unit tests for all components',
                'Add integration tests',
                'Create test data fixtures',
                'Add performance benchmarks'
            ]
        })
        
        return sorted(components, key=lambda x: x['priority'])
    
    def _create_validation_plan(
        self,
        technical_content: Dict[str, Any],
        analysis_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create plan for validating implementation"""
        
        algorithms = technical_content.get('algorithms', [])
        requirements = analysis_results.get('implementation_requirements', {})
        metrics = requirements.get('evaluation_metrics', [])
        
        validation_plan = {
            'testing_strategy': [
                'Unit tests for individual components',
                'Integration tests for complete workflows',
                'Performance benchmarking',
                'Correctness validation against paper results'
            ],
            'test_categories': [
                {
                    'name': 'Algorithm Correctness',
                    'description': 'Verify algorithms produce expected outputs',
                    'methods': [
                        'Test with known input/output pairs',
                        'Compare with reference implementations',
                        'Validate mathematical properties'
                    ]
                },
                {
                    'name': 'Performance Testing',
                    'description': 'Ensure acceptable performance characteristics',
                    'methods': [
                        'Benchmark execution time',
                        'Profile memory usage',
                        'Test scalability with large inputs'
                    ]
                }
            ],
            'validation_data': {
                'synthetic_data': 'Generate synthetic test cases',
                'reference_data': 'Use datasets mentioned in paper if available',
                'edge_cases': 'Test boundary conditions and error cases'
            },
            'success_criteria': [
                'All unit tests pass',
                'Algorithm outputs match expected behavior',
                'Performance within acceptable bounds',
                'Code follows best practices'
            ]
        }
        
        if metrics:
            validation_plan['evaluation_metrics'] = metrics
            validation_plan['test_categories'].append({
                'name': 'Metric Validation',
                'description': 'Verify evaluation metrics implementation',
                'methods': [f'Test {metric} calculation' for metric in metrics]
            })
        
        return validation_plan
    
    def _plan_environment_setup(
        self,
        requirements: Dict[str, Any],
        repository_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Plan environment and dependency setup"""
        
        language = requirements.get('programming_language', 'python')
        frameworks = requirements.get('frameworks', [])
        dependencies = requirements.get('dependencies', [])
        
        if language == 'python':
            setup = {
                'language': 'Python 3.8+',
                'package_manager': 'pip',
                'virtual_environment': 'python -m venv venv',
                'activation': {
                    'linux': 'source venv/bin/activate',
                    'windows': 'venv\\Scripts\\activate'
                },
                'installation': 'pip install -r requirements.txt',
                'core_dependencies': [
                    'numpy>=1.19.0',
                    'scipy>=1.7.0'
                ]
            }
            
            # Add framework-specific dependencies
            if 'pytorch' in frameworks:
                setup['core_dependencies'].extend([
                    'torch>=1.9.0',
                    'torchvision>=0.10.0'
                ])
            
            if 'tensorflow' in frameworks:
                setup['core_dependencies'].extend([
                    'tensorflow>=2.6.0'
                ])
            
            # Add detected dependencies
            setup['core_dependencies'].extend(dependencies)
            
        else:
            # Generic setup for other languages
            setup = {
                'language': language,
                'notes': f'Setup instructions for {language} will depend on specific requirements',
                'core_dependencies': dependencies
            }
        
        return setup
    
    def _create_implementation_strategy(
        self,
        components: List[Dict[str, Any]],
        complexity: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create step-by-step implementation strategy"""
        
        # Group components by priority
        phases = {}
        for component in components:
            priority = component.get('priority', 3)
            phase_name = f"Phase {priority}"
            
            if phase_name not in phases:
                phases[phase_name] = []
            
            phases[phase_name].append(component)
        
        strategy = {
            'approach': 'Incremental development with continuous testing',
            'phases': {},
            'development_workflow': [
                'Set up development environment',
                'Implement core components first',
                'Add tests for each component',
                'Integrate components gradually',
                'Validate against paper requirements',
                'Optimize and refine implementation'
            ],
            'risk_mitigation': [
                'Start with simplest components',
                'Validate each component before proceeding',
                'Keep fallback implementations for complex algorithms',
                'Regular testing throughout development'
            ]
        }
        
        # Define phase descriptions
        phase_descriptions = {
            'Phase 1': 'Core Infrastructure - Essential files and main entry points',
            'Phase 2': 'Algorithm Implementation - Core algorithms and models',
            'Phase 3': 'Supporting Components - Utilities and data processing',
            'Phase 4': 'Testing and Documentation - Comprehensive testing and docs'
        }
        
        for phase_name in sorted(phases.keys()):
            strategy['phases'][phase_name] = {
                'description': phase_descriptions.get(phase_name, 'Implementation phase'),
                'components': phases[phase_name],
                'estimated_time': self._estimate_phase_time(phases[phase_name], complexity)
            }
        
        return strategy
    
    def _generate_requirements_list(
        self,
        requirements: Dict[str, Any],
        repository_results: Dict[str, Any]
    ) -> List[str]:
        """Generate list of package requirements"""
        
        deps = set()
        
        # Add base dependencies
        language = requirements.get('programming_language', 'python')
        
        if language == 'python':
            deps.update([
                'numpy>=1.19.0',
                'scipy>=1.7.0'
            ])
            
            # Add framework dependencies
            frameworks = requirements.get('frameworks', [])
            for framework in frameworks:
                if framework.lower() in ['pytorch', 'torch']:
                    deps.add('torch>=1.9.0')
                elif framework.lower() in ['tensorflow', 'tf']:
                    deps.add('tensorflow>=2.6.0')
                elif framework.lower() in ['sklearn', 'scikit-learn']:
                    deps.add('scikit-learn>=1.0.0')
                elif framework.lower() == 'pandas':
                    deps.add('pandas>=1.3.0')
                elif framework.lower() == 'matplotlib':
                    deps.add('matplotlib>=3.4.0')
            
            # Add detected dependencies
            detected_deps = requirements.get('dependencies', [])
            for dep in detected_deps:
                if dep not in ['numpy', 'scipy']:  # Avoid duplicates with base
                    deps.add(dep)
            
            # Add testing dependencies
            deps.update([
                'pytest>=6.0.0',
                'pytest-cov>=2.12.0'
            ])
        
        return sorted(list(deps))
    
    def _sanitize_filename(self, name: str) -> str:
        """Sanitize name for use as filename"""
        import re
        
        # Convert to lowercase and replace spaces/special chars with underscores
        name = re.sub(r'[^a-zA-Z0-9\s]', '', name)
        name = re.sub(r'\s+', '_', name)
        name = name.lower().strip('_')
        
        if not name:
            name = 'component'
        
        return name
    
    def _get_file_extension(self, language: str) -> str:
        """Get file extension for programming language"""
        
        extensions = {
            'python': 'py',
            'java': 'java',
            'cpp': 'cpp',
            'c': 'c',
            'javascript': 'js',
            'r': 'R',
            'matlab': 'm'
        }
        
        return extensions.get(language.lower(), 'py')
    
    def _infer_dependencies(self, component: Dict[str, Any]) -> List[str]:
        """Infer likely dependencies for a component"""
        
        deps = []
        name = component.get('name', '').lower()
        
        if 'network' in name or 'neural' in name:
            deps.extend(['torch', 'tensorflow'])  # One will be filtered out later
        
        if 'data' in name:
            deps.append('pandas')
        
        if 'plot' in name or 'vis' in name:
            deps.append('matplotlib')
        
        return deps
    
    def _estimate_phase_time(self, components: List[Dict[str, Any]], complexity: Dict[str, Any]) -> str:
        """Estimate time needed for implementation phase"""
        
        base_time = len(components) * 0.5  # 0.5 days per component
        
        complexity_multiplier = {
            'Low': 1.0,
            'Medium': 1.5, 
            'High': 2.0
        }
        
        level = complexity.get('level', 'Medium')
        adjusted_time = base_time * complexity_multiplier.get(level, 1.5)
        
        if adjusted_time < 1:
            return "< 1 day"
        elif adjusted_time < 3:
            return "1-2 days"
        elif adjusted_time < 7:
            return "3-6 days"
        else:
            return "1+ weeks"
