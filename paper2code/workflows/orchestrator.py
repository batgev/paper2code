"""
Workflow Orchestrator for Paper2Code

Coordinates the entire paper-to-code transformation pipeline using specialized agents.
"""

import asyncio
import time
from pathlib import Path
from typing import Dict, Any, Optional, Callable

from ..config.manager import ConfigManager
from ..utils.file_processor import FileProcessor
from ..utils.logger import get_logger
from ..agents.document_analyzer import DocumentAnalysisAgent
from ..agents.code_planner import CodePlanningAgent  
from ..agents.code_generator import CodeGeneratorAgent
from ..agents.repository_finder import RepositoryFinderAgent

logger = get_logger(__name__)


class WorkflowOrchestrator:
    """
    Main workflow orchestrator for Paper2Code.
    
    Coordinates the multi-agent pipeline from paper analysis to code generation.
    """
    
    def __init__(self, config: ConfigManager):
        """
        Initialize orchestrator.
        
        Args:
            config: Configuration manager instance
        """
        self.config = config
        self.file_processor = FileProcessor()
        
        # Initialize agents
        self.document_analyzer = DocumentAnalysisAgent(config)
        self.code_planner = CodePlanningAgent(config)
        self.code_generator = CodeGeneratorAgent(config) 
        self.repository_finder = RepositoryFinderAgent(config)
        
        # Processing state
        self.current_task = None
        self.processing_stats = {}
    
    async def process_paper(
        self,
        input_path: Path,
        output_path: Path,
        options: Dict[str, Any],
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Process a research paper through the complete pipeline.
        
        Args:
            input_path: Path to input paper
            output_path: Output directory path
            options: Processing options
            progress_callback: Optional progress callback function
            
        Returns:
            Processing result dictionary
        """
        start_time = time.time()
        
        try:
            logger.info(f"🚀 Starting paper processing pipeline")
            logger.info(f"📖 Input: {input_path}")
            logger.info(f"📁 Output: {output_path}")
            logger.info(f"⚙️ Options: {options}")
            
            # Initialize processing state
            self.current_task = {
                'input_path': input_path,
                'output_path': output_path,
                'start_time': start_time,
                'stage': 'initialization'
            }
            
            # Create output structure
            if progress_callback:
                progress_callback(5, "📁 Creating output structure...")
            
            # output_path is already paper-specific directory; avoid double-nesting
            output_structure = await self.file_processor.create_output_structure(
                output_path
            )
            
            # Phase 1: Document Analysis
            if progress_callback:
                progress_callback(15, "📊 Analyzing research document...")
            
            analysis_results = await self._run_document_analysis(
                input_path, output_structure, options
            )
            
            # Phase 2: Repository Discovery (if enabled)
            repository_results = {}
            if options.get('enable_repository_search', True):
                if progress_callback:
                    progress_callback(35, "🔍 Discovering relevant repositories...")
                
                repository_results = await self._run_repository_discovery(
                    analysis_results, output_structure, options
                )
            else:
                logger.info("⚡ Repository search disabled (fast mode)")
            
            # Phase 3: Code Planning
            if progress_callback:
                progress_callback(55, "🏗️ Creating implementation plan...")
            
            planning_results = await self._run_code_planning(
                analysis_results, repository_results, output_structure, options
            )
            
            # Phase 4: Code Generation
            if progress_callback:
                progress_callback(75, "💻 Generating code implementation...")
            
            generation_results = await self._run_code_generation(
                planning_results, output_structure, options
            )
            
            # Phase 5: Finalization
            if progress_callback:
                progress_callback(95, "📝 Finalizing output...")
            
            finalization_results = await self._finalize_output(
                output_structure, 
                analysis_results,
                planning_results, 
                generation_results,
                options
            )
            
            # Complete processing
            processing_time = time.time() - start_time
            
            success_result = {
                'status': 'success',
                'output_path': str(output_path),
                'processing_time': processing_time,
                'generated_files': finalization_results.get('files', []),
                'metadata': {
                    'input_file': str(input_path),
                    'processing_mode': options.get('mode', 'comprehensive'),
                    'stages_completed': ['analysis', 'planning', 'generation', 'finalization'],
                    'analysis_summary': analysis_results.get('summary', {}),
                    'planning_summary': planning_results.get('summary', {}),
                    'generation_summary': generation_results.get('summary', {})
                }
            }
            
            if progress_callback:
                progress_callback(100, "✅ Processing completed successfully!")
            
            logger.info(f"🎉 Processing completed in {processing_time:.2f}s")
            return success_result
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Pipeline error: {str(e)}"
            logger.error(error_msg, exc_info=True)
            
            return {
                'status': 'error',
                'error': error_msg,
                'processing_time': processing_time,
                'stage': self.current_task.get('stage', 'unknown') if self.current_task else 'initialization'
            }
    
    async def _run_document_analysis(
        self,
        input_path: Path,
        output_structure: Dict[str, Path],
        options: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run document analysis phase"""
        logger.info("📊 Phase 1: Document Analysis")
        self.current_task['stage'] = 'document_analysis'
        
        try:
            # Read document content
            content = await self.file_processor.read_file_content(input_path)
            
            # Check if segmentation should be used
            use_segmentation = options.get('enable_segmentation', True)
            threshold = self.config.get('document_segmentation.size_threshold_chars', 50000)
            
            should_segment, reason = self.file_processor.should_use_segmentation(
                content, threshold
            )
            
            if use_segmentation and should_segment:
                logger.info(f"🔧 Using document segmentation: {reason}")
                analysis_results = await self.document_analyzer.analyze_with_segmentation(
                    content, input_path
                )
            else:
                logger.info(f"📖 Using full document analysis: {reason}")
                analysis_results = await self.document_analyzer.analyze_full_document(
                    content, input_path
                )
            
            # Save analysis results
            await self.file_processor.save_analysis_results(
                output_structure['paper_dir'],
                {'document_analysis': analysis_results}
            )
            
            logger.info("✅ Document analysis completed")
            return analysis_results
            
        except Exception as e:
            logger.error(f"❌ Document analysis failed: {e}")
            raise
    
    async def _run_repository_discovery(
        self,
        analysis_results: Dict[str, Any],
        output_structure: Dict[str, Path],
        options: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run repository discovery phase"""
        logger.info("🔍 Phase 2: Repository Discovery")
        self.current_task['stage'] = 'repository_discovery'
        
        try:
            # Extract search terms from analysis
            search_terms = self._extract_search_terms(analysis_results)
            
            # Find relevant repositories
            repository_results = await self.repository_finder.find_repositories(
                search_terms, analysis_results
            )
            
            # Save repository results
            await self.file_processor.save_analysis_results(
                output_structure['paper_dir'],
                {'repository_discovery': repository_results}
            )
            
            logger.info(f"✅ Found {len(repository_results.get('repositories', []))} relevant repositories")
            return repository_results
            
        except Exception as e:
            logger.error(f"❌ Repository discovery failed: {e}")
            # Continue without repositories in case of failure
            return {'repositories': [], 'error': str(e)}
    
    async def _run_code_planning(
        self,
        analysis_results: Dict[str, Any],
        repository_results: Dict[str, Any],
        output_structure: Dict[str, Path],
        options: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run code planning phase"""
        logger.info("🏗️ Phase 3: Code Planning")
        self.current_task['stage'] = 'code_planning'
        
        try:
            # Create implementation plan
            planning_results = await self.code_planner.create_implementation_plan(
                analysis_results, repository_results, options
            )
            
            # Save planning results
            await self.file_processor.save_analysis_results(
                output_structure['paper_dir'],
                {'implementation_plan': planning_results}
            )
            
            logger.info("✅ Implementation plan created")
            return planning_results
            
        except Exception as e:
            logger.error(f"❌ Code planning failed: {e}")
            raise
    
    async def _run_code_generation(
        self,
        planning_results: Dict[str, Any],
        output_structure: Dict[str, Path],
        options: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run code generation phase"""
        logger.info("💻 Phase 4: Code Generation")
        self.current_task['stage'] = 'code_generation'
        
        try:
            # Generate code based on plan
            generation_results = await self.code_generator.generate_implementation(
                planning_results, output_structure['code_dir'], options
            )
            
            logger.info(f"✅ Generated {len(generation_results.get('files', []))} code files")
            return generation_results
            
        except Exception as e:
            logger.error(f"❌ Code generation failed: {e}")
            raise
    
    async def _finalize_output(
        self,
        output_structure: Dict[str, Path],
        analysis_results: Dict[str, Any],
        planning_results: Dict[str, Any],
        generation_results: Dict[str, Any],
        options: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Finalize output and create documentation"""
        logger.info("📝 Phase 5: Finalization")
        self.current_task['stage'] = 'finalization'
        
        try:
            all_files = []
            
            # Create README for the generated code
            if self.config.get('output.create_readme', True):
                readme_content = self._create_readme_content(
                    analysis_results, planning_results, generation_results
                )
                
                readme_path = output_structure['code_dir'] / 'README.md'
                with open(readme_path, 'w', encoding='utf-8') as f:
                    f.write(readme_content)
                
                all_files.append(str(readme_path))
            
            # Create requirements.txt if applicable
            requirements = planning_results.get('requirements', [])
            if requirements:
                requirements_path = output_structure['code_dir'] / 'requirements.txt'
                with open(requirements_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(requirements))
                
                all_files.append(str(requirements_path))
            
            # Add generated files
            all_files.extend(generation_results.get('files', []))
            
            # Create project summary
            summary = {
                'project_name': output_structure['paper_dir'].name,
                'created': time.time(),
                'total_files': len(all_files),
                'processing_mode': options.get('mode', 'comprehensive'),
                'analysis_complexity': analysis_results.get('complexity', {}),
                'implementation_language': planning_results.get('primary_language', 'python')
            }
            
            logger.info("✅ Finalization completed")
            return {
                'files': all_files,
                'summary': summary
            }
            
        except Exception as e:
            logger.error(f"❌ Finalization failed: {e}")
            # Return partial results
            return {
                'files': generation_results.get('files', []),
                'summary': {'error': str(e)}
            }
    
    def _extract_search_terms(self, analysis_results: Dict[str, Any]) -> list:
        """Extract search terms for repository discovery"""
        terms = []
        
        # Extract from paper title
        doc_title = (
            analysis_results.get('document_info', {}).get('title')
            or analysis_results.get('summary', {}).get('title')
            or ''
        )
        if doc_title:
            terms.extend(doc_title.split())
        
        # Extract technical terms
        algorithms = analysis_results.get('algorithms', [])
        for alg in algorithms:
            if isinstance(alg, dict) and 'name' in alg:
                terms.append(alg['name'])
        
        # Extract key concepts
        concepts = analysis_results.get('key_concepts', [])
        terms.extend(concepts)
        
        return list(set(terms))  # Remove duplicates
    
    def _create_readme_content(
        self,
        analysis_results: Dict[str, Any],
        planning_results: Dict[str, Any],
        generation_results: Dict[str, Any]
    ) -> str:
        """Create README content for generated code"""
        paper_title = (
            analysis_results.get('document_info', {}).get('title')
            or analysis_results.get('summary', {}).get('title')
            or 'Research Paper Implementation'
        )
        
        readme_content = f"""# {paper_title}

*This implementation was automatically generated from the research paper using Paper2Code.*

## Overview

{analysis_results.get('abstract', 'Implementation of research paper algorithms and methods.')}

## Generated Components

"""
        
        # List generated files
        files = generation_results.get('files', [])
        for file_path in files:
            filename = Path(file_path).name
            readme_content += f"- `{filename}`\n"
        
        readme_content += f"""
## Requirements

Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

```python
# Example usage
from main import main

# Run implementation
result = main()
print(result)
```

## Implementation Details

This code implements the following algorithms from the paper:

"""
        
        # List algorithms
        algorithms = analysis_results.get('algorithms', [])
        for alg in algorithms:
            if isinstance(alg, dict):
                name = alg.get('name', 'Unknown Algorithm')
                readme_content += f"- **{name}**: {alg.get('description', 'Algorithm implementation')}\n"
        
        readme_content += f"""
## Paper Information

- **Processing Mode**: {planning_results.get('mode', 'comprehensive')}
- **Primary Language**: {planning_results.get('primary_language', 'python')}
- **Generated**: {time.strftime('%Y-%m-%d %H:%M:%S')}

## Notes

This is an automatically generated implementation. Please review and test the code before using in production.
"""
        
        return readme_content
    
    def get_processing_status(self) -> Optional[Dict[str, Any]]:
        """Get current processing status"""
        if not self.current_task:
            return None
        
        elapsed = time.time() - self.current_task['start_time']
        return {
            'input_path': str(self.current_task['input_path']),
            'output_path': str(self.current_task['output_path']),
            'current_stage': self.current_task['stage'],
            'elapsed_time': elapsed,
            'start_time': self.current_task['start_time']
        }
    
    async def cancel_processing(self):
        """Cancel current processing"""
        if self.current_task:
            logger.warning("⚠️ Processing cancellation requested")
            # Add cancellation logic here
            self.current_task = None
