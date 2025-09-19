"""
Basic tests for Paper2Code standalone functionality
"""

import pytest
import tempfile
from pathlib import Path

# Test imports
def test_imports():
    """Test that main modules can be imported"""
    try:
        import paper2code
        assert hasattr(paper2code, 'Paper2CodeProcessor')
        assert hasattr(paper2code, 'ConfigManager')
        assert hasattr(paper2code, 'FileProcessor')
    except ImportError as e:
        pytest.fail(f"Failed to import paper2code: {e}")

def test_processor_initialization():
    """Test processor can be initialized"""
    from paper2code import Paper2CodeProcessor
    
    processor = Paper2CodeProcessor()
    assert processor is not None
    assert hasattr(processor, 'config')
    assert hasattr(processor, 'process_paper')

def test_config_manager():
    """Test configuration manager"""
    from paper2code.config import ConfigManager
    
    config = ConfigManager()
    assert config is not None
    
    # Test getting default values
    assert config.get('processing.mode') == 'comprehensive'
    assert config.get('llm.preferred_provider') == 'anthropic'

def test_file_processor():
    """Test file processor basic functionality"""
    from paper2code.utils import FileProcessor
    
    processor = FileProcessor()
    assert processor is not None
    
    # Test with temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("# Test Paper\n\nThis is a test paper content.")
        temp_path = Path(f.name)
    
    try:
        # Test complexity estimation
        content = temp_path.read_text()
        complexity = processor.estimate_complexity(content)
        
        assert 'level' in complexity
        assert 'score' in complexity
        assert complexity['level'] in ['Low', 'Medium', 'High']
        
    finally:
        temp_path.unlink()

def test_document_analyzer_agent():
    """Test document analyzer agent"""
    from paper2code.agents import DocumentAnalysisAgent
    from paper2code.config import ConfigManager
    
    config = ConfigManager()
    analyzer = DocumentAnalysisAgent(config)
    
    assert analyzer is not None
    assert hasattr(analyzer, 'analyze_full_document')
    assert hasattr(analyzer, 'analyze_with_segmentation')

@pytest.mark.asyncio
async def test_process_sample_content():
    """Test processing sample content"""
    from paper2code.agents import DocumentAnalysisAgent
    from paper2code.config import ConfigManager
    
    config = ConfigManager()
    analyzer = DocumentAnalysisAgent(config)
    
    # Sample academic content
    sample_content = """
# Neural Network Optimization in Deep Learning

## Abstract
This paper presents a novel approach to optimizing neural networks using advanced gradient descent techniques.

## Introduction
Deep learning has revolutionized machine learning through the use of neural networks with multiple hidden layers.

## Method
We propose Algorithm 1 for improved gradient descent:

Algorithm 1: Enhanced Gradient Descent
1. Initialize parameters θ
2. For each iteration:
   2.1 Compute gradient ∇θ L(θ)
   2.2 Update θ ← θ - α∇θ L(θ)
3. Return optimized parameters

## Results
Our method achieves 95% accuracy on the test dataset.
    """
    
    # Test full document analysis
    result = await analyzer.analyze_full_document(sample_content, Path("test_paper.md"))
    
    assert result is not None
    assert 'analysis_type' in result
    assert 'technical_content' in result
    assert 'complexity' in result
    assert 'summary' in result
    
    # Check technical content extraction
    tech_content = result['technical_content']
    assert 'algorithms' in tech_content
    assert len(tech_content['algorithms']) > 0
    
    # Check complexity assessment
    complexity = result['complexity']
    assert 'level' in complexity
    assert 'score' in complexity

if __name__ == "__main__":
    pytest.main([__file__])
