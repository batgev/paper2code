# Paper2Code - AI-Powered Research Paper Implementation Tool

<div align="center">

<h3>🧬 Transform Research Papers into Working Code</h3>

[![Python](https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![AI](https://img.shields.io/badge/AI-Multi--Agent-9b59b6?style=for-the-badge&logo=brain&logoColor=white)](https://github.com)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

*Automatically convert academic research papers into production-ready code implementations*

**Created by Hesham Haroon - AI Lead**

</div>

---

## 🚀 **Overview**

**Paper2Code** is a cutting-edge AI-powered tool that transforms research papers into working code implementations using advanced multi-agent architecture. Built by Hesham Haroon, this standalone system analyzes academic papers and generates complete, testable code implementations with state-of-the-art accuracy.

## ✨ **Key Features**

### 📄 **Intelligent Document Processing**
- **Multi-format Support**: PDF, DOCX, HTML, TXT, Markdown
- **Smart Segmentation**: Handles large papers with token-efficient processing
- **Academic Focus**: Optimized for research paper structure and content

### 🤖 **Multi-Agent Architecture**
- **Document Analysis Agent**: Extracts algorithms, formulas, and technical details
- **Code Planning Agent**: Creates comprehensive implementation roadmaps
- **Implementation Agent**: Generates working code with proper structure
- **Repository Agent**: Discovers and integrates relevant GitHub repositories

### ⚡ **Advanced Processing**
- **Configurable Modes**: Fast mode vs. comprehensive analysis
- **Token Optimization**: Intelligent segmentation for large documents
- **Error Handling**: Robust processing with fallback mechanisms

## 🏗️ **Architecture**

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Research Paper │ -> │  Multi-Agent AI  │ -> │  Working Code   │
│                 │    │                  │    │                 │
│ • PDF/DOCX/etc  │    │ • Document Agent │    │ • Complete Impl │
│ • Algorithms    │    │ • Planning Agent │    │ • Tests         │
│ • Formulas      │    │ • Code Agent     │    │ • Documentation │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### **Processing Pipeline:**
1. **📖 Document Analysis**: Extract structure, algorithms, and technical details
2. **🎯 Implementation Planning**: Create comprehensive code implementation plan
3. **🔍 Repository Discovery**: Find relevant GitHub repositories for reference
4. **💻 Code Generation**: Produce complete, runnable implementations
5. **🧪 Testing & Validation**: Generate tests and verify functionality

## 📦 **Installation**

### **Prerequisites**
- Python 3.9 or higher
- Node.js 16+ (for MCP servers)
- Git

### **Install from Source**
```bash
# Clone the repository
git clone https://github.com/h9-tec/paper2code.git
cd paper2code

# Install Python dependencies
pip install -r requirements.txt

# Install Node.js MCP servers
npm install -g @modelcontextprotocol/server-brave-search
npm install -g @modelcontextprotocol/server-filesystem
```

### **Configuration**
1. Copy configuration templates:
```bash
cp paper2code/config/config.yaml.template paper2code/config/config.yaml
cp paper2code/config/secrets.yaml.template paper2code/config/secrets.yaml
```

2. Edit `paper2code/config/secrets.yaml` with your API keys:
```yaml
openai:
  api_key: "your_openai_api_key_here"
  base_url: "https://api.openai.com/v1"

anthropic:
  api_key: "your_anthropic_api_key_here"

brave_search:
  api_key: "your_brave_api_key_here"  # Optional
```

## 🎯 **Usage**

### **Command Line Interface**

```bash
# Process a research paper PDF
python -m paper2code --file paper.pdf

# Process from URL
python -m paper2code --url https://arxiv.org/pdf/2301.12345.pdf

# Fast mode (skip repository indexing)
python -m paper2code --file paper.pdf --fast

# Custom output directory
python -m paper2code --file paper.pdf --output /path/to/output

# Enable debug mode
python -m paper2code --file paper.pdf --debug
```

### **Web Interface**

```bash
# 1) Run API server (serves frontend build at /)
paper2code-api

# 2) Run frontend dev (HMR) in another terminal
cd frontend && npm run dev

# 3) Build frontend for production and serve from API static mount
cd frontend && npm run build
paper2code-api
```

### **Interactive Mode**
```bash
# Start interactive session
python -m paper2code

# Follow the prompts to:
# 1. Select input method (file/URL)
# 2. Choose processing mode
# 3. Configure output options
```

### **Python API**
```python
from paper2code import Paper2CodeProcessor

# Initialize processor
processor = Paper2CodeProcessor()

# Process a paper
result = await processor.process_paper("path/to/paper.pdf")

# Check results
if result.success:
    print(f"✅ Code generated at: {result.output_path}")
    print(f"📊 Files created: {len(result.files)}")
else:
    print(f"❌ Error: {result.error}")
```

## ⚙️ **Configuration**

### **Processing Modes**
```yaml
# config.yaml
processing:
  mode: "comprehensive"  # or "fast"
  enable_repository_search: true
  enable_segmentation: true
  segmentation_threshold: 50000  # characters
```

### **Document Segmentation**
```yaml
document_segmentation:
  enabled: true
  size_threshold_chars: 50000
  overlap_chars: 1000
  max_segments: 10
```

### **Output Configuration**
```yaml
output:
  base_directory: "./output"
  create_readme: true
  include_analysis: true
  code_format: "python"  # primary language
```

## 📁 **Output Structure**

```
output/
├── paper_name/
│   ├── analysis/
│   │   ├── document_analysis.md
│   │   ├── algorithm_extraction.yaml
│   │   └── implementation_plan.yaml
│   ├── code/
│   │   ├── __init__.py
│   │   ├── main.py
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   └── core_model.py
│   │   ├── algorithms/
│   │   │   ├── __init__.py
│   │   │   └── main_algorithm.py
│   │   ├── utils/
│   │   │   ├── __init__.py
│   │   │   └── helpers.py
│   │   └── tests/
│   │       ├── __init__.py
│   │       ├── test_main.py
│   │       └── test_models.py
│   ├── docs/
│   │   ├── README.md
│   │   ├── implementation_notes.md
│   │   └── api_documentation.md
│   └── references/
│       ├── github_repositories.json
│       └── reference_papers.txt
```

## 🔧 **Advanced Features**

### **Custom Prompts**
Modify prompts in `paper2code/prompts/` for specialized domains:
- `paper_analysis.py`: Document analysis prompts
- `code_generation.py`: Code implementation prompts
- `planning.py`: Implementation planning prompts

### **MCP Server Extensions**
Add custom tools in `paper2code/tools/`:
```python
# custom_tool_server.py
from paper2code.tools.base import BaseMCPServer

class CustomToolServer(BaseMCPServer):
    def setup_tools(self):
        # Define custom tools
        pass
```

### **Environment Variables**
```bash
export PAPER2CODE_CONFIG_PATH="/custom/config/path"
export PAPER2CODE_OUTPUT_DIR="/custom/output/path"
export PAPER2CODE_LOG_LEVEL="DEBUG"
export PAPER2CODE_CACHE_DIR="/custom/cache/path"
```

## 🧪 **Testing**

```bash
# Run all tests
python -m pytest tests/

# Run specific test categories
python -m pytest tests/test_agents.py
python -m pytest tests/test_workflows.py

# Test with sample paper
python -m paper2code --file samples/sample_paper.pdf --test-mode
```

## 📊 **Supported Paper Types**

- **Machine Learning**: Neural networks, algorithms, architectures
- **Computer Vision**: Image processing, detection, recognition
- **Natural Language Processing**: Text analysis, language models
- **Reinforcement Learning**: Agents, environments, policies
- **Systems**: Distributed systems, databases, protocols
- **Theory**: Mathematical proofs, algorithms, complexity

## 🔍 **Troubleshooting**

### **Common Issues**

1. **API Rate Limits**
   ```bash
   # Solution: Configure rate limiting in config.yaml
   api:
     rate_limit_delay: 1.0  # seconds between requests
   ```

2. **Large Papers (Token Limits)**
   ```bash
   # Solution: Enable document segmentation
   python -m paper2code --file large_paper.pdf --segment-threshold 30000
   ```

3. **Memory Issues**
   ```bash
   # Solution: Use fast mode
   python -m paper2code --file paper.pdf --fast --no-repository-search
   ```

### **Debug Mode**
```bash
# Enable detailed logging
python -m paper2code --file paper.pdf --debug --verbose

# Check logs
tail -f logs/paper2code.log
```

## 🎛️ **Performance Tuning**

### **Fast Mode Options**
```yaml
# Optimize for speed
performance:
  fast_mode: true
  skip_repository_search: true
  parallel_processing: true
  cache_enabled: true
  max_concurrent_agents: 3
```

### **Memory Optimization**
```yaml
# Optimize for memory usage
memory:
  max_document_size: 100000  # chars
  clear_cache_frequency: 100  # operations
  use_streaming: true
```

## 🤝 **Contributing**

1. **Fork the Repository**
   ```bash
   git fork https://github.com/h9-tec/paper2code
   ```

2. **Create Feature Branch**
   ```bash
   git checkout -b feature/amazing-feature
   ```

3. **Make Changes**
   - Add tests for new features
   - Update documentation
   - Follow code style guidelines

4. **Submit Pull Request**
   ```bash
   git commit -m 'Add amazing feature'
   git push origin feature/amazing-feature
   ```

### **Development Setup**
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run tests before committing
python -m pytest tests/
```

## 📝 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 **Support**

- **🐛 Issues**: [GitHub Issues](https://github.com/h9-tec/paper2code/issues)
- **💬 Discussions**: [GitHub Discussions](https://github.com/h9-tec/paper2code/discussions)
- **📧 Email**: support@paper2code.dev

## 🙏 **Acknowledgments**

- **Hesham Haroon**: AI Lead and Creator of Paper2Code
- **DeepCode Project**: Original multi-agent architecture by Data Intelligence Lab @ HKU
- **MCP Protocol**: Model Context Protocol for tool integration
- **OpenAI & Anthropic**: AI model providers
- **Research Community**: For providing papers to test and improve the system

---

<div align="center">

**🧬 Transform Research Into Reality 🧬**

*Accelerating scientific progress through automated paper reproduction*

[**⭐ Star this repo**](https://github.com/h9-tec/paper2code) • [**🚀 Try it now**](#installation)

</div>