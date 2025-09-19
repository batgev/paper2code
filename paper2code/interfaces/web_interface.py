"""
Streamlit Web Interface for Paper2Code

Provides a modern, user-friendly web interface for paper-to-code transformation.
"""

import streamlit as st
import asyncio
import tempfile
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

from ..processor import Paper2CodeProcessor
from ..utils.logger import get_logger

logger = get_logger(__name__)


class WebInterface:
    """Streamlit-based web interface for Paper2Code"""
    
    def __init__(self):
        """Initialize web interface"""
        self.setup_page_config()
        self.initialize_session_state()
    
    def setup_page_config(self):
        """Configure Streamlit page"""
        st.set_page_config(
            page_title="Paper2Code - Research to Implementation",
            page_icon="🧬",
            layout="wide",
            initial_sidebar_state="expanded",
            menu_items={
                'Get Help': 'https://github.com/paper2code/paper2code-standalone',
                'Report a bug': "https://github.com/paper2code/paper2code-standalone/issues",
                'About': "Transform research papers into working code using AI"
            }
        )
    
    def initialize_session_state(self):
        """Initialize Streamlit session state"""
        if 'processing_history' not in st.session_state:
            st.session_state.processing_history = []
        
        if 'current_result' not in st.session_state:
            st.session_state.current_result = None
        
        if 'processing_complete' not in st.session_state:
            st.session_state.processing_complete = False
        
        if 'processor' not in st.session_state:
            st.session_state.processor = Paper2CodeProcessor()
    
    def run(self):
        """Run the web interface"""
        self.render_header()
        self.render_sidebar()
        self.render_main_content()
        self.render_footer()
    
    def render_header(self):
        """Render page header"""
        st.markdown("""
        <div style="text-align: center; padding: 2rem 0;">
            <h1 style="color: #1f77b4; margin-bottom: 0.5rem;">
                🧬 Paper2Code
            </h1>
            <p style="font-size: 1.2rem; color: #666; margin-bottom: 2rem;">
                Transform Research Papers into Working Code
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    def render_sidebar(self):
        """Render sidebar with controls"""
        with st.sidebar:
            st.markdown("### ⚙️ Configuration")
            
            # Processing mode
            mode = st.selectbox(
                "🎯 Processing Mode",
                ["comprehensive", "fast"],
                index=0,
                help="Comprehensive: Full analysis with repository search\nFast: Quick processing without indexing"
            )
            
            # Document segmentation
            enable_segmentation = st.checkbox(
                "📖 Smart Document Segmentation",
                value=True,
                help="Automatically handle large documents by segmenting them"
            )
            
            if enable_segmentation:
                segmentation_threshold = st.slider(
                    "📏 Segmentation Threshold",
                    min_value=10000,
                    max_value=100000,
                    value=50000,
                    step=5000,
                    help="Document size (in characters) to trigger segmentation"
                )
            else:
                segmentation_threshold = 50000
            
            # Output directory
            output_dir = st.text_input(
                "📁 Output Directory",
                value="./output",
                help="Directory where generated code will be saved"
            )
            
            # Advanced options
            with st.expander("🔧 Advanced Options"):
                verbose_logging = st.checkbox("📝 Verbose Logging", value=False)
                generate_tests = st.checkbox("🧪 Generate Tests", value=True)
                generate_docs = st.checkbox("📚 Generate Documentation", value=True)
            
            # Store configuration in session state
            st.session_state.config = {
                'mode': mode,
                'enable_segmentation': enable_segmentation,
                'segmentation_threshold': segmentation_threshold,
                'output_dir': output_dir,
                'verbose_logging': verbose_logging,
                'generate_tests': generate_tests,
                'generate_docs': generate_docs
            }
            
            # Processing history
            if st.session_state.processing_history:
                st.markdown("### 📜 Recent Processing")
                for entry in st.session_state.processing_history[-3:]:
                    timestamp = datetime.fromisoformat(entry['timestamp'])
                    status_icon = "✅" if entry['success'] else "❌"
                    
                    with st.expander(f"{status_icon} {timestamp.strftime('%H:%M')}"):
                        st.text(f"Input: {Path(entry['input_source']).name}")
                        st.text(f"Mode: {entry['mode'].title()}")
                        st.text(f"Duration: {entry['duration']:.1f}s")
                        if entry['success']:
                            st.text(f"Files: {entry.get('file_count', 0)}")
    
    def render_main_content(self):
        """Render main content area"""
        # Input method selector
        input_method = st.radio(
            "📖 Choose Input Method:",
            ["📄 Upload File", "🌐 Enter URL"],
            horizontal=True
        )
        
        input_source = None
        
        if input_method == "📄 Upload File":
            input_source = self.render_file_upload()
        else:
            input_source = self.render_url_input()
        
        # Processing section
        if input_source:
            self.render_processing_section(input_source)
        
        # Results section
        if st.session_state.current_result:
            self.render_results_section()
    
    def render_file_upload(self) -> Optional[str]:
        """Render file upload interface"""
        st.markdown("### 📄 Upload Research Paper")
        
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['pdf', 'docx', 'doc', 'txt', 'md', 'html', 'htm'],
            help="Supported formats: PDF, Word, Text, Markdown, HTML"
        )
        
        if uploaded_file:
            # Show file info
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("📄 File Name", uploaded_file.name)
            with col2:
                st.metric("📊 Size", f"{uploaded_file.size / 1024:.1f} KB")
            with col3:
                st.metric("📋 Type", uploaded_file.type)
            
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                tmp_file.write(uploaded_file.read())
                return tmp_file.name
        
        return None
    
    def render_url_input(self) -> Optional[str]:
        """Render URL input interface"""
        st.markdown("### 🌐 Enter Paper URL")
        
        url = st.text_input(
            "Paper URL",
            placeholder="https://arxiv.org/pdf/2301.12345.pdf",
            help="Enter URL to a research paper (arXiv, direct PDF links, etc.)"
        )
        
        if url:
            if url.startswith(('http://', 'https://')):
                st.success(f"✅ Valid URL: {url}")
                return url
            else:
                st.error("❌ Please enter a valid URL starting with http:// or https://")
        
        return None
    
    def render_processing_section(self, input_source: str):
        """Render processing section"""
        st.markdown("### 🚀 Process Paper")
        
        # Display processing configuration
        config = st.session_state.config
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info(f"🎯 Mode: **{config['mode'].title()}**")
        with col2:
            st.info(f"📖 Segmentation: **{'Enabled' if config['enable_segmentation'] else 'Disabled'}**")
        with col3:
            st.info(f"📁 Output: **{config['output_dir']}**")
        
        # Processing button
        if st.button("🚀 Start Processing", type="primary", use_container_width=True):
            self.process_paper(input_source)
    
    def process_paper(self, input_source: str):
        """Process the paper"""
        config = st.session_state.config
        
        # Show processing status
        status_container = st.empty()
        progress_bar = st.progress(0)
        progress_text = st.empty()
        
        try:
            status_container.info("🔄 Initializing processing...")
            
            # Create progress callback
            def progress_callback(progress: int, message: str):
                progress_bar.progress(progress / 100)
                progress_text.text(f"[{progress}%] {message}")
            
            # Process paper (we need to run this in async context)
            result = asyncio.run(self._process_paper_async(
                input_source, config, progress_callback
            ))
            
            # Store result
            st.session_state.current_result = result
            st.session_state.processing_complete = True
            
            # Add to history
            st.session_state.processing_history.append({
                'timestamp': datetime.now().isoformat(),
                'input_source': input_source,
                'mode': config['mode'],
                'success': result.success,
                'duration': result.processing_time or 0,
                'file_count': len(result.files or []) if result.success else 0,
                'output_path': result.output_path if result.success else None
            })
            
            if result.success:
                status_container.success("✅ Processing completed successfully!")
                progress_bar.progress(100)
                progress_text.text("✅ Complete!")
            else:
                status_container.error(f"❌ Processing failed: {result.error}")
                progress_bar.progress(0)
                progress_text.text("❌ Failed")
        
        except Exception as e:
            status_container.error(f"❌ Error: {str(e)}")
            st.session_state.current_result = None
            logger.error(f"Processing error: {e}")
    
    async def _process_paper_async(self, input_source: str, config: Dict[str, Any], progress_callback):
        """Async wrapper for paper processing"""
        processor = st.session_state.processor
        
        return await processor.process_paper(
            input_source=input_source,
            output_dir=Path(config['output_dir']),
            mode=config['mode'],
            enable_segmentation=config['enable_segmentation'],
            segmentation_threshold=config['segmentation_threshold'],
            progress_callback=progress_callback
        )
    
    def render_results_section(self):
        """Render results section"""
        result = st.session_state.current_result
        
        if not result:
            return
        
        st.markdown("### 📊 Processing Results")
        
        if result.success:
            # Success metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("✅ Status", "SUCCESS")
            with col2:
                st.metric("📄 Files Generated", len(result.files or []))
            with col3:
                st.metric("⏱️ Processing Time", f"{result.processing_time:.1f}s")
            with col4:
                st.metric("📁 Output Directory", Path(result.output_path).name)
            
            # Output location
            st.success(f"🎉 Code generated successfully!")
            st.info(f"📁 **Output Location:** `{result.output_path}`")
            
            # Generated files
            if result.files:
                st.markdown("#### 📋 Generated Files")
                
                # Group files by type
                files_by_type = {
                    'Python Files': [],
                    'Configuration': [],
                    'Documentation': [],
                    'Tests': [],
                    'Other': []
                }
                
                for file_path in result.files:
                    file_name = Path(file_path).name
                    rel_path = str(Path(file_path).relative_to(Path(result.output_path)))
                    
                    if file_name.endswith('.py'):
                        files_by_type['Python Files'].append(rel_path)
                    elif file_name in ['requirements.txt', 'setup.py', 'config.yaml']:
                        files_by_type['Configuration'].append(rel_path)
                    elif file_name.endswith(('.md', '.rst', '.txt')) and 'doc' in file_name.lower():
                        files_by_type['Documentation'].append(rel_path)
                    elif 'test' in file_name.lower():
                        files_by_type['Tests'].append(rel_path)
                    else:
                        files_by_type['Other'].append(rel_path)
                
                # Display files in expandable sections
                for category, files in files_by_type.items():
                    if files:
                        with st.expander(f"{category} ({len(files)} files)"):
                            for file_path in files:
                                st.text(f"📄 {file_path}")
            
            # Action buttons
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("📂 View Output Folder", use_container_width=True):
                    st.info(f"Open this folder on your system: `{result.output_path}`")
            
            with col2:
                if st.button("📋 Copy Output Path", use_container_width=True):
                    st.code(result.output_path)
                    st.success("Path displayed above - copy from the code block")
            
            with col3:
                if st.button("🔄 Process Another Paper", use_container_width=True):
                    st.session_state.current_result = None
                    st.session_state.processing_complete = False
                    st.experimental_rerun()
        
        else:
            # Error display
            st.error(f"❌ **Processing Failed**")
            st.error(f"💥 **Error:** {result.error}")
            
            if result.processing_time:
                st.info(f"⏱️ **Duration:** {result.processing_time:.1f} seconds")
            
            # Troubleshooting tips
            with st.expander("🔧 Troubleshooting Tips"):
                st.markdown("""
                **Common Issues:**
                1. **API Key Missing**: Check your `secrets.yaml` file
                2. **Large Document**: Try enabling segmentation or fast mode
                3. **Network Issues**: Check internet connection for URL downloads
                4. **File Format**: Ensure file is a supported format (PDF, DOCX, etc.)
                
                **Solutions:**
                - Try Fast mode for quicker processing
                - Enable document segmentation for large papers
                - Check the logs for detailed error information
                """)
            
            if st.button("🔄 Try Again", use_container_width=True):
                st.session_state.current_result = None
                st.session_state.processing_complete = False
                st.experimental_rerun()
    
    def render_footer(self):
        """Render page footer"""
        st.markdown("---")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### 🔗 Links")
            st.markdown("[📚 Documentation](https://github.com/paper2code/paper2code-standalone)")
            st.markdown("[🐛 Report Issues](https://github.com/paper2code/paper2code-standalone/issues)")
        
        with col2:
            st.markdown("### 💡 Tips")
            st.markdown("• Use **Comprehensive** mode for best results")
            st.markdown("• Enable **Segmentation** for large papers")
            st.markdown("• Check generated **tests** to validate code")
        
        with col3:
            st.markdown("### ⚙️ Configuration")
            st.markdown("Edit `secrets.yaml` to add API keys")
            st.markdown("Adjust processing options in sidebar")
        
        st.markdown("""
        <div style="text-align: center; padding: 2rem; color: #666;">
            <p>🧬 Paper2Code Standalone - Transform research into reality</p>
        </div>
        """, unsafe_allow_html=True)


def run_web_interface():
    """Main function to run the web interface"""
    interface = WebInterface()
    interface.run()


if __name__ == "__main__":
    run_web_interface()
