"""
Jupyter Notebook Interface for Paper2Code

Provides interactive widgets and tools for Jupyter notebook environments.
"""

from IPython.display import display, HTML, clear_output
import ipywidgets as widgets
from pathlib import Path
import asyncio
import json
from datetime import datetime
from typing import Dict, Any, Optional

from ..processor import Paper2CodeProcessor
from ..utils.logger import get_logger

logger = get_logger(__name__)


class NotebookInterface:
    """Interactive Jupyter notebook interface for Paper2Code"""
    
    def __init__(self):
        """Initialize notebook interface"""
        self.processor = Paper2CodeProcessor()
        self.current_result = None
        self.processing_history = []
        
        # Create UI components
        self.create_widgets()
        self.setup_layout()
    
    def create_widgets(self):
        """Create interactive widgets"""
        # Header
        self.header = widgets.HTML("""
        <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; color: white; margin-bottom: 20px;">
            <h1 style="margin: 0; font-size: 2.5em;">🧬 Paper2Code</h1>
            <p style="margin: 10px 0 0 0; font-size: 1.2em;">Transform Research Papers into Working Code</p>
        </div>
        """)
        
        # Input method selector
        self.input_method = widgets.RadioButtons(
            options=[('📄 Upload File', 'file'), ('🌐 Enter URL', 'url')],
            value='file',
            description='Input Method:',
            style={'description_width': 'initial'}
        )
        
        # File upload
        self.file_upload = widgets.FileUpload(
            accept='.pdf,.docx,.doc,.txt,.md,.html,.htm',
            multiple=False,
            description='Select Paper:',
            style={'description_width': 'initial'}
        )
        
        # URL input
        self.url_input = widgets.Text(
            placeholder='https://arxiv.org/pdf/2301.12345.pdf',
            description='Paper URL:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='500px')
        )
        
        # Processing options
        self.mode_select = widgets.Dropdown(
            options=[
                ('🧠 Comprehensive (Recommended)', 'comprehensive'),
                ('⚡ Fast Mode', 'fast')
            ],
            value='comprehensive',
            description='Mode:',
            style={'description_width': 'initial'}
        )
        
        self.segmentation_toggle = widgets.Checkbox(
            value=True,
            description='📖 Smart Document Segmentation',
            style={'description_width': 'initial'}
        )
        
        self.segmentation_threshold = widgets.IntSlider(
            value=50000,
            min=10000,
            max=100000,
            step=5000,
            description='Threshold:',
            style={'description_width': 'initial'}
        )
        
        self.output_dir = widgets.Text(
            value='./output',
            description='Output Dir:',
            style={'description_width': 'initial'}
        )
        
        # Process button
        self.process_button = widgets.Button(
            description='🚀 Process Paper',
            button_style='primary',
            layout=widgets.Layout(width='200px', height='40px')
        )
        
        # Progress bar
        self.progress_bar = widgets.IntProgress(
            value=0,
            min=0,
            max=100,
            description='Progress:',
            bar_style='info',
            style={'bar_color': '#1f77b4'},
            layout=widgets.Layout(width='500px')
        )
        
        # Status output
        self.status_output = widgets.Output()
        
        # Results display
        self.results_output = widgets.Output()
        
        # History display
        self.history_output = widgets.Output()
        
        # Setup event handlers
        self.input_method.observe(self.on_input_method_change, names='value')
        self.segmentation_toggle.observe(self.on_segmentation_toggle, names='value')
        self.process_button.on_click(self.on_process_click)
        
        # Initially hide URL input
        self.url_input.layout.display = 'none'
    
    def setup_layout(self):
        """Setup widget layout"""
        # Configuration section
        config_section = widgets.VBox([
            widgets.HTML("<h3>⚙️ Configuration</h3>"),
            self.mode_select,
            self.segmentation_toggle,
            self.segmentation_threshold,
            self.output_dir
        ])
        
        # Input section
        input_section = widgets.VBox([
            widgets.HTML("<h3>📖 Input Source</h3>"),
            self.input_method,
            self.file_upload,
            self.url_input
        ])
        
        # Processing section
        processing_section = widgets.VBox([
            widgets.HTML("<h3>🚀 Processing</h3>"),
            self.process_button,
            self.progress_bar,
            self.status_output
        ])
        
        # Main layout
        self.main_layout = widgets.VBox([
            self.header,
            widgets.HBox([input_section, config_section]),
            processing_section,
            widgets.HTML("<h3>📊 Results</h3>"),
            self.results_output,
            widgets.HTML("<h3>📜 Processing History</h3>"),
            self.history_output
        ])
    
    def display(self):
        """Display the interface"""
        display(self.main_layout)
        self.update_history_display()
    
    def on_input_method_change(self, change):
        """Handle input method change"""
        if change['new'] == 'file':
            self.file_upload.layout.display = None
            self.url_input.layout.display = 'none'
        else:
            self.file_upload.layout.display = 'none'
            self.url_input.layout.display = None
    
    def on_segmentation_toggle(self, change):
        """Handle segmentation toggle"""
        self.segmentation_threshold.disabled = not change['new']
    
    def on_process_click(self, button):
        """Handle process button click"""
        # Clear previous results
        with self.status_output:
            clear_output()
        with self.results_output:
            clear_output()
        
        # Get input source
        input_source = self.get_input_source()
        if not input_source:
            with self.status_output:
                print("❌ Please select a file or enter a URL")
            return
        
        # Disable process button during processing
        self.process_button.disabled = True
        self.process_button.description = '⏳ Processing...'
        
        # Start processing
        try:
            # Run processing in asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(self.process_paper_async(input_source))
            loop.close()
            
            # Display results
            self.display_results(result)
            
        except Exception as e:
            with self.status_output:
                print(f"❌ Error: {str(e)}")
        finally:
            # Re-enable process button
            self.process_button.disabled = False
            self.process_button.description = '🚀 Process Paper'
    
    def get_input_source(self) -> Optional[str]:
        """Get input source from widgets"""
        if self.input_method.value == 'file':
            if self.file_upload.value:
                # Save uploaded file and return path
                uploaded_file = list(self.file_upload.value.values())[0]
                temp_path = Path(f"temp_{uploaded_file['metadata']['name']}")
                temp_path.write_bytes(uploaded_file['content'])
                return str(temp_path)
            return None
        else:
            url = self.url_input.value.strip()
            return url if url.startswith(('http://', 'https://')) else None
    
    async def process_paper_async(self, input_source: str):
        """Process paper asynchronously"""
        # Progress callback
        def progress_callback(progress: int, message: str):
            self.progress_bar.value = progress
            with self.status_output:
                clear_output(wait=True)
                print(f"[{progress:3d}%] {message}")
        
        # Get configuration
        config = {
            'mode': self.mode_select.value,
            'enable_segmentation': self.segmentation_toggle.value,
            'segmentation_threshold': self.segmentation_threshold.value
        }
        
        # Process paper
        result = await self.processor.process_paper(
            input_source=input_source,
            output_dir=Path(self.output_dir.value),
            mode=config['mode'],
            enable_segmentation=config['enable_segmentation'],
            segmentation_threshold=config['segmentation_threshold']
        )
        
        # Add to history
        self.processing_history.append({
            'timestamp': datetime.now().isoformat(),
            'input_source': input_source,
            'mode': config['mode'],
            'success': result.success,
            'duration': result.processing_time or 0,
            'file_count': len(result.files or []) if result.success else 0,
            'output_path': result.output_path if result.success else None,
            'error': result.error if not result.success else None
        })
        
        return result
    
    def display_results(self, result):
        """Display processing results"""
        with self.results_output:
            clear_output()
            
            if result.success:
                # Success display
                display(HTML(f"""
                <div style="padding: 15px; background-color: #d4edda; border: 1px solid #c3e6cb; border-radius: 5px; color: #155724;">
                    <h4 style="margin: 0 0 10px 0;">✅ Processing Completed Successfully!</h4>
                    <p><strong>📁 Output:</strong> <code>{result.output_path}</code></p>
                    <p><strong>📄 Files Generated:</strong> {len(result.files or [])}</p>
                    <p><strong>⏱️ Processing Time:</strong> {result.processing_time:.1f} seconds</p>
                </div>
                """))
                
                # File list
                if result.files:
                    file_list_html = "<h5>📋 Generated Files:</h5><ul>"
                    for file_path in result.files[:10]:  # Show first 10
                        rel_path = Path(file_path).relative_to(Path(result.output_path))
                        file_list_html += f"<li><code>{rel_path}</code></li>"
                    
                    if len(result.files) > 10:
                        file_list_html += f"<li><em>... and {len(result.files) - 10} more files</em></li>"
                    
                    file_list_html += "</ul>"
                    display(HTML(file_list_html))
                
                # Code preview (show main.py if exists)
                main_py_path = Path(result.output_path) / "main.py"
                if main_py_path.exists():
                    code_content = main_py_path.read_text(encoding='utf-8')
                    preview_length = 1000
                    
                    if len(code_content) > preview_length:
                        preview_code = code_content[:preview_length] + "\n# ... (truncated)"
                    else:
                        preview_code = code_content
                    
                    display(HTML("<h5>👀 Code Preview (main.py):</h5>"))
                    
                    # Create code widget
                    code_widget = widgets.HTML(f"""
                    <div style="background-color: #f8f9fa; padding: 10px; border-radius: 5px; border: 1px solid #e9ecef;">
                        <pre style="margin: 0; font-family: 'Monaco', 'Consolas', monospace; font-size: 12px; white-space: pre-wrap;">{preview_code}</pre>
                    </div>
                    """)
                    display(code_widget)
            
            else:
                # Error display
                display(HTML(f"""
                <div style="padding: 15px; background-color: #f8d7da; border: 1px solid #f5c6cb; border-radius: 5px; color: #721c24;">
                    <h4 style="margin: 0 0 10px 0;">❌ Processing Failed</h4>
                    <p><strong>💥 Error:</strong> {result.error}</p>
                    <p><strong>⏱️ Duration:</strong> {result.processing_time:.1f} seconds</p>
                </div>
                """))
                
                # Troubleshooting tips
                display(HTML("""
                <div style="padding: 10px; background-color: #fff3cd; border: 1px solid #ffeaa7; border-radius: 5px; color: #856404; margin-top: 10px;">
                    <h5>🔧 Troubleshooting Tips:</h5>
                    <ul>
                        <li>Check if API keys are configured in <code>secrets.yaml</code></li>
                        <li>Try Fast mode for quicker processing</li>
                        <li>Enable document segmentation for large papers</li>
                        <li>Verify file format is supported</li>
                    </ul>
                </div>
                """))
        
        # Update progress bar
        self.progress_bar.value = 100 if result.success else 0
        self.progress_bar.bar_style = 'success' if result.success else 'danger'
        
        # Update history
        self.update_history_display()
    
    def update_history_display(self):
        """Update processing history display"""
        with self.history_output:
            clear_output()
            
            if not self.processing_history:
                print("📝 No processing history yet.")
                return
            
            # Show last 5 entries
            recent_history = self.processing_history[-5:]
            
            for i, entry in enumerate(reversed(recent_history), 1):
                timestamp = datetime.fromisoformat(entry['timestamp'])
                status_icon = "✅" if entry['success'] else "❌"
                
                input_name = Path(entry['input_source']).name if not entry['input_source'].startswith('http') else entry['input_source'][:40] + "..."
                
                history_html = f"""
                <div style="padding: 10px; margin: 5px 0; border: 1px solid #ddd; border-radius: 5px; background-color: #f8f9fa;">
                    <strong>{status_icon} {timestamp.strftime('%Y-%m-%d %H:%M')}</strong><br/>
                    📖 {input_name}<br/>
                    🎯 {entry['mode'].title()} mode • ⏱️ {entry['duration']:.1f}s
                """
                
                if entry['success']:
                    history_html += f"<br/>📄 {entry.get('file_count', 0)} files generated"
                else:
                    history_html += f"<br/>💥 {entry.get('error', 'Unknown error')[:50]}..."
                
                history_html += "</div>"
                
                display(HTML(history_html))
            
            if len(self.processing_history) > 5:
                display(HTML(f"<p><em>... and {len(self.processing_history) - 5} more entries</em></p>"))
    
    def create_quick_start_example(self):
        """Create a quick start example cell"""
        example_code = '''
# Paper2Code Quick Start Example
from paper2code.interfaces import NotebookInterface

# Create and display the interface
interface = NotebookInterface()
interface.display()

# Alternative: Direct API usage
from paper2code import Paper2CodeProcessor
import asyncio

async def process_paper_direct():
    processor = Paper2CodeProcessor()
    result = await processor.process_paper("path/to/paper.pdf")
    
    if result.success:
        print(f"✅ Success! Code generated at: {result.output_path}")
    else:
        print(f"❌ Failed: {result.error}")

# Run direct processing
# asyncio.run(process_paper_direct())
        '''
        
        return example_code


def create_notebook_interface():
    """Create and return notebook interface"""
    return NotebookInterface()


def show_quick_start():
    """Display quick start guide"""
    interface = NotebookInterface()
    
    display(HTML("""
    <div style="padding: 20px; background-color: #e3f2fd; border-radius: 10px; margin: 10px 0;">
        <h2>🧬 Paper2Code - Jupyter Notebook Interface</h2>
        <p>Transform research papers into working code directly in your notebook!</p>
        
        <h4>🚀 Quick Start:</h4>
        <ol>
            <li>Upload a research paper or enter a URL</li>
            <li>Choose your processing mode (Comprehensive recommended)</li>
            <li>Configure options as needed</li>
            <li>Click "Process Paper" and wait for results</li>
            <li>View generated code and files</li>
        </ol>
        
        <h4>💡 Tips:</h4>
        <ul>
            <li>Use <strong>Comprehensive mode</strong> for best results</li>
            <li>Enable <strong>Smart Segmentation</strong> for large papers</li>
            <li>Check the <strong>generated tests</strong> to validate code</li>
            <li>Explore the <strong>output directory</strong> for all files</li>
        </ul>
    </div>
    """))
    
    interface.display()


# Export main function
__all__ = ["NotebookInterface", "create_notebook_interface", "show_quick_start"]
