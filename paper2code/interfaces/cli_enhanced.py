"""
Enhanced CLI Interactive Interface for Paper2Code

Provides a rich, user-friendly command-line interface with features like:
- File browser
- Configuration management  
- Real-time progress tracking
- Result visualization
- History management
"""

import asyncio
import os
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

from ..processor import Paper2CodeProcessor
from ..utils.logger import get_logger

logger = get_logger(__name__)


class EnhancedCLIInterface:
    """Enhanced command-line interface for Paper2Code"""
    
    def __init__(self):
        """Initialize enhanced CLI"""
        self.processor = Paper2CodeProcessor()
        self.history = []
        self.config = {}
        self.current_dir = Path.cwd()
    
    async def run(self):
        """Run the enhanced interactive interface"""
        self.print_banner()
        await self.load_history()
        
        try:
            while True:
                choice = self.show_main_menu()
                
                if choice == '1':
                    await self.process_paper_workflow()
                elif choice == '2':
                    self.browse_files()
                elif choice == '3':
                    self.view_history()
                elif choice == '4':
                    self.configure_settings()
                elif choice == '5':
                    self.show_help()
                elif choice == '6':
                    print("\n👋 Thank you for using Paper2Code!")
                    break
                else:
                    print("❌ Invalid choice. Please try again.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
        except Exception as e:
            print(f"\n❌ Error: {e}")
        finally:
            await self.save_history()
    
    def print_banner(self):
        """Print enhanced startup banner"""
        banner = """
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║    🧬 Paper2Code - Enhanced Interactive Interface           ║
║                                                              ║
║    Transform Research Papers → Working Code                 ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝

🎯 Welcome to the intelligent paper-to-code transformation system!
📚 Supported formats: PDF, DOCX, TXT, MD, HTML, URLs
⚡ Choose your processing mode for optimal results
"""
        print(banner)
    
    def show_main_menu(self) -> str:
        """Show main menu and get user choice"""
        print("\n" + "="*60)
        print("📋 MAIN MENU")
        print("="*60)
        print("1. 🚀 Process Research Paper")
        print("2. 📁 Browse Files")
        print("3. 📜 View Processing History")
        print("4. ⚙️  Configure Settings")
        print("5. ❓ Help & Documentation")
        print("6. 🚪 Exit")
        print("-"*60)
        
        return input("Choose an option (1-6): ").strip()
    
    async def process_paper_workflow(self):
        """Complete paper processing workflow"""
        print("\n🚀 PAPER PROCESSING WORKFLOW")
        print("="*50)
        
        # Step 1: Get input source
        input_source = self.get_input_source()
        if not input_source:
            return
        
        # Step 2: Configure processing options
        options = self.configure_processing_options()
        
        # Step 3: Set output location
        output_dir = self.get_output_directory()
        
        # Step 4: Review and confirm
        if not self.confirm_processing(input_source, options, output_dir):
            return
        
        # Step 5: Process the paper
        await self.execute_processing(input_source, options, output_dir)
    
    def get_input_source(self) -> Optional[str]:
        """Get input source from user"""
        print("\n📖 SELECT INPUT SOURCE")
        print("-" * 30)
        print("1. 📄 Local file")
        print("2. 🌐 URL")
        print("3. 📁 Browse files")
        print("4. ⬅️  Back to main menu")
        
        choice = input("\nChoose input method (1-4): ").strip()
        
        if choice == '1':
            return self.get_local_file()
        elif choice == '2':
            return self.get_url_input()
        elif choice == '3':
            return self.browse_and_select_file()
        elif choice == '4':
            return None
        else:
            print("❌ Invalid choice")
            return self.get_input_source()
    
    def get_local_file(self) -> Optional[str]:
        """Get local file path from user"""
        print("\n📄 Enter file path:")
        print("💡 Tip: Use Tab for auto-completion, drag & drop file, or type path")
        
        file_path = input("File path: ").strip().strip('"\'')
        
        if not file_path:
            return None
        
        path = Path(file_path)
        if not path.exists():
            print(f"❌ File not found: {file_path}")
            retry = input("🔄 Try again? (y/n): ").strip().lower()
            return self.get_local_file() if retry == 'y' else None
        
        if not path.is_file():
            print(f"❌ Not a file: {file_path}")
            return self.get_local_file()
        
        # Show file info
        size_mb = path.stat().st_size / 1024 / 1024
        print(f"✅ File selected: {path.name}")
        print(f"📊 Size: {size_mb:.2f} MB")
        print(f"📅 Modified: {datetime.fromtimestamp(path.stat().st_mtime).strftime('%Y-%m-%d %H:%M')}")
        
        return str(path)
    
    def get_url_input(self) -> Optional[str]:
        """Get URL input from user"""
        print("\n🌐 Enter paper URL:")
        print("💡 Examples: arXiv, research gate, direct PDF links")
        
        url = input("URL: ").strip()
        
        if not url:
            return None
        
        if not url.startswith(('http://', 'https://')):
            print("❌ Invalid URL format")
            retry = input("🔄 Try again? (y/n): ").strip().lower()
            return self.get_url_input() if retry == 'y' else None
        
        print(f"✅ URL selected: {url}")
        return url
    
    def browse_and_select_file(self) -> Optional[str]:
        """Browse and select file"""
        return self.browse_files(select_mode=True)
    
    def browse_files(self, select_mode: bool = False) -> Optional[str]:
        """Browse files in current directory"""
        print(f"\n📁 BROWSING: {self.current_dir}")
        print("-" * 50)
        
        try:
            items = list(self.current_dir.iterdir())
            dirs = [item for item in items if item.is_dir() and not item.name.startswith('.')]
            files = [item for item in items if item.is_file() and not item.name.startswith('.')]
            
            # Sort items
            dirs.sort(key=lambda x: x.name.lower())
            files.sort(key=lambda x: x.name.lower())
            
            # Display directories
            print("📁 Directories:")
            for i, dir_item in enumerate(dirs[:10], 1):  # Show max 10
                print(f"  {i:2}. 📁 {dir_item.name}/")
            
            if len(dirs) > 10:
                print(f"     ... and {len(dirs) - 10} more directories")
            
            # Display files
            print("\n📄 Files:")
            supported_extensions = {'.pdf', '.docx', '.doc', '.txt', '.md', '.html', '.htm'}
            
            file_count = 0
            for file_item in files:
                if file_count >= 15:  # Show max 15 files
                    break
                
                size_mb = file_item.stat().st_size / 1024 / 1024
                icon = "📄" if file_item.suffix.lower() in supported_extensions else "📋"
                
                print(f"  {len(dirs) + file_count + 1:2}. {icon} {file_item.name} ({size_mb:.1f} MB)")
                file_count += 1
            
            if len(files) > 15:
                print(f"     ... and {len(files) - 15} more files")
            
            # Navigation options
            print(f"\n🔝 Navigation:")
            if self.current_dir.parent != self.current_dir:
                print(f"  {len(dirs) + file_count + 1}. ⬆️  Parent directory")
            
            print(f"  0. ⬅️  Back")
            
            if select_mode:
                print("\n💡 Enter number to select file, or navigate to find your file")
            else:
                print("\n💡 Enter number to navigate, or 0 to go back")
            
            choice = input("\nYour choice: ").strip()
            
            if choice == '0':
                return None
            
            try:
                choice_num = int(choice)
                
                # Directory navigation
                if 1 <= choice_num <= len(dirs):
                    selected_dir = dirs[choice_num - 1]
                    self.current_dir = selected_dir
                    return self.browse_files(select_mode)
                
                # File selection
                elif len(dirs) < choice_num <= len(dirs) + len(files):
                    file_index = choice_num - len(dirs) - 1
                    if file_index < len(files):
                        selected_file = files[file_index]
                        if select_mode:
                            return str(selected_file)
                        else:
                            print(f"📄 Selected: {selected_file.name}")
                            input("Press Enter to continue...")
                            return None
                
                # Parent directory
                elif choice_num == len(dirs) + file_count + 1 and self.current_dir.parent != self.current_dir:
                    self.current_dir = self.current_dir.parent
                    return self.browse_files(select_mode)
                
                else:
                    print("❌ Invalid choice")
                    return self.browse_files(select_mode)
                    
            except ValueError:
                print("❌ Please enter a valid number")
                return self.browse_files(select_mode)
                
        except PermissionError:
            print(f"❌ Permission denied to access {self.current_dir}")
            return None
    
    def configure_processing_options(self) -> Dict[str, Any]:
        """Configure processing options"""
        print("\n⚙️  PROCESSING OPTIONS")
        print("-" * 30)
        
        # Processing mode
        print("🎯 Processing Mode:")
        print("1. 🧠 Comprehensive (recommended) - Full analysis with repository search")
        print("2. ⚡ Fast - Quick processing without repository indexing")
        
        mode_choice = input("\nChoose mode (1-2, default=1): ").strip() or '1'
        mode = "comprehensive" if mode_choice == '1' else "fast"
        
        # Document segmentation
        print(f"\n📖 Document Segmentation:")
        print("1. ✅ Enabled (recommended) - Smart handling of large documents")
        print("2. ❌ Disabled - Process entire document at once")
        
        seg_choice = input("\nChoose option (1-2, default=1): ").strip() or '1'
        enable_segmentation = seg_choice == '1'
        
        # Advanced options
        advanced = input("\n🔧 Configure advanced options? (y/n, default=n): ").strip().lower()
        
        options = {
            'mode': mode,
            'enable_segmentation': enable_segmentation,
        }
        
        if advanced == 'y':
            # Segmentation threshold
            if enable_segmentation:
                threshold = input("📏 Segmentation threshold (characters, default=50000): ").strip()
                try:
                    options['segmentation_threshold'] = int(threshold) if threshold else 50000
                except ValueError:
                    options['segmentation_threshold'] = 50000
        
        # Display selected options
        print(f"\n✅ CONFIGURED OPTIONS:")
        print(f"   🎯 Mode: {mode.title()}")
        print(f"   📖 Segmentation: {'Enabled' if enable_segmentation else 'Disabled'}")
        if 'segmentation_threshold' in options:
            print(f"   📏 Threshold: {options['segmentation_threshold']:,} chars")
        
        return options
    
    def get_output_directory(self) -> str:
        """Get output directory from user"""
        print(f"\n📁 OUTPUT DIRECTORY")
        print("-" * 25)
        print(f"💡 Current directory: {Path.cwd()}")
        
        default_output = "./output"
        output_dir = input(f"Output directory (default={default_output}): ").strip() or default_output
        
        # Validate/create output directory
        output_path = Path(output_dir)
        try:
            output_path.mkdir(parents=True, exist_ok=True)
            print(f"✅ Output directory: {output_path.resolve()}")
        except Exception as e:
            print(f"❌ Cannot create directory: {e}")
            return self.get_output_directory()
        
        return str(output_path)
    
    def confirm_processing(self, input_source: str, options: Dict[str, Any], output_dir: str) -> bool:
        """Confirm processing with user"""
        print(f"\n📋 PROCESSING SUMMARY")
        print("=" * 30)
        print(f"📖 Input: {Path(input_source).name if not input_source.startswith('http') else input_source}")
        print(f"🎯 Mode: {options['mode'].title()}")
        print(f"📁 Output: {output_dir}")
        print(f"📖 Segmentation: {'Yes' if options.get('enable_segmentation', True) else 'No'}")
        
        confirm = input(f"\n🚀 Proceed with processing? (y/n): ").strip().lower()
        return confirm == 'y'
    
    async def execute_processing(self, input_source: str, options: Dict[str, Any], output_dir: str):
        """Execute the paper processing"""
        print(f"\n🚀 PROCESSING STARTED")
        print("=" * 30)
        
        start_time = datetime.now()
        
        # Progress tracking
        last_progress = 0
        progress_bar_width = 50
        
        def progress_callback(progress: int, message: str):
            nonlocal last_progress
            last_progress = progress
            
            # Create progress bar
            filled = int(progress_bar_width * progress / 100)
            bar = "█" * filled + "▒" * (progress_bar_width - filled)
            
            print(f"\r[{progress:3d}%] {bar} {message}", end="", flush=True)
            
            if progress == 100:
                print()  # New line when complete
        
        try:
            # Process the paper
            result = await self.processor.process_paper(
                input_source=input_source,
                output_dir=Path(output_dir),
                mode=options['mode'],
                progress_callback=progress_callback,
                **{k: v for k, v in options.items() if k != 'mode'}
            )
            
            end_time = datetime.now()
            duration = end_time - start_time
            
            # Display results
            print(f"\n📊 PROCESSING RESULTS")
            print("=" * 30)
            
            if result.success:
                print(f"✅ Status: SUCCESS")
                print(f"📁 Output: {result.output_path}")
                print(f"📄 Files: {len(result.files or [])}")
                print(f"⏱️  Duration: {duration.total_seconds():.1f} seconds")
                
                if result.files:
                    print(f"\n📋 Generated Files:")
                    for file_path in result.files[:10]:  # Show first 10
                        rel_path = Path(file_path).relative_to(Path(result.output_path))
                        print(f"   📄 {rel_path}")
                    
                    if len(result.files) > 10:
                        print(f"   ... and {len(result.files) - 10} more files")
                
                # Add to history
                self.history.append({
                    'timestamp': start_time.isoformat(),
                    'input_source': input_source,
                    'output_path': result.output_path,
                    'mode': options['mode'],
                    'duration': duration.total_seconds(),
                    'success': True,
                    'file_count': len(result.files or [])
                })
                
                # Ask if user wants to open output directory
                open_dir = input(f"\n📂 Open output directory? (y/n): ").strip().lower()
                if open_dir == 'y':
                    self.open_directory(result.output_path)
            
            else:
                print(f"❌ Status: FAILED")
                print(f"💥 Error: {result.error}")
                print(f"⏱️  Duration: {duration.total_seconds():.1f} seconds")
                
                # Add to history
                self.history.append({
                    'timestamp': start_time.isoformat(),
                    'input_source': input_source,
                    'error': result.error,
                    'mode': options['mode'],
                    'duration': duration.total_seconds(),
                    'success': False
                })
        
        except Exception as e:
            print(f"\n❌ Processing failed: {e}")
        
        input(f"\nPress Enter to continue...")
    
    def view_history(self):
        """View processing history"""
        print(f"\n📜 PROCESSING HISTORY")
        print("=" * 40)
        
        if not self.history:
            print("📝 No processing history found.")
            input("Press Enter to continue...")
            return
        
        # Show recent history (last 10 entries)
        recent_history = self.history[-10:]
        
        for i, entry in enumerate(reversed(recent_history), 1):
            timestamp = datetime.fromisoformat(entry['timestamp'])
            status = "✅" if entry['success'] else "❌"
            duration = entry.get('duration', 0)
            
            input_name = Path(entry['input_source']).name if not entry['input_source'].startswith('http') else entry['input_source'][:50] + "..."
            
            print(f"\n{i:2}. {status} {timestamp.strftime('%Y-%m-%d %H:%M')}")
            print(f"    📖 {input_name}")
            print(f"    🎯 {entry.get('mode', 'unknown').title()} mode")
            print(f"    ⏱️  {duration:.1f}s")
            
            if entry['success']:
                print(f"    📁 {entry.get('output_path', 'unknown')}")
                print(f"    📄 {entry.get('file_count', 0)} files generated")
            else:
                print(f"    💥 {entry.get('error', 'Unknown error')}")
        
        if len(self.history) > 10:
            print(f"\n... and {len(self.history) - 10} more entries")
        
        input(f"\nPress Enter to continue...")
    
    def configure_settings(self):
        """Configure application settings"""
        print(f"\n⚙️  CONFIGURATION")
        print("=" * 25)
        print("1. 🔑 API Keys")
        print("2. 🎯 Default Processing Mode")
        print("3. 📁 Default Output Directory")
        print("4. 🧹 Clear History")
        print("5. ⬅️  Back")
        
        choice = input("\nChoose option (1-5): ").strip()
        
        if choice == '1':
            self.configure_api_keys()
        elif choice == '2':
            self.configure_default_mode()
        elif choice == '3':
            self.configure_default_output()
        elif choice == '4':
            self.clear_history()
        elif choice == '5':
            return
        else:
            print("❌ Invalid choice")
        
        self.configure_settings()
    
    def configure_api_keys(self):
        """Configure API keys"""
        print(f"\n🔑 API KEY CONFIGURATION")
        print("-" * 30)
        print("📝 Edit the configuration file to add your API keys:")
        
        config_file = Path("paper2code/config/secrets.yaml")
        print(f"📄 File: {config_file}")
        
        if config_file.exists():
            print("✅ Configuration file exists")
        else:
            print("❌ Configuration file not found")
            print("💡 Run the installation script to create it")
        
        input("Press Enter to continue...")
    
    def configure_default_mode(self):
        """Configure default processing mode"""
        current_mode = self.config.get('default_mode', 'comprehensive')
        
        print(f"\n🎯 DEFAULT PROCESSING MODE")
        print("-" * 35)
        print(f"📊 Current: {current_mode.title()}")
        print("1. 🧠 Comprehensive")
        print("2. ⚡ Fast")
        
        choice = input(f"\nChoose new default (1-2): ").strip()
        
        if choice == '1':
            self.config['default_mode'] = 'comprehensive'
            print("✅ Default mode set to Comprehensive")
        elif choice == '2':
            self.config['default_mode'] = 'fast'
            print("✅ Default mode set to Fast")
        else:
            print("❌ Invalid choice")
    
    def configure_default_output(self):
        """Configure default output directory"""
        current_output = self.config.get('default_output', './output')
        
        print(f"\n📁 DEFAULT OUTPUT DIRECTORY")
        print("-" * 35)
        print(f"📊 Current: {current_output}")
        
        new_output = input(f"New default directory (press Enter to keep current): ").strip()
        
        if new_output:
            try:
                Path(new_output).mkdir(parents=True, exist_ok=True)
                self.config['default_output'] = new_output
                print(f"✅ Default output set to: {new_output}")
            except Exception as e:
                print(f"❌ Error: {e}")
        else:
            print("📊 Keeping current setting")
    
    def clear_history(self):
        """Clear processing history"""
        if not self.history:
            print("📝 No history to clear")
            input("Press Enter to continue...")
            return
        
        confirm = input(f"🗑️  Clear {len(self.history)} history entries? (y/n): ").strip().lower()
        
        if confirm == 'y':
            self.history.clear()
            print("✅ History cleared")
        else:
            print("📊 History preserved")
        
        input("Press Enter to continue...")
    
    def show_help(self):
        """Show help and documentation"""
        help_text = """
📖 PAPER2CODE HELP & DOCUMENTATION
════════════════════════════════════

🎯 WHAT IS PAPER2CODE?
Paper2Code automatically transforms research papers into working code implementations
using advanced AI agents specialized in document analysis and code generation.

🚀 SUPPORTED FORMATS:
• PDF documents (.pdf)
• Word documents (.docx, .doc) 
• Text files (.txt, .md)
• HTML files (.html, .htm)
• URLs (arXiv, research papers, etc.)

⚙️ PROCESSING MODES:
• Comprehensive: Full analysis with repository search and indexing
• Fast: Quick processing without repository discovery (faster)

🧠 AI AGENTS:
• Document Analyzer: Extracts algorithms, formulas, and technical details
• Code Planner: Creates implementation roadmaps
• Code Generator: Produces working Python code
• Repository Finder: Discovers relevant GitHub repositories

📁 OUTPUT STRUCTURE:
output/paper_name/
├── main.py              # Main application
├── src/algorithms/      # Algorithm implementations  
├── src/models/          # Model components
├── src/utils/          # Utility functions
├── tests/              # Test suite
└── requirements.txt    # Dependencies

🔧 CONFIGURATION:
Edit paper2code/config/secrets.yaml to add your API keys:
• OpenAI API key (for GPT models)
• Anthropic API key (for Claude models)

💡 TIPS:
• Use Comprehensive mode for research papers
• Use Fast mode for quick prototyping
• Large papers benefit from document segmentation
• Check the output directory after processing
• Review generated tests to validate implementation

🆘 SUPPORT:
• GitHub: https://github.com/paper2code/paper2code-standalone
• Documentation: README.md in project root
"""
        print(help_text)
        input("Press Enter to continue...")
    
    def open_directory(self, path: str):
        """Open directory in file explorer"""
        try:
            import subprocess
            import platform
            
            if platform.system() == "Windows":
                subprocess.Popen(f'explorer "{path}"')
            elif platform.system() == "Darwin":  # macOS
                subprocess.Popen(["open", path])
            else:  # Linux
                subprocess.Popen(["xdg-open", path])
        except Exception as e:
            print(f"❌ Could not open directory: {e}")
    
    async def load_history(self):
        """Load processing history"""
        history_file = Path("paper2code_history.json")
        if history_file.exists():
            try:
                with open(history_file, 'r') as f:
                    data = json.load(f)
                    self.history = data.get('history', [])
                    self.config = data.get('config', {})
            except Exception as e:
                logger.warning(f"Could not load history: {e}")
    
    async def save_history(self):
        """Save processing history"""
        history_file = Path("paper2code_history.json")
        try:
            data = {
                'history': self.history,
                'config': self.config,
                'last_saved': datetime.now().isoformat()
            }
            with open(history_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save history: {e}")


# CLI entry point
async def run_enhanced_cli():
    """Run the enhanced CLI interface"""
    interface = EnhancedCLIInterface()
    await interface.run()


if __name__ == "__main__":
    asyncio.run(run_enhanced_cli())
