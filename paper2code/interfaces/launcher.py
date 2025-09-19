"""
Interface Launcher for Paper2Code

Allows users to choose and launch different interactive interfaces.
"""

import asyncio
import sys
import subprocess
from pathlib import Path


def print_banner():
    """Print interface selection banner"""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║    🧬 Paper2Code - Interactive Interface Launcher           ║
║                                                              ║
║    Choose your preferred interface:                          ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
"""
    print(banner)


def show_interface_menu():
    """Show interface selection menu"""
    print("🎯 Available Interfaces:")
    print("-" * 50)
    print("1. 🖥️  Enhanced CLI - Rich terminal interface")
    print("2. 🌐 Web Interface - Browser-based GUI")
    print("3. 📓 Jupyter Notebook - Interactive notebook widgets")
    print("4. ⚡ Basic CLI - Simple command-line interface")
    print("5. ❓ Help - Learn about each interface")
    print("6. 🚪 Exit")
    print("-" * 50)
    
    return input("Choose interface (1-6): ").strip()


def show_help():
    """Show interface descriptions and help"""
    help_text = """
🎯 PAPER2CODE INTERFACES GUIDE
════════════════════════════════

🖥️ ENHANCED CLI INTERFACE
• Rich terminal-based interface with menus and file browsing
• Features: File browser, progress tracking, history management
• Best for: Command-line users who want a full-featured experience
• Usage: python -m paper2code.interfaces.launcher (select option 1)

🌐 WEB INTERFACE
• Modern browser-based graphical interface
• Features: Drag & drop files, real-time progress, visual results
• Best for: Users who prefer graphical interfaces
• Requirements: Streamlit (pip install streamlit)
• Usage: streamlit run paper2code/interfaces/web_interface.py

📓 JUPYTER NOTEBOOK INTERFACE
• Interactive widgets for Jupyter environments
• Features: Inline results, code previews, notebook integration
• Best for: Researchers working in Jupyter notebooks
• Requirements: ipywidgets (pip install ipywidgets)
• Usage: Import and use in Jupyter notebooks

⚡ BASIC CLI INTERFACE
• Simple command-line interface
• Features: Direct processing, minimal interaction
• Best for: Scripts, automation, CI/CD pipelines
• Usage: python -m paper2code --interactive

🔧 COMPARISON TABLE
                │ Enhanced CLI │ Web GUI │ Notebook │ Basic CLI │
────────────────┼──────────────┼─────────┼──────────┼───────────┤
Ease of Use     │     ★★★★     │  ★★★★★  │   ★★★★   │    ★★     │
Features        │     ★★★★★    │  ★★★★★  │   ★★★★   │    ★★     │
Visual Appeal   │     ★★★      │  ★★★★★  │   ★★★★   │    ★      │
Performance     │     ★★★★     │   ★★★   │   ★★★    │   ★★★★★   │
Automation      │     ★★       │    ★    │    ★★    │   ★★★★★   │

💡 RECOMMENDATIONS:
• First time users: Web Interface
• Command-line experts: Enhanced CLI
• Researchers: Jupyter Notebook Interface
• Automation/Scripts: Basic CLI

🚀 GETTING STARTED:
1. Choose an interface from the main menu
2. Follow the interface-specific setup if needed
3. Process your first research paper
4. Explore the generated code and documentation

Press Enter to return to the main menu...
"""
    print(help_text)
    input()


async def launch_enhanced_cli():
    """Launch enhanced CLI interface"""
    print("🖥️ Launching Enhanced CLI Interface...")
    try:
        from .cli_enhanced import run_enhanced_cli
        await run_enhanced_cli()
    except ImportError as e:
        print(f"❌ Error importing enhanced CLI: {e}")
        print("Make sure all dependencies are installed.")
    except Exception as e:
        print(f"❌ Error launching enhanced CLI: {e}")


def launch_web_interface():
    """Launch web interface using Streamlit"""
    print("🌐 Launching Web Interface...")
    print("💡 This will open in your default web browser")
    
    try:
        # Check if streamlit is available
        import streamlit
        
        # Get the path to web_interface.py
        web_interface_path = Path(__file__).parent / "web_interface.py"
        
        # Launch streamlit
        print(f"🚀 Starting Streamlit server...")
        print(f"📁 Interface file: {web_interface_path}")
        
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            str(web_interface_path),
            "--server.address", "localhost",
            "--server.port", "8501",
            "--browser.gatherUsageStats", "false"
        ])
        
    except ImportError:
        print("❌ Streamlit not found. Install with: pip install streamlit")
        print("💡 Then try again.")
        input("Press Enter to continue...")
    except Exception as e:
        print(f"❌ Error launching web interface: {e}")
        input("Press Enter to continue...")


def launch_notebook_guide():
    """Show Jupyter notebook interface guide"""
    print("📓 Jupyter Notebook Interface Guide")
    print("=" * 50)
    
    guide_text = """
🚀 JUPYTER NOTEBOOK SETUP:

1. Install required packages:
   pip install ipywidgets

2. Enable widgets (if needed):
   jupyter nbextension enable --py widgetsnbextension

3. In your Jupyter notebook, use:

   # Import and create interface
   from paper2code.interfaces import show_quick_start
   show_quick_start()

   # Or use directly
   from paper2code.interfaces import NotebookInterface
   interface = NotebookInterface()
   interface.display()

4. Alternative - Direct API usage:
   
   from paper2code import Paper2CodeProcessor
   import asyncio
   
   async def process_paper():
       processor = Paper2CodeProcessor()
       result = await processor.process_paper("paper.pdf")
       return result
   
   result = await process_paper()

📋 EXAMPLE NOTEBOOK CELL:
"""
    
    print(guide_text)
    
    # Show example code
    example_code = '''
# Paper2Code Notebook Example
from paper2code.interfaces import show_quick_start

# This creates a full interactive interface
show_quick_start()
'''
    
    print("```python")
    print(example_code)
    print("```")
    
    print("\n💡 Copy the code above into a Jupyter notebook cell and run it!")
    input("\nPress Enter to continue...")


def launch_basic_cli():
    """Launch basic CLI interface"""
    print("⚡ Launching Basic CLI Interface...")
    try:
        from ...__main__ import interactive_mode
        asyncio.run(interactive_mode())
    except ImportError as e:
        print(f"❌ Error importing basic CLI: {e}")
    except Exception as e:
        print(f"❌ Error launching basic CLI: {e}")


def main():
    """Main launcher function"""
    print_banner()
    
    while True:
        choice = show_interface_menu()
        
        if choice == '1':
            asyncio.run(launch_enhanced_cli())
        elif choice == '2':
            launch_web_interface()
        elif choice == '3':
            launch_notebook_guide()
        elif choice == '4':
            asyncio.run(launch_basic_cli())
        elif choice == '5':
            show_help()
        elif choice == '6':
            print("👋 Thank you for using Paper2Code!")
            break
        else:
            print("❌ Invalid choice. Please try again.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
