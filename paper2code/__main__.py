"""
Command-line interface for Paper2Code standalone.

Entry point for the paper2code command.
"""

import argparse
import asyncio
import sys
from pathlib import Path

from .processor import Paper2CodeProcessor
from .utils.logger import setup_logging, get_logger

logger = get_logger(__name__)


def create_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser"""
    
    parser = argparse.ArgumentParser(
        description="Paper2Code - Transform research papers into working code",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  paper2code --file paper.pdf                    # Process PDF paper
  paper2code --url https://arxiv.org/pdf/...     # Process from URL  
  paper2code --file paper.pdf --fast             # Fast mode (skip repo search)
  paper2code --file paper.pdf --output ./code    # Custom output directory
  paper2code --interactive                       # Basic interactive mode
  paper2code --interface                         # Launch interface selector
  
For more information, visit: https://github.com/paper2code/paper2code-standalone
        """
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=False)
    input_group.add_argument(
        '--file', '-f',
        type=str,
        help='Path to research paper file (PDF, DOCX, TXT, MD, HTML)'
    )
    input_group.add_argument(
        '--url', '-u', 
        type=str,
        help='URL to research paper'
    )
    input_group.add_argument(
        '--interactive', '-i',
        action='store_true',
        help='Start basic interactive mode'
    )
    input_group.add_argument(
        '--interface', '-I',
        action='store_true',
        help='Launch interface selector (Enhanced CLI, Web, Notebook)'
    )
    
    # Processing options
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='./output',
        help='Output directory (default: ./output)'
    )
    parser.add_argument(
        '--mode', '-m',
        choices=['comprehensive', 'fast'],
        default='comprehensive',
        help='Processing mode: comprehensive (default) or fast'
    )
    parser.add_argument(
        '--fast',
        action='store_true',
        help='Enable fast mode (equivalent to --mode fast)'
    )
    
    # Configuration options
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration file'
    )
    parser.add_argument(
        '--no-segmentation',
        action='store_true',
        help='Disable document segmentation'
    )
    parser.add_argument(
        '--segment-threshold',
        type=int,
        default=50000,
        help='Document size threshold for segmentation (default: 50000 chars)'
    )
    
    # Output options
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode'
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress output (except errors)'
    )
    
    return parser


def setup_logging_from_args(args):
    """Setup logging based on command line arguments"""
    
    if args.quiet:
        level = "ERROR"
        console = True
    elif args.debug:
        level = "DEBUG"
        console = True
    elif args.verbose:
        level = "INFO"
        console = True
    else:
        level = "WARNING"
        console = True
    
    # Setup logging
    setup_logging(
        level=level,
        console=console,
        log_file="./logs/paper2code.log" if args.debug else None
    )


async def process_paper_async(args) -> int:
    """Process paper asynchronously"""
    
    try:
        # Initialize processor
        processor = Paper2CodeProcessor(config_path=args.config)
        
        # Determine input source
        if args.file:
            input_source = args.file
            if not Path(input_source).exists():
                print(f"❌ Error: File not found: {input_source}")
                return 1
        elif args.url:
            input_source = args.url
        else:
            print("❌ Error: No input source specified. Use --file or --url.")
            return 1
        
        # Determine processing mode
        mode = "fast" if args.fast else args.mode
        
        # Configure processing options
        options = {
            'enable_segmentation': not args.no_segmentation,
            'segmentation_threshold': args.segment_threshold,
        }
        
        # Setup progress callback
        def progress_callback(progress: int, message: str):
            if not args.quiet:
                print(f"[{progress:3d}%] {message}")
        
        # Process the paper
            result = await processor.process_paper(
            input_source=input_source,
            output_dir=Path(args.output),
            mode=mode,
                progress_callback=progress_callback,
                **options
        )
        
        # Handle results
        if result.success:
            if not args.quiet:
                print(f"\n🎉 Processing completed successfully!")
                print(f"📁 Output directory: {result.output_path}")
                print(f"📊 Generated files: {len(result.files or [])}")
                print(f"⏱️  Processing time: {result.processing_time:.1f} seconds")
                
                if args.verbose and result.files:
                    print("\n📄 Generated files:")
                    for file_path in result.files:
                        print(f"  - {file_path}")
            return 0
        else:
            print(f"❌ Processing failed: {result.error}")
            return 1
            
    except KeyboardInterrupt:
        print("\n⚠️  Processing interrupted by user")
        return 130
    except Exception as e:
        if args.debug:
            import traceback
            traceback.print_exc()
        else:
            print(f"❌ Unexpected error: {e}")
        return 1


async def interactive_mode() -> int:
    """Run interactive mode"""
    
    print("🧬 Paper2Code - Interactive Mode")
    print("=" * 50)
    
    try:
        processor = Paper2CodeProcessor()
        
        while True:
            print("\nChoose input method:")
            print("1. Local file")
            print("2. URL")
            print("3. Exit")
            
            choice = input("\nEnter choice (1-3): ").strip()
            
            if choice == '3':
                print("👋 Goodbye!")
                return 0
            elif choice == '1':
                file_path = input("Enter file path: ").strip()
                if not Path(file_path).exists():
                    print(f"❌ File not found: {file_path}")
                    continue
                input_source = file_path
            elif choice == '2':
                url = input("Enter URL: ").strip()
                if not url.startswith(('http://', 'https://')):
                    print("❌ Invalid URL")
                    continue
                input_source = url
            else:
                print("❌ Invalid choice")
                continue
            
            # Processing options
            print("\nProcessing options:")
            print("1. Comprehensive (default)")
            print("2. Fast mode")
            
            mode_choice = input("Enter mode (1-2, default=1): ").strip() or '1'
            mode = "fast" if mode_choice == '2' else "comprehensive"
            
            output_dir = input("Output directory (default=./output): ").strip() or "./output"
            
            # Progress callback
            def progress_callback(progress: int, message: str):
                print(f"[{progress:3d}%] {message}")
            
            print(f"\n🚀 Processing {input_source}...")
            print(f"📊 Mode: {mode}")
            print(f"📁 Output: {output_dir}")
            
            # Process
            result = await processor.process_paper(
                input_source=input_source,
                output_dir=Path(output_dir),
                mode=mode
            )
            
            if result.success:
                print(f"\n✅ Success! Files generated in: {result.output_path}")
                print(f"⏱️  Processing time: {result.processing_time:.1f} seconds")
            else:
                print(f"\n❌ Failed: {result.error}")
            
            # Continue or exit
            continue_choice = input("\nProcess another paper? (y/n): ").strip().lower()
            if continue_choice not in ('y', 'yes'):
                print("👋 Goodbye!")
                return 0
                
    except KeyboardInterrupt:
        print("\n⚠️  Interactive mode interrupted")
        return 130
    except Exception as e:
        print(f"❌ Error in interactive mode: {e}")
        return 1


async def launch_interface_selector() -> int:
    """Launch the interactive interface selector"""
    
    try:
        from .interfaces import launch_interface_selector
        launch_interface_selector()
        return 0
    except ImportError as e:
        print(f"❌ Error: Could not import interface selector: {e}")
        print("💡 Install missing dependencies and try again.")
        return 1
    except KeyboardInterrupt:
        print("\n⚠️  Interface selector interrupted")
        return 130
    except Exception as e:
        print(f"❌ Error in interface selector: {e}")
        return 1


def validate_args(args) -> bool:
    """Validate command line arguments"""
    
    # Check for input source in non-interactive mode
    if not args.interactive and not args.interface and not args.file and not args.url:
        print("❌ Error: Must specify --file, --url, --interactive, or --interface")
        return False
    
    # Check file exists
    if args.file and not Path(args.file).exists():
        print(f"❌ Error: File not found: {args.file}")
        return False
    
    # Check output directory is valid
    try:
        output_path = Path(args.output)
        output_path.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"❌ Error: Cannot create output directory {args.output}: {e}")
        return False
    
    return True


async def main():
    """Main entry point"""
    
    parser = create_parser()
    args = parser.parse_args()
    
    # Setup logging
    setup_logging_from_args(args)
    
    # Print banner (unless quiet)
    if not args.quiet:
        print("🧬 Paper2Code - Standalone Research Paper Implementation Tool")
        print("Transform research papers into working code using AI")
        print("-" * 60)
    
    # Validate arguments
    if not validate_args(args):
        return 1
    
    # Run appropriate mode
    if args.interactive:
        return await interactive_mode()
    elif args.interface:
        return await launch_interface_selector()
    else:
        return await process_paper_async(args)


def main_sync():
    """Synchronous entry point for setuptools"""
    try:
        return asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
        return 130
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main_sync())
