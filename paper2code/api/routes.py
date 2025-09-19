"""
API routes for Paper2Code
"""

from typing import Optional, List, Dict, Any
from pathlib import Path
import os

import asyncio
import uuid

from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel, Field

from ..processor import Paper2CodeProcessor, ProcessingResult
from ..config.manager import ConfigManager
import subprocess


router = APIRouter(prefix="/api", tags=["paper2code"])


class ProcessRequest(BaseModel):
    input_source: str = Field(..., description="Path to file or URL")
    mode: str = Field("comprehensive", pattern="^(comprehensive|fast)$")
    output_dir: Optional[str] = Field(None, description="Custom output directory")
    enable_segmentation: Optional[bool] = True
    segmentation_threshold: Optional[int] = 50000
    llm_provider: Optional[str] = Field("ollama", description="ollama|openai|anthropic")
    llm_model: Optional[str] = Field(None, description="model name, e.g., deepseek-r1:8b")
    openai_api_key: Optional[str] = Field(None, description="OpenAI API key if using openai provider")
    anthropic_api_key: Optional[str] = Field(None, description="Anthropic API key if using anthropic provider")


class ProcessResponse(BaseModel):
    success: bool
    output_path: Optional[str] = None
    files: Optional[List[str]] = None
    error: Optional[str] = None
    processing_time: Optional[float] = None


@router.post("/process", response_model=ProcessResponse)
async def process_paper(req: ProcessRequest) -> ProcessResponse:
    processor = Paper2CodeProcessor()

    def progress_cb(progress: int, message: str):
        # Placeholder for future streaming/WebSocket updates
        return None

    result: ProcessingResult = await processor.process_paper(
        input_source=req.input_source,
        output_dir=Path(req.output_dir) if req.output_dir else None,
        mode=req.mode,
        enable_segmentation=req.enable_segmentation,
        segmentation_threshold=req.segmentation_threshold,
        progress_callback=progress_cb,
        llm_provider=req.llm_provider,
        llm_model=req.llm_model,
    )

    return ProcessResponse(
        success=result.success,
        output_path=result.output_path,
        files=result.files,
        error=result.error,
        processing_time=result.processing_time,
    )


# -----------------
# Background tasks & progress tracking
# -----------------
TASKS: Dict[str, Dict[str, Any]] = {}
_TASK_LOCK = asyncio.Lock()

# Enhanced logging for deep process visibility
PROCESS_LOGS: Dict[str, List[Dict[str, Any]]] = {}


class StartResponse(BaseModel):
    task_id: str
    status: str


@router.post("/process/start", response_model=StartResponse)
async def start_process(req: ProcessRequest) -> StartResponse:
    task_id = uuid.uuid4().hex

    async with _TASK_LOCK:
        TASKS[task_id] = {
            'status': 'running',
            'progress': 0,
            'message': 'Starting…',
            'result': None,
            'error': None,
        }

    async def run():
        processor = Paper2CodeProcessor()
        
        # Initialize process logs
        PROCESS_LOGS[task_id] = []

        def progress_cb(progress: int, message: str):
            # Update task progress
            data = TASKS.get(task_id)
            if data is not None:
                data['progress'] = int(progress)
                data['message'] = str(message)
                
            # Add to detailed logs
            import time
            log_entry = {
                'timestamp': time.time(),
                'progress': int(progress),
                'message': str(message),
                'phase': _determine_phase(progress, message)
            }
            PROCESS_LOGS.setdefault(task_id, []).append(log_entry)

        try:
            # Set API keys if provided
            if req.openai_api_key:
                processor.config.set('openai.api_key', req.openai_api_key)
            if req.anthropic_api_key:
                processor.config.set('anthropic.api_key', req.anthropic_api_key)
                
            result: ProcessingResult = await processor.process_paper(
                input_source=req.input_source,
                output_dir=Path(req.output_dir) if req.output_dir else None,
                mode=req.mode,
                enable_segmentation=req.enable_segmentation,
                segmentation_threshold=req.segmentation_threshold,
                progress_callback=progress_cb,
                llm_provider=req.llm_provider,
                llm_model=req.llm_model,
            )
            TASKS[task_id]['status'] = 'completed' if result.success else 'error'
            TASKS[task_id]['result'] = {
                'success': result.success,
                'output_path': result.output_path,
                'files': result.files,
                'error': result.error,
                'processing_time': result.processing_time,
            }
        except Exception as e:
            TASKS[task_id]['status'] = 'error'
            TASKS[task_id]['error'] = str(e)

    asyncio.create_task(run())
    return StartResponse(task_id=task_id, status='started')


@router.get("/progress/{task_id}")
async def get_progress(task_id: str) -> Dict[str, Any]:
    data = TASKS.get(task_id)
    if not data:
        raise HTTPException(status_code=404, detail="Task not found")
    return {
        'task_id': task_id,
        'status': data['status'],
        'progress': data['progress'],
        'message': data['message'],
        'error': data.get('error'),
    }


@router.get("/result/{task_id}")
async def get_result(task_id: str) -> Dict[str, Any]:
    data = TASKS.get(task_id)
    if not data:
        raise HTTPException(status_code=404, detail="Task not found")
    return {
        'task_id': task_id,
        'status': data['status'],
        'result': data.get('result'),
        'error': data.get('error'),
    }


@router.get("/logs/{task_id}")
async def get_process_logs(task_id: str) -> Dict[str, Any]:
    """Get detailed process logs for a task."""
    logs = PROCESS_LOGS.get(task_id, [])
    return {
        'task_id': task_id,
        'logs': logs,
        'total_entries': len(logs)
    }


def _determine_phase(progress: int, message: str) -> str:
    """Determine processing phase from progress and message."""
    msg_lower = message.lower()
    
    if progress < 15 or 'analyzing' in msg_lower or 'document' in msg_lower:
        return 'analysis'
    elif progress < 35 or 'repository' in msg_lower or 'discovery' in msg_lower:
        return 'discovery'
    elif progress < 55 or 'planning' in msg_lower or 'implementation' in msg_lower:
        return 'planning'
    elif progress < 75 or 'generating' in msg_lower or 'code' in msg_lower:
        return 'generation'
    elif progress < 95 or 'finalizing' in msg_lower:
        return 'finalization'
    else:
        return 'complete'


@router.get("/llm/models/{provider}")
async def list_models(provider: str) -> Dict[str, Any]:
    """Return available models for the specified provider."""
    if provider == 'ollama':
        try:
            out = subprocess.check_output(['ollama', 'list'], text=True)
            lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
            models = []
            for ln in lines[1:]:
                parts = [p for p in ln.split(' ') if p]
                if parts:
                    models.append({'name': parts[0], 'size': parts[2] if len(parts) > 2 else 'Unknown'})
            return {'provider': 'ollama', 'models': models}
        except Exception as e:
            return {'provider': 'ollama', 'models': [], 'error': str(e)}
    
    elif provider == 'openai':
        return {
            'provider': 'openai',
            'models': [
                {'name': 'gpt-4', 'description': 'Most capable model'},
                {'name': 'gpt-4-turbo', 'description': 'Fast and capable'},
                {'name': 'gpt-3.5-turbo', 'description': 'Fast and efficient'}
            ]
        }
    
    elif provider == 'anthropic':
        return {
            'provider': 'anthropic', 
            'models': [
                {'name': 'claude-3-5-sonnet-20241022', 'description': 'Most capable'},
                {'name': 'claude-3-haiku-20240307', 'description': 'Fast and efficient'}
            ]
        }
    
    else:
        raise HTTPException(status_code=400, detail="Unsupported provider")


@router.get("/llm/providers")
async def list_providers() -> Dict[str, Any]:
    """List all available LLM providers and their status."""
    providers = []
    
    # Check Ollama
    try:
        subprocess.check_output(['ollama', 'list'], text=True)
        providers.append({
            'name': 'ollama',
            'display_name': 'Ollama (Local)',
            'available': True,
            'requires_api_key': False
        })
    except:
        providers.append({
            'name': 'ollama',
            'display_name': 'Ollama (Local)',
            'available': False,
            'requires_api_key': False,
            'error': 'Ollama not installed or not running'
        })
    
    # OpenAI
    providers.append({
        'name': 'openai',
        'display_name': 'OpenAI',
        'available': True,
        'requires_api_key': True
    })
    
    # Anthropic
    providers.append({
        'name': 'anthropic',
        'display_name': 'Anthropic',
        'available': True,
        'requires_api_key': True
    })
    
    return {'providers': providers}


@router.post("/upload")
async def upload_file(file: UploadFile = File(...)) -> dict:
    """
    Receive a file upload and store it, returning the saved path.
    """
    try:
        # Try local uploads directory first, fallback to temp
        try:
            uploads_dir = Path("./uploads").resolve()
            uploads_dir.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            # Fallback to temp directory
            import tempfile
            uploads_dir = Path(tempfile.gettempdir()) / "paper2code_uploads"
            uploads_dir.mkdir(parents=True, exist_ok=True)

        # Sanitize filename
        filename = Path(file.filename).name if file.filename else "uploaded_file.pdf"
        save_path = uploads_dir / filename

        # Avoid collisions by appending number
        counter = 1
        stem = save_path.stem
        suffix = save_path.suffix
        while save_path.exists():
            save_path = uploads_dir / f"{stem}_{counter}{suffix}"
            counter += 1

        content = await file.read()
        save_path.write_bytes(content)

        return {"path": str(save_path)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {e}")


@router.get("/recent")
async def list_recent(limit: int = 10) -> dict:
    processor = Paper2CodeProcessor()
    items = await processor.list_recent_outputs(limit=limit)
    return {"items": items}


@router.get("/file/content/{task_id}")
async def get_file_content(task_id: str, file_path: str) -> dict:
    """Get content of a specific generated file."""
    task_data = TASKS.get(task_id)
    if not task_data or not task_data.get('result'):
        raise HTTPException(status_code=404, detail="Task not found")
    
    result = task_data['result']
    if not result.get('success'):
        raise HTTPException(status_code=400, detail="Task did not complete successfully")
    
    try:
        full_path = Path(result['output_path']) / file_path
        if not full_path.exists():
            raise HTTPException(status_code=404, detail="File not found")
        
        content = full_path.read_text(encoding='utf-8')
        return {
            'file_path': file_path,
            'content': content,
            'size': len(content)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading file: {e}")


@router.get("/analysis/{task_id}")
async def get_analysis_data(task_id: str) -> dict:
    """Get detailed analysis data for visualization."""
    task_data = TASKS.get(task_id)
    if not task_data or not task_data.get('result'):
        raise HTTPException(status_code=404, detail="Task not found")
    
    result = task_data['result']
    if not result.get('success'):
        raise HTTPException(status_code=400, detail="Task did not complete successfully")
    
    try:
        output_path = Path(result['output_path'])
        analysis_dir = output_path / 'analysis'
        
        analysis_data = {}
        
        # Load analysis files
        analysis_files = ['document_analysis.json', 'implementation_plan.json', 'repository_discovery.json']
        for file_name in analysis_files:
            file_path = analysis_dir / file_name
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    analysis_data[file_name.replace('.json', '')] = json.load(f)
        
        return {
            'task_id': task_id,
            'analysis': analysis_data,
            'output_path': str(output_path)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading analysis: {e}")


@router.get("/download/{task_id}")
async def download_project_zip(task_id: str):
    """Download the entire generated project as a ZIP file."""
    task_data = TASKS.get(task_id)
    if not task_data or not task_data.get('result'):
        raise HTTPException(status_code=404, detail="Task not found")
    
    result = task_data['result']
    if not result.get('success'):
        raise HTTPException(status_code=400, detail="Task did not complete successfully")
    
    try:
        import zipfile
        import tempfile
        from fastapi.responses import FileResponse
        
        output_path = Path(result['output_path'])
        if not output_path.exists():
            raise HTTPException(status_code=404, detail="Output directory not found")
        
        # Create temporary zip file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_file:
            zip_path = tmp_file.name
        
        # Create zip archive
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in output_path.rglob('*'):
                if file_path.is_file():
                    arcname = file_path.relative_to(output_path)
                    zipf.write(file_path, arcname)
        
        project_name = output_path.name
        return FileResponse(
            zip_path,
            media_type='application/zip',
            filename=f"{project_name}.zip",
            background=None  # Keep file until response is sent
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating zip: {e}")


@router.get("/files/{task_id}")
async def list_project_files(task_id: str) -> dict:
    """List all files in the generated project with metadata."""
    task_data = TASKS.get(task_id)
    if not task_data or not task_data.get('result'):
        raise HTTPException(status_code=404, detail="Task not found")
    
    result = task_data['result']
    if not result.get('success'):
        raise HTTPException(status_code=400, detail="Task did not complete successfully")
    
    try:
        output_path = Path(result['output_path'])
        if not output_path.exists():
            raise HTTPException(status_code=404, detail="Output directory not found")
        
        files = []
        for file_path in output_path.rglob('*'):
            if file_path.is_file():
                relative_path = file_path.relative_to(output_path)
                stat = file_path.stat()
                
                files.append({
                    'path': str(relative_path).replace('\\', '/'),
                    'name': file_path.name,
                    'size': stat.st_size,
                    'modified': stat.st_mtime,
                    'extension': file_path.suffix,
                    'type': 'code' if file_path.suffix in ['.py', '.js', '.ts', '.java', '.cpp'] else 'other'
                })
        
        # Sort by path
        files.sort(key=lambda x: x['path'])
        
        return {
            'task_id': task_id,
            'files': files,
            'total_files': len(files),
            'output_path': str(output_path)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing files: {e}")


@router.get("/view/{task_id}")
async def view_file_content(task_id: str, file_path: str = None) -> dict:
    """View content of a specific file in the generated project."""
    if not file_path:
        raise HTTPException(status_code=400, detail="file_path parameter required")
    
    task_data = TASKS.get(task_id)
    if not task_data or not task_data.get('result'):
        raise HTTPException(status_code=404, detail="Task not found")
    
    result = task_data['result']
    if not result.get('success'):
        raise HTTPException(status_code=400, detail="Task did not complete successfully")
    
    try:
        output_path = Path(result['output_path'])
        full_path = output_path / file_path
        
        if not full_path.exists() or not full_path.is_file():
            raise HTTPException(status_code=404, detail="File not found")
        
        # Check if file is within output directory (security)
        if not str(full_path.resolve()).startswith(str(output_path.resolve())):
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Read file content
        try:
            content = full_path.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            # Try binary read for non-text files
            content = full_path.read_bytes().hex()
            is_binary = True
        else:
            is_binary = False
        
        stat = full_path.stat()
        
        return {
            'task_id': task_id,
            'file_path': file_path,
            'name': full_path.name,
            'content': content,
            'size': stat.st_size,
            'modified': stat.st_mtime,
            'extension': full_path.suffix,
            'is_binary': is_binary
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading file: {e}")


