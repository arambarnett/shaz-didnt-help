import logging
import threading
from collections import defaultdict

logger = logging.getLogger(__name__)

# Store progress information
progress_store = defaultdict(lambda: {
    'stage': 'upload',
    'progress': 0,
    'details': 'Starting...'
})
progress_lock = threading.Lock()

def update_progress(task_id: str, stage: str, progress: int, details: str):
    """Update the progress for a given task"""
    with progress_lock:
        progress_store[task_id].update({
            'stage': stage,
            'progress': progress,
            'details': details
        })
        logger.info(f"Progress update for {task_id}: {stage} - {progress}% - {details}")

def get_progress(task_id: str) -> int:
    """Get the current progress percentage for a task"""
    with progress_lock:
        return progress_store[task_id].get('progress', 0)

def get_stage_details(task_id: str) -> str:
    """Get the current stage details for a task"""
    with progress_lock:
        return progress_store[task_id].get('details', '')

def get_current_stage(task_id: str) -> str:
    """Get the current processing stage for a task"""
    with progress_lock:
        return progress_store[task_id].get('stage', 'Processing') 