

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Union

def sanitize_upload_filename(filename: str) -> str:

    clean_name = re.sub(r"[^a-zA-Z0-9.\-_]", "_", filename)
    
    clean_name = re.sub(r"\.+", ".", clean_name)
    clean_name = clean_name.strip("._-")
    
    if not clean_name:
        clean_name = "uploaded_mri_scan.jpg"
    return clean_name

def ensure_safe_path(base_dir: Union[str, Path], target_path: Union[str, Path]) -> Path:
    
    resolved_base = Path(base_dir).resolve()
    resolved_target = Path(target_path).resolve()

    try:
        resolved_target.relative_to(resolved_base)
    except ValueError:
        raise ValueError("Security violation: Path traversal detected.")

    return resolved_target
