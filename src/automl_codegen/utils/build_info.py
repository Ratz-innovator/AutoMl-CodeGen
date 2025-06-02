"""
Build Information Module

Provides build metadata and version information for AutoML-CodeGen.
"""

import sys
import platform
from typing import Dict, Any

def get_build_info() -> Dict[str, Any]:
    """Get comprehensive build information."""
    return {
        'version': '1.0.0',
        'python_version': sys.version,
        'platform': platform.platform(),
        'architecture': platform.architecture(),
        'build_date': '2024-01-01',
        'git_commit': 'dev',
        'dependencies': {
            'torch': _get_package_version('torch'),
            'tensorflow': _get_package_version('tensorflow'),
            'numpy': _get_package_version('numpy'),
        }
    }

def _get_package_version(package_name: str) -> str:
    """Get version of an installed package."""
    try:
        import importlib.metadata
        return importlib.metadata.version(package_name)
    except:
        try:
            module = __import__(package_name)
            return getattr(module, '__version__', 'unknown')
        except:
            return 'not installed'

def get_build_info_fast() -> Dict[str, Any]:
    """Get build info quickly (same as get_build_info but with fast alias)."""
    return get_build_info() 