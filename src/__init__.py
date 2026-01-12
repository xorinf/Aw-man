"""
SENTINEL - Autonomous AI Cyber Defense Platform
================================================

A production-ready AI-powered cybersecurity platform for:
- Zero-day attack detection using unsupervised ML
- Attack behavior prediction using sequence models
- Explainable AI for threat analysis
- AI-powered red team simulation

Author: xorinf (https://github.com/xorinf)
License: MIT
Version: 1.0.0
Repository: https://github.com/xorinf/Aw-man
"""

__version__ = "1.0.0"
__author__ = "xorinf"
__license__ = "MIT"

# Package exports
from .config import settings

__all__ = ["settings", "__version__", "__author__"]
