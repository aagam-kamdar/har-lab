"""
reid package
------------
Local vector-based Person Re-Identification for Student Productivity Tracker.
"""

from .embedding_model import BaseReIDModel, PersonReIDExtractor
from .faiss_manager import FAISSManager
from .gallery import StudentGallery
from .identity_resolver import IdentityResolver

__all__ = [
    "BaseReIDModel",
    "PersonReIDExtractor",
    "FAISSManager",
    "StudentGallery",
    "IdentityResolver",
]
