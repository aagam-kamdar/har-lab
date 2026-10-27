"""
reid package
------------
Local vector-based Person Re-Identification for Student Productivity Tracker.

Exports the original pipeline (StudentGallery / IdentityResolver) AND
the new named enrollment pipeline (EnrolledGallery / NamedIdentityResolver).
"""

# ---- Original pipeline (untouched, still runnable) ----
from .embedding_model import BaseReIDModel, PersonReIDExtractor
from .faiss_manager import FAISSManager
from .gallery import StudentGallery
from .identity_resolver import IdentityResolver

# ---- New named enrollment pipeline ----
from .face_embedder import FaceEmbedder
from .enrolled_gallery import EnrolledGallery
from .named_identity_resolver import NamedIdentityResolver

__all__ = [
    # Original
    "BaseReIDModel",
    "PersonReIDExtractor",
    "FAISSManager",
    "StudentGallery",
    "IdentityResolver",
    # New enrollment pipeline
    "FaceEmbedder",
    "EnrolledGallery",
    "NamedIdentityResolver",
]
