"""
Xctopus - Capa Clustering

Sistema de organización orgánica de conocimiento mediante Knowledge Nodes.
"""

__version__ = "0.1.0"

# Inicializar logging al importar el paquete
from .logger_config import setup_logging

# Configurar logging automáticamente
setup_logging()

# Exports principales
from .repository import KNRepository
from .filter_bayesian import FilterBayesian
from .knowledgenode import KnowledgeNode
from .orchestrator import Orchestrator

__all__ = [
    "setup_logging",
    "KNRepository",
    "FilterBayesian",
    "KnowledgeNode",
    "Orchestrator",
    # Se completará en fases siguientes
]
