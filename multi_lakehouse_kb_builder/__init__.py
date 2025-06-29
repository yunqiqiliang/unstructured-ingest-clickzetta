"""
ClickZetta 多Lakehouse知识库批量部署系统
"""

from .multi_lakehouse_kb_builder import (
    LakehouseConnectionManager,
    LakehouseSchemaManager,
    KnowledgeBaseBuilder,
    BatchKnowledgeBaseDeployer,
    get_complete_deployment_sql
)

from .kb_data_validator import (
    KnowledgeBaseValidator,
    BatchKnowledgeBaseValidator
)

__version__ = "1.1.0"
__all__ = [
    "LakehouseConnectionManager",
    "LakehouseSchemaManager", 
    "KnowledgeBaseBuilder",
    "BatchKnowledgeBaseDeployer",
    "KnowledgeBaseValidator",
    "BatchKnowledgeBaseValidator",
    "get_complete_deployment_sql"
]