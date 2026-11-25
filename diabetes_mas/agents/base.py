from abc import ABC, abstractmethod

from ..kb.rag import RAG
from ..kb.store import KnowledgeBase
from ..schemas import AgentResult, AnalysisContext


class BaseAgent(ABC):
    name, kind = "BaseAgent", ""  # name: trùng tên lớp; kind: cách agent ra quyết định, hiện trên báo cáo

    def __init__(self, kb: KnowledgeBase, rag: RAG):
        self.kb, self.rag = kb, rag

    @abstractmethod
    def run(self, ctx: AnalysisContext) -> AgentResult: ...
