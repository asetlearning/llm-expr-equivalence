from abc import ABC, abstractmethod

from llm_expr_equivalence.db.data_loader import DataLoader
from llm_expr_equivalence.schemas.common import Trichotomy


class LLMEquivalenceBase(ABC):
    @abstractmethod
    def __equal__(self, expr1, expr2) -> Trichotomy | None:
        pass

    def equal(self, expr1, expr2) -> Trichotomy | None:
        return self.__equal__(expr1, expr2)



class LLMEquivalenceBatchBase(ABC):
    @abstractmethod
    def to_batch_json(self, hash: str, expr1: str, expr2: str) -> dict:
        pass

    @abstractmethod
    def submit_batch(self, data: DataLoader):
        pass

    @abstractmethod
    def get_batch_result(self):
        pass
