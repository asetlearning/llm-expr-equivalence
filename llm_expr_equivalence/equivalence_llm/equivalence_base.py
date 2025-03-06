from abc import ABC, abstractmethod

from llm_expr_equivalence.schemas.common import Trichotomy


class LLMEquivalenceBase(ABC):
    @abstractmethod
    def __equal__(self, expr1, expr2) -> Trichotomy:
        pass

    def equal(self, expr1, expr2) -> Trichotomy:
        return self.__equal__(expr1, expr2)
