from pydantic import BaseModel
from enum import StrEnum

class Trichotomy(StrEnum):
    YES = "YES"
    NO = "NO"
    DONOTKNOW = "DONTKNOW"