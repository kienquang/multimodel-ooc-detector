from pydantic import BaseModel, Field
from typing import List

RELATIONS = ["PERFORMS", "LOCATED_IN", "OCCURRED_ON", "TARGETS", "HAS_STATE", "SAME_AS"]

class SVOTriplet(BaseModel):
    subject: str = Field(..., description="Main entity")
    relation: str = Field(..., description=f"Must be one of: {RELATIONS}")
    object: str = Field(..., description="Value or target")

class SVOList(BaseModel):
    triplets: List[SVOTriplet] = Field(..., description="List of S-V-O triplets")