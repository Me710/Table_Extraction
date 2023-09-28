from pydantic import BaseModel
from typing import List


class Output(BaseModel):
    image_name: str
    bbox: List[float]
