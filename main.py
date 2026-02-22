from typing import List, Dict, Optional
import os
from pydantic import BaseModel, Field

class YtGraphState(BaseModel):
    summary: str = Field(...  description="The summary of the video")