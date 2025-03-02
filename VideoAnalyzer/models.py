from enum import Enum
from typing import Any, Literal, Optional, List, TypedDict
from pydantic import BaseModel
from typing import Literal

FILE_TYPE = Literal[
  "pdf",
  "png",
  "jpeg",
  "jpg",
  "svg",
  "xlsx",
  "xls",
  "mp3",
  "flac",
  "mp4",
  "mpeg",
  "mpga",
  "m4a",
  "ogg",
  "wav",
  "webm",
   "txt",
]

PROCESS_TYPE = Literal[
  "text",
  "image",
  "excel",
  "text_with_image",
  "audio",
  "survey_excel",
  "sql_excel",
  "video",
]

class ProcessType(str, Enum):
    HYBRID = "HYBRID"
    NON_HYBRID = "NON-HYBRID"


class StatusRequestDto(BaseModel):
    request_id: int
    response_data_api_path: str

class FileInjestionRequestDto(StatusRequestDto):
    pre_signed_url: str
    file_name: str
    namespace: str
    original_file_name: str
    process_type: PROCESS_TYPE = "text"
    file_type: FILE_TYPE = "pdf"
    metadata: List[dict[str, str]] = []
    params: dict[str, Any] = {}
    search_type: ProcessType = ProcessType.HYBRID


class FileInjestionResponseDto(BaseModel):
    title: Optional[str] = None
    author: Optional[str] = None
    file_name: Optional[str] = None
    original_file_name: Optional[str] = None
    total_pages: Optional[int] = None
    thumbnail_object_path: Optional[str] = None

