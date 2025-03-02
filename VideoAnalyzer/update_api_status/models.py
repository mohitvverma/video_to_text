from enum import Enum
from pydantic import BaseModel

class RequestStatusEnum(str, Enum):
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class ApiNameEnum(str, Enum):
    INJEST_DOC = "injest-doc"
    SCRAPE = "scrape"
    DELETE_FILE = "delete-file"
    PROFILE_DETAILS = "profile-details"
    AUTO_TAGGING = 'auto-tagging'


class RequestStatus(BaseModel):
    request_id: int
    status: RequestStatusEnum
    api_name: ApiNameEnum = None
    data_json: dict = None
    error_detail: str = None
