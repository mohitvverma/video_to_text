from pydantic import BaseModel
from typing import Optional, Any
from VideoAnalyzer.settings import config_settings


class MilvusConnectionRequest(BaseModel):
    host: str = '0.0.0.0'
    port: int = 19530
    user: Optional[str] = None
    password: Optional[str] = None
    db_name: str = "dev"
    connection_alias: str = "default"


class MilvusConnectionResponse(BaseModel):
    status: bool = False
    message: Optional[str] = None


class MilvusCollectionRequest(BaseModel):
    collection_name: str = config_settings.MILVUS_COLLECTION_NAME_DEV
    collection_schema: Any = None
    drop_collection_status: bool = config_settings.MILVUS_DROP_COLLECTION_STATUS


class MilvusCollectionResponse(BaseModel):
    status : bool = None
    message : Optional[str] = None
    collection : Optional[Any] = None