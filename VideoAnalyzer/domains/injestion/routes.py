from fastapi import APIRouter, HTTPException, BackgroundTasks, Header
from VideoAnalyzer.domains.injestion.utils import extract_metadata_from_video
from VideoAnalyzer.models import (
    FileInjestionRequestDto,
    FileInjestionResponseDto,

)
from VideoAnalyzer.settings import config_settings
from loguru import logger
from typing import Tuple, Any, Callable


router = APIRouter(tags=["injestion"])


@router.post(
    path="/injestion",
    summary="Injest the document into database",
    description="Injest the document into database as per their different formats",
)
def injest_doc(
        request: FileInjestionRequestDto,
        background_tasks: BackgroundTasks,
        token: str = Header(alias="authorization"),
) -> FileInjestionResponseDto:
    logger.info(f"Injesting the document into database")

    try:
        logger.info("Extracting the metadata")
        if request.process_type == "video":
            metadata_dict = extract_metadata_from_video(
                pre_signed_url=request.pre_signed_url,
                file_name=request.file_name,
                original_file_name=request.original_file_name,
                bucket_name=config_settings.BUCKET_NAME,
            )

            response = FileInjestionResponseDto(
                title=metadata_dict["title"],
                author=metadata_dict["author"],
                file_name=metadata_dict["file_name"],
                original_file_name=metadata_dict["original_file_name"],
                total_pages=metadata_dict["total_pages"],
                thumbnail_object_path=metadata_dict["thumbnail_object_path"],
            )

        elif request.process_type == "audio":
            response = FileInjestionResponseDto(
                title=request.original_file_name,
                author="",
                file_name=request.file_name,
                original_file_name=request.original_file_name,
                total_pages=None,
                thumbnail_object_path="",
            )

        background_tasks.add_task(
            request,

        )
        return response

    except ModuleNotFoundError:
        raise HTTPException()
