from loguru import logger
from typing import Any, Tuple
from VideoAnalyzer.models import FileInjestionRequestDto
from VideoAnalyzer.vector_db.push_vector import push_to_database
from VideoAnalyzer.settings import config_settings
from langchain_core.documents import Document
from VideoAnalyzer.domains.injestion.doc_loaders import file_loader
from VideoAnalyzer.exception import VideoException
from VideoAnalyzer.update_api_status.models import RequestStatus, RequestStatusEnum, ApiNameEnum


def load_file(
    pre_signed_url: str,
    file_name: str,
    original_file_name: str,
    file_type: str,
    process_type: str,
    params: dict[str, Any],
    metadata,
    request_id: int,
    response_data_api_path: str,
    token: str,
) -> Tuple[list[Document], str, Any]:
    logger.info(f"Received file type: {file_type}")
    try:
        return file_loader(
            pre_signed_url,
            file_name,
            original_file_name,
            file_type,
            process_type,
            request_id,
            response_data_api_path,
            params,
            metadata,
            token,
        )
    except Exception as e:
        logger.exception("Exception during file load")
        raise VideoException(f"Exception during file load", e) from e



def load_file_and_push_to_database_and_update_status(
    request: FileInjestionRequestDto, token: str
) -> None:
    logger.info(
        f"Starting background task for {request.file_name} and process_type: {request.process_type}"
    )

    try:
        documents, summary, transcription_json = load_file(
            request.pre_signed_url,
            request.file_name,
            request.original_file_name,
            request.file_type,
            request.process_type,
            request.params,
            request.metadata,
            request.request_id,
            request.response_data_api_path,
            token,
        )
        push_to_database(documents, config_settings.INDEX_NAME, request.namespace)

    except Exception as e:
        logger.exception("Failed")
        error_detail = f"Failed when process_type is {request.process_type}: {e}"
        status = RequestStatus(
            request_id=request.request_id,
            api_name=ApiNameEnum.INJEST_DOC,
            status=RequestStatusEnum.FAILED,
            error_detail=error_detail,
        )

    else:
        logger.info("Completed")
        error_detail = ""

        # Prepare response data with summary
        response_data = {"summary": summary}

        # Add transcript only if it exists
        if transcription_json and "transcript" in transcription_json:
            response_data["transcript"] = transcription_json["transcript"]

        # Create status object
        status = RequestStatus(
            request_id=request.request_id,
            api_name=ApiNameEnum.INJEST_DOC,
            status=RequestStatusEnum.COMPLETED,
            data_json=response_data,
        )

    logger.info(
        f"Completed injest-doc for file_name: {request.file_name}"
        f" with status: {status.status}"
        f" to backend service for file_name: {request.file_name}"
    )

    return status
