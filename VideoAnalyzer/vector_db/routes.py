from loguru import logger
from typing import Optional, Any, Tuple
from VideoAnalyzer.vector_db.models import FileInjestionRequestDto



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

        if request.process_type == "excel":
            logger.info(f"Excel file is present")
            push_df_to_database(
                user_id=str(request.request_id),
                asset_id=request.request_id,
                file_name=request.original_file_name,
                original_file_name=request.file_name,
                file_path=request.pre_signed_url,
            )

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

    call_update_status_api(request.response_data_api_path, status, token)