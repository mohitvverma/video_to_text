from loguru import logger
from typing import Optional, Any, Tuple
from VideoAnalyzer.models import FileInjestionRequestDto



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



def file_loader(
    pre_signed_url: str,
    file_name: str,
    original_file_name: str,
    file_type: str,
    process_type: str,
    request_id: int,
    response_data_api_path: str,
    params: dict[str, Any],
    metadata: list[dict[str, str]] = [{}],
    token: str = "",
) -> Tuple[list[Document], str, Any]:
    if file_type not in get_args(FILE_TYPE):
        raise FileLoaderException(f"{file_type} is not a supported file type")

    loaders: dict[str, Callable[[], BaseLoader]] = {
        "text": lambda: PDFLoaderExtended(pre_signed_url, extract_images=False),
        "image": lambda: ImageLoader(pre_signed_url, file_type),
        "audio": lambda: MediaProcessor(pre_signed_url, file_type),
        "video": lambda: MediaProcessor(pre_signed_url, file_type),
    }

    if (loader := loaders.get(process_type)) is None:
        raise FileNotFoundError("Unsupported process_type")

    documents = loader().load()
    logger.info(f"documents loaded {len(documents)}")

    # Format transcript for audio/video files
    transcript_json = None
    if process_type in ["audio", "video"]:
        transcript_json = {
            "transcript": [
                {
                    "text": doc.page_content,
                    "start_time": doc.metadata["start_time"],
                    "end_time": doc.metadata["end_time"],
                }
                for doc in documents
            ]
        }

    parsed_documents: list[Document] = []
    tags = params.get("tags") or []
    synonyms = params.get("synonyms") or []

    # Process SQL Excel-specific logic
    if process_type == "excel":
        logger.info("Inside the SQL Excel Data Processing")

        document_content = documents["documents"]
        logger.info(f"Before split total: {len(document_content)} documents")

        sheets_summary = documents["sheet_summary"]
        logger.info(f"Total sheet summary: {len(sheets_summary)} sheets")
        logger.info(f"sheet summay: {sheets_summary}")

        documents = split_text(document_content, CHUNK_SIZE, CHUNK_OVERLAP)
        logger.info(f"Total documents after split: {len(documents)}")

        for doc in documents:
            parsed_documents.append(
                Document(
                    page_content=json.dumps({
                        "file_name": original_file_name,
                        "file_type": file_type,
                        "tags": list(set(tags + synonyms)),
                        "page_content": doc.page_content,
                    }),
                    metadata=doc.metadata,
                )
            )

        # Generate summary if required
        if params.get("summary"):
            document_summary = params["summary"]
        else:
            chain = load_summarize_chain(
                get_chat_model(model_key="SUMMARIZE_LLM_MODEL"),
                chain_type="stuff",
                verbose=True,
            )
            summary_result = chain.invoke(input={"input_documents": sheets_summary})
            document_summary = summary_result.get("output_text", "")
            logger.info(f"Document summary generated for SQL Excel: {document_summary}")

            # Add additional metadata
        additional_metadata = {
            "original_file_name": original_file_name,
            "file_name": file_name,
            "file_type": file_type,
            "process_type": process_type,
            "tags": tags,
            "synonyms": synonyms,
        }
    else:
        # Split documents for embedding
        documents = split_text(documents, CHUNK_SIZE, CHUNK_OVERLAP)

        for doc in documents:
            document = Document(
                page_content=json.dumps(
                    {
                        "file_name": original_file_name,
                        "file_type": file_type,
                        "tags": list(set(tags + synonyms)),
                        "page_content": doc.page_content,
                    }
                ),
                metadata=doc.metadata,
            )
            parsed_documents.append(document)

        # Generate summary
        if params.get("summary", False):
            document_summary = params.get("summary", "")
        else:
            llm = get_chat_model(model_key="SUMMARIZE_LLM_MODEL")
            if llm:
                chain = load_summarize_chain(
                    llm,
                    chain_type="stuff",
                    verbose=True,
                )
                summary = chain.invoke(
                    input={
                        "input_documents": parsed_documents[
                            : config_settings.INITIAL_NUMBER_OF_PAGES_TO_RETRIEVE_FOR_SUMMARIZATION
                        ]
                    }
                )
            document_summary = summary.get("output_text", "")

        additional_metadata = {
            "original_file_name": original_file_name,
            "file_name": file_name,
            "file_type": file_type,
            "process_type": process_type,
            "document_summary": document_summary,
            "tags": tags,
            "synonyms": synonyms,
        }

    if metadata:
        for i in metadata:
            additional_metadata.update(i)

    for document in parsed_documents:
        document.metadata |= additional_metadata | {
            "title": document.metadata.get(TITLE_KEY) or original_file_name
        }

    return parsed_documents, document_summary, transcript_json




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
        raise FileLoaderException(f"Exception during file load {e}") from e