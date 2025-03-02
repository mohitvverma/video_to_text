from loguru import logger
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from VideoAnalyzer.settings import config_settings
from VideoAnalyzer.domains.injestion.utils import (
    is_valid_url,
    download_file,
    extract_audio_from_video,
    extract_metadata_from_video,
    compress_audio,
    transcribe_and_combine_chunks,
    format_transcription,
    cleanup_temp_files,
)
import os
import uuid
import pprint
from openai import OpenAI
from typing import Iterator, List, Any, Tuple
from VideoAnalyzer.domains.injestion.exception import FileLoaderException
from langchain_community.document_loaders import TextLoader, PyPDFLoader

from typing import get_args, Callable
from VideoAnalyzer.models import FILE_TYPE
from VideoAnalyzer.vector_db.utils import split_text
from langchain.chains.summarize import load_summarize_chain
import json
from VideoAnalyzer.utils import get_chat_model


class MediaProcessor(BaseLoader):
    # Constants for file paths
    # TEMP_DIR = os.path.join(os.path.dirname(__file__), "downloaded_audio_data_files")
    TEMP_DIR = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), "downloaded_audio_data_files")
    logger.info(f"Current working directory: {os.getcwd()}. Real path: {os.path.realpath(os.getcwd())}")
    logger.info(f"TEMP_DIR: {TEMP_DIR}")

    EXTRACTED_AUDIO_TEMPLATE = "extracted_audio_{unique_id}.ogg"
    COMPRESSED_AUDIO_TEMPLATE = "compressed_audio_{unique_id}.ogg"
    TRANSCRIPT_TXT_TEMPLATE = "transcript_{unique_id}.txt"
    TRANSCRIPT_JSON_TEMPLATE = "transcript_{unique_id}.json"

    def __init__(self, file_path: str, file_type: str) -> None:
        """
        Initialize MediaProcessor with a file URL and type
        Args:
            file_path (str): Pre-signed URL of the media file
            file_type (str): File type/extension (e.g., 'mp3', 'mp4')
        """
        self.file_path = file_path
        self.file_type = file_type.lower()
        self.client = OpenAI(api_key=config_settings.OPENAI_API_KEY)

        # Validate URL
        if not is_valid_url(self.file_path) and not os.path.isfile(self.file_path):
            raise ValueError(f"Upload file url is invalid")

        # Ensure the temporary directory exists
        os.makedirs(self.TEMP_DIR, exist_ok=True)

        # Generate a unique ID for this processing session
        self.unique_id = str(uuid.uuid4())[:8]

        # Define output file paths using constants
        self.extracted_audio = os.path.join(
            self.TEMP_DIR,
            self.EXTRACTED_AUDIO_TEMPLATE.format(unique_id=self.unique_id),
        )
        self.compressed_audio = os.path.join(
            self.TEMP_DIR,
            self.COMPRESSED_AUDIO_TEMPLATE.format(unique_id=self.unique_id),
        )
        self.transcript_txt = os.path.join(
            self.TEMP_DIR, self.TRANSCRIPT_TXT_TEMPLATE.format(unique_id=self.unique_id)
        )
        self.transcript_json = os.path.join(
            self.TEMP_DIR,
            self.TRANSCRIPT_JSON_TEMPLATE.format(unique_id=self.unique_id),
        )
        super().__init__()

    def lazy_load(self) -> Iterator[Document]:
        """Process media file and yield Document objects with transcription segments."""
        try:
            logger.info(f"Starting media processing for file type: {self.file_type}")

            # Download file
            temp_input_file = os.path.join(
                self.TEMP_DIR, f"input_{self.unique_id}.{self.file_type}"
            )
            logger.info(f"Downloading file to temporary location: {temp_input_file}")

            if not os.path.isfile(self.file_path):
                logger.info(f"Downloading file from {self.file_path}")
                download_file(self.file_path, temp_input_file, logger)

            # Verify the file exists after download
            if not os.path.exists(temp_input_file):
                raise FileNotFoundError(f"Downloaded file does not exist: {temp_input_file}")

            # Determine file type and processing path
            is_video = self.file_type in ["mp4", "mkv", "avi", "mov"]
            logger.info(f"File identified as: {'video' if is_video else 'audio'}")

            # Process based on file type
            if is_video:
                logger.info("Starting video processing workflow...")
                extract_audio_from_video(temp_input_file, self.extracted_audio, logger)
                audio_to_process = self.extracted_audio
                audio_final = audio_to_process
                logger.info("Video processing workflow completed")
            else:
                logger.info("Starting audio processing workflow...")
                audio_to_process = temp_input_file
                logger.info("Compressing audio file...")
                compress_audio(audio_to_process, self.compressed_audio, logger)
                audio_final = self.compressed_audio
                logger.info("Audio processing workflow completed")

            # Transcribe audio
            logger.info("Starting transcription process...")
            file_size = os.path.getsize(audio_final) / (1024 * 1024)  # Convert to MB
            logger.info(f"Processing audio file of size: {file_size:.2f}MB")

            all_segments = transcribe_and_combine_chunks(
                audio_final, self.TEMP_DIR, self.unique_id, self.client, logger
            )
            logger.info(
                f"Transcription completed. Generated {len(all_segments)} segments"
            )

            # Format transcription
            logger.info("Formatting transcription into documents...")
            documents = format_transcription(all_segments, logger)
            logger.info(f"Created {len(documents)} document objects")

            # Yield documents
            logger.info("Starting document yield process...")
            doc_count = 0
            for doc in documents:
                doc_count += 1
                if doc_count % 100 == 0:  # Log progress every 100 documents
                    logger.info(f"Yielded {doc_count}/{len(documents)} documents")
                yield doc

            logger.info(
                f"Successfully completed processing. Total documents yielded: {doc_count}"
            )

        except Exception as e:
            logger.error(
                f"Critical error during media processing: {str(e)}", exc_info=True
            )
            raise
        finally:
            logger.info("Cleaning up temporary files...")
            cleanup_temp_files(self.TEMP_DIR)
            logger.info("Cleanup completed")

    def load(self) -> List[Document]:
        """Implementation of load for BaseLoader."""
        return list(self.lazy_load())


class TextFileLoader:
    def __init__(self, file_path: str, process_type: str = "text"):
        self.file_path = file_path
        self.process_type = process_type
        # Validate the process type
        self._validate_process_type()

    def _validate_process_type(self) -> None:
        """Checks if the process_type is valid."""
        valid_types = ["text", "pdf"]

        if self.process_type not in valid_types:
            raise ValueError(f"Invalid process type: {self.process_type}. Supported types are: {', '.join(valid_types)}")


    def _validate_file_path(self) -> None:
        """Checks if the file path exists."""
        if not os.path.isfile(self.file_path):
            raise FileNotFoundError(f"File not found: {self.file_path}")

    def load(self) -> Document | str:
        """
        Loads the file based on the specified process type.

        Returns:
            Document or str: Loaded Document object or an error message string.
        """
        try:
            logger.info(f"{self.__class__.__name__}.load(): Attempting to load file from {self.file_path}")

            # Validate file path
            self._validate_file_path()

            if self.process_type == "text":
                text_loader = TextLoader(
                    file_path=self.file_path
                )

                file_contents=text_loader.load()
                return file_contents

            else:
                raise ValueError(f"Unsupported process type: {self.process_type}")

        except FileNotFoundError as fnf_error:
            logger.error(f"{self.__class__.__name__}.load(): File not found - {fnf_error}")
            return "Error: File does not exist."
        except ValueError as val_error:
            logger.error(f"{self.__class__.__name__}.load(): Value error - {val_error}")
            return str(val_error)
        except Exception as e:
            logger.error(f"{self.__class__.__name__}.load(): Unexpected error - {e}")
            return "Error: An unexpected error occurred while loading the file."


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
) -> Tuple[list[Document], str, Any]:

    if file_type not in get_args(FILE_TYPE):
        raise FileLoaderException(f"{file_type} is not a supported file type")

    loaders: dict[str, Callable[[], BaseLoader]] = {
        "text": lambda: TextFileLoader(pre_signed_url, process_type="text"),
        "pdf": lambda: TextFileLoader(pre_signed_url, process_type="pdf"),
        "audio": lambda: MediaProcessor(pre_signed_url, file_type),
        "video": lambda: MediaProcessor(pre_signed_url, file_type),
    }

    if (loader := loaders.get(process_type)) is None:
        raise FileNotFoundError("Unsupported process_type")

    loaded_documents = loader().load()
    logger.info(f"documents loaded {len(loaded_documents)}")

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
                for doc in loaded_documents
            ]
        }

    parsed_documents: list[Document] = []
    tags = params.get("tags") or []
    synonyms = params.get("synonyms") or []
    document_summary=""

    parsed_documents = split_text(
        text=loaded_documents,
        CHUNK_SIZE=config_settings.CHUNK_SIZE,
        CHUNK_OVERLAP=config_settings.CHUNK_OVERLAP
    )

    # Generate summary
    if params.get("summary", False):
        document_summary = params.get("summary", "")
    else:
        logger.info("Generating document summary")
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
        "tags": tags,
        "synonyms": synonyms,
    }

    if metadata:
        for i in metadata:
            additional_metadata.update(i)

    for document in parsed_documents:
        document.metadata |= additional_metadata | {
            "title": document.metadata.get("title") or original_file_name
        }

    return parsed_documents, document_summary, transcript_json


if __name__ == "__main__":
    res=file_loader(
        file_name="test.mp3",
        original_file_name="test.mp3",
        file_type="mp3",
        process_type="audio",
        pre_signed_url="https://mohitver1999.s3.eu-north-1.amazonaws.com/128-Ishq%20Mein%20-%20Nadaaniyan%20128%20Kbps.mp3?response-content-disposition=inline&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Security-Token=IQoJb3JpZ2luX2VjEIn%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCmV1LW5vcnRoLTEiRzBFAiEA5Qjy6j2gjdw%2BR5JAtxwCSjoL%2Flc41%2Bf441jjt%2FfAJFECIEEVN0ledbjtql26q7W6kEm7NM7ZVlQs%2BGySah4o%2BeJSKtADCML%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEQABoMNDQyNDI2ODg0NzE3IgwCmSHNVL7QkLmIZZYqpAMyx4aO1kMP2ekCiUcl1O4n1fBU6oMo5Ae7GpgmKlz%2BNwTCwcC2%2FdXoC3iJE%2FEdeRByAU5XLkFk%2F8xnNZF%2BcVgd4xSh65%2FWSyt%2Fb5ozWJKqArMsMxpQmYu8thCJOqHqtNuQV8xTR%2B83QVMEocm1hOA8wX%2F%2Fxp3QlvTYaj%2Bkqfm3LLDIUVzuiyqkUuixFIBfz4N4zM3valtNDjsd9LaDvA1LUdM5RafI8jJafm1tl0Z2oqU8qDASoYX9KlXU5u2xVcK9PZnHa8DzytSDmTcogYFAr0dMW9jTGAj8Ks6%2B4zj3%2F1ttcefcZBbQ38edMezr%2FTBTT9UV%2B5rX8Eznk5GsB4%2FVmYM9IifKQrsBC5bpx36fXzIWJ5zn5nqaTR%2BqH8W345a1d0ObR3uPpjspwizTQA5aVgl9jym7yaGgarMWlXY%2FweO3r%2BFEGYa71J6eAa7H1L9gjV3z4pG0eRrgv8%2BLYEjDQGPZLxGE18NR7lZfx6S%2B7hi3ISppLGKvA0gYQyhyU1kEp1XWvZ5GOCv%2BFOF4OMcyc7fa9TU8pYEz9Wiia78eMma0NlYwzZ2SvgY65AL8GcETj36Xb1XsycnR7xCv4KEH5C1ge9yDCeHb7ScBPvflR3NknEA84Aq%2BSF28gdFgdGReOSCVQllBWjnBItTHLDJFTUfPgPLTU8QIL1dssviv4HT7msaKZTClJknil0WsFGK6kh367cW08OgMW7FGhYwbZrBv4Zv9jM2mClqV1m4xE9UpLNIobz6kDzQDNvPGmWI2tr7QZZaftrz5Rxdlx5VnRLKOP8FJtt1w817O1cmuTh%2Fj8lbdK530JUR1xa4o8yzEN54ObsCMs3G82fYWBX8%2F9k3VRD%2FGHHjzCqONVSj1UVv6PQmCwTga4ZjqGd9eRJ6Rkno3eut%2BNTCoxcaKFxAzczFKJrgMWHmfBzKWlpAtVJMqoKcV3XpiEymPTjqw%2BALyF2MDkO%2F2XwykdC8kT3JjQY06EkB8lw6%2BrY39TD7FyB5s4A02h9ujQ2oBW3uIUALGXIPaprJhUBLABKyKRpDu8g%3D%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=ASIAWOAVSQJWWTTL6GKM%2F20250302%2Feu-north-1%2Fs3%2Faws4_request&X-Amz-Date=20250302T170152Z&X-Amz-Expires=32460&X-Amz-SignedHeaders=host&X-Amz-Signature=4a4e1b1eb65f64ce1da6e82d79eadadb32d73c3d8bec98eb8185fd5765dfc6e4",
        request_id=1,
        response_data_api_path="",
        params={"summary": False},
        metadata=[]
    )
    pprint.pprint(res)
    # print(os.path.dirname(os.path.dirname(os.getcwd())))
    # print(os.path.relpath(os.getcwd(), os.path.dirname(os.path.dirname(os.getcwd()))))
    # print(os.path.realpath(os.path.dirname(__file__)))
    # print(os.path.relpath(os.getcwd()))
