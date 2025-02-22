from loguru import logger
from pydantic import BaseModel
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from VideoAnalyzer.settings import config_settings
from VideoAnalyzer.domains.injestion.utils import (
    is_valid_url,
    download_file,
    extract_audio_from_video,
    extract_metadata_from_video,
    compress_audio,

)
import os
import uuid
from openai import OpenAI
from typing import Iterator, List


class MediaProcessor(BaseLoader):
    # Constants for file paths
    TEMP_DIR = os.path.join(os.path.dirname(__file__), "temp")
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
        if not is_valid_url(self.file_path):
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
            download_file(self.file_path, temp_input_file, logger)

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

if __name__ == "__main__":
