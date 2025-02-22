from VideoAnalyzer.domains.models import FileMetadata
from VideoAnalyzer.settings import config_settings
from VideoAnalyzer.models import FileInjestionRequestDto
from loguru import logger
from urllib.parse import urlparse
from pydub import AudioSegment
from VideoAnalyzer.exception import VideoException
import os
from subprocess import run, CalledProcessError
import subprocess
from io import BytesIO
import requests
import math
import time
from typing import List, Any, BinaryIO
import boto3
from pathlib import Path
import botocore.exceptions


def get_s3_client(
        region_name: str,
        endpoint_url: str,
        aws_access_key_id: str,
        aws_secret_access_key: str,
) -> Any:
    return boto3.session.Session().client(
        "s3",
        region_name=region_name,
        endpoint_url=endpoint_url,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
    )


def upload_to_spaces(
        client: Any, file: BinaryIO, bucket_name: str, file_name: str, content_type: str
) -> None:
    try:
        client.upload_fileobj(
            file,
            bucket_name,
            file_name,
            ExtraArgs={"ContentType": content_type},
        )
    except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
        logger.exception("Failed to upload to spaces")
        raise Exception(f"Failed to upload to spaces: {e}") from e


def generate_video_thumbnail(pre_signed_url: str) -> BytesIO:
    """
    Generate a thumbnail from a video URL and return it as BytesIO object
    """
    thumbnail_command = [
        'ffmpeg',
        '-ss', '00:00:05',
        '-i', pre_signed_url,
        '-vframes', '1',
        '-f', 'image2pipe',
        '-vcodec', 'mjpeg',
        '-'
    ]

    thumbnail_result = subprocess.run(
        thumbnail_command,
        capture_output=True,
        check=True
    )

    return BytesIO(thumbnail_result.stdout)


def extract_metadata_from_video(
        pre_signed_url: str,
        file_name: str,
        original_file_name: str,
        bucket_name: str = config_settings.BUCKET_NAME,
) -> FileMetadata:
    """
    Extract metadata from video file and generate thumbnail
    """
    try:
        if not pre_signed_url:
            raise Exception("Failed to download from pre_signed_url")

        try:
            #Try to generate and upload thumbnail
            s3_client = get_s3_client(
                config_settings.REGION_NAME,
                config_settings.ENDPOINT_URL,
                config_settings.AWS_ACCESS_KEY_ID,
                config_settings.AWS_SECRET_ACCESS_KEY,
            )

            thumbnail_object_path = str(Path(file_name).with_suffix(".jpg"))

            upload_to_spaces(
                s3_client,
                generate_video_thumbnail(pre_signed_url),
                bucket_name,
                thumbnail_object_path,
                "image/jpeg",
            )
        except Exception:
            # If thumbnail generation fails, just return None for thumbnail_object_path
            thumbnail_object_path = None
            logger.error(f"Failed to generate thumbnail for {original_file_name}")

        return FileMetadata(
            title=original_file_name,
            author=None,
            file_name=file_name,
            original_file_name=original_file_name,
            total_pages=None,
            thumbnail_object_path=thumbnail_object_path,
        )
    except Exception as e:
        raise VideoException("Failed to download from pre_signed_url", error_detail=e)


def compress_audio(input_file, output_file, logger):
    """Compress audio file using FFmpeg"""
    try:
        logger.info(f"Compressing audio: {input_file}")
        start_time = time.time()
        command = [
            "ffmpeg",
            "-i",
            input_file,
            "-vn",
            "-map_metadata",
            "-1",
            "-ac",
            "1",
            "-c:a",
            "libopus",
            "-b:a",
            "12k",
            "-application",
            "voip",
            output_file,
        ]
        run(command, check=True)
        logger.info(
            f"Compression completed in {time.time() - start_time:.2f} seconds: {output_file}"
        )
    except Exception as e:
        logger.error(f"An error occurred during audio compression: {str(e)}")
        raise


def split_audio_into_chunks(
    input_file, temp_dir, chunk_length_ms=1800000, unique_id=""
):
    """Split audio file into chunks"""
    try:
        # Create temp directory if it doesn't exist
        os.makedirs(temp_dir, exist_ok=True)

        # Load the audio file
        audio = AudioSegment.from_file(input_file)

        # Calculate number of chunks
        num_chunks = math.ceil(len(audio) / chunk_length_ms)
        chunk_files = []

        for i in range(num_chunks):
            start = i * chunk_length_ms
            end = min((i + 1) * chunk_length_ms, len(audio))
            chunk = audio[start:end]

            chunk_name = os.path.join(temp_dir, f"chunk_{unique_id}_{i}.ogg")
            # Use libopus instead of libvorbis
            chunk.export(chunk_name, format="ogg", codec="libopus")
            chunk_files.append(chunk_name)

        return chunk_files
    except Exception as e:
        logger.error(f"Error in split_audio_into_chunks: {str(e)}")
        raise


def transcribe_audio(file_path, client, logger):
    """Transcribe audio using OpenAI's Whisper model"""
    try:
        logger.info(f"Starting transcription of: {file_path}")
        start_time = time.time()
        with open(file_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1", file=audio_file, response_format="verbose_json"
            )
        logger.info(
            f"Transcription completed in {time.time() - start_time:.2f} seconds"
        )
        return transcript
    except Exception as e:
        logger.error(f"An error occurred during transcription: {str(e)}")
        raise


def transcribe_and_combine_chunks(
    compressed_audio, temp_dir, unique_id, client, logger
):
    """Transcribe audio chunks and combine the results"""
    try:
        chunks = split_audio_into_chunks(
            compressed_audio, temp_dir, unique_id=unique_id
        )
        all_segments = []
        for i, chunk in enumerate(chunks):
            logger.info(f"Processing chunk {i + 1}/{len(chunks)}")
            transcript = transcribe_audio(chunk, client, logger)
            chunk_start_time = i * 30 * 60  # 30 minutes in seconds
            for segment in transcript.segments:
                segment.start += chunk_start_time
                segment.end += chunk_start_time
            all_segments.extend(transcript.segments)
            os.remove(chunk)
            logger.info(f"Processed and removed chunk: {chunk}")
        return all_segments
    except Exception as e:
        logger.error(f"An error occurred during chunk transcription: {str(e)}")
        raise


def extract_audio_from_video(input_video, output_audio, logger):
    """Extract audio from video file using FFmpeg"""
    try:
        if not os.path.exists(input_video):
            logger.error(f"Input file does not exist: {input_video}")
            raise FileNotFoundError(f"Input file does not exist: {input_video}")

        logger.info(f"Extracting audio from video: {input_video}")
        command = [
            "ffmpeg",
            "-i",
            input_video,
            "-vn",
            "-ac",
            "1",
            "-c:a",
            "libopus",
            "-b:a",
            "12k",
            "-application",
            "voip",
            output_audio,
        ]
        run(command, check=True)
        logger.info(f"Audio extracted and saved to: {output_audio}")
    except Exception as e:
        logger.error(f"An error occurred during audio extraction: {str(e)}")
        raise


def download_file(url: str, output_path: str, logger) -> str:
    """Download file from URL to local path with progress tracking"""
    try:
        response = requests.get(url, stream=True, timeout=30)
        if response.status_code != 200:
            raise ValueError(
                f"Failed to download file: status code {response.status_code}"
            )

        total_size = int(response.headers.get("content-length", 0))
        chunk_size = 131072  # 128KB chunks
        downloaded = 0

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)

                    if total_size > 0 and downloaded % (total_size // 20) < chunk_size:
                        logger.info(
                            f"Download progress: {(downloaded / total_size) * 100:.1f}% ({downloaded/(1024*1024):.1f}MB)"
                        )

        logger.info(
            f"File downloaded successfully to: {output_path} (Total: {total_size/(1024*1024):.1f}MB)"
        )
        return output_path

    except Exception as e:
        logger.error(f"Error downloading file: {str(e)}")
        raise


def is_valid_url(url: str) -> bool:
    """Validate if the provided string is a valid URL"""
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except:
        return False
