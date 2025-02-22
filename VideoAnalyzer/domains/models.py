from typing import TypedDict

class FileMetadata(TypedDict):
    title: str | None
    author: str | None
    file_name: str | None
    original_file_name: str | None
    total_pages: int | None
    thumbnail_object_path: str | None