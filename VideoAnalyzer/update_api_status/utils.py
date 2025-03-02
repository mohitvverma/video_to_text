import requests
from loguru import logger

from VideoAnalyzer.update_api_status.models import RequestStatus
from VideoAnalyzer.settings import config_settings


def call_update_status_api(
        status_api_path: str,
        request_status: RequestStatus,
        token: str) -> None:
    try:
        status_api_url = f"{config_settings.API_HOSTNAME}/{status_api_path}"
        logger.info(
            f"Calling update status API with "
            f"URL: {status_api_url} and "
            f"API_PATH: {status_api_path} and "
            f"auth_token: {token} and "
            f"data: {request_status}")
        response = requests.post(
            status_api_url,
            json=request_status.model_dump(),
            headers={"Authorization": token},
        )
        if response.status_code == 200:
            logger.info(
                f'Successfully sent request for '
                f'URL: {status_api_path} for '
                f'data: "{request_status}" with '
                f'response: "{response.text}"'
            )
        else:
            logger.error(
                f'Request failed for '
                f'URL: {status_api_path} for '
                f'data: "{request_status}" with '
                f'status code: "{response.status_code}" and '
                f'response: "{response.text}"'
            )
    except requests.exceptions.RequestException as e:
        logger.exception("Error occurred during the API request", e)
