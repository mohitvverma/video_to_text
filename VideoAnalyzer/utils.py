from openai import OpenAI, AzureOpenAI
from langchain_openai import (
    ChatOpenAI,
    AzureChatOpenAI,
    AzureOpenAIEmbeddings,
)
from langchain_ollama import ChatOllama
from langchain_ollama.llms import OllamaLLM
from VideoAnalyzer.settings import config_settings
from loguru import logger


def get_chat_model(model_key: str = "RAG_LLM_MODEL", temperature: float = 0.0):
    if config_settings.LLM_SERVICE == "openai":
        return ChatOpenAI(
            model=config_settings.LLMS.get(model_key, ""), temperature=temperature,
            stream_usage=True
        )

    elif config_settings.LLM_SERVICE == "azure-openai":
        return AzureChatOpenAI(
            azure_endpoint=config_settings.AZURE_OPENAI_SETTINGS[model_key]["ENDPOINT"],
            azure_deployment=config_settings.AZURE_OPENAI_SETTINGS[model_key][
                "DEPLOYMENT"
            ],
            api_key=config_settings.AZURE_OPENAI_SETTINGS[model_key]["API_KEY"],
            api_version=config_settings.AZURE_OPENAI_SETTINGS[model_key]["API_VERSION"],
            model=config_settings.LLMS.get(model_key, ""),
            temperature=temperature,
        )

    elif config_settings.LLM_SERVICE == "ollama":
        return ChatOllama(
            model=config_settings.OLLAMA_LLM_SETTING.get(model_key, ""), temperature=temperature
        )


def get_openai_client(model_key: str):
    if config_settings.LLM_SERVICE == "openai":
        return OpenAI(api_key=config_settings.OPENAI_API_KEY)

    elif config_settings.LLM_SERVICE == "azure-openai":
        return AzureOpenAI(
            azure_endpoint=config_settings.AZURE_OPENAI_SETTINGS[model_key]["ENDPOINT"],
            azure_deployment=config_settings.AZURE_OPENAI_SETTINGS[model_key][
                "DEPLOYMENT"
            ],
            api_key=config_settings.AZURE_OPENAI_SETTINGS[model_key]["API_KEY"],
            api_version=config_settings.AZURE_OPENAI_SETTINGS[model_key]["API_VERSION"],
        )

    elif config_settings.LLM_SERVICE == "ollama":
        return OllamaLLM(
            model=config_settings.OLLAMA_LLM_SETTING.get(model_key, "")
        )
