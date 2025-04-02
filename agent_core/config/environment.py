from typing import Annotated

from dotenv import load_dotenv
from pydantic import HttpUrl, AfterValidator, Field
from pydantic_settings import BaseSettings

load_dotenv()


def http_url(value: str) -> str:
    HttpUrl(value)
    return value


HttpUrlStr = Annotated[str, AfterValidator(http_url)]


class Environment(BaseSettings):

    agent_core_log_level: str = Field(
        "WARNING", validation_alias="agent_core_log_level"
    )
    openai_api_base: HttpUrlStr = Field(validation_alias="openai_api_base")
    openai_api_key: str = Field(validation_alias="openai_api_key")
    default_model: str = Field(validation_alias="default_model")
