# interface the openai api

import openai
import base64
import os

from tenacity import (
    retry,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

OPENAI_API_ENGINE = None
OPENAI_API_MODEL = None

def openai_api_setup(model=None, engine=None):
    global OPENAI_API_ENGINE, OPENAI_API_MODEL
    if "OPENAI_API_TYPE" in os.environ:
        openai.api_type = os.environ["OPENAI_API_TYPE"]
    if "OPENAI_API_BASE" in os.environ:
        openai.api_base = os.environ["OPENAI_API_BASE"]
    if "OPENAI_API_VERSION" in os.environ:
        openai.api_version = os.environ["OPENAI_API_VERSION"]
    openai.organization = os.environ["OPENAI_API_ORG"]
    openai.api_key = os.environ["OPENAI_API_KEY"]
    if model == "gpt-4":
        OPENAI_API_MODEL = "gpt-4-0613"
    elif model == "gpt-3.5-turbo":
        OPENAI_API_MODEL = "gpt-3.5-turbo-16k-0613"


@retry(
    retry=retry_if_not_exception_type(openai.error.InvalidRequestError),
    wait=wait_random_exponential(min=1, max=60),
    stop=stop_after_attempt(7),
)
def send_chat_completion(messages, temperature, max_tokens):
    global OPENAI_API_ENGINE, OPENAI_API_MODEL
    response = openai.ChatCompletion.create(
        engine=OPENAI_API_ENGINE,
        model=OPENAI_API_MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response