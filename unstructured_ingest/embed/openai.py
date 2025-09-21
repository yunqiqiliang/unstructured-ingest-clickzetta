import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from pydantic import Field, SecretStr

from unstructured_ingest.embed.interfaces import (
    AsyncBaseEmbeddingEncoder,
    BaseEmbeddingEncoder,
    EmbeddingConfig,
)
from unstructured_ingest.error import (
    ProviderError,
    QuotaError,
    RateLimitError,
    UserAuthError,
    UserError,
    is_internal_error,
)
from unstructured_ingest.logger import logger
from unstructured_ingest.utils.dep_check import requires_dependencies
from unstructured_ingest.utils.tls import ssl_context_with_optional_ca_override

if TYPE_CHECKING:
    from openai import AsyncOpenAI, OpenAI


class OpenAIEmbeddingConfig(EmbeddingConfig):
    api_key: SecretStr = Field(description="API key for OpenAI")
    embedder_model_name: str = Field(
        default="text-embedding-ada-002", alias="model_name", description="OpenAI model name"
    )
    base_url: Optional[str] = Field(default=None, description="optional override for the base url")

    def __init__(self, **data):
        super().__init__(**data)
        # 如果没有提供 base_url，尝试从环境变量获取
        if self.base_url is None:
            env_base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE")
            if env_base_url:
                self.base_url = env_base_url
                logger.info(f"Using base_url from environment: {self.base_url}")

    @requires_dependencies(["openai"], extras="openai")
    def wrap_error(self, e: Exception) -> Exception:
        if is_internal_error(e=e):
            return e
        # https://platform.openai.com/docs/guides/error-codes/api-errors
        from openai import APIStatusError

        if not isinstance(e, APIStatusError):
            logger.error(f"unhandled exception from openai: {e}", exc_info=True)
            raise e
        error_code = e.code
        if 400 <= e.status_code < 500:
            # user error
            if e.status_code == 401:
                return UserAuthError(e.message)
            if e.status_code == 429:
                # 429 indicates rate limit exceeded and quote exceeded
                if error_code == "insufficient_quota":
                    return QuotaError(e.message)
                else:
                    return RateLimitError(e.message)
            return UserError(e.message)
        if e.status_code >= 500:
            return ProviderError(e.message)
        logger.error(f"unhandled exception from openai: {e}", exc_info=True)
        return e

    @requires_dependencies(["openai"], extras="openai")
    def get_models(self) -> Optional[list[str]]:
        # In case the list model endpoint isn't exposed, don't break
        from openai import APIStatusError

        client = self.get_client()
        try:
            models = [m.id for m in list(client.models.list())]
            logger.info(f"Available models from {self.base_url or 'OpenAI'}: {models}")
            return models
        except APIStatusError as e:
            if e.status_code == 404:
                logger.warning(f"Models list endpoint not available for {self.base_url or 'OpenAI'}")
                return None
        except Exception as e:
            logger.error(f"Error getting models from {self.base_url or 'OpenAI'}: {e}")
            raise self.wrap_error(e=e)

    def run_precheck(self) -> None:
        try:
            logger.info(f"Running precheck for model '{self.embedder_model_name}' with base_url: {self.base_url}")
            models = self.get_models()
            if models is None:
                logger.warning("Model list not available, skipping model validation")
                return
            if self.embedder_model_name not in models:
                logger.error(f"Model '{self.embedder_model_name}' not found in available models: {models}")
                # 对于通义千问等第三方API，如果模型列表中没有嵌入模型，但实际支持，则跳过检查
                if self.base_url and "dashscope.aliyuncs.com" in self.base_url:
                    logger.warning(f"Skipping model validation for DashScope API. Model '{self.embedder_model_name}' may still work.")
                    return
                raise UserError(
                    "model '{}' not found: {}".format(self.embedder_model_name, ", ".join(models))
                )
            else:
                logger.info(f"Model '{self.embedder_model_name}' found in available models")
        except Exception as e:
            # 对于通义千问等API，如果是模型不存在的错误，我们提供更友好的错误信息
            if "not found" in str(e) and self.base_url and "dashscope.aliyuncs.com" in self.base_url:
                logger.error(f"通义千问DashScope API不支持嵌入模型 '{self.embedder_model_name}'")
                logger.info("建议使用本地BGE模型: BAAI/bge-large-zh-v1.5")
                raise UserError(
                    f"通义千问DashScope API暂不支持嵌入模型 '{self.embedder_model_name}'。"
                    f"请使用本地BGE模型 'BAAI/bge-large-zh-v1.5' 或其他支持的嵌入API。"
                )
            raise self.wrap_error(e=e)

    @requires_dependencies(["openai"], extras="openai")
    def get_client(self) -> "OpenAI":
        from openai import DefaultHttpxClient, OpenAI

        client = DefaultHttpxClient(verify=ssl_context_with_optional_ca_override())
        client_args = {
            "api_key": self.api_key.get_secret_value(),
            "http_client": client
        }
        if self.base_url:
            client_args["base_url"] = self.base_url
            logger.info(f"Creating OpenAI client with custom base_url: {self.base_url}")
        else:
            logger.info("Creating OpenAI client with default endpoint")

        return OpenAI(**client_args)

    @requires_dependencies(["openai"], extras="openai")
    def get_async_client(self) -> "AsyncOpenAI":
        from openai import AsyncOpenAI, DefaultAsyncHttpxClient

        client = DefaultAsyncHttpxClient(verify=ssl_context_with_optional_ca_override())
        client_args = {
            "api_key": self.api_key.get_secret_value(),
            "http_client": client
        }
        if self.base_url:
            client_args["base_url"] = self.base_url
            logger.info(f"Creating AsyncOpenAI client with custom base_url: {self.base_url}")
        else:
            logger.info("Creating AsyncOpenAI client with default endpoint")

        return AsyncOpenAI(**client_args)


@dataclass
class OpenAIEmbeddingEncoder(BaseEmbeddingEncoder):
    config: OpenAIEmbeddingConfig

    def precheck(self):
        self.config.run_precheck()

    def wrap_error(self, e: Exception) -> Exception:
        return self.config.wrap_error(e=e)

    def get_client(self) -> "OpenAI":
        return self.config.get_client()

    def embed_batch(self, client: "OpenAI", batch: list[str]) -> list[list[float]]:
        try:
            logger.debug(f"Embedding batch of {len(batch)} texts with model '{self.config.embedder_model_name}'")
            response = client.embeddings.create(input=batch, model=self.config.embedder_model_name)
            embeddings = [data.embedding for data in response.data]
            logger.debug(f"Successfully generated {len(embeddings)} embeddings, dimension: {len(embeddings[0]) if embeddings else 0}")
            return embeddings
        except Exception as e:
            logger.error(f"Error in embed_batch: {e}")
            raise self.wrap_error(e)


@dataclass
class AsyncOpenAIEmbeddingEncoder(AsyncBaseEmbeddingEncoder):
    config: OpenAIEmbeddingConfig

    def precheck(self):
        self.config.run_precheck()

    def wrap_error(self, e: Exception) -> Exception:
        return self.config.wrap_error(e=e)

    def get_client(self) -> "AsyncOpenAI":
        return self.config.get_async_client()

    async def embed_batch(self, client: "AsyncOpenAI", batch: list[str]) -> list[list[float]]:
        try:
            logger.debug(f"Async embedding batch of {len(batch)} texts with model '{self.config.embedder_model_name}'")
            response = await client.embeddings.create(
                input=batch, model=self.config.embedder_model_name
            )
            embeddings = [data.embedding for data in response.data]
            logger.debug(f"Successfully generated {len(embeddings)} embeddings, dimension: {len(embeddings[0]) if embeddings else 0}")
            return embeddings
        except Exception as e:
            logger.error(f"Error in async embed_batch: {e}")
            raise self.wrap_error(e)