import dashscope
from dashscope import TextEmbedding
from typing import List, Optional
import numpy as np
import logging
import time
import random

from unstructured_ingest.embed.interfaces import BaseEmbeddingEncoder, EmbeddingConfig, EMBEDDINGS_KEY


class DashScopeEmbeddingConfig(EmbeddingConfig):
    api_key: str
    model_name: str = "text-embedding-v4"  # DashScope 默认嵌入模型
    max_retries: int = 3  # 最大重试次数
    retry_delay: float = 1.0  # 重试延迟 (秒)
    enable_debug_logging: bool = False  # 是否启用调试日志


class DashScopeEmbeddingEncoder(BaseEmbeddingEncoder):
    def __init__(self, config: DashScopeEmbeddingConfig):
        super().__init__(config=config)
        dashscope.api_key = config.api_key
        self.model_name = config.model_name
        self.max_retries = getattr(config, 'max_retries', 3)
        self.retry_delay = getattr(config, 'retry_delay', 1.0)
        self.enable_debug_logging = getattr(config, 'enable_debug_logging', False)
        
        # 设置日志
        self.logger = logging.getLogger(__name__)
        if self.enable_debug_logging:
            self.logger.setLevel(logging.DEBUG)
        
        # 统计信息
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'zero_vector_count': 0,
            'empty_text_count': 0
        }

    def embed_documents(self, elements: list[dict]) -> list[dict]:
        """
        Embed a list of text elements using DashScope TextEmbedding API.
        """
        # 复制 elements 避免修改原始数据
        elements = elements.copy()
        elements_with_text = [e for e in elements if e.get("text")]
        
        if not elements_with_text:
            return elements
        
        # 提取文本
        texts = []
        for element in elements_with_text:
            text = self.get_text_from_element(element)
            texts.append(text if text else "")
        
        # 获取 embeddings
        embeddings = []
        for i, text in enumerate(texts):
            if text.strip():  # 只处理非空文本
                embedding = self._embed_text_with_retry(text)
                embeddings.append(embedding)
            else:
                # 空文本返回零向量
                self.stats['empty_text_count'] += 1
                embeddings.append(self._get_zero_vector())
        
        # 将 embeddings 添加到对应的 elements 中
        for element, embedding in zip(elements_with_text, embeddings):
            element[EMBEDDINGS_KEY] = embedding
        
        return elements

    def get_client(self):
        """DashScope doesn't need a client object, API calls are direct"""
        return None
    
    def embed_batch(self, client, batch: list[str]) -> list[list[float]]:
        """Batch embedding for compatibility with base interface"""
        embeddings = []
        for text in batch:
            if text.strip():
                embedding = self._embed_text_with_retry(text)
                embeddings.append(embedding)
            else:
                self.stats['empty_text_count'] += 1
                embeddings.append(self._get_zero_vector())
        return embeddings

    def _embed_text_with_retry(self, text: str) -> List[float]:
        """带重试机制的文本嵌入"""
        self.stats['total_requests'] += 1
        
        for attempt in range(self.max_retries):
            try:
                # 添加随机延迟以避免并发冲突
                if attempt > 0:
                    delay = self.retry_delay * (2 ** (attempt - 1)) + random.uniform(0, 0.5)
                    time.sleep(delay)
                
                response = TextEmbedding.call(
                    model=self.model_name,
                    input=text
                )
                
                if response.status_code == 200:
                    embedding = response.output['embeddings'][0]['embedding']
                    self.stats['successful_requests'] += 1
                    if self.enable_debug_logging:
                        self.logger.debug(f"Successfully embedded text (attempt {attempt + 1})")
                    return embedding
                else:
                    error_msg = f"API error on attempt {attempt + 1}: {response.status_code} - {getattr(response, 'message', 'Unknown error')}"
                    if self.enable_debug_logging:
                        self.logger.warning(error_msg)
                    
                    # 如果是限流错误，增加延迟
                    if response.status_code == 429:  # Too Many Requests
                        time.sleep(self.retry_delay * 2)
                        
            except Exception as e:
                error_msg = f"Exception on attempt {attempt + 1}: {type(e).__name__}: {str(e)}"
                if self.enable_debug_logging:
                    self.logger.warning(error_msg)
        
        # 所有重试都失败了
        self.stats['failed_requests'] += 1
        self.stats['zero_vector_count'] += 1
        if self.enable_debug_logging:
            self.logger.error(f"Failed to embed text after {self.max_retries} attempts, returning zero vector")
        return self._get_zero_vector()

    def _embed_query(self, query: str) -> List[float]:
        """Internal method for embedding a single query"""
        if not query.strip():
            self.stats['empty_text_count'] += 1
            return self._get_zero_vector()
            
        return self._embed_text_with_retry(query)

    def embed_query(self, query: str) -> List[float]:
        """
        Embed a single query string using DashScope TextEmbedding API.
        """
        return self._embed_query(query)

    def _get_zero_vector(self) -> List[float]:
        """返回零向量"""
        if self.model_name == "text-embedding-v2":
            embedding_dim = 1536
        elif self.model_name == "text-embedding-v1":
            embedding_dim = 1536
        elif self.model_name == "text-embedding-v4":
            embedding_dim = 1024
        else:
            embedding_dim = 1024  # 默认值
        return [0.0] * embedding_dim

    def is_unit_vector(self) -> bool:
        """DashScope embeddings are not unit vectors by default."""
        return False

    def num_of_dimensions(self) -> tuple[int, ...]:
        """Return the number of dimensions for DashScope embeddings."""
        if self.model_name in ["text-embedding-v2", "text-embedding-v1"]:
            return (1536,)
        elif self.model_name == "text-embedding-v4":
            return (1024,)
        else:
            return (1024,)  # 默认值

    @staticmethod
    def get_text_from_element(element) -> str:
        """Extract text from an element."""
        if hasattr(element, 'text'):
            return element.text
        elif isinstance(element, str):
            return element
        elif isinstance(element, dict) and 'text' in element:
            return element['text']
        else:
            return str(element)

    def get_stats(self) -> dict:
        """获取嵌入统计信息"""
        total = self.stats['total_requests']
        if total > 0:
            success_rate = (self.stats['successful_requests'] / total) * 100
            failure_rate = (self.stats['failed_requests'] / total) * 100
            zero_vector_rate = (self.stats['zero_vector_count'] / total) * 100
        else:
            success_rate = failure_rate = zero_vector_rate = 0.0
            
        return {
            **self.stats,
            'success_rate_percent': round(success_rate, 2),
            'failure_rate_percent': round(failure_rate, 2),
            'zero_vector_rate_percent': round(zero_vector_rate, 2)
        }

    def reset_stats(self):
        """重置统计信息"""
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'zero_vector_count': 0,
            'empty_text_count': 0
        }