import dashscope
from dashscope import TextEmbedding
from typing import List, Optional
import numpy as np

from unstructured_ingest.embed.interfaces import BaseEmbeddingEncoder, EmbeddingConfig


class DashScopeEmbeddingConfig(EmbeddingConfig):
    api_key: str
    model_name: str = "text-embedding-v4"  # DashScope 默认嵌入模型


class DashScopeEmbeddingEncoder(BaseEmbeddingEncoder):
    def __init__(self, config: DashScopeEmbeddingConfig):
        super().__init__(config=config)
        dashscope.api_key = config.api_key
        self.model_name = config.model_name

    def embed_documents(self, elements) -> List[List[float]]:
        """
        Embed a list of text elements using DashScope TextEmbedding API.
        """
        embeddings = []
        
        # 批量处理文本 - 参考 OpenAI 的实现方式
        texts = []
        for element in elements:
            text = self.get_text_from_element(element)
            texts.append(text if text else "")
        
        # 批量调用 DashScope API
        for text in texts:
            if text.strip():  # 只处理非空文本
                try:
                    response = TextEmbedding.call(
                        model=self.model_name,
                        input=text
                    )
                    
                    if response.status_code == 200:
                        embedding = response.output['embeddings'][0]['embedding']
                        embeddings.append(embedding)
                    else:
                        print(f"DashScope API error: {response.message}")
                        embeddings.append(self._get_zero_vector())
                        
                except Exception as e:
                    print(f"Error embedding text with DashScope: {e}")
                    embeddings.append(self._get_zero_vector())
            else:
                # 空文本返回零向量
                embeddings.append(self._get_zero_vector())
                
        return embeddings

    def embed_query(self, query: str) -> List[float]:
        """
        Embed a single query string using DashScope TextEmbedding API.
        """
        if not query.strip():
            return self._get_zero_vector()
            
        try:
            response = TextEmbedding.call(
                model=self.model_name,
                input=query
            )
            
            if response.status_code == 200:
                return response.output['embeddings'][0]['embedding']
            else:
                print(f"DashScope API error: {response.message}")
                return self._get_zero_vector()
                
        except Exception as e:
            print(f"Error embedding query with DashScope: {e}")
            return self._get_zero_vector()

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