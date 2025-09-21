import json
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generator, Optional, List, Dict, Tuple, Union

import numpy as np
import pandas as pd
from pydantic import Field, Secret

from unstructured_ingest.utils.data_prep import split_dataframe
from unstructured_ingest.utils.dep_check import requires_dependencies
from unstructured_ingest.data_types.file_data import FileData
from unstructured_ingest.logger import logger
from unstructured_ingest.processes.connector_registry import (
    DestinationRegistryEntry,
    SourceRegistryEntry,
)
from unstructured_ingest.processes.connectors.sql.sql import (
    _DATE_COLUMNS,
    SQLAccessConfig,
    SqlBatchFileData,
    SQLConnectionConfig,
    SQLDownloader,
    SQLDownloaderConfig,
    SQLIndexer,
    SQLIndexerConfig,
    SQLUploader,
    SQLUploaderConfig,
    SQLUploadStager,
    SQLUploadStagerConfig,
    parse_date_string,
)

if TYPE_CHECKING:
    from clickzetta.connector import connect
    
from clickzetta.zettapark.session import Session
from clickzetta.connector.sqlalchemy.datatype import VECTOR
from sqlalchemy.types import BIGINT
import clickzetta.zettapark.types as T

CONNECTOR_TYPE = "clickzetta"

_ARRAY_COLUMNS = (
    "embeddings",
    "languages",
    "link_urls",
    "link_texts",
    "sent_from",
    "sent_to",
    "emphasized_text_contents",
    "emphasized_text_tags",
)

# ClickZetta表的所有列定义（33列）
_REQUIRED_COLUMNS = [
    "id", "record_locator", "type", "record_id", "element_id", "filetype", "file_directory",
    "filename", "last_modified", "languages", "page_number", "text", "embeddings", "parent_id",
    "is_continuation", "orig_elements", "element_type", "coordinates", "link_texts", "link_urls",
    "email_message_id", "sent_from", "sent_to", "subject", "url", "version", "date_created",
    "date_modified", "date_processed", "text_as_html", "emphasized_text_contents", "emphasized_text_tags",
    "documents_original_source"
]
def _ensure_required_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    确保DataFrame包含所有必需的列，并按正确顺序排列

    Args:
        df: 输入的DataFrame

    Returns:
        包含所有必需列的DataFrame
    """
    # 补齐缺失列
    for col in _REQUIRED_COLUMNS:
        if col not in df.columns:
            df[col] = None

    # 保证列顺序一致并清理NaN值
    return df[_REQUIRED_COLUMNS].copy().replace({np.nan: None})


def generate_df_schema(df: pd.DataFrame) -> T.StructType:
    """
    Generate a schema definition for a DataFrame in the format of T.StructType.

    Args:
        df (pd.DataFrame): The DataFrame for which to generate the schema.

    Returns:
        T.StructType: The schema definition.
    """
    type_mapping = {
        "int64": T.IntegerType(),
        "float64": T.FloatType(),
        "object": T.StringType(),
        "bool": T.BooleanType(),
        "datetime64[ns]": T.TimestampType(),
        "vector('float',512)": T.VectorType('float',512),
        "vector('float',768)": T.VectorType('float',768),
        "vector('float',1024)": T.VectorType('float',1024),
        "vector('float',1536)": T.VectorType('float',1536),
        "bigint": T.LongType(),
        "string": T.StringType(),
        "array": T.ArrayType()
    }

    fields = []
    for column_name, dtype in df.dtypes.items():
        field_type = type_mapping.get(str(dtype), T.StringType())  # Default to StringType if type is unknown
        fields.append(T.StructField(column_name, field_type))

    return T.StructType(fields)

class ClickzettaAccessConfig(SQLAccessConfig):
    password: Optional[str] = Field(default=None, description="DB password")


class ClickzettaConnectionConfig(SQLConnectionConfig):
    schema: str = Field(default=None, description="Schema name for Clickzetta.")
    access_config: Secret[ClickzettaAccessConfig] = Field(
        default=ClickzettaAccessConfig(), validate_default=True
    )
    service: str = Field(
        default=None,
        description="Your service url. "
        "Your service url.",
    )
    username: Optional[str] = Field(default=None, description="username")
    instance: Optional[str] = Field(default=None, description="instance id")
    workspace: Optional[str] = Field(default=None, description="workspace/database name")
    vcluster: str = Field(
        default=None,
        description="vcluster name.",
    )
    connector_type: str = Field(default=CONNECTOR_TYPE, init=False)
    # Pydantic v2: 使用 model_config 替代 class Config 以消除弃用警告
    model_config = {
        "populate_by_name": True
    }

    @contextmanager
    @requires_dependencies(["clickzetta"], extras="clickzetta")
    def get_session(self) -> Generator["Session", None, None]:
        from clickzetta.zettapark.session import Session

        connect_kwargs = {
            "service": self.service,
            "username": self.username,
            "instance": self.instance,
            "workspace": self.workspace,
            "vcluster": self.vcluster,
            "schema": self.schema,
            "password": self.access_config.get_secret_value().password,
        }
        active_kwargs = {k: v for k, v in connect_kwargs.items() if v is not None}
        session = None  # 防止finally报错
        try:
            session = Session.builder.configs(active_kwargs).create()
            session.sql("select 'Initialize session to the Clickzetta by unstructured ingest Tool';").collect()
            yield session
        finally:
            if session:
                session.close()

    @contextmanager
    @requires_dependencies(["clickzetta"], extras="clickzetta")
    def get_connection(self) -> Generator[Any, None, None]:
        from clickzetta.connector import connect
        # 明确映射字段，避免 model_dump 可能的嵌套或大小写问题
        connect_kwargs = {
            "service": self.service,
            "username": self.username,
            "instance": self.instance,
            "workspace": self.workspace,
            "vcluster": self.vcluster,
            "schema": self.schema,
            "password": self.access_config.get_secret_value().password,
            "paramstyle": "qmark",
        }
        # 自动 strip 所有字符串参数，并打印类型和值
        for k, v in connect_kwargs.items():
            if isinstance(v, str):
                connect_kwargs[k] = v.strip()
        active_kwargs = {k: v for k, v in connect_kwargs.items() if v is not None}
        connection = connect(**active_kwargs)
        try:
            yield connection
        finally:
            connection.commit()
            connection.close()

    @contextmanager
    def get_cursor(self) -> Generator[Any, None, None]:
        with self.get_connection() as connection:
            cursor = connection.cursor()
            try:
                yield cursor
            finally:
                cursor.close()


class ClickzettaIndexerConfig(SQLIndexerConfig):
    pass


@dataclass
class ClickzettaIndexer(SQLIndexer):
    connection_config: ClickzettaConnectionConfig
    index_config: ClickzettaIndexerConfig
    connector_type: str = CONNECTOR_TYPE


class ClickzettaDownloaderConfig(SQLDownloaderConfig):
    pass


@dataclass
class ClickzettaDownloader(SQLDownloader):
    connection_config: ClickzettaConnectionConfig
    download_config: ClickzettaDownloaderConfig
    connector_type: str = CONNECTOR_TYPE
    # 修正：ClickZetta 不支持 IN (?, ?, ?) 占位符，需直接拼接 id 列表
    @requires_dependencies(["clickzetta"], extras="clickzetta")
    def query_db(self, file_data: SqlBatchFileData) -> tuple[list[dict], list[str]]:
        table_name = file_data.additional_metadata.table_name
        id_column = file_data.additional_metadata.id_column
        ids = [item.identifier for item in file_data.batch_items]

        # 直接拼接 id 列表为字符串，防止 SQL 语法错误
        id_list_str = ",".join([f"'{str(i)}'" for i in ids]) if ids else "''"
        query = f"SELECT {','.join(self.download_config.fields) if self.download_config.fields else '*'} FROM {table_name} WHERE {id_column} IN ({id_list_str})"
        with self.connection_config.get_session() as session:
            result = session.sql(query).to_pandas()
            rows = result.to_dict(orient="records")
            columns = list(rows[0].keys()) if rows else []
            return rows, columns


class ClickzettaUploadStagerConfig(SQLUploadStagerConfig):
    pass


class ClickzettaUploadStager(SQLUploadStager):
    upload_stager_config: ClickzettaUploadStagerConfig


class ClickzettaUploaderConfig(SQLUploaderConfig):
    documents_original_source: str = Field(default="unknown", description="Source of the documents")



@dataclass
class ClickzettaUploader(SQLUploader):
    upload_config: ClickzettaUploaderConfig = field(default_factory=ClickzettaUploaderConfig)
    connection_config: ClickzettaConnectionConfig
    connector_type: str = CONNECTOR_TYPE
    values_delimiter: str = "?"

    def prepare_data(
        self, columns: List[str], data: Tuple[Tuple[Any, ...], ...]
    ) -> List[Tuple[Any, ...]]:
        output = []
        for row in data:
            parsed = []
            for column_name, value in zip(columns, row):
                if column_name in _DATE_COLUMNS:
                    if value is None or pd.isna(value):
                        parsed.append(None)
                    else:
                        parsed.append(parse_date_string(value))
                elif column_name in _ARRAY_COLUMNS:
                    if not isinstance(value, list) and (value is None or pd.isna(value)):
                        parsed.append(None)
                    else:
                        parsed.append(json.dumps(value))
                else:
                    parsed.append(value)
            output.append(tuple(parsed))
        return output

    def __post_init__(self):
        # 如果没有设置batch_size，使用默认值1000
        if not hasattr(self.upload_config, 'batch_size') or self.upload_config.batch_size is None:
            self.upload_config.batch_size = 1000
        self._batch_buffer = []  # 批量缓冲区
        self._buffer_size = 0

    def is_batch(self) -> bool:
        """启用批量处理模式"""
        return True

    def run_batch(self, contents: list, **kwargs) -> None:
        """批量处理多个文件的数据"""
        
        all_data = []
        all_file_data = []
        
        # 收集所有文件的数据
        for content in contents:
            try:
                from unstructured_ingest.utils.data_prep import get_json_data
                data = get_json_data(path=content.path)
                if data:
                    all_data.extend(data)
                    # 为每条记录保存文件数据引用
                    all_file_data.extend([content.file_data] * len(data))
            except Exception as e:
                logger.warning(f"Failed to load data from {content.path}: {e}")
                continue
        
        if all_data:
            # 使用第一个文件的 file_data 作为代表（因为批量上传需要一个 file_data）
            representative_file_data = contents[0].file_data if contents else None

            # 直接调用优化的批量上传方法
            self._upload_data_batch(data=all_data, file_data=representative_file_data)

    def _upload_data_batch(self, data: list[dict], file_data: FileData) -> None:
        """批量上传，只保留目标表字段（id/text），兼容 stager 输出"""
        import pandas as pd
        import numpy as np
        # 只保留 id/text 两列，兼容 stager 输出
        filtered_data = []
        for item in data:
            if isinstance(item, dict):
                filtered_data.append({
                    "id": item.get("id"),
                    "text": item.get("text")
                })
        df = pd.DataFrame(filtered_data)
        # 补齐缺失列
        for col in ["id", "text"]:
            if col not in df.columns:
                df[col] = None
        df = df[["id", "text"]].copy()
        df = df.replace({np.nan: None})
        df_schema = generate_df_schema(df)
        columns = list(df.columns)
        with self.connection_config.get_session() as session:
            batch_count = 0
            for rows in split_dataframe(df=df, chunk_size=self.upload_config.batch_size):
                batch_count += 1
                batch_size = len(rows)
                values = self.prepare_data(columns, tuple(rows.itertuples(index=False, name=None)))
                values_df = pd.DataFrame(values, columns=columns)
                zetta_df = session.create_dataframe(values_df, schema=df_schema)
                zetta_df.write.mode("append").save_as_table(self.upload_config.table_name)
    
    def _parse_values(self, columns: List[str]) -> str:
        return ",".join([self.values_delimiter for _ in columns])

    def upload_dataframe(self, df: pd.DataFrame, file_data: FileData) -> None:
        """🔧 智能缓冲：小批量立即刷新，大批量使用缓冲"""
        # 将数据添加到缓冲区
        self._batch_buffer.append(df)
        self._buffer_size += len(df)
        
        # 🔧 强制立即刷新策略：确保数据不丢失
        # 对于任何数据都立即刷新，避免缓冲区导致的数据丢失问题
        self._flush_buffer()
        
        # 注册atexit回调，确保程序退出时刷新缓冲区
        if not hasattr(self, '_atexit_registered'):
            import atexit
            atexit.register(self._safe_flush)
            self._atexit_registered = True
    
    def _flush_buffer(self):
        """刷新缓冲区，批量上传所有数据"""
        if not self._batch_buffer:
            return
            
        try:
            # 合并所有DataFrame
            combined_df = pd.concat(self._batch_buffer, ignore_index=True)
        except Exception as e:
            logger.error(f"合并DataFrame失败: {e}")
            # 如果concat失败，尝试逐个处理
            try:
                for df in self._batch_buffer:
                    self._write_single_df(df)
                self._batch_buffer = []
                self._buffer_size = 0
                return
            except Exception as e2:
                logger.error(f"逐个处理DataFrame也失败: {e2}")
                return
        
        # 补齐缺失列并保证列顺序一致
        combined_df = _ensure_required_columns(combined_df)

        df_schema = generate_df_schema(combined_df)
        columns = list(combined_df.columns)

        # 使用单个session处理所有数据
        with self.connection_config.get_session() as session:

            for rows in split_dataframe(df=combined_df, chunk_size=self.upload_config.batch_size):
                values = self.prepare_data(columns, tuple(rows.itertuples(index=False, name=None)))
                values_df = pd.DataFrame(values, columns=columns)

                # 应用列补齐逻辑，确保所有33列都存在
                values_df = _ensure_required_columns(values_df)

                # 将 embeddings 列转换为 vector 类型
                if "embeddings" in values_df.columns:
                    def to_vector(val):
                        if val is None:
                            return None
                        # 如果是字符串，先转为list
                        if isinstance(val, str):
                            try:
                                val = json.loads(val)
                            except Exception:
                                return None
                        # 转为float数组
                        return [float(x) for x in val]
                    values_df["embeddings"] = values_df["embeddings"].apply(to_vector)

                # 设置 documents_original_source 列的值
                if "documents_original_source" in values_df.columns:
                    values_df["documents_original_source"] = self.upload_config.documents_original_source

                try:
                    zetta_df = session.create_dataframe(values_df, schema=df_schema)
                    zetta_df.write.mode("append").save_as_table(self.upload_config.table_name)
                except Exception as e:
                    logger.error(f"写入失败 - {e}")
                    raise

            # 清空缓冲区
            self._batch_buffer = []
            self._buffer_size = 0

    def run_data(self, data: list[dict], file_data: FileData, **kwargs) -> None:
        """重写run_data方法，确保最后刷新缓冲区"""
        df = pd.DataFrame(data)
        self.upload_dataframe(df=df, file_data=file_data)
        
        # 确保在处理完成后刷新缓冲区
        # 这个方法通常是处理单个文件的最后步骤
        # 注意：这可能导致小批次上传，但能确保数据不丢失
        if hasattr(self, '_batch_buffer') and self._batch_buffer and self._buffer_size < self.upload_config.batch_size:
            # 如果是最后一批数据（缓冲区未满），等待更多数据
            # 但如果等待时间过长，应该刷新
            pass
    
    def finish(self):
        """完成上传，刷新所有剩余的缓冲区数据"""
        if hasattr(self, '_batch_buffer') and self._batch_buffer:
            self._flush_buffer()
    
    def _safe_flush(self):
        """安全的刷新方法，用于atexit回调"""
        try:
            if hasattr(self, '_batch_buffer') and self._batch_buffer:
                self._flush_buffer()
        except Exception as e:
            logger.error(f"安全刷新失败: {e}")
    
    def _write_single_df(self, df):
        """写入单个DataFrame到数据库"""
        
        # 补齐缺失列并保证列顺序一致
        df = _ensure_required_columns(df)

        df_schema = generate_df_schema(df)
        table_full_name = f"{self.workspace}.{self.db_schema}.{self.table_name}"
        try:
            self.connection.write_pandas(
                df=df,
                table_name=table_full_name,
                if_exists="append",
                df_schema=df_schema
            )
        except Exception as e:
            logger.error(f"写入DataFrame失败: {e}")
            raise

    def __del__(self):
        """析构函数，确保刷新剩余的缓冲区数据"""
        if hasattr(self, '_batch_buffer') and self._batch_buffer:
            try:
                self._flush_buffer()
            except Exception as e:
                logger.error(f"析构函数中刷新缓冲区失败: {e}")



clickzetta_source_entry = SourceRegistryEntry(
    connection_config=ClickzettaConnectionConfig,
    indexer_config=ClickzettaIndexerConfig,
    indexer=ClickzettaIndexer,
    downloader_config=ClickzettaDownloaderConfig,
    downloader=ClickzettaDownloader,
)

clickzetta_destination_entry = DestinationRegistryEntry(
    connection_config=ClickzettaConnectionConfig,
    uploader=ClickzettaUploader,
    uploader_config=ClickzettaUploaderConfig,
    upload_stager=ClickzettaUploadStager,
    upload_stager_config=ClickzettaUploadStagerConfig,
)
