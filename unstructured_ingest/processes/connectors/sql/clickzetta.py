import json
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generator, List, Optional, Tuple

import numpy as np
import pandas as pd
from pydantic import Field, Secret

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
from unstructured_ingest.utils.data_prep import split_dataframe
from unstructured_ingest.utils.dep_check import requires_dependencies

if TYPE_CHECKING:
    pass
    
import clickzetta.zettapark.types as T
from clickzetta.zettapark.session import Session

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
    "date_modified", "date_processed", "text_as_html", "emphasized_text_contents",
    "emphasized_text_tags",
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
        # 特殊处理embeddings列：检查是否为float数组
        if column_name == "embeddings":
            # 检查实际数据来确定vector类型
            sample_data = df[column_name].dropna()
            if len(sample_data) > 0:
                sample_val = sample_data.iloc[0]
                if isinstance(sample_val, list) and len(sample_val) > 0:
                    vector_size = len(sample_val)
                    if vector_size in [512, 768, 1024, 1536]:
                        field_type = T.VectorType('float', vector_size)
                        logger.debug(f"检测到embeddings为vector类型，维度: {vector_size}")
                    else:
                        field_type = T.ArrayType()
                        logger.warning(
                            f"⚠️ embeddings维度({vector_size})不在支持列表中，使用ArrayType"
                        )
                else:
                    field_type = T.StringType()
                    logger.warning("⚠️ embeddings数据类型异常，使用StringType")
            else:
                field_type = T.VectorType('float', 1024)  # 默认1024维
                logger.debug("embeddings列无数据，使用默认vector(float,1024)")
        else:
            field_type = type_mapping.get(str(dtype), T.StringType())

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
            session.sql(
                "select 'Initialize session to the Clickzetta by unstructured ingest Tool';"
            ).collect()
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
        fields_str = (
            ','.join(self.download_config.fields)
            if self.download_config.fields
            else '*'
        )
        query = (
            f"SELECT {fields_str} FROM {table_name} "
            f"WHERE {id_column} IN ({id_list_str})"
        )
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
        import numpy as np
        import pandas as pd
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
            for batch_count, rows in enumerate(
                split_dataframe(df=df, chunk_size=self.upload_config.batch_size), 1
            ):
                values = self.prepare_data(columns, tuple(rows.itertuples(index=False, name=None)))
                values_df = pd.DataFrame(values, columns=columns)

                # 使用直接SQL插入方法
                logger.debug(
                    f"DataFrame列数: {len(values_df.columns)}, "
                    f"Schema字段数: {len(df_schema.fields)}"
                )

                # 使用直接SQL插入，绕过save_as_table可能的问题
                self._direct_sql_insert(session, values_df, self.upload_config.table_name)
    
    def _parse_values(self, columns: List[str]) -> str:
        return ",".join([self.values_delimiter for _ in columns])

    def upload_dataframe(self, df: pd.DataFrame, file_data: FileData) -> None:
        """智能批处理：累积到batch_size后再刷新，提高效率"""
        # 将数据添加到缓冲区
        self._batch_buffer.append(df)
        self._buffer_size += len(df)

        # 智能批处理策略：只有达到batch_size时才刷新
        if self._buffer_size >= self.upload_config.batch_size:
            logger.info(f"达到批处理阈值，开始批量写入 ({self._buffer_size} 行)")
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

        # 移除pandas自动生成的索引列（如__index_level_0__）
        auto_index_columns = [
            col for col in combined_df.columns if col.startswith('__index_level_')
        ]
        if auto_index_columns:
            combined_df = combined_df.drop(columns=auto_index_columns)
            # 重新应用列清理以确保列数正确
            combined_df = _ensure_required_columns(combined_df)

        # 在列清理后生成schema，确保schema与实际数据列数一致
        df_schema = generate_df_schema(combined_df)
        columns = list(combined_df.columns)

        # 使用单个session处理所有数据
        with self.connection_config.get_session() as session:

            for rows in split_dataframe(df=combined_df, chunk_size=self.upload_config.batch_size):
                values = self.prepare_data(columns, tuple(rows.itertuples(index=False, name=None)))
                values_df = pd.DataFrame(values, columns=columns)

                # 确保values_df列顺序与schema一致（已在combined_df级别完成清理）
                values_df = _ensure_required_columns(values_df)

                # 移除pandas自动生成的索引列（如__index_level_0__）
                auto_index_columns = [
                    col for col in values_df.columns
                    if col.startswith('__index_level_')
                ]
                if auto_index_columns:
                    logger.debug(f"移除pandas自动生成的索引列: {auto_index_columns}")
                    values_df = values_df.drop(columns=auto_index_columns)

                # 将 embeddings 列从JSON字符串转换为float数组（vector类型需要）
                if "embeddings" in values_df.columns:
                    def json_to_vector(val):
                        if val is None or pd.isna(val):
                            return None
                        # 如果是字符串，先转为list
                        if isinstance(val, str):
                            try:
                                val = json.loads(val)
                            except Exception:
                                return None
                        # 确保是list类型并转为float数组
                        if isinstance(val, list):
                            return [float(x) for x in val]
                        return None

                    logger.debug("转换embeddings列：从JSON字符串到float数组")
                    values_df["embeddings"] = values_df["embeddings"].apply(json_to_vector)

                    # 检查转换结果
                    non_null_embeddings = values_df["embeddings"].dropna()
                    if len(non_null_embeddings) > 0:
                        sample_embedding = non_null_embeddings.iloc[0]
                        logger.debug(f"embeddings转换结果: 类型={type(sample_embedding)}, 长度={len(sample_embedding) if sample_embedding else 0}")

                # 设置 documents_original_source 列的值
                if "documents_original_source" in values_df.columns:
                    values_df["documents_original_source"] = self.upload_config.documents_original_source

                # 强制确保列数匹配：重新生成schema以匹配当前数据
                if len(values_df.columns) != len(df_schema.fields):
                    logger.warning(f"列数不匹配，重新生成schema：DataFrame({len(values_df.columns)}) vs Schema({len(df_schema.fields)})")
                    # 重新应用列清理，确保exactly 33列
                    values_df = _ensure_required_columns(values_df)
                    # 重新生成schema
                    current_schema = generate_df_schema(values_df)
                    df_schema = current_schema
                    logger.warning(f"已重新生成schema，新的字段数: {len(df_schema.fields)}")

                # 详细调试列数不匹配问题
                logger.debug("准备写入数据")
                logger.info(f"  values_df.shape: {values_df.shape}")
                logger.info(f"  values_df.columns.tolist(): {values_df.columns.tolist()}")
                logger.info(f"  df_schema.fields: {len(df_schema.fields)}")
                logger.info(f"  df_schema field names: {[f.name for f in df_schema.fields]}")

                # 检查是否有重复列
                if len(values_df.columns) != len(set(values_df.columns)):
                    duplicate_cols = [col for col in values_df.columns if values_df.columns.tolist().count(col) > 1]
                    logger.error(f"发现重复列: {duplicate_cols}")

                # 检查列名与schema的差异（移除schema字段名的反引号进行比较）
                df_cols = set(values_df.columns)
                schema_cols = set(f.name.strip('`') for f in df_schema.fields)  # 移除反引号
                if df_cols != schema_cols:
                    logger.error("列名不匹配:")
                    logger.error(f"  仅在DataFrame中: {df_cols - schema_cols}")
                    logger.error(f"  仅在Schema中: {schema_cols - df_cols}")
                else:
                    logger.debug("DataFrame与Schema列名一致（除了反引号格式）")

                try:
                    # 方法1（主要）：使用schema，为vector类型提供正确的类型定义
                    logger.debug("尝试方法1：使用schema（优先，支持vector类型）")

                    # 使用更简洁的方法：确保DataFrame没有索引列并重置索引
                    final_clean_df = values_df.copy()

                    # 移除任何索引列
                    index_cols = [col for col in final_clean_df.columns
                                 if col.startswith('__index_level_') or col.startswith('level_') or 'index' in col.lower()]
                    if index_cols:
                        logger.debug(f"移除索引列: {index_cols}")
                        final_clean_df = final_clean_df.drop(columns=index_cols)

                    # 强制重置索引为简单的RangeIndex，避免任何索引相关问题
                    final_clean_df = final_clean_df.reset_index(drop=True)

                    logger.debug(f"传递给ClickZetta的DataFrame: shape={final_clean_df.shape}, index_type={type(final_clean_df.index).__name__}")

                    zetta_df_with_schema = session.create_dataframe(final_clean_df, schema=df_schema)

                    # 检查zetta_df的列数
                    zetta_cols_with_schema = zetta_df_with_schema.columns
                    logger.debug(f"使用schema - zetta_df.columns: {len(zetta_cols_with_schema)}")
                    logger.debug(f"使用schema - zetta_df.columns names: {zetta_cols_with_schema}")

                    zetta_df_with_schema.write.mode("append").save_as_table(self.upload_config.table_name)
                    logger.info(f"成功写入 {len(values_df)} 行数据")

                except Exception as e:
                    logger.error(f"方法1失败: {e}")

                    # 方法2（后备）：不使用schema的自动推断
                    logger.debug("尝试方法2：不使用schema的自动推断（后备）")
                    try:
                        # 方法2也使用相同的清理逻辑
                        final_clean_df2 = values_df.copy()

                        # 移除任何索引列
                        index_cols2 = [col for col in final_clean_df2.columns
                                      if col.startswith('__index_level_') or col.startswith('level_') or 'index' in col.lower()]
                        if index_cols2:
                            logger.debug(f"方法2移除索引列: {index_cols2}")
                            final_clean_df2 = final_clean_df2.drop(columns=index_cols2)

                        # 强制重置索引
                        final_clean_df2 = final_clean_df2.reset_index(drop=True)

                        logger.debug(f"方法2传递给ClickZetta的DataFrame: shape={final_clean_df2.shape}, index_type={type(final_clean_df2.index).__name__}")

                        zetta_df = session.create_dataframe(final_clean_df2)

                        # 检查zetta_df的列数
                        zetta_cols = zetta_df.columns
                        logger.debug(f"自动推断 - zetta_df.columns: {len(zetta_cols)}")
                        logger.debug(f"自动推断 - zetta_df.columns names: {zetta_cols}")

                        zetta_df.write.mode("append").save_as_table(self.upload_config.table_name)
                        logger.info(f"成功写入 {len(values_df)} 行数据")

                    except Exception as e2:
                        logger.error(f"方法2也失败: {e2}")
                        logger.error("最终调试信息:")
                        logger.error(f"  DataFrame shape: {values_df.shape}")
                        logger.error(f"  Embeddings列样本: {values_df['embeddings'].iloc[0] if 'embeddings' in values_df.columns and len(values_df) > 0 else 'N/A'}")
                        raise

            # 清空缓冲区
            self._batch_buffer = []
            self._buffer_size = 0

    def run_data(self, data: list[dict], file_data: FileData, **kwargs) -> None:
        """重写run_data方法，支持智能批处理"""
        df = pd.DataFrame(data)
        self.upload_dataframe(df=df, file_data=file_data)

        # 不在此处强制刷新，让upload_dataframe的批处理逻辑处理
        # 只在最后通过finish()方法刷新剩余数据
    
    def finish(self):
        """完成上传，刷新所有剩余的缓冲区数据"""
        if hasattr(self, '_batch_buffer') and self._batch_buffer:
            logger.info(f"Pipeline完成，刷新剩余数据: {self._buffer_size} 行")
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

    def _direct_sql_insert(self, session, df: pd.DataFrame, table_name: str):
        """直接使用 SQL INSERT 语句插入数据，绕过 save_as_table 可能的问题"""

        if len(df) == 0:
            return

        # 确保列数正确
        df = _ensure_required_columns(df)

        # 生成 INSERT SQL
        columns_str = ", ".join([f"`{col}`" for col in df.columns])
        placeholders = ", ".join(["?" for _ in df.columns])

        insert_sql = f"""
        INSERT INTO {table_name} ({columns_str})
        VALUES ({placeholders})
        """

        logger.info(f"使用直接SQL插入，列数: {len(df.columns)}")

        # 准备数据
        values_list = []
        for _, row in df.iterrows():
            values = []
            for col in df.columns:
                val = row[col]

                # 处理特殊数据类型
                if col == "embeddings" and val is not None:
                    if isinstance(val, str):
                        try:
                            val = json.loads(val)
                        except Exception:
                            val = None
                    if isinstance(val, list):
                        val = json.dumps(val)
                elif col in _ARRAY_COLUMNS and val is not None:
                    if not isinstance(val, str):
                        val = json.dumps(val) if val else None
                elif col in _DATE_COLUMNS and val is not None:
                    if pd.isna(val):
                        val = None
                    else:
                        # 处理不同类型的日期数据
                        if isinstance(val, pd.Timestamp):
                            # 如果是pandas Timestamp，直接转换为字符串
                            val = val.strftime('%Y-%m-%d %H:%M:%S')
                        elif isinstance(val, str):
                            # 如果是字符串，使用parse_date_string处理
                            val = parse_date_string(val)
                        else:
                            # 其他类型尝试转换为字符串后解析
                            try:
                                val = parse_date_string(str(val))
                            except Exception:
                                val = None
                elif pd.isna(val):
                    val = None

                values.append(val)

            values_list.append(tuple(values))

        # 批量插入
        batch_size = 100
        for i in range(0, len(values_list), batch_size):
            batch = values_list[i:i + batch_size]

            with self.connection_config.get_cursor() as cursor:
                cursor.executemany(insert_sql, batch)

        logger.info(f"成功插入 {len(values_list)} 行数据")

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
