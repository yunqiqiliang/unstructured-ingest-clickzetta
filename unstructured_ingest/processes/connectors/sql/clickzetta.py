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

    class Config:
        populate_by_name = True

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

        connect_kwargs = self.model_dump()
        connect_kwargs.pop("access_configs", None)
        connect_kwargs["password"] = self.access_config.get_secret_value().password
        connect_kwargs["paramstyle"] = "qmark"
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
    values_delimiter: str = "?"

    # The actual clickzetta module package name is: clickzetta-connector-python
    @requires_dependencies(["clickzetta"], extras="clickzetta")
    # def query_db(self, file_data: SqlBatchFileData) -> tuple[list[tuple], list[str]]:
    #     table_name = file_data.additional_metadata.table_name
    #     id_column = file_data.additional_metadata.id_column
    #     ids = [item.identifier for item in file_data.batch_items]

    #     with self.connection_config.get_cursor() as cursor:
    #         query = """SELECT {fields} FROM {table_name} WHERE {id_column} IN ({values})""".format(
    #             table_name=table_name,
    #             id_column=id_column,
    #             fields=(
    #                 ",".join(self.download_config.fields) if self.download_config.fields else "*"
    #             ),
    #             values=",".join([self.values_delimiter for _ in ids]),
    #         )
    #         # logger.debug(f"running query: {query}\nwith values: {ids}")
    #         cursor.execute(query, binding_params=ids)
    #         # cursor.execute(query)
    #         rows = [
    #             tuple(row.values()) if isinstance(row, dict) else row for row in cursor.fetchall()
    #         ]
    #         columns = [col[0] for col in cursor.description]
    #         return rows, columns
    def query_db(self, file_data: SqlBatchFileData) -> tuple[list[tuple], list[str]]:
        table_name = file_data.additional_metadata.table_name
        id_column = file_data.additional_metadata.id_column
        ids = [item.identifier for item in file_data.batch_items]

        with self.connection_config.get_session() as session:
            query = """SELECT {fields} FROM {table_name} WHERE {id_column} IN ({values})""".format(
                table_name=table_name,
                id_column=id_column,
                fields=(
                    ",".join(self.download_config.fields) if self.download_config.fields else "*"
                ),
                values=",".join([self.values_delimiter for _ in ids]),
            )
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
        self.upload_config.batch_size = 1000
        self._batch_buffer = []  # 批量缓冲区
        self._buffer_size = 0

    def is_batch(self) -> bool:
        """启用批量处理模式"""
        return True

    def run_batch(self, contents: list, **kwargs) -> None:
        """批量处理多个文件的数据"""
        logger.info(f"Processing batch of {len(contents)} files")
        
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
            logger.info(f"Batch processing {len(all_data)} total elements from {len(contents)} files")
            # 使用第一个文件的 file_data 作为代表（因为批量上传需要一个 file_data）
            representative_file_data = contents[0].file_data if contents else None
            self.run_data(data=all_data, file_data=representative_file_data)
        else:
            logger.warning("No data found in batch to process")

    def _parse_values(self, columns: List[str]) -> str:
        return ",".join([self.values_delimiter for _ in columns])

    def upload_dataframe(self, df: pd.DataFrame, file_data: FileData) -> None:
        import numpy as np

        logger.info(f"Processing {len(df)} elements for upload")

        # 1. 获取目标表所有字段名（建议硬编码或通过元数据获取）
        required_columns = [
            "id", "record_locator", "type", "record_id", "element_id", "filetype", "file_directory",
            "filename", "last_modified", "languages", "page_number", "text", "embeddings", "parent_id",
            "is_continuation", "orig_elements", "element_type", "coordinates", "link_texts", "link_urls",
            "email_message_id", "sent_from", "sent_to", "subject", "url", "version", "date_created",
            "date_modified", "date_processed", "text_as_html", "emphasized_text_contents", "emphasized_text_tags","documents_source"
        ]

        # 2. 补齐缺失列
        for col in required_columns:
            if col not in df.columns:
                df[col] = None

        # 3. 保证列顺序一致
        df = df[required_columns]

        # if self.can_delete():
        #     self.delete_by_record_id(file_data=file_data)
        # else:
        #     logger.warning(
        #         f"table doesn't contain expected "
        #         f"record id column "
        #         f"{self.upload_config.record_id_key}, skipping delete"
        #     )
        df.replace({np.nan: None}, inplace=True)
        
        # Skip _fit_to_schema for ClickZetta as it will auto-create table with save_as_table
        # self._fit_to_schema(df=df)
        
        df_schema = generate_df_schema(df)
        columns = list(df.columns)

        logger.info(
            f"Uploading {len(df)} elements in batches of {self.upload_config.batch_size} to table {self.upload_config.table_name}"
        )

        batch_count = 0
        for rows in split_dataframe(df=df, chunk_size=self.upload_config.batch_size):
            batch_count += 1
            batch_size = len(rows)
            logger.debug(f"Processing batch {batch_count} with {batch_size} records")
            
            with self.connection_config.get_session() as session:
                values = self.prepare_data(columns, tuple(rows.itertuples(index=False, name=None)))
                values_df = pd.DataFrame(values, columns=columns)
                # --- 新增：将 embeddings 列转换为 vector 类型 ---
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
                # --- 设置 documents_source 列的值 ---
                if "documents_source" in values_df.columns:
                    values_df["documents_source"] = self.upload_config.documents_original_source
                # --- end ---
                zetta_df = session.create_dataframe(values_df, schema=df_schema)
                zetta_df.write.mode("append").save_as_table(self.upload_config.table_name)
                logger.debug(f"Successfully uploaded batch {batch_count} with {batch_size} records")

        logger.info(f"Completed upload of {len(df)} elements in {batch_count} batches")


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
