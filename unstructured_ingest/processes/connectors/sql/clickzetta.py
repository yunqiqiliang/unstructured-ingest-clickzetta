import json
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generator, Optional

import numpy as np
import pandas as pd
from pydantic import Field, Secret

from unstructured_ingest.utils.data_prep import split_dataframe
from unstructured_ingest.utils.dep_check import requires_dependencies
from unstructured_ingest.v2.interfaces.file_data import FileData
from unstructured_ingest.v2.logger import logger
from unstructured_ingest.v2.processes.connector_registry import (
    DestinationRegistryEntry,
    SourceRegistryEntry,
)
from unstructured_ingest.v2.processes.connectors.sql.sql import (
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
# -- old code
# if TYPE_CHECKING:
#     from clickzetta.connector import ClickzettaConnection
#     from clickzetta.connector.cursor import ClickzettaCursor

# --modified
if TYPE_CHECKING:
    from clickzetta.connector import connect
    # from clickzetta.connector.cursor import ClickzettaCursor

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


class ClickzettaAccessConfig(SQLAccessConfig):
    password: Optional[str] = Field(default=None, description="DB password")


class ClickzettaConnectionConfig(SQLConnectionConfig):
    access_config: Secret[ClickzettaAccessConfig] = Field(
        default=ClickzettaAccessConfig(), validate_default=True
    )
    service: str = Field(
        default=None,
        description="Your service url. "
        "Your service url.",
    )
    username: Optional[str] = Field(default=None, description="DB username")
    instance: Optional[str] = Field(default=None, description="instance id")
    workspace: Optional[str] = Field(default=None, description="workspace/database name")
    vcluster: str = Field(
        default=None,
        description="vcluster name.",
    )
    schema: str = Field(default=None, description="Database schema.", alias="schema")

    connector_type: str = Field(default=CONNECTOR_TYPE, init=False)

    @contextmanager
    # The actual clickzetta module package name is: clickzetta-connector-python
    @requires_dependencies(["clickzetta"], extras="clickzetta")
    def get_connection(self) -> Generator["ClickzettaConnection", None, None]:
        # https://docs.snowflake.com/en/developer-guide/python-connector/python-connector-api#label-snowflake-connector-methods-connect
        from clickzetta.connector import connect

        connect_kwargs = self.model_dump()
        # connect_kwargs["schema"] = connect_kwargs.pop("dbschema")
        connect_kwargs.pop("access_configs", None)
        connect_kwargs["password"] = self.access_config.get_secret_value().password
        # https://peps.python.org/pep-0249/#paramstyle
        connect_kwargs["paramstyle"] = "qmark"
        # remove anything that is none
        active_kwargs = {k: v for k, v in connect_kwargs.items() if v is not None}

        connection = connect(**active_kwargs)
        try:
            yield connection
        finally:
            connection.commit()
            connection.close()

    @contextmanager
    def get_cursor(self) -> Generator["ClickzettaCursor", None, None]:
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
    def query_db(self, file_data: SqlBatchFileData) -> tuple[list[tuple], list[str]]:
        table_name = file_data.additional_metadata.table_name
        id_column = file_data.additional_metadata.id_column
        ids = [item.identifier for item in file_data.batch_items]

        with self.connection_config.get_cursor() as cursor:
            query = "SELECT {fields} FROM {table_name} WHERE {id_column} IN ({values})".format(
                table_name=table_name,
                id_column=id_column,
                fields=(
                    ",".join(self.download_config.fields) if self.download_config.fields else "*"
                ),
                values=",".join([self.values_delimiter for _ in ids]),
            )
            logger.debug(f"running query: {query}\nwith values: {ids}")
            cursor.execute(query, binding_params=ids)
            # cursor.execute(query)
            rows = [
                tuple(row.values()) if isinstance(row, dict) else row for row in cursor.fetchall()
            ]
            columns = [col[0] for col in cursor.description]
            return rows, columns


class ClickzettaUploadStagerConfig(SQLUploadStagerConfig):
    pass


class ClickzettaUploadStager(SQLUploadStager):
    upload_stager_config: ClickzettaUploadStagerConfig


class ClickzettaUploaderConfig(SQLUploaderConfig):
    pass


@dataclass
class ClickzettaUploader(SQLUploader):
    upload_config: ClickzettaUploaderConfig = field(default_factory=ClickzettaUploaderConfig)
    connection_config: ClickzettaConnectionConfig
    connector_type: str = CONNECTOR_TYPE
    values_delimiter: str = "?"

    def prepare_data(
        self, columns: list[str], data: tuple[tuple[Any, ...], ...]
    ) -> list[tuple[Any, ...]]:
        output = []
        for row in data:
            parsed = []
            for column_name, value in zip(columns, row):
                if column_name in _DATE_COLUMNS:
                    if value is None or pd.isna(value):  # pandas is nan
                        parsed.append(None)
                    else:
                        parsed.append(parse_date_string(value))
                elif column_name in _ARRAY_COLUMNS:
                    if not isinstance(value, list) and (
                        value is None or pd.isna(value)
                    ):  # pandas is nan
                        parsed.append(None)
                    else:
                        parsed.append(json.dumps(value))
                else:
                    parsed.append(value)
            output.append(tuple(parsed))
        return output

    # def _parse_values(self, columns: list[str]) -> str:
    #     return ",".join(
    #         [
    #             (
    #                 f"PARSE_JSON({self.values_delimiter})"
    #                 if col in _ARRAY_COLUMNS
    #                 else self.values_delimiter
    #             )
    #             for col in columns
    #         ]
    #     )
    def _parse_values(self, columns: list[str]) -> str:
        return ",".join(
            [
                (
                    f"({self.values_delimiter})"
                    if col in _ARRAY_COLUMNS
                    else self.values_delimiter
                )
                for col in columns
            ]
        )

    def upload_dataframe(self, df: pd.DataFrame, file_data: FileData) -> None:
        if self.can_delete():
            self.delete_by_record_id(file_data=file_data)
        else:
            logger.warning(
                f"table doesn't contain expected "
                f"record id column "
                f"{self.upload_config.record_id_key}, skipping delete"
            )
        df.replace({np.nan: None}, inplace=True)
        self._fit_to_schema(df=df)

        columns = list(df.columns)
        stmt = "INSERT INTO {table_name} ({columns}) SELECT {values}".format(
            table_name=self.upload_config.table_name,
            columns=",".join(columns),
            values=self._parse_values(columns),
        )
        logger.info(
            f"writing a total of {len(df)} elements via"
            f" document batches to destination"
            f" table named {self.upload_config.table_name}"
            f" with batch size {self.upload_config.batch_size}"
        )
        for rows in split_dataframe(df=df, chunk_size=self.upload_config.batch_size):
            with self.connection_config.get_cursor() as cursor:
                values = self.prepare_data(columns, tuple(rows.itertuples(index=False, name=None)))
                # TODO: executemany break on 'Binding data in type (list) is not supported'
                for val in values:
                    logger.debug(f"running query: {stmt}\nwith values: {val}")
                    cursor.execute(stmt, binding_params=val)



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
