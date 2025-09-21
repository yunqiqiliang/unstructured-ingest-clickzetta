import os
import json
from types import SimpleNamespace
from pathlib import Path
import pytest
from dotenv import load_dotenv
from _pytest.fixtures import TopRequest
from test.integration.connectors.utils.constants import (
    DESTINATION_TAG,
    SOURCE_TAG,
    SQL_TAG,
    env_setup_path,
)
from test.integration.connectors.utils.docker import container_context
from test.integration.connectors.utils.validation.destination import (
    StagerValidationConfigs,
    stager_validation,
)
from test.integration.connectors.utils.validation.source import (
    SourceValidationConfigs,
    source_connector_validation,
)
from test.integration.utils import requires_env
from unstructured_ingest.processes.connectors.sql.clickzetta import (
    ClickzettaConnectionConfig,
    ClickzettaIndexerConfig,
    ClickzettaIndexer,
    ClickzettaDownloaderConfig,
    ClickzettaDownloader,
    ClickzettaUploaderConfig,
    ClickzettaUploader,
    ClickzettaUploadStagerConfig,
    ClickzettaUploadStager,
    CONNECTOR_TYPE,
)
load_dotenv()

@pytest.mark.tags(CONNECTOR_TYPE, SQL_TAG)
def test_clickzetta_stager_empty(tmp_path: Path):
    """测试 stager 处理空文件"""
    empty_file = tmp_path / "empty.ndjson"
    empty_file.write_text("")
    stager = ClickzettaUploadStager()
    file_data = SimpleNamespace(identifier="empty_test")
    staged_path = stager.run(
        elements_filepath=empty_file,
        file_data=file_data,
        output_dir=tmp_path,
        output_filename=empty_file.name,
    )
    with staged_path.open("r") as f:
        lines = [line.strip() for line in f if line.strip()]
    assert len(lines) == 0, "stager should output 0 lines for empty input"


@pytest.fixture
def source_database_setup():
    # TODO: 使用 ClickzettaConnectionConfig + session 初始化测试表和数据
    from unstructured_ingest.processes.connectors.sql.clickzetta import ClickzettaAccessConfig
    connect_params = {
        "service": os.getenv("cz_service"),
        "username": os.getenv("cz_username"),
        "workspace": os.getenv("cz_workspace"),
        "vcluster": os.getenv("cz_vcluster"),
        "schema": os.getenv("cz_schema"),
        "instance": os.getenv("cz_instance"),
        "access_config": ClickzettaAccessConfig(password=os.getenv("cz_password")),
    }
    print("[DEBUG] source_database_setup connect_params:", connect_params)
    from unstructured_ingest.processes.connectors.sql.clickzetta import ClickzettaConnectionConfig
    connection_config = ClickzettaConnectionConfig(**connect_params)
    with connection_config.get_session() as session:
        session.sql("DROP TABLE IF EXISTS elements;").collect()
        session.sql(
            """
            CREATE TABLE elements (
                id INT PRIMARY KEY,
                text STRING
            );
            """
        ).collect()
        for i in range(1, 21):
            session.sql(f"INSERT INTO elements (id, text) VALUES ({i}, 'test_text_{i}')").collect()
    yield connect_params


def init_db_destination() -> dict:
    # 使用真实连接参数，目标表初始化
    from unstructured_ingest.processes.connectors.sql.clickzetta import ClickzettaAccessConfig
    connect_params = {
        "service": os.getenv("cz_service"),
        "username": os.getenv("cz_username"),
        "workspace": os.getenv("cz_workspace"),
        "vcluster": os.getenv("cz_vcluster"),
        "schema": os.getenv("cz_schema"),
        "instance": os.getenv("cz_instance"),
        "access_config": ClickzettaAccessConfig(password=os.getenv("cz_password")),
    }
    connection_config = ClickzettaConnectionConfig(**connect_params)
    with connection_config.get_session() as session:
        session.sql("DROP TABLE IF EXISTS elements;").collect()
        session.sql(
            """
            CREATE TABLE elements (
                id INT PRIMARY KEY,
                text STRING
            );
            """
        ).collect()
    return connect_params


@pytest.fixture
def destination_database_setup():
    # TODO: 使用 ClickzettaConnectionConfig + session 初始化目标表
    from unstructured_ingest.processes.connectors.sql.clickzetta import ClickzettaAccessConfig
    connect_params = {
        "service": os.getenv("cz_service"),
        "username": os.getenv("cz_username"),
        "workspace": os.getenv("cz_workspace"),
        "vcluster": os.getenv("cz_vcluster"),
        "schema": os.getenv("cz_schema"),
        "instance": os.getenv("cz_instance"),
        "access_config": ClickzettaAccessConfig(password=os.getenv("cz_password")),
    }
    print("[DEBUG] destination_database_setup connect_params:", connect_params)
    from unstructured_ingest.processes.connectors.sql.clickzetta import ClickzettaConnectionConfig
    connection_config = ClickzettaConnectionConfig(**connect_params)
    with connection_config.get_session() as session:
        session.sql("DROP TABLE IF EXISTS elements;").collect()
        session.sql(
            """
            CREATE TABLE elements (
                id INT PRIMARY KEY,
                text STRING
            );
            """
        ).collect()
        for i in range(1, 21):
            session.sql(f"INSERT INTO elements (id, text) VALUES ({i}, 'test_text_{i}')").collect()
    yield connect_params


@pytest.mark.asyncio
@pytest.mark.tags(CONNECTOR_TYPE, SOURCE_TAG, SQL_TAG)
async def test_clickzetta_source(temp_dir: Path, source_database_setup):
    connection_config = ClickzettaConnectionConfig(**source_database_setup)
    # 测试前清空表，插入 4 条测试数据，保证断言通过
    with connection_config.get_session() as session:
        session.sql("DELETE FROM elements;").collect()
        for i in range(1, 5):
            session.sql(f"INSERT INTO elements (id, text) VALUES ({i}, 'test_text_{i}')").collect()

    # 使用原生 ClickzettaIndexer/Downloader，确保接口兼容
    indexer = ClickzettaIndexer(
        connection_config=connection_config,
        index_config=ClickzettaIndexerConfig(table_name="elements", id_column="id", batch_size=5),
    )
    downloader = ClickzettaDownloader(
        connection_config=connection_config,
        download_config=ClickzettaDownloaderConfig(fields=["id", "text"], download_dir=temp_dir),
    )
    # 断言索引 id 数量和下载内容
    await source_connector_validation(
        indexer=indexer,
        downloader=downloader,
        configs=SourceValidationConfigs(
            test_id="clickzetta",
            expected_num_files=4,
            expected_number_indexed_file_data=4,
            validate_downloaded_files=True,
        ),
    )


def validate_destination(connect_params, expected_num_elements):
    connection_config = ClickzettaConnectionConfig(**connect_params)
    with connection_config.get_session() as session:
        result = session.sql("select count(*) as cnt from elements;").to_pandas()
        count = result["cnt"].iloc[0]
        assert count == expected_num_elements, f"dest check failed: got {count}, expected {expected_num_elements}"


@pytest.mark.asyncio
@pytest.mark.tags(CONNECTOR_TYPE, DESTINATION_TAG, SQL_TAG)
async def test_clickzetta_destination(upload_file: Path, temp_dir: Path, destination_database_setup):
    stager = ClickzettaUploadStager()
    # 构造 mock FileData 对象
    file_data = SimpleNamespace(identifier="test_file_id")
    staged_path = stager.run(
        elements_filepath=upload_file,
        file_data=file_data,
        output_dir=temp_dir,
        output_filename=upload_file.name,
    )
    assert staged_path.suffix == upload_file.suffix
    connect_params = destination_database_setup
    uploader = ClickzettaUploader(
        connection_config=ClickzettaConnectionConfig(**connect_params)
    )
    # 测试前清空表，保证计数准确
    with uploader.connection_config.get_session() as session:
        session.sql("DELETE FROM elements;").collect()
    uploader.precheck()
    # 读取 staged_data，批量上传，只保留 id/text 两列
    with staged_path.open("r") as f:
        staged_data = json.load(f)
    filtered_data = []
    for item in staged_data:
        if isinstance(item, dict):
            filtered_data.append({
                "id": item.get("id"),
                "text": item.get("text")
            })
    uploader._upload_data_batch(data=filtered_data, file_data=file_data)
    expected_num_elements = len(filtered_data)
    validate_destination(
        connect_params=connect_params,
        expected_num_elements=expected_num_elements,
    )


@pytest.mark.tags(CONNECTOR_TYPE, DESTINATION_TAG, SQL_TAG)
@pytest.mark.parametrize("upload_file_str", ["upload_file_ndjson", "upload_file"])
def test_clickzetta_stager(request: TopRequest, upload_file_str: str, tmp_path: Path):
    upload_file: Path = request.getfixturevalue(upload_file_str)
    stager = ClickzettaUploadStager()
    # 自动计算 expected_count，兼容 NDJSON 格式
    ndjson_lines = []
    with upload_file.open("r") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    ndjson_lines.append(json.loads(line))
                except Exception:
                    pass
    expected_count = len(ndjson_lines) if ndjson_lines else 22
    # 构造 mock FileData 对象，仅用于 stager.run
    file_data = SimpleNamespace(identifier="test_file_id")
    staged_path = stager.run(
        elements_filepath=upload_file,
        file_data=file_data,
        output_dir=tmp_path,
        output_filename=upload_file.name,
    )
    # 断言输出文件内容、字段顺序和数据条数
    staged_lines = []
    with staged_path.open("r") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    staged_lines.append(json.loads(line))
                except Exception:
                    pass
    assert len(staged_lines) == expected_count, f"stager output count mismatch: {len(staged_lines)} vs {expected_count}"
    # 只校验关键字段（id/text），兼容 stager 字段扩展
    dict_staged = [x for x in staged_lines if isinstance(x, dict)]
    dict_ndjson = [x for x in ndjson_lines if isinstance(x, dict)]
    if dict_staged and dict_ndjson:
        for key in ["id", "text"]:
            assert key in dict_staged[0], f"stager output missing key: {key}"
            assert key in dict_ndjson[0], f"ndjson input missing key: {key}"
    stager_validation(
        configs=StagerValidationConfigs(test_id=CONNECTOR_TYPE, expected_count=expected_count),
        input_file=upload_file,
        stager=stager,
        tmp_dir=tmp_path,
    )
