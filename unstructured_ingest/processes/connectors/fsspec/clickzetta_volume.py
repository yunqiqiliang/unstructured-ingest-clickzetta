from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generator, Optional, List, Dict
import os
from pathlib import Path

from pydantic import Field, Secret, BaseModel

from unstructured_ingest.data_types.file_data import FileDataSourceMetadata
from unstructured_ingest.errors_v2 import UserAuthError, UserError
from unstructured_ingest.logger import logger
from unstructured_ingest.processes.connector_registry import (
    DestinationRegistryEntry,
    SourceRegistryEntry,
)
from unstructured_ingest.processes.connectors.fsspec.fsspec import (
    FsspecAccessConfig,
    FsspecConnectionConfig,
    FsspecDownloader,
    FsspecDownloaderConfig,
    FsspecIndexer,
    FsspecIndexerConfig,
    FsspecUploader,
    FsspecUploaderConfig,
)
from unstructured_ingest.processes.utils.blob_storage import (
    BlobStoreUploadStager,
    BlobStoreUploadStagerConfig,
)
from unstructured_ingest.utils.dep_check import requires_dependencies

CONNECTOR_TYPE = "clickzetta_volume"

if TYPE_CHECKING:
    from clickzetta.zettapark.session import Session

# 工具函数区

def build_remote_url(volume: str, remote_path: Optional[str] = None) -> str:
    """拼接 ClickZetta Volume 协议的 remote_url"""
    if not volume:
        raise ValueError("volume 不能为空")

    # 根据 ClickZetta 产品文档使用正确的协议格式
    if volume.lower() == "user":
        # User Volume: volume:user://~/filename
        if remote_path:
            return f"volume:user://~/{remote_path.lstrip('/')}"
        return "volume:user://~/"
    elif volume.lower().startswith("table_"):
        # Table Volume: volume:table://table_name/file
        table_name = volume[6:]  # 去掉 "table_" 前缀
        if remote_path:
            return f"volume:table://{table_name}/{remote_path.lstrip('/')}"
        return f"volume:table://{table_name}/"
    else:
        # Named Volume: volume://volume_name/path
        if remote_path:
            return f"volume://{volume}/{remote_path.lstrip('/')}"
        return f"volume://{volume}/"

def build_sql(action: str, volume: str, file_path: Optional[str] = None, is_table: bool = False, is_user: bool = False, regexp: Optional[str] = None) -> str:
    """统一 SQL 拼接"""
    if is_user:
        prefix = "USER VOLUME"
    elif is_table:
        prefix = f"TABLE VOLUME {volume[6:]}"
    else:
        prefix = f"VOLUME {volume}"
    if action == "list":
        sql = f"LIST {prefix}"
        # 🔧 修复：REGEXP和SUBDIRECTORY是互斥的，优先使用REGEXP
        if regexp:
            sql += f" REGEXP = '{regexp}'"
        elif file_path:
            sql += f" SUBDIRECTORY '{file_path.rstrip('/')}'"
    elif action == "get":
        sql = f"GET {prefix} FILE '{file_path}' TO '{{local_path}}'"
    elif action == "put":
        sql = f"PUT '{{local_path}}' TO {prefix} FILE '{file_path}'"
    elif action == "remove_file":
        sql = f"REMOVE {prefix} FILE '{file_path}'"
    elif action == "remove_dir":
        sql = f"REMOVE {prefix} SUBDIRECTORY '{file_path}'"
    elif action == "remove_all":
        sql = f"REMOVE {prefix} SUBDIRECTORY '/'"
    else:
        raise ValueError(f"未知 action: {action}")
    return sql

def inherit_param(*args) -> Optional[Any]:
    """参数自动继承，返回第一个非 None 的值"""
    for v in args:
        if v is not None:
            return v
    return None

def get_env_multi(key: str) -> str:
    """优先查找 cz_、CLICKZETTA_、无前缀三种环境变量，支持大写/小写，取到第一个非空值，返回原始值"""
    candidates = [key.lower(), key.upper()]
    for prefix in ["CLICKZETTA_", "CZ_", "cz_", ""]:
        for k in candidates:
            v = os.getenv(f"{prefix}{k}")
            if v:
                return v  # 保持原始值，不转小写
    return None

# 配置类统一用 BaseModel
class ClickzettaVolumeAccessConfig(FsspecAccessConfig):
    pass

class ClickzettaVolumeDeleterConfig(BaseModel):
    delete_volume_type: str = Field(..., description="Volume类型: 'user', 'table', 'named'")
    delete_volume_name: Optional[str] = Field(default=None, description="Volume名称，user volume不需要，table volume需要表名，named volume需要卷名")

    @property
    def volume(self) -> str:
        """构建完整的volume标识符"""
        if self.delete_volume_type == "user":
            return "user"
        elif self.delete_volume_type == "table":
            return f"table_{self.delete_volume_name}"
        else:  # named
            return self.delete_volume_name

class ClickzettaVolumeConnectionConfig(FsspecConnectionConfig):
    supported_protocols: List[str] = Field(default_factory=lambda: ["s3", "s3a"], init=False)
    access_config: Secret[ClickzettaVolumeAccessConfig] = Field(default=ClickzettaVolumeAccessConfig(), validate_default=True)
    connector_type: str = Field(default=CONNECTOR_TYPE, init=False)

    @requires_dependencies(["clickzetta"], extras="clickzetta")
    @contextmanager
    def get_client(self, protocol: str = "s3") -> Generator["Session", None, None]:
        from clickzetta.zettapark.session import Session
        # 参数名全部转小写，值保持原始
        config = {k.lower(): get_env_multi(k) for k in ["username", "password", "service", "instance", "workspace", "schema", "vcluster"]}
        
        # 🔧 修复：添加详细的配置调试信息
        logger.info("ClickZetta Volume Connector - 创建会话:")
        for k, v in config.items():
            if k == 'password':
                logger.info(f"  {k}: {'***' if v else 'None'}")
            else:
                logger.info(f"  {k}: {v}")
        
        missing = [k for k, v in config.items() if not v]
        if missing:
            logger.error(f"缺少必需的环境变量: {missing}")
            logger.error("请设置以下环境变量: CLICKZETTA_USERNAME, CLICKZETTA_PASSWORD, CLICKZETTA_SERVICE, CLICKZETTA_INSTANCE, CLICKZETTA_WORKSPACE, CLICKZETTA_SCHEMA, CLICKZETTA_VCLUSTER")
            raise UserAuthError(f"Missing required environment variables for clickzetta: {missing}")
        try:
            logger.info("正在创建ClickZetta会话...")
            session = Session.builder.configs(config).create()
            logger.info("ClickZetta会话创建成功")
            yield session
        except Exception as e:
            logger.error(f"创建ClickZetta会话失败: {e}")
            raise UserAuthError(f"Failed to create clickzetta session: {e}")

    def wrap_error(self, e: Exception) -> Exception:
        if isinstance(e, PermissionError):
            return UserAuthError(e)
        if isinstance(e, FileNotFoundError):
            return UserError(f"File not found: {e}")
        logger.error(f"unhandled exception from clickzetta ({type(e)}): {e}", exc_info=True)
        return e

class ClickzettaVolumeIndexerConfig(FsspecIndexerConfig):
    index_volume_type: str = Field(..., description="Volume类型: 'user', 'table', 'named'")
    index_volume_name: Optional[str] = Field(default=None, description="Volume名称，user volume不需要，table volume需要表名，named volume需要卷名")
    index_remote_path: Optional[str] = Field(default=None, description="卷内相对路径，如 'image1/' 或 'image1/file.png'，无需协议和卷名前缀")
    remote_url: Optional[str] = None
    index_regexp: Optional[str] = Field(default=None, description="正则过滤，生成 SQL REGEXP = 'pattern'")

    def __init__(self, **data):
        # 兼容旧字段名
        if "volume_type" in data and "index_volume_type" not in data:
            data["index_volume_type"] = data["volume_type"]
        if "volume_name" in data and "index_volume_name" not in data:
            data["index_volume_name"] = data["volume_name"]
        if "remote_path" in data and "index_remote_path" not in data:
            data["index_remote_path"] = data["remote_path"]
        if "regexp" in data and "index_regexp" not in data:
            data["index_regexp"] = data["regexp"]

        # 验证配置
        volume_type = data.get("index_volume_type")
        volume_name = data.get("index_volume_name")

        if volume_type == "table" and not volume_name:
            raise ValueError("table volume必须指定volume_name（表名）")
        elif volume_type == "named" and not volume_name:
            raise ValueError("named volume必须指定volume_name（卷名）")

        # 构建完整的volume标识符用于内部使用
        if volume_type == "user":
            full_volume = "user"
        elif volume_type == "table":
            full_volume = f"table_{volume_name}"
        else:  # named
            full_volume = volume_name

        if "remote_url" not in data and full_volume is not None:
            data["remote_url"] = build_remote_url(full_volume, data.get("index_remote_path", ""))
        super().__init__(**data)

    @property
    def volume(self) -> str:
        """构建完整的volume标识符"""
        if self.index_volume_type == "user":
            return "user"
        elif self.index_volume_type == "table":
            return f"table_{self.index_volume_name}"
        else:  # named
            return self.index_volume_name

    # 兼容属性
    @property
    def volume_type(self) -> str:
        return self.index_volume_type

    @property
    def volume_name(self) -> Optional[str]:
        return self.index_volume_name

    @property
    def remote_path(self) -> Optional[str]:
        return self.index_remote_path

    @property
    def regexp(self) -> Optional[str]:
        return self.index_regexp

class ClickzettaVolumeDownloaderConfig(FsspecDownloaderConfig):
    # 使用不同的字段名避免CLI选项冲突
    download_volume_type: Optional[str] = Field(default=None, description="Volume类型: 'user', 'table', 'named'")
    download_volume_name: Optional[str] = Field(default=None, description="Volume名称")
    download_remote_path: Optional[str] = Field(default=None, description="远程路径")
    remote_url: Optional[str] = None
    download_regexp: Optional[str] = None

    @property
    def volume(self) -> str:
        """构建完整的volume标识符"""
        if self.download_volume_type == "user":
            return "user"
        elif self.download_volume_type == "table":
            return f"table_{self.download_volume_name}"
        else:  # named
            return self.download_volume_name

class ClickzettaVolumeUploaderConfig(FsspecUploaderConfig):
    # 简化配置避免CLI选项冲突
    upload_volume_type: Optional[str] = Field(default=None, description="Volume类型: 'user', 'table', 'named'")
    upload_volume_name: Optional[str] = Field(default=None, description="Volume名称")
    upload_remote_path: Optional[str] = Field(default=None, description="远程路径")
    remote_url: Optional[str] = None
    upload_regexp: Optional[str] = None

    def __init__(self, **data):
        # 兼容旧字段名
        if "volume_type" in data and "upload_volume_type" not in data:
            data["upload_volume_type"] = data["volume_type"]
        if "volume_name" in data and "upload_volume_name" not in data:
            data["upload_volume_name"] = data["volume_name"]
        if "remote_path" in data and "upload_remote_path" not in data:
            data["upload_remote_path"] = data["remote_path"]

        # 构建remote_url如果没有提供
        if "remote_url" not in data:
            volume_type = data.get("upload_volume_type")
            volume_name = data.get("upload_volume_name")
            remote_path = data.get("upload_remote_path", "")

            if volume_type == "user":
                volume = "user"
            elif volume_type == "table":
                volume = f"table_{volume_name}"
            else:  # named
                volume = volume_name

            data["remote_url"] = build_remote_url(volume, remote_path)
        super().__init__(**data)

    @property
    def volume(self) -> str:
        """构建完整的volume标识符"""
        if self.upload_volume_type == "user":
            return "user"
        elif self.upload_volume_type == "table":
            return f"table_{self.upload_volume_name}"
        else:  # named
            return self.upload_volume_name

    # 兼容性属性
    @property
    def volume_type(self) -> Optional[str]:
        return self.upload_volume_type

    @property
    def volume_name(self) -> Optional[str]:
        return self.upload_volume_name

    @property
    def remote_path(self) -> Optional[str]:
        return self.upload_remote_path

@dataclass
class ClickzettaVolumeIndexer(FsspecIndexer):
    connection_config: ClickzettaVolumeConnectionConfig
    index_config: ClickzettaVolumeIndexerConfig
    connector_type: str = CONNECTOR_TYPE

    def precheck(self) -> None:
        """跳过标准 fsspec 校验，因为 ClickZetta Volume 使用自定义连接逻辑"""
        return

    def list_files(self) -> List[Dict[str, Any]]:
        """列举卷内文件，支持正则过滤"""
        with self.connection_config.get_client() as session:
            try:
                volume = self.index_config.volume
                remote_path = self.index_config.index_remote_path
                regexp = self.index_config.index_regexp
                is_user = volume.lower() == "user"
                is_table = volume.lower().startswith("table_")
                sql = build_sql("list", volume, remote_path, is_table, is_user, regexp)
                
                # 🔧 修复：添加调试日志
                logger.info(f"ClickZetta Volume Indexer - 执行SQL: {sql}")
                logger.info(f"Volume: {volume}, remote_path: {remote_path}, regexp: {regexp}")
                
                result = session.sql(sql).collect()
                
                # 🔧 修复：检查查询结果
                logger.info(f"SQL查询返回 {len(result)} 条记录")
                if len(result) == 0:
                    logger.warning(f"Volume '{volume}' 中未找到匹配的文件")
                    logger.warning(f"请检查: 1) Volume是否存在 2) 文件路径是否正确 3) 正则表达式是否有效")
                
                files = []
                # 兼容 tuple/list/dict
                for i, row in enumerate(result):
                    logger.debug(f"处理第{i+1}行数据: type={type(row)}, value={row}")
                    
                    if isinstance(row, tuple):
                        full_path = row[0]
                        path, name = os.path.split(full_path)
                        file_info = {
                            "name": name,
                            "path": path,
                            "full_name": full_path,
                            "size": row[2] if len(row) > 2 else None,
                            "last_modified": row[3] if len(row) > 3 else None,
                            "url": row[1] if len(row) > 1 else None,
                            "relative_path": row[0],
                        }
                        logger.debug(f"Tuple格式文件: {file_info}")
                        files.append(file_info)
                    else:
                        full_path = row.get("name")
                        path, name = os.path.split(full_path) if full_path else ("", row.get("name", ""))
                        file_info = {
                            "name": name,
                            "path": path,
                            "full_name": full_path,
                            "size": row.get("size"),
                            "last_modified": row.get("last_modified_time") or row.get("last_modified"),
                            "url": row.get("url"),
                            "relative_path": row.get("relative_path"),
                        }
                        logger.debug(f"Dict格式文件: {file_info}")
                        files.append(file_info)
                
                logger.info(f"成功解析 {len(files)} 个文件")
                return files
            except Exception as e:
                raise self.connection_config.wrap_error(e)

    def get_file_info(self) -> List[Dict[str, Any]]:
        return self.list_files()

    def get_metadata(self, file_info: Dict[str, Any]) -> FileDataSourceMetadata:
        url = file_info.get("url")
        if url is not None and not isinstance(url, str):
            url = str(url)
        return FileDataSourceMetadata(
            size=file_info.get("size"),
            last_modified=file_info.get("last_modified"),
            url=url,
            relative_path=file_info.get("relative_path"),
        )

@dataclass
class ClickzettaVolumeDownloader(FsspecDownloader):
    protocol: str = "s3"
    connection_config: ClickzettaVolumeConnectionConfig
    connector_type: str = CONNECTOR_TYPE
    download_config: Optional[ClickzettaVolumeDownloaderConfig] = field(default_factory=ClickzettaVolumeDownloaderConfig)
    index_config: Optional[ClickzettaVolumeIndexerConfig] = None

    def __post_init__(self):
        # 自动继承 volume_type/volume_name/remote_path/regexp，优先级：download_config > index_config > connection_config
        dc = self.download_config
        ic = self.index_config
        cc = self.connection_config
        if dc is not None:
            dc.download_volume_type = inherit_param(dc.download_volume_type, ic.index_volume_type if ic else None, cc.volume_type if hasattr(cc, "volume_type") else None)
            dc.download_volume_name = inherit_param(dc.download_volume_name, ic.index_volume_name if ic else None, cc.volume_name if hasattr(cc, "volume_name") else None)
            dc.download_remote_path = inherit_param(dc.download_remote_path, ic.index_remote_path if ic else None, cc.remote_path if hasattr(cc, "remote_path") else None)
            dc.download_regexp = inherit_param(dc.download_regexp, ic.index_regexp if ic else None)
            if not dc.remote_url and dc.volume:
                dc.remote_url = build_remote_url(dc.volume, dc.download_remote_path or "")

    def is_async(self) -> bool:
        return False

    def download_file(self, remote_path: str, local_path: str, file_info: dict = None, session=None) -> None:
        # 优先从 file_info 自动推断 volume
        volume = None
        if file_info and 'volume' in file_info and file_info['volume']:
            volume = file_info['volume']
        if not volume:
            volume = self.download_config.volume
        if not volume and self.index_config is not None:
            volume = self.index_config.volume
        if not volume and hasattr(self.connection_config, "volume_type"):
            if self.connection_config.volume_type == "user":
                volume = "user"
            elif self.connection_config.volume_type == "table":
                volume = f"table_{self.connection_config.volume_name}" if hasattr(self.connection_config, "volume_name") else None
            else:  # named
                volume = self.connection_config.volume_name if hasattr(self.connection_config, "volume_name") else None
        # 尝试从 remote_path 拆分 volume
        if not volume and remote_path and isinstance(remote_path, str) and "/" in remote_path:
            volume = remote_path.split("/")[0]
        if not volume:
            raise ValueError("volume 不能为空，且未能自动继承，请检查配置")
        from pathlib import Path
        import shutil
        dir_name = os.path.dirname(str(local_path))
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        if Path(local_path).exists() and Path(local_path).is_dir():
            shutil.rmtree(local_path)
        logger.info(f"下载文件 '{remote_path}' 到 '{local_path}'")

        # 如果没有传入session，创建新的session；否则使用共享session
        if session is None:
            with self.connection_config.get_client() as new_session:
                self._execute_download(volume, remote_path, local_path, new_session)
        else:
            self._execute_download(volume, remote_path, local_path, session)

    def _execute_download(self, volume: str, remote_path: str, local_path: str, session) -> None:
        """执行实际的下载操作"""
        try:
            is_user = volume.lower() == "user"
            is_table = volume.lower().startswith("table_")
            sql = build_sql("get", volume, remote_path, is_table, is_user)

            # ClickZetta客户端会在local_path基础上自动添加文件名
            # 所以我们需要传递目录路径，而不是完整文件路径
            local_dir = str(Path(local_path).parent)
            sql = sql.replace("{local_path}", local_dir)

            logger.info(f"执行 SQL: {sql}")
            logger.debug(f"ClickZetta会将文件下载到: {local_dir} + 文件名")

            session.sql(sql).collect()

            # 检查文件是否下载成功
            if not os.path.exists(local_path):
                raise FileNotFoundError(f"下载文件失败: {remote_path} -> {local_path}")
            if Path(local_path).stat().st_size == 0:
                logger.warning(f"下载的文件为空: {local_path}")
        except Exception as e:
            logger.error(f"下载失败: {str(e)}")
            raise self.connection_config.wrap_error(e)

    def run(self, files: Optional[List[Dict[str, Any]]] = None, docs=None, file_list=None, **kwargs) -> List[Dict[str, Any]]:
        candidates = [files, docs, file_list]
        file_data_obj = kwargs.get("file_data")
        for v in kwargs.values():
            if isinstance(v, list) and v and isinstance(v[0], dict) and "name" in v[0]:
                candidates.append(v)
            elif hasattr(v, "additional_metadata") and hasattr(v, "source_identifiers"):
                file_info = v.additional_metadata.copy()
                file_info["name"] = v.source_identifiers.filename
                candidates.append([file_info])
        files = next((c for c in candidates if c is not None), None)
        # 自动获取 remote_path 下的文件列表
        if files is None:
            index_config = ClickzettaVolumeIndexerConfig(
                volume_type=self.download_config.download_volume_type or "user",
                volume_name=self.download_config.download_volume_name,
                remote_path=self.download_config.download_remote_path
            )
            indexer = ClickzettaVolumeIndexer(
                connection_config=self.connection_config,
                index_config=index_config
            )
            files = indexer.list_files()
        # 只保留一个 for file_info in files 循环，始终走 volume 推断链
        for file_info in files:
            volume = (
                (self.download_config.volume if hasattr(self.download_config, "volume") else None)
                or (self.index_config.volume if self.index_config and hasattr(self.index_config, "volume") else None)
                or (self.connection_config.volume if hasattr(self.connection_config, "volume") else None)
            )
            if not volume:
                url = file_info.get("url", "")
                volume = _extract_volume_from_url(url)
                if not volume:
                    volume = _extract_volume_from_path(file_info.get("full_name", ""))
                if not volume:
                    volume = _extract_volume_from_path(file_info.get("relative_path", ""))
            if volume:
                file_info["volume"] = volume
            if not file_info.get("volume"):
                logger.error(
                    f"file_info 缺少 volume 字段，file_info={file_info}，"
                    f"download_config.volume={self.download_config.volume if hasattr(self.download_config, 'volume') else None}，"
                    f"index_config.volume={self.index_config.volume if self.index_config and hasattr(self.index_config, 'volume') else None}，"
                    f"connection_config.volume={self.connection_config.volume if hasattr(self.connection_config, 'volume') else None}，"
                    f"url={file_info.get('url', None)}，full_name={file_info.get('full_name', None)}，relative_path={file_info.get('relative_path', None)}"
                )
                raise ValueError(
                    f"file_info 缺少 volume 字段，file_info={file_info}，"
                    f"download_config.volume={self.download_config.volume if hasattr(self.download_config, 'volume') else None}，"
                    f"index_config.volume={self.index_config.volume if self.index_config and hasattr(self.index_config, 'volume') else None}，"
                    f"connection_config.volume={self.connection_config.volume if hasattr(self.connection_config, 'volume') else None}"
                )
        if not files:
            logger.warning(f"No files to download, kwargs={kwargs}")
            return []
        logger.info(f"准备下载 {len(files)} 个文件")
        logger.debug(f"待下载文件详情: {files}")
        all_files_info = {}
        try:
            index_config = ClickzettaVolumeIndexerConfig(
                volume_type=self.download_config.download_volume_type or "user",
                volume_name=self.download_config.download_volume_name,
                remote_url=self.download_config.remote_url if hasattr(self.download_config, "remote_url") else None
            )
            indexer = ClickzettaVolumeIndexer(
                connection_config=self.connection_config,
                index_config=index_config
            )
            listed_files = indexer.list_files()
            # 目录过滤
            prefix = index_config.path_without_protocol if hasattr(index_config, "path_without_protocol") else None
            if prefix:
                prefix = prefix.lstrip("/")
                listed_files = [f for f in listed_files if f.get("path", "").startswith(prefix)]
            logger.info(f"从卷 {self.download_config.volume} 中找到 {len(listed_files)} 个文件")
            logger.debug(f"卷中所有文件: {listed_files}")
            for file_info in listed_files:
                for key in ["relative_path", "full_name", "name"]:
                    if key in file_info and file_info[key]:
                        all_files_info[file_info[key]] = file_info
        except Exception as e:
            logger.warning(f"获取文件列表失败，将使用简单路径: {str(e)}")
            listed_files = []
        results = []
        # 修正：下载目录使用 self.download_dir，兼容 pipeline 传递的 download_dir
        from pathlib import Path
        download_dir = Path(self.download_dir)

        # 🔧 优化：创建共享session，避免每次下载都创建新session
        logger.info(f"🔧 创建共享ClickZetta会话，即将下载 {len(files)} 个文件")
        with self.connection_config.get_client() as shared_session:
            logger.info(f"✅ 共享ClickZetta会话创建成功")
            for file_info in files:
                # 强制覆盖 volume 字段：只要继承链有值就赋值，彻底避免 None 泄漏
                inherited_volume = (
                    self.download_config.volume
                    or (self.index_config.volume if self.index_config else None)
                    or (self.connection_config.volume if hasattr(self.connection_config, 'volume') else None)
                )
                # 只要继承链有值就覆盖，无论 file_info['volume'] 是否存在、是否为 None
                if inherited_volume is not None:
                    file_info['volume'] = inherited_volume
                # 兜底校验，volume 必须有值，否则强制再推断一次
                if not file_info.get('volume'):
                    # 最后一次强制推断
                    url = file_info.get('url', '')
                    full_name = file_info.get('full_name', '')
                    relative_path = file_info.get('relative_path', '')
                    volume = _extract_volume_from_url(url) or _extract_volume_from_path(full_name) or _extract_volume_from_path(relative_path)
                    if volume:
                        file_info['volume'] = volume
                # 兜底校验，volume 必须有值，否则详细报错
                if not file_info.get('volume'):
                    logger.error(
                        f"file_info 缺少 volume 字段，file_info={file_info}，"
                        f"download_config.volume={self.download_config.volume if hasattr(self.download_config, 'volume') else None}，"
                        f"index_config.volume={self.index_config.volume if self.index_config and hasattr(self.index_config, 'volume') else None}，"
                        f"connection_config.volume={self.connection_config.volume if hasattr(self.connection_config, 'volume') else None}，"
                        f"url={file_info.get('url', None)}，full_name={file_info.get('full_name', None)}，relative_path={file_info.get('relative_path', None)}"
                    )
                    raise ValueError(
                        f"file_info 缺少 volume 字段，file_info={file_info}，"
                        f"download_config.volume={self.download_config.volume if hasattr(self.download_config, 'volume') else None}，"
                        f"index_config.volume={self.index_config.volume if self.index_config and hasattr(self.index_config, 'volume') else None}，"
                        f"connection_config.volume={self.connection_config.volume if hasattr(self.connection_config, 'volume') else None}"
                    )
                remote_path = file_info["name"]
            
            logger.info(f"处理文件: {remote_path}")
            logger.debug(f"文件详情: {file_info}")
            
            # 尝试多种方式匹配文件信息
            detected_file_info = self._find_file_info(file_info, all_files_info, listed_files)
            
            # 整合文件路径信息
            name, path, full_name, relative_path = self._extract_file_paths(file_info, detected_file_info)
            
            # 确定远程路径，优先使用相对路径，其次是完整名称
            if relative_path:
                original_remote_path = relative_path
            elif full_name:
                original_remote_path = full_name
            elif path:
                # 检查path是否已经包含文件名，避免重复拼接
                if path.endswith(name):
                    original_remote_path = path
                else:
                    original_remote_path = path + "/" + name
            else:
                original_remote_path = name
            
            # 构建目标路径
            if path:
                # 修正：检查path是否已经是完整文件路径（包含文件名）
                if path.endswith(name):
                    # path已包含文件名，去掉文件名部分作为目录路径
                    dir_path = path[:-len(name)].rstrip('/')
                    if dir_path:
                        local_path = download_dir / dir_path / name
                        logger.debug(f"路径已包含文件名，提取目录: {dir_path}, 本地路径={local_path}")
                    else:
                        local_path = download_dir / name
                        logger.debug(f"路径仅为文件名，使用根目录: 本地路径={local_path}")
                else:
                    local_path = download_dir / path / name
                    logger.debug(f"路径不含文件名: {path}, 本地路径={local_path}")
            else:
                local_path = download_dir / name
                logger.debug(f"使用根目录: 名称={name}, 本地路径={local_path}")
            
            try:
                # 确保父目录存在
                local_path.parent.mkdir(parents=True, exist_ok=True)
                
                logger.info(f"下载文件: {original_remote_path} -> {local_path}")

                # 下载文件，传递 file_info 以便自动推断 volume，使用共享session
                self.download_file(original_remote_path, str(local_path), file_info=file_info, session=shared_session)

                # 先修复路径问题（ClickZetta可能创建目录而不是文件）
                self._fix_nested_file_issue(local_path, original_remote_path)

                # 处理文件下载失败的情况
                if not os.path.exists(str(local_path)):
                    if not self._recover_missing_file(original_remote_path, name, local_path, download_dir):
                        raise FileNotFoundError(f"下载后文件不存在: {local_path}, 已尝试所有修复方法")

                # 检查下载的文件内容是否为错误响应
                if os.path.exists(str(local_path)) and os.path.isfile(str(local_path)):
                    with open(str(local_path), 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read(1000)  # 读取前1000字符检查
                        if '<Error>' in content and '<Code>NoSuchKey</Code>' in content:
                            # ClickZetta返回了错误XML，删除这个错误文件并抛出异常
                            os.remove(str(local_path))
                            raise FileNotFoundError(f"文件不存在于Volume中: {original_remote_path}")
                        elif '<Error>' in content and '<Code>' in content:
                            # 其他类型的错误
                            error_msg = content[:500]  # 截取前500字符作为错误信息
                            os.remove(str(local_path))
                            raise Exception(f"下载文件时发生错误: {error_msg}")
                
                results.append({
                    "remote_path": original_remote_path,
                    "local_path": local_path,
                    "path": local_path,  # 这是 pipeline 需要的
                    "status": "success",
                    "file_data": file_data_obj
                })
            except Exception as e:
                logger.error(f"下载失败: {str(e)}")
                results.append({
                    "remote_path": original_remote_path,
                    "local_path": local_path,
                    "path": local_path,
                    "status": "failed",
                    "error": str(e),
                    "file_data": file_data_obj
                })
        return results
        
    def _find_file_info(self, file_info, all_files_info, listed_files):
        """查找匹配的文件信息"""
        detected_file_info = None
        remote_path = file_info["name"]
        
        # 尝试多种键进行匹配
        for key in ["relative_path", "full_name", "name"]:
            if key in file_info and file_info[key] and file_info[key] in all_files_info:
                detected_file_info = all_files_info[file_info[key]]
                logger.debug(f"通过 {key} 匹配到文件: {detected_file_info}")
                break
                
        # 遍历查找
        if not detected_file_info and listed_files:
            for info in listed_files:
                if info.get("name") == remote_path:
                    detected_file_info = info
                    logger.debug(f"通过遍历匹配到文件: {info}")
                    break
                    
        return detected_file_info
        
    def _extract_file_paths(self, file_info, detected_file_info):
        """提取文件路径信息"""
        remote_path = file_info["name"]
        name = file_info.get("name", remote_path)
        path = file_info.get("path", "")
        full_name = file_info.get("full_name", "")
        relative_path = file_info.get("relative_path", "")
        
        # 如果找到了更详细的信息，则使用查找到的信息
        if detected_file_info:
            name = detected_file_info.get("name", name)
            path = detected_file_info.get("path", path) 
            full_name = detected_file_info.get("full_name", full_name)
            relative_path = detected_file_info.get("relative_path", relative_path)
            
        return name, path, full_name, relative_path
        
    def _recover_missing_file(self, original_remote_path, name, local_path, download_dir):
        """尝试恢复丢失的文件"""
        import shutil
        
        logger.info(f"文件下载失败，检查其他可能的位置")
        
        # 可能的文件位置
        possible_locations = [
            download_dir / Path(original_remote_path).name,
            download_dir / original_remote_path / name,
            download_dir / original_remote_path,
        ]
        
        # 在所有可能的位置中查找
        found = False
        for possible_path in possible_locations:
            if os.path.exists(str(possible_path)) and os.path.isfile(str(possible_path)):
                logger.info(f"找到文件在: {possible_path}")
                os.makedirs(os.path.dirname(str(local_path)), exist_ok=True)
                shutil.copy2(str(possible_path), str(local_path))
                logger.info(f"复制文件: {possible_path} -> {local_path}")
                found = True
                break
        
        # 如果还找不到，进行全局搜索
        if not found:
            # 在整个下载目录中查找匹配的文件名
            logger.info(f"全局搜索文件名: {name}")
            for root, _, files in os.walk(str(download_dir)):
                if name in files:
                    found_path = os.path.join(root, name)
                    logger.info(f"找到文件: {found_path}")
                    os.makedirs(os.path.dirname(str(local_path)), exist_ok=True)
                    shutil.copy2(found_path, str(local_path))
                    logger.info(f"复制文件: {found_path} -> {local_path}")
                    found = True
                    break
        
        # 尝试重新下载
        if not found:
            logger.info(f"尝试重新下载文件")
            os.makedirs(os.path.dirname(str(local_path)), exist_ok=True)
            self.download_file(original_remote_path, str(local_path))
            found = os.path.exists(str(local_path))
            
        return found
        
    def _fix_nested_file_issue(self, local_path, remote_path):
        """修复嵌套文件问题"""
        import shutil, os
        
        nested_file = local_path / Path(remote_path).name
        if local_path.is_dir() and os.path.exists(str(nested_file)) and os.path.isfile(str(nested_file)):
            logger.info(f"检测到嵌套文件问题: {nested_file}")
            # 创建临时文件
            import secrets
            temp_file = local_path.parent / f"temp_{secrets.token_hex(4)}{Path(remote_path).suffix}"
            # 将嵌套文件复制到临时文件
            shutil.copy2(str(nested_file), str(temp_file))
            # 删除原目录
            shutil.rmtree(str(local_path))
            # 将临时文件移回原位置
            shutil.move(str(temp_file), str(local_path))
            logger.info(f"修复嵌套路径: {nested_file} -> {local_path}")
                
@dataclass
class ClickzettaVolumeUploader(FsspecUploader):
    connector_type: str = CONNECTOR_TYPE
    connection_config: ClickzettaVolumeConnectionConfig
    upload_config: ClickzettaVolumeUploaderConfig = field(default=None)

    def upload_file(self, local_path: str, remote_path: Optional[str] = None) -> None:
        """上传文件到指定卷"""
        import os
        logger.info(f"上传文件: {local_path} -> {remote_path}")
        with self.connection_config.get_client() as session:
            try:
                volume = self.upload_config.volume
                remote_path = remote_path or self.upload_config.upload_remote_path
                if not remote_path:
                    raise ValueError("remote_path 不能为空")
                if remote_path.endswith("/"):
                    filename = os.path.basename(local_path)
                    remote_path = remote_path + filename

                # 清理可能的重复路径问题
                remote_path = self._clean_file_path(remote_path)
                is_user = volume.lower() == "user"
                is_table = volume.lower().startswith("table_")
                sql = build_sql("put", volume, remote_path, is_table, is_user)
                sql = sql.replace("{local_path}", local_path)
                logger.info(f"执行SQL: {sql}")
                session.sql(sql).collect()
                logger.info(f"文件上传成功: {remote_path}")
            except Exception as e:
                logger.error(f"文件上传失败 {local_path} -> {remote_path}: {str(e)}")
                raise self.connection_config.wrap_error(e)

    def _clean_file_path(self, file_path: str) -> str:
        """清理文件路径，避免重复路径问题"""
        if not file_path:
            return file_path

        # 检查是否有文件名重复的情况，如 'dir/file.md/file.md'
        parts = file_path.split('/')
        if len(parts) >= 2 and parts[-1] == parts[-2]:
            # 移除最后一个重复的文件名
            cleaned_path = '/'.join(parts[:-1])
            logger.debug(f"清理重复路径: {file_path} -> {cleaned_path}")
            return cleaned_path
        return file_path

@dataclass
class ClickzettaVolumeDeleter:
    """删除卷中文件的类"""
    connection_config: ClickzettaVolumeConnectionConfig
    deleter_config: ClickzettaVolumeDeleterConfig

    @property
    def volume(self) -> str:
        return self.deleter_config.volume

    def delete_file(self, file_path: str) -> bool:
        """删除卷中指定路径的文件"""
        # 清理可能的重复路径问题
        file_path = self._clean_file_path(file_path)

        with self.connection_config.get_client() as session:
            try:
                volume = self.volume
                is_user = volume.lower() == "user"
                is_table = volume.lower().startswith("table_")
                sql = build_sql("remove_file", volume, file_path, is_table, is_user)
                logger.info(f"执行SQL: {sql}")
                session.sql(sql).collect()
                logger.info(f"成功删除文件: {file_path}")
                return True
            except Exception as e:
                logger.error(f"删除文件 {file_path} 失败: {str(e)}")
                raise self.connection_config.wrap_error(e)

    def delete_directory(self, directory_path: str) -> bool:
        """删除卷中指定目录及其下所有文件"""
        with self.connection_config.get_client() as session:
            try:
                volume = self.volume
                is_user = volume.lower() == "user"
                is_table = volume.lower().startswith("table_")
                sql = build_sql("remove_dir", volume, directory_path, is_table, is_user)
                logger.info(f"执行SQL: {sql}")
                session.sql(sql).collect()
                return True
            except Exception as e:
                logger.error(f"删除目录 {directory_path} 失败: {str(e)}")
                raise self.connection_config.wrap_error(e)

    def delete_all(self) -> bool:
        """删除卷中所有文件和目录"""
        with self.connection_config.get_client() as session:
            try:
                volume = self.volume
                is_user = volume.lower() == "user"
                is_table = volume.lower().startswith("table_")
                sql = build_sql("remove_all", volume, None, is_table, is_user)
                logger.info(f"执行SQL: {sql}")
                session.sql(sql).collect()
                return True
            except Exception as e:
                logger.error(f"删除卷 {self.volume} 中所有内容失败: {str(e)}")
                raise self.connection_config.wrap_error(e)

clickzetta_volume_source_entry = SourceRegistryEntry(
    indexer=ClickzettaVolumeIndexer,
    indexer_config=ClickzettaVolumeIndexerConfig,
    downloader=ClickzettaVolumeDownloader,
    downloader_config=ClickzettaVolumeDownloaderConfig,
    connection_config=ClickzettaVolumeConnectionConfig,
)

clickzetta_volume_destination_entry = DestinationRegistryEntry(
    uploader=ClickzettaVolumeUploader,
    uploader_config=ClickzettaVolumeUploaderConfig,
    connection_config=ClickzettaVolumeConnectionConfig,
    upload_stager_config=BlobStoreUploadStagerConfig,
    upload_stager=BlobStoreUploadStager,
)

def _extract_volume_from_url(url: str) -> str | None:
    import re
    m = re.search(r'oss://([^/]+)/', url or "")
    if m:
        bucket = m.group(1)
        # 兼容 oss://unstructured-etl/xxx 和 oss://unstructured_etl/xxx
        if bucket.replace("-", "_").startswith("unstructured_etl"):
            return "unstructured_etl_volume_test"
        if bucket.startswith("user"):
            return "user"
        if bucket.startswith("table"):
            return "table_elements"
    # 兜底：兼容 oss://.../internal_volume/{volume}_xxx/...
    m2 = re.search(r'/internal_volume/([a-zA-Z0-9_]+)_', url or "")
    if m2:
        prefix = m2.group(1)
        if prefix.startswith("user"):
            return "user"
        if prefix.startswith("table"):
            return "table_elements"
        if prefix.startswith("unstructured_etl_volume_test"):
            return "unstructured_etl_volume_test"
    return None

def _extract_volume_from_path(path: str) -> str | None:
    # 允许 table_xxx 自动识别为 table_elements
    if path.startswith("table") or "/table" in path:
        return "table_elements"
    for v in ["user", "unstructured_etl_volume_test"]:
        if f"/{v}/" in path or path.startswith(f"{v}/"):
            return v
    return None