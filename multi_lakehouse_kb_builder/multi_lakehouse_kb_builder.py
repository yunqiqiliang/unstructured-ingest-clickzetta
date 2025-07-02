#!/usr/bin/env python3
"""
多Lakehouse知识库批量构建系统

功能：
1. 从connections.json读取所有Lakehouse连接
2. 检查并创建clickzetta_doc_kb schema
3. 检查并清空（不删除）Raw表和Silver表
4. 批量部署知识库到多个Lakehouse
"""

import os
import sys

# 设置环境变量以抑制unstructured的日志
os.environ['UNSTRUCTURED_LOG_LEVEL'] = 'WARNING'
import json
import logging
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback
import time
import multiprocessing
import platform

# 在文件顶部强制设置multiprocessing的启动方法
# 这必须在任何其他multiprocessing操作之前完成
if platform.system() == 'Linux':
    # 在Linux上强制使用spawn，避免fork的问题
    try:
        multiprocessing.set_start_method('spawn', force=True)
        # 移除print语句，避免破坏MCP的stdio协议
    except RuntimeError as e:
        # 可能已经设置过了
        pass

# 修复 Streamlit 环境中的 __main__ 问题
if '__main__' not in sys.modules:
    import types
    sys.modules['__main__'] = types.ModuleType('__main__')

# 转换规则引擎现在在同一目录中

# 配置日志
# 确保logs目录存在
logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(logs_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(logs_dir, f'kb_deployment_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 抑制 unstructured_ingest 的详细日志
unstructured_logger = logging.getLogger('unstructured_ingest')
unstructured_logger.setLevel(logging.WARNING)

# 抑制各个子模块的日志
for module in ['unstructured_ingest.pipeline', 'unstructured_ingest.processes', 
               'unstructured_ingest.pipeline.interfaces', 'unstructured_ingest.pipeline.steps',
               'MainProcess', 'unstructured_ingest.processes.partitioner',
               'unstructured_ingest.processes.embedder', 'unstructured_ingest.processes.chunker',
               'unstructured_ingest.processes.filter', 'unstructured_ingest.processes.connectors']:
    module_logger = logging.getLogger(module)
    module_logger.setLevel(logging.ERROR)  # 只显示错误级别的日志

# 抑制 clickzetta 的一些日志
clickzetta_logger = logging.getLogger('clickzetta.connector')
clickzetta_logger.setLevel(logging.WARNING)

# 导入必要的库
try:
    from clickzetta.connector import connect
    import dashscope
    from dashscope import TextEmbedding
    from unstructured_ingest.interfaces import ProcessorConfig
    from unstructured_ingest.pipeline.pipeline import Pipeline
    from unstructured_ingest.processes.chunker import ChunkerConfig
    from unstructured_ingest.processes.connectors.local import (
        LocalIndexerConfig,
        LocalDownloaderConfig,
        LocalConnectionConfig
    )
    from unstructured_ingest.processes.embedder import EmbedderConfig
    from unstructured_ingest.processes.partitioner import PartitionerConfig
    from unstructured_ingest.processes.filter import FiltererConfig
    from unstructured_ingest.processes.connectors.sql.clickzetta import (
        ClickzettaConnectionConfig,
        ClickzettaAccessConfig,
        ClickzettaUploadStagerConfig,
        ClickzettaUploaderConfig
    )
except ImportError as e:
    logger.error(f"导入依赖失败: {e}")
    logger.error("请确保已安装所有必要的依赖包")
    sys.exit(1)


class LakehouseConnectionManager:
    """Lakehouse连接管理器"""
    
    def __init__(self, config_path: str = "~/.clickzetta/connections.json"):
        self.config_path = os.path.expanduser(config_path)
        self.config = self.load_config()
        self.connections = self.config.get('connections', [])
        self.system_config = self.config.get('system_config', {})
        logger.info(f"加载了 {len(self.connections)} 个Lakehouse连接")
    
    def load_config(self) -> Dict[str, Any]:
        """加载完整配置文件"""
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            
            return config
            
        except FileNotFoundError:
            logger.error(f"配置文件不存在: {self.config_path}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"配置文件格式错误: {e}")
            raise
    
    def load_connections(self) -> List[Dict[str, Any]]:
        """加载所有Lakehouse连接配置"""
        connections = self.config.get('connections', [])
        if not connections:
            logger.warning(f"配置文件 {self.config_path} 中没有找到连接配置")
            return []
        
        # 验证每个连接的必要字段
        valid_connections = []
        for conn in connections:
            if self._validate_connection(conn):
                valid_connections.append(conn)
            else:
                logger.warning(f"跳过无效连接: {conn.get('connection_name', 'unnamed')}")
        
        return valid_connections
    
    def _validate_connection(self, conn: Dict[str, Any]) -> bool:
        """验证连接配置的必要字段"""
        required_fields = ['service', 'username', 'password', 'instance']
        
        for field in required_fields:
            if field not in conn or not conn[field]:
                logger.warning(f"连接缺少必要字段: {field}")
                return False
        
        return True
    
    def get_active_connections(self, 
                             filter_pattern: Optional[str] = None,
                             exclude_pattern: Optional[str] = None) -> List[Dict[str, Any]]:
        """获取活跃的连接（支持过滤）"""
        active_conns = []
        
        for conn in self.connections:
            conn_name = conn.get('connection_name', 'unnamed')
            
            # 应用过滤规则
            if filter_pattern and filter_pattern not in conn_name:
                continue
            if exclude_pattern and exclude_pattern in conn_name:
                continue
            
            active_conns.append(conn)
        
        logger.info(f"筛选出 {len(active_conns)} 个活跃连接")
        return active_conns
    
    def get_dashscope_api_key(self) -> Optional[str]:
        """从配置中获取DashScope API密钥"""
        embedding_config = self.system_config.get('embedding', {})
        dashscope_config = embedding_config.get('dashscope', {})
        api_key = dashscope_config.get('api_key')
        
        if api_key:
            logger.info("从配置文件中读取到DashScope API密钥")
        else:
            # 尝试从环境变量获取
            api_key = os.getenv("DASHSCOPE_API_KEY")
            if api_key:
                logger.info("从环境变量中读取到DashScope API密钥")
            else:
                # 使用默认值
                api_key = "sk-7d178531cbd14ce6bba2d16fe3948239"
                logger.warning("使用默认的DashScope API密钥")
        
        return api_key


class LakehouseSchemaManager:
    """Lakehouse Schema和表管理器"""
    
    def __init__(self, connection_params: Dict[str, Any]):
        self.connection_params = connection_params
        self.conn = None
        self.workspace = connection_params.get('workspace', 'default')
        self.schema_name = "clickzetta_doc_kb"
        self.raw_table_name = "dashscope_v4_1024_2048_20250611_yunqi_raw_elements"
        self.silver_table_name = "dashscope_v4_1024_2048_20250611_yunqi_elements"
        self.embeddings_dimensions = 1024
        
        # 连接标识
        # 兼容两种命名方式
        self.conn_name = connection_params.get('name') or connection_params.get('connection_name', 'unnamed')
        
    def create_connection(self):
        """创建数据库连接"""
        try:
            self.conn = connect(
                password=self.connection_params['password'],
                username=self.connection_params['username'],
                service=self.connection_params['service'],
                instance=self.connection_params['instance'],
                workspace=self.connection_params.get('workspace', 'default'),
                schema=self.connection_params.get('schema', 'default'),
                vcluster=self.connection_params.get('vcluster', 'default')
            )
            logger.info(f"[{self.conn_name}] 成功创建连接")
            return self.conn
        except Exception as e:
            logger.error(f"[{self.conn_name}] 创建连接失败: {e}")
            raise
    
    def execute_sql(self, sql_statement: str) -> List[Any]:
        """执行SQL语句"""
        if not self.conn:
            self.create_connection()
        
        try:
            with self.conn.cursor() as cur:
                cur.execute(sql_statement)
                if cur.description:  # 有返回结果的查询
                    results = cur.fetchall()
                else:
                    results = [['OPERATION SUCCEED']]
                # 对DDL语句（CREATE, DROP, TRUNCATE等）进行提交
                sql_upper = sql_statement.strip().upper()
                if any(sql_upper.startswith(ddl) for ddl in ['CREATE', 'DROP', 'TRUNCATE', 'ALTER']):
                    try:
                        self.conn.commit()
                        logger.debug(f"[{self.conn_name}] DDL语句已提交")
                    except AttributeError:
                        # 如果连接不支持commit，忽略
                        logger.debug(f"[{self.conn_name}] 连接不支持显式commit")
                return results
        except Exception as e:
            logger.error(f"[{self.conn_name}] SQL执行失败: {sql_statement}")
            logger.error(f"错误: {e}")
            raise
    
    def check_and_create_schema(self) -> bool:
        """检查并创建schema"""
        try:
            # 检查schema是否存在
            check_sql = f"SHOW SCHEMAS LIKE '{self.schema_name}'"
            results = self.execute_sql(check_sql)
            
            schema_exists = any(self.schema_name in str(row) for row in results)
            
            if not schema_exists:
                logger.info(f"[{self.conn_name}] Schema {self.schema_name} 不存在，创建中...")
                create_sql = f"CREATE SCHEMA IF NOT EXISTS {self.schema_name}"
                self.execute_sql(create_sql)
                logger.info(f"[{self.conn_name}] Schema {self.schema_name} 创建成功")
            else:
                logger.info(f"[{self.conn_name}] Schema {self.schema_name} 已存在")
            
            return True
            
        except Exception as e:
            logger.error(f"[{self.conn_name}] 检查/创建schema失败: {e}")
            return False
    
    def check_table_exists(self, table_name: str) -> bool:
        """检查表是否存在"""
        try:
            # 尝试查询表的结构来检查表是否存在
            # 使用完整路径：workspace.schema.table
            check_sql = f"SELECT 1 FROM {self.workspace}.{self.schema_name}.{table_name} LIMIT 1"
            try:
                self.execute_sql(check_sql)
                return True
            except Exception as e:
                # 如果查询失败，说明表不存在，这是正常情况
                # 只记录debug日志，不记录错误
                logger.debug(f"[{self.conn_name}] 表 {self.workspace}.{self.schema_name}.{table_name} 不存在: {e}")
                return False
        except Exception as e:
            logger.error(f"[{self.conn_name}] 检查表 {table_name} 失败: {e}")
            return False
    
    def get_table_ddl(self) -> Tuple[str, str]:
        """获取Raw表和Silver表的DDL"""
        raw_table_ddl = f"""
        CREATE TABLE IF NOT EXISTS {self.workspace}.{self.schema_name}.{self.raw_table_name} (
            `id` STRING,
            `record_locator` STRING,
            `type` STRING,
            `record_id` STRING,
            `element_id` STRING,
            `filetype` STRING,
            `file_directory` STRING,
            `filename` STRING,
            `last_modified` TIMESTAMP,
            `languages` STRING,
            `page_number` STRING,
            `text` STRING,
            `embeddings` VECTOR({self.embeddings_dimensions}),
            `parent_id` STRING,
            `is_continuation` BOOLEAN,
            `orig_elements` STRING,
            `element_type` STRING,
            `coordinates` STRING,
            `link_texts` STRING,
            `link_urls` STRING,
            `email_message_id` STRING,
            `sent_from` STRING,
            `sent_to` STRING,
            `subject` STRING,
            `url` STRING,
            `version` STRING,
            `date_created` TIMESTAMP,
            `date_modified` TIMESTAMP,
            `date_processed` TIMESTAMP,
            `text_as_html` STRING,
            `emphasized_text_contents` STRING,
            `emphasized_text_tags` STRING,
            `documents_source` STRING
        ) USING PARQUET;
        """
        
        silver_table_ddl = f"""
        CREATE TABLE IF NOT EXISTS {self.workspace}.{self.schema_name}.{self.silver_table_name} (
            `id` STRING,
            `record_locator` STRING,
            `type` STRING,
            `record_id` STRING,
            `element_id` STRING,
            `filetype` STRING,
            `file_directory` STRING,
            `filename` STRING,
            `last_modified` TIMESTAMP,
            `languages` STRING,
            `page_number` STRING,
            `text` STRING,
            `embeddings` VECTOR({self.embeddings_dimensions}),
            `parent_id` STRING,
            `is_continuation` BOOLEAN,
            `orig_elements` STRING,
            `element_type` STRING,
            `coordinates` STRING,
            `link_texts` STRING,
            `link_urls` STRING,
            `email_message_id` STRING,
            `sent_from` STRING,
            `sent_to` STRING,
            `subject` STRING,
            `url` STRING,
            `version` STRING,
            `date_created` TIMESTAMP,
            `date_modified` TIMESTAMP,
            `date_processed` TIMESTAMP,
            `text_as_html` STRING,
            `emphasized_text_contents` STRING,
            `emphasized_text_tags` STRING,
            `documents_source` STRING,
            INDEX `dashscope_v4_inverted_text_index_yunqi_cn_{self._generate_random_suffix()}` (`text`) Inverted PROPERTIES('analyzer'='unicode'),
            INDEX `dashscope_v4_embeddings_vec_index_yunqi_cn_{self._generate_random_suffix()}` (`embeddings`) Vector PROPERTIES('scalar.type'='f32','distance.function'='cosine_distance')
        ) USING PARQUET;
        """
        
        return raw_table_ddl, silver_table_ddl
    
    def _generate_random_suffix(self) -> str:
        """生成6位随机字符串"""
        import random
        import string
        return ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
    
    def check_table_columns(self, table_name: str, required_columns: List[str]) -> List[str]:
        """检查表中缺失的列
        
        Args:
            table_name: 表名
            required_columns: 必需的列名列表
            
        Returns:
            缺失的列名列表
        """
        try:
            # 获取表的列信息
            desc_sql = f"DESCRIBE {self.schema_name}.{table_name}"
            results = self.execute_sql(desc_sql)
            
            # 提取现有列名
            existing_columns = set()
            for row in results:
                if row and len(row) > 0:
                    existing_columns.add(row[0].lower())
            
            # 检查缺失的列
            missing_columns = []
            for col in required_columns:
                if col.lower() not in existing_columns:
                    missing_columns.append(col)
            
            return missing_columns
            
        except Exception as e:
            logger.error(f"[{self.conn_name}] 检查表列失败: {e}")
            return required_columns  # 假设所有列都缺失
    
    def check_and_prepare_tables(self, append_mode: bool = False) -> bool:
        """检查表并准备数据（支持覆盖和追加模式）
        
        Args:
            append_mode: 是否为追加模式
                - True: Raw表清空（避免重复），Silver表保留（追加新数据）
                - False: Raw表和Silver表都清空（完全重建）
        """
        try:
            raw_ddl, silver_ddl = self.get_table_ddl()
            
            # 定义必需的列（所有重要列）
            required_columns = [
                'id', 'record_locator', 'type', 'record_id', 'element_id',
                'filetype', 'file_directory', 'filename', 'last_modified', 'languages',
                'page_number', 'text', 'embeddings', 'parent_id', 'is_continuation',
                'orig_elements', 'element_type', 'coordinates', 'link_texts', 'link_urls',
                'email_message_id', 'sent_from', 'sent_to', 'subject', 'url',
                'version', 'date_created', 'date_modified', 'date_processed',
                'text_as_html', 'emphasized_text_contents', 'emphasized_text_tags', 'documents_source'
            ]
            
            # 处理Raw表 - 无论什么模式都需要清空，因为Pipeline会重新处理所有文档
            if self.check_table_exists(self.raw_table_name):
                # 检查是否缺少必需的列
                missing_columns = self.check_table_columns(self.raw_table_name, required_columns)
                if missing_columns:
                    error_msg = f"Raw表 {self.raw_table_name} 缺少必需的列: {missing_columns}。请手动删除该表或添加缺失的列后重试。"
                    logger.error(f"[{self.conn_name}] {error_msg}")
                    raise ValueError(error_msg)
                else:
                    logger.info(f"[{self.conn_name}] Raw表存在，清空数据（避免重复处理）...")
                    truncate_sql = f"TRUNCATE TABLE {self.workspace}.{self.schema_name}.{self.raw_table_name}"
                    self.execute_sql(truncate_sql)
                    logger.info(f"[{self.conn_name}] Raw表数据已清空")
            else:
                logger.info(f"[{self.conn_name}] Raw表不存在，创建中...")
                self.execute_sql(raw_ddl)
                logger.info(f"[{self.conn_name}] Raw表创建成功")
            
            # 处理Silver表 - 根据模式决定是否保留数据
            if self.check_table_exists(self.silver_table_name):
                # 检查是否缺少必需的列
                missing_columns = self.check_table_columns(self.silver_table_name, required_columns)
                # 检查embeddings列是否为正确的VECTOR类型
                logger.info(f"[{self.conn_name}] 开始检查Silver表embeddings列类型...")
                embeddings_type_correct = self._check_embeddings_column_type(self.silver_table_name)
                logger.info(f"[{self.conn_name}] embeddings类型检查结果: {embeddings_type_correct}")
                
                if missing_columns or not embeddings_type_correct:
                    error_details = []
                    if missing_columns:
                        error_details.append(f"缺少列: {missing_columns}")
                    if not embeddings_type_correct:
                        error_details.append(f"embeddings列类型不正确，应为VECTOR({self.embeddings_dimensions})")
                    
                    error_msg = f"Silver表 {self.silver_table_name} 结构不匹配: {'; '.join(error_details)}。请手动删除该表或修改表结构后重试。"
                    logger.error(f"[{self.conn_name}] {error_msg}")
                    raise ValueError(error_msg)
                else:
                    logger.info(f"[{self.conn_name}] Silver表结构验证通过")
                    if append_mode:
                        logger.info(f"[{self.conn_name}] Silver表存在，追加模式 - 保留现有数据")
                    else:
                        logger.info(f"[{self.conn_name}] Silver表存在，覆盖模式 - 清空数据...")
                        truncate_sql = f"TRUNCATE TABLE {self.workspace}.{self.schema_name}.{self.silver_table_name}"
                        self.execute_sql(truncate_sql)
                        logger.info(f"[{self.conn_name}] Silver表数据已清空")
            else:
                logger.info(f"[{self.conn_name}] Silver表不存在，创建中...")
                try:
                    # 使用正常的DDL（包含VECTOR和INDEX）
                    logger.info(f"[{self.conn_name}] 创建Silver表...")
                    logger.info(f"[{self.conn_name}] Silver表完整DDL: \n{silver_ddl}")
                    self.execute_sql(silver_ddl)
                    logger.info(f"[{self.conn_name}] Silver表创建SQL执行完成")
                    
                    # 等待一下让表创建完成
                    import time
                    time.sleep(1)
                    
                    # 验证表是否真的创建成功
                    if self.check_table_exists(self.silver_table_name):
                        logger.info(f"[{self.conn_name}] 验证：Silver表确实已创建")
                    else:
                        logger.error(f"[{self.conn_name}] 验证失败：Silver表创建后仍不存在")
                        # 尝试查看所有表
                        logger.info(f"[{self.conn_name}] 尝试查看所有表...")
                        try:
                            show_tables_sql = f"SHOW TABLES IN {self.workspace}.{self.schema_name}"
                            tables = self.execute_sql(show_tables_sql)
                            logger.info(f"[{self.conn_name}] Schema {self.workspace}.{self.schema_name} 中的表: {tables}")
                        except Exception as e:
                            logger.error(f"[{self.conn_name}] 无法列出表: {e}")
                            
                except Exception as create_error:
                    logger.error(f"[{self.conn_name}] 创建Silver表失败: {create_error}")
                    logger.error(f"[{self.conn_name}] 错误类型: {type(create_error).__name__}")
                    logger.error(f"[{self.conn_name}] 错误详情: {str(create_error)}")
                    raise create_error
            
            return True
            
        except Exception as e:
            logger.error(f"[{self.conn_name}] 准备表失败: {e}")
            return False
    
    def _check_embeddings_column_type(self, table_name: str) -> bool:
        """检查embeddings列是否为正确的VECTOR类型"""
        try:
            # 查询表结构
            desc_sql = f"DESCRIBE {self.workspace}.{self.schema_name}.{table_name}"
            rows = self.execute_sql(desc_sql)
            
            logger.info(f"[{self.conn_name}] 检查表 {table_name} 的列结构...")
            
            # 查找embeddings列
            for row in rows:
                if len(row) >= 2:
                    col_name = str(row[0]).lower()
                    col_type = str(row[1]).upper()
                    logger.debug(f"[{self.conn_name}] 列: {col_name}, 类型: {col_type}")
                    
                    if col_name == 'embeddings':
                        # 检查是否为VECTOR类型且维度正确
                        expected_type = f"VECTOR({self.embeddings_dimensions})"
                        logger.info(f"[{self.conn_name}] 发现embeddings列，类型: {col_type}, 期望: {expected_type}")
                        
                        if expected_type in col_type or f"VECTOR(FLOAT,{self.embeddings_dimensions})" in col_type:
                            logger.info(f"[{self.conn_name}] embeddings列类型正确: {col_type}")
                            return True
                        else:
                            logger.warning(f"[{self.conn_name}] embeddings列类型不正确: {col_type}, 期望包含: {expected_type}")
                            return False
            
            # 如果没找到embeddings列
            logger.warning(f"[{self.conn_name}] 表 {table_name} 中未找到embeddings列")
            return False
            
        except Exception as e:
            logger.error(f"[{self.conn_name}] 检查embeddings列类型失败: {e}")
            return False
    
    def close(self):
        """关闭连接"""
        if self.conn:
            try:
                self.conn.close()
                logger.info(f"[{self.conn_name}] 连接已关闭")
            except:
                pass


class KnowledgeBaseBuilder:
    """知识库构建器"""
    
    def __init__(self, connection_params: Dict[str, Any], doc_path: str, api_key: Optional[str] = None, append_mode: bool = False, file_glob: Optional[str] = None, recursive: bool = True, documents_original_source: Optional[str] = None):
        self.connection_params = connection_params
        self.doc_path = doc_path
        # 兼容两种命名方式
        self.conn_name = connection_params.get('name') or connection_params.get('connection_name', 'unnamed')
        self.append_mode = append_mode
        self.workspace = connection_params.get('workspace', 'default')
        
        # 文件过滤模式和递归设置
        self.file_glob = file_glob or "**/*.md"  # 默认只处理.md文件
        self.recursive = recursive  # 是否递归搜索子目录
        self.documents_original_source = documents_original_source or "https://yunqi.tech/documents"  # 文档原始来源URL
        
        # 嵌入配置
        self.embedding_config = {
            "provider": "dashscope",
            "model": "text-embedding-v4",
            "api_key": api_key ,
            "dimensions": 1024,
            "chunk_size": 2048,
            "chunk_overlap": 512
        }
        
        # 表名
        self.schema_name = "clickzetta_doc_kb"
        self.raw_table_name = "dashscope_v4_1024_2048_20250611_yunqi_raw_elements"
        self.silver_table_name = "dashscope_v4_1024_2048_20250611_yunqi_elements"
        
        # 转换规则
        self.transformation_rules = []
        self.transformation_engine = None
        
        # 尝试导入转换规则引擎
        try:
            # 尝试相对导入
            try:
                from .kb_transformation_rules import TransformationRuleEngine
            except ImportError:
                # 如果相对导入失败，尝试绝对导入
                from kb_transformation_rules import TransformationRuleEngine
            
            self.transformation_engine = TransformationRuleEngine()
            logger.info(f"[{self.conn_name}] 转换规则引擎已加载")
        except ImportError as e:
            logger.warning(f"[{self.conn_name}] 无法导入转换规则引擎: {e}")
            logger.warning(f"[{self.conn_name}] 将使用默认的转换逻辑")
    
    def set_transformation_rules(self, rules: List[Dict[str, Any]]):
        """设置转换规则"""
        self.transformation_rules = rules
        logger.info(f"[{self.conn_name}] 设置了 {len(rules)} 个转换规则")
        
        # 验证规则
        if self.transformation_engine:
            logger.info(f"[{self.conn_name}] 转换引擎存在，开始验证规则")
            is_valid, errors = self.transformation_engine.validate_rules(rules)
            if not is_valid:
                logger.warning(f"[{self.conn_name}] 转换规则验证失败: {errors}")
            else:
                logger.info(f"[{self.conn_name}] 转换规则验证通过")
        else:
            logger.warning(f"[{self.conn_name}] 转换引擎不存在，无法验证规则")
    
    def build_knowledge_base(self) -> Dict[str, Any]:
        """构建知识库的主流程"""
        start_time = datetime.now()
        result = {
            "connection_name": self.conn_name,
            "status": "pending",
            "start_time": start_time,
            "end_time": None,
            "duration": None,
            "error": None,
            "stats": {}
        }
        
        try:
            logger.info(f"[{self.conn_name}] 开始构建知识库...")
            logger.info(f"[{self.conn_name}] 文档路径: {self.doc_path}")
            logger.info(f"[{self.conn_name}] 目标表: {self.schema_name}.{self.raw_table_name}")
            
            # 设置 unstructured_ingest 的日志级别
            import logging as _logging
            unstructured_logger = _logging.getLogger('unstructured_ingest')
            unstructured_logger.setLevel(_logging.WARNING)
            
            # 检测是否在Streamlit环境中
            import threading
            current_thread = threading.current_thread().name
            is_streamlit = "ScriptRunner" in current_thread
            
            logger.info(f"[{self.conn_name}] 当前线程: {current_thread}")
            logger.info(f"[{self.conn_name}] 是否为Streamlit环境: {is_streamlit}")
            
            if is_streamlit:
                logger.info(f"[{self.conn_name}] 检测到Streamlit环境 (线程: {current_thread})，使用subprocess方式运行Pipeline")
                logger.info(f"[{self.conn_name}] 准备调用_run_pipeline_in_subprocess()...")
                
                # 使用subprocess运行Pipeline
                subprocess_result = self._run_pipeline_in_subprocess()
                logger.info(f"[{self.conn_name}] _run_pipeline_in_subprocess返回: {subprocess_result}")
                
                if not subprocess_result:
                    raise Exception("Pipeline在subprocess中执行失败")
                logger.info(f"[{self.conn_name}] Pipeline在subprocess中执行成功")
                
            else:
                # 直接在主线程中创建和运行Pipeline
                logger.info(f"[{self.conn_name}] 非Streamlit环境，直接创建Pipeline...")
                try:
                    # 创建Pipeline
                    pipeline = self._create_pipeline()
                    logger.info(f"[{self.conn_name}] Pipeline创建成功")
                
                    # 运行Pipeline
                    logger.info(f"[{self.conn_name}] 开始执行pipeline.run()...")
                    logger.info(f"[{self.conn_name}] 文档目录: {self.doc_path}")
                    logger.info(f"[{self.conn_name}] 文件过滤模式: {self.file_glob}")
                    
                    # 执行Pipeline
                    pipeline_result = pipeline.run()
                    
                    logger.info(f"[{self.conn_name}] pipeline.run()执行完成")
                    if pipeline_result is not None:
                        logger.info(f"[{self.conn_name}] Pipeline返回结果: {type(pipeline_result)}")
                    
                    # 清理Pipeline对象
                    logger.info(f"[{self.conn_name}] 清理Pipeline对象...")
                    del pipeline
                    gc.collect()
                    
                except Exception as e:
                    logger.error(f"[{self.conn_name}] Pipeline创建或执行失败: {type(e).__name__}: {str(e)}")
                    logger.error(f"[{self.conn_name}] 堆栈跟踪:\n{traceback.format_exc()}")
                    raise
            
            # 3. 确保数据写入完成
            logger.info(f"[{self.conn_name}] Pipeline执行完成，确保数据写入...")
            import time
            import gc
            
            # Pipeline对象已经在子线程中被清理，这里只需要做全局垃圾回收
            # 强制垃圾回收，确保所有缓冲区都被刷新
            logger.info(f"[{self.conn_name}] 执行全局垃圾回收...")
            gc.collect()
            time.sleep(2)
            
            # 再次垃圾回收，确保所有对象都被清理
            logger.info(f"[{self.conn_name}] 第二次全局垃圾回收...")
            gc.collect()
            time.sleep(3)
            
            # 检查Raw表的数据量
            logger.info(f"[{self.conn_name}] 检查Raw表数据...")
            raw_count = self._get_table_count(self.raw_table_name)
            logger.info(f"[{self.conn_name}] Raw表记录数: {raw_count}")
            
            # 如果还是没有数据，可能需要更长时间
            if raw_count == 0:
                logger.warning(f"[{self.conn_name}] Raw表暂时没有数据，等待10秒后再检查...")
                time.sleep(10)
                raw_count = self._get_table_count(self.raw_table_name)
                logger.info(f"[{self.conn_name}] 再次检查Raw表记录数: {raw_count}")
            
            # 最终检查Raw表的数据量
            raw_count = self._get_table_count(self.raw_table_name)
            logger.info(f"[{self.conn_name}] Raw表最终记录数: {raw_count}")
            
            # 如果没有数据，警告并跳过转换
            if raw_count == 0:
                logger.warning(f"[{self.conn_name}] Raw表没有数据，跳过数据转换步骤")
                logger.warning(f"[{self.conn_name}] 可能的原因：")
                logger.warning(f"[{self.conn_name}]   1. Pipeline尚未完成数据写入")
                logger.warning(f"[{self.conn_name}]   2. 文档目录为空或没有匹配的文件")
                logger.warning(f"[{self.conn_name}]   3. 文档处理出错")
                result["status"] = "warning"
                result["error"] = "Raw表没有数据"
            else:
                # 3. 执行数据转换（从Raw表到Silver表）
                logger.info(f"[{self.conn_name}] 准备执行数据转换，Raw表有 {raw_count} 条记录")
                self._transform_data()
            
            # 检查Silver表的数据量
            silver_count = self._get_table_count(self.silver_table_name)
            logger.info(f"[{self.conn_name}] Silver表转换后记录数: {silver_count}")
            
            if raw_count != silver_count:
                logger.warning(f"[{self.conn_name}] 数据行数不匹配！Raw: {raw_count}, Silver: {silver_count}, 差异: {raw_count - silver_count}")
            
            # 4. 收集统计信息
            stats = self._collect_stats()
            result["stats"] = stats
            
            result["status"] = "success"
            logger.info(f"[{self.conn_name}] 知识库构建成功")
            
        except Exception as e:
            result["status"] = "failed"
            result["error"] = str(e)
            logger.error(f"[{self.conn_name}] 知识库构建失败: {e}")
            logger.error(traceback.format_exc())
        
        finally:
            end_time = datetime.now()
            result["end_time"] = end_time
            result["duration"] = (end_time - start_time).total_seconds()
        
        return result
    
    def _get_table_count(self, table_name: str) -> int:
        """获取表的记录数"""
        logger.info(f"[{self.conn_name}] 开始获取表 {table_name} 的记录数...")
        # print(f"[DEBUG] {self.conn_name} - 创建新的数据库连接来获取表记录数...")
        
        conn = connect(
            password=self.connection_params['password'],
            username=self.connection_params['username'],
            service=self.connection_params['service'],
            instance=self.connection_params['instance'],
            workspace=self.connection_params.get('workspace'),
            schema=self.connection_params.get('schema'),
            vcluster=self.connection_params.get('vcluster')
        )
        
        try:
            with conn.cursor() as cur:
                sql = f"SELECT COUNT(*) FROM {self.workspace}.{self.schema_name}.{table_name}"
                logger.info(f"[{self.conn_name}] 执行SQL: {sql}")
                # print(f"[DEBUG] {self.conn_name} - 执行COUNT查询...")
                cur.execute(sql)
                count = cur.fetchone()[0]
                logger.info(f"[{self.conn_name}] 表 {table_name} 记录数: {count}")
                # print(f"[DEBUG] {self.conn_name} - COUNT查询完成，记录数: {count}")
                return count
        except Exception as e:
            logger.error(f"[{self.conn_name}] 获取表 {table_name} 记录数失败: {e}")
            # print(f"[DEBUG] {self.conn_name} - COUNT查询失败: {e}")
            return -1
        finally:
            logger.info(f"[{self.conn_name}] 关闭数据库连接")
            # print(f"[DEBUG] {self.conn_name} - 关闭数据库连接")
            conn.close()
    
    def _parse_file_glob(self, file_glob: str) -> List[str]:
        """解析文件过滤模式字符串为列表
        
        支持的格式：
        - 单个模式: "*.md"
        - 多个模式（逗号分隔）: "*.md,*.txt"
        - 大括号扩展: "**/*.{md,txt,pdf}"
        """
        import re
        
        # 处理大括号扩展，如 **/*.{md,txt,pdf}
        if '{' in file_glob and '}' in file_glob:
            # 提取大括号中的内容
            match = re.match(r'(.*)\{([^}]+)\}(.*)', file_glob)
            if match:
                prefix, extensions, suffix = match.groups()
                # 分割扩展名并生成模式列表
                patterns = []
                for ext in extensions.split(','):
                    patterns.append(f"{prefix}{ext.strip()}{suffix}")
                logger.info(f"[{self.conn_name}] 解析文件模式 '{file_glob}' 为: {patterns}")
                return patterns
        
        # 处理逗号分隔的多个模式
        if ',' in file_glob:
            patterns = [p.strip() for p in file_glob.split(',')]
            logger.info(f"[{self.conn_name}] 解析文件模式 '{file_glob}' 为: {patterns}")
            return patterns
        
        # 单个模式
        patterns = [file_glob.strip()]
        logger.info(f"[{self.conn_name}] 使用文件模式: {patterns}")
        return patterns
    
    def _create_pipeline(self) -> Pipeline:
        """创建处理Pipeline"""
        # 设置环境变量以进一步抑制日志
        import os
        os.environ['UNSTRUCTURED_LOG_LEVEL'] = 'WARNING'
        
        # 设置其他环境变量来避免潜在问题
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # 禁用tokenizer并行
        os.environ['OMP_NUM_THREADS'] = '1'  # 限制OpenMP线程数
        
        logger.info(f"[{self.conn_name}] 开始构建Pipeline配置...")
        logger.info(f"[{self.conn_name}] 文档路径: {self.doc_path}")
        logger.info(f"[{self.conn_name}] 文件模式: {self.file_glob}")
        
        # 检查文档路径是否存在
        if not os.path.exists(self.doc_path):
            logger.error(f"[{self.conn_name}] 文档路径不存在: {self.doc_path}")
            raise ValueError(f"文档路径不存在: {self.doc_path}")
        
        logger.info(f"[{self.conn_name}] 创建ProcessorConfig...")
        context = ProcessorConfig(
            verbose=False,
            tqdm=False,
            num_processes=1,
            disable_parallelism=True,  # 完全禁用并行处理
        )
        logger.info(f"[{self.conn_name}] ProcessorConfig创建成功")
        
        logger.info(f"[{self.conn_name}] 创建LocalIndexerConfig...")
        indexer_config = LocalIndexerConfig(
            input_path=self.doc_path,
            recursive=self.recursive
        )
        logger.info(f"[{self.conn_name}] LocalIndexerConfig创建成功, recursive={self.recursive}")
        
        logger.info(f"[{self.conn_name}] 创建其他Local配置...")
        downloader_config = LocalDownloaderConfig()
        source_connection_config = LocalConnectionConfig()
        logger.info(f"[{self.conn_name}] Local配置创建成功")
        
        logger.info(f"[{self.conn_name}] 创建FiltererConfig...")
        file_patterns = self._parse_file_glob(self.file_glob)
        logger.info(f"[{self.conn_name}] 文件模式: {file_patterns}")
        
        # 检查文档目录中是否有匹配的文件
        import glob
        matching_files = []
        # 对每个模式进行匹配
        for pattern in file_patterns:
            test_pattern = os.path.join(self.doc_path, pattern)
            files = glob.glob(test_pattern, recursive=self.recursive)
            matching_files.extend(files)
        
        # 去重
        matching_files = list(set(matching_files))
        logger.info(f"[{self.conn_name}] 找到 {len(matching_files)} 个匹配的文件")
        if len(matching_files) > 0:
            logger.info(f"[{self.conn_name}] 第一个匹配文件: {matching_files[0]}")
        
        filterer_config = FiltererConfig(
            file_glob=file_patterns
        )
        logger.info(f"[{self.conn_name}] FiltererConfig创建成功")
        
        logger.info(f"[{self.conn_name}] 创建PartitionerConfig...")
        partitioner_config = PartitionerConfig(
            partition_by_api=False,
            strategy="hi_res",
            additional_partition_args={
                "split_pdf_page": True,
                "split_pdf_allow_failed": True,
                "split_pdf_concurrency_level": 1
            }
        )
        logger.info(f"[{self.conn_name}] PartitionerConfig创建成功")
        
        logger.info(f"[{self.conn_name}] 创建ChunkerConfig...")
        chunker_config = ChunkerConfig(
            chunking_strategy="by_title",
            chunk_max_characters=self.embedding_config["chunk_size"],
            chunk_overlap=self.embedding_config["chunk_overlap"],
            chunk_combine_text_under_n_chars=200,
        )
        logger.info(f"[{self.conn_name}] ChunkerConfig创建成功")
        
        logger.info(f"[{self.conn_name}] 创建EmbedderConfig...")
        embedder_config = EmbedderConfig(
            embedding_provider=self.embedding_config["provider"],
            embedding_model_name=self.embedding_config["model"],
            embedding_api_key=self.embedding_config["api_key"],
        )
        logger.info(f"[{self.conn_name}] EmbedderConfig创建成功")
        
        logger.info(f"[{self.conn_name}] 创建ClickzettaConnectionConfig...")
        destination_connection_config = ClickzettaConnectionConfig(
            access_config=ClickzettaAccessConfig(
                password=self.connection_params['password']
            ),
            username=self.connection_params['username'],
            service=self.connection_params['service'],
            instance=self.connection_params['instance'],
            workspace=self.connection_params.get('workspace', 'default'),
            schema=self.schema_name,
            vcluster=self.connection_params.get('vcluster', 'default'),
        )
        logger.info(f"[{self.conn_name}] ClickzettaConnectionConfig创建成功")
        
        logger.info(f"[{self.conn_name}] 创建Stager和Uploader配置...")
        stager_config = ClickzettaUploadStagerConfig()
        uploader_config = ClickzettaUploaderConfig(
            table_name=self.raw_table_name,
            batch_size=1000,
            documents_original_source=self.documents_original_source
        )
        logger.info(f"[{self.conn_name}] Stager和Uploader配置创建成功")
        
        logger.info(f"[{self.conn_name}] 调用Pipeline.from_configs...")
        
        # 添加更多诊断信息
        import threading
        import gc  # 在顶部导入gc，避免后面引用错误
        
        logger.info(f"[{self.conn_name}] 当前线程: {threading.current_thread().name}")
        logger.info(f"[{self.conn_name}] 活跃线程数: {threading.active_count()}")
        
        # 检查multiprocessing状态
        import multiprocessing
        current_method = None
        try:
            current_method = multiprocessing.get_start_method()
            logger.info(f"[{self.conn_name}] Multiprocessing start method: {current_method}")
        except:
            logger.info(f"[{self.conn_name}] Multiprocessing start method not set")
        
        # 尝试创建Pipeline
        try:
            # 在创建Pipeline前清理一些可能的状态
            gc.collect()
            
            # 如果是fork模式，尝试切换到spawn
            if current_method == 'fork':
                logger.warning(f"[{self.conn_name}] 检测到fork模式，这在多线程环境中可能有问题")
                # 不能在这里改变启动方法，因为已经太晚了
            
            logger.info(f"[{self.conn_name}] 开始调用Pipeline.from_configs...")
            
            # 尝试分步创建，看看是哪个配置导致的问题
            logger.info(f"[{self.conn_name}] 尝试创建基础Pipeline...")
            
            # 第一步：只用最基础的配置
            try:
                basic_pipeline = Pipeline.from_configs(
                    context=context,
                    indexer_config=indexer_config,
                    downloader_config=downloader_config,
                    source_connection_config=source_connection_config,
                    partitioner_config=partitioner_config
                )
                logger.info(f"[{self.conn_name}] 基础Pipeline创建成功!")
                del basic_pipeline
                gc.collect()
            except Exception as e:
                logger.error(f"[{self.conn_name}] 基础Pipeline创建失败: {e}")
                raise
            
            # 第二步：创建完整Pipeline
            logger.info(f"[{self.conn_name}] 创建完整Pipeline...")
            pipeline = Pipeline.from_configs(
                context=context,
                indexer_config=indexer_config,
                downloader_config=downloader_config,
                source_connection_config=source_connection_config,
                filterer_config=filterer_config,
                partitioner_config=partitioner_config,
                chunker_config=chunker_config,
                embedder_config=embedder_config,
                destination_connection_config=destination_connection_config,
                stager_config=stager_config,
                uploader_config=uploader_config
            )
            logger.info(f"[{self.conn_name}] Pipeline.from_configs调用成功!")
            return pipeline
        except Exception as e:
            logger.error(f"[{self.conn_name}] Pipeline.from_configs失败: {type(e).__name__}: {str(e)}")
            import traceback
            logger.error(f"[{self.conn_name}] 堆栈跟踪:\n{traceback.format_exc()}")
            raise
    
    def _run_pipeline_in_subprocess(self) -> bool:
        """在子进程中运行Pipeline，避免Streamlit环境的线程冲突"""
        logger.info(f"[{self.conn_name}] 进入_run_pipeline_in_subprocess函数")
        
        import subprocess
        import tempfile
        import json
        
        # 创建临时Python脚本
        script_content = '''#!/usr/bin/env python3
import os
import sys
import json
import site

# 设置环境
os.environ['UNSTRUCTURED_LOG_LEVEL'] = 'WARNING'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OMP_NUM_THREADS'] = '1'

# 设置日志级别以减少输出
import logging
logging.getLogger().setLevel(logging.WARNING)
logging.getLogger('unstructured_ingest').setLevel(logging.WARNING)
logging.getLogger('MainProcess').setLevel(logging.WARNING)

# 读取配置
config = json.loads(sys.argv[1])

# 添加路径 - 支持Docker和本地环境
if os.path.exists('/app/unstructured-ingest-clickzetta'):
    # Docker环境
    sys.path.insert(0, '/app/unstructured-ingest-clickzetta')
    sys.path.insert(0, '/app/unstructured-ingest-clickzetta/multi_lakehouse_kb_builder')
else:
    # 本地环境 - 使用相对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    sys.path.insert(0, parent_dir)
    sys.path.insert(0, script_dir)

# 调试输出（使用stderr避免干扰MCP）
sys.stderr.write(f"Python executable: {sys.executable}\\n")
sys.stderr.write(f"Python version: {sys.version}\\n")
sys.stderr.write(f"sys.path: {sys.path[:3]}...\\n")

# 在Docker环境中，确保添加虚拟环境的site-packages
if os.path.exists('/app/.venv'):
    # 获取Python版本
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    site_packages = f"/app/.venv/lib/python{python_version}/site-packages"
    if os.path.exists(site_packages) and site_packages not in sys.path:
        sys.path.insert(0, site_packages)
        sys.stderr.write(f"Added venv site-packages: {site_packages}\\n")

# 尝试导入必要的模块
try:
    from unstructured_ingest.pipeline.pipeline import Pipeline
except ImportError as e:
    sys.stderr.write(f"Failed to import Pipeline: {e}\\n")
    sys.stderr.write(f"Trying to import unstructured_ingest...\\n")
    try:
        import unstructured_ingest
        sys.stderr.write(f"unstructured_ingest location: {unstructured_ingest.__file__}\\n")
    except ImportError:
        sys.stderr.write("unstructured_ingest module not found!\\n")
    raise
from unstructured_ingest.interfaces import ProcessorConfig
from unstructured_ingest.processes.connectors.local import *
from unstructured_ingest.processes.filter import FiltererConfig
from unstructured_ingest.processes.partitioner import PartitionerConfig
from unstructured_ingest.processes.chunker import ChunkerConfig
from unstructured_ingest.processes.embedder import EmbedderConfig
from unstructured_ingest.processes.connectors.sql.clickzetta import *

try:
    # 创建Pipeline
    pipeline = Pipeline.from_configs(
        context=ProcessorConfig(
            verbose=False,
            tqdm=False,
            num_processes=1,
            disable_parallelism=True,
        ),
        indexer_config=LocalIndexerConfig(
            input_path=config['doc_path'],
            recursive=config.get('recursive', True)
        ),
        downloader_config=LocalDownloaderConfig(),
        source_connection_config=LocalConnectionConfig(),
        filterer_config=FiltererConfig(
            file_glob=config['file_patterns']  # 使用解析后的模式列表
        ),
        partitioner_config=PartitionerConfig(
            partition_by_api=False,
            strategy="hi_res",
            additional_partition_args={
                "split_pdf_page": True,
                "split_pdf_allow_failed": True,
                "split_pdf_concurrency_level": 1
            }
        ),
        chunker_config=ChunkerConfig(
            chunking_strategy="by_title",
            chunk_max_characters=config['chunk_size'],
            chunk_overlap=config['chunk_overlap'],
            chunk_combine_text_under_n_chars=200,
        ),
        embedder_config=EmbedderConfig(
            embedding_provider=config['embedding_provider'],
            embedding_model_name=config['embedding_model'],
            embedding_api_key=config['api_key'],
        ),
        destination_connection_config=ClickzettaConnectionConfig(
            access_config=ClickzettaAccessConfig(
                password=config['password']
            ),
            username=config['username'],
            service=config['service'],
            instance=config['instance'],
            workspace=config.get('workspace', 'default'),
            schema=config['schema_name'],
            vcluster=config.get('vcluster', 'default'),
        ),
        stager_config=ClickzettaUploadStagerConfig(),
        uploader_config=ClickzettaUploaderConfig(
            table_name=config['raw_table_name'],
            batch_size=1000,
            documents_original_source=config.get('documents_original_source', 'https://yunqi.tech/documents')
        ),
    )
    
    # 运行Pipeline
    # 调试信息使用stderr，但SUCCESS标记必须输出到stdout
    sys.stderr.write("Pipeline created successfully, running...\\n")
    pipeline.run()
    print("SUCCESS")  # 必须输出到stdout，用于检测成功
    
except Exception as e:
    sys.stderr.write(f"ERROR: {type(e).__name__}: {str(e)}\\n")
    import traceback
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
'''
        
        # 准备配置
        # 解析file_glob为列表
        file_patterns = self._parse_file_glob(self.file_glob)
        
        config = {
            'doc_path': self.doc_path,
            'file_glob': self.file_glob,
            'file_patterns': file_patterns,  # 添加解析后的模式列表
            'recursive': self.recursive,
            'documents_original_source': self.documents_original_source,
            'chunk_size': self.embedding_config['chunk_size'],
            'chunk_overlap': self.embedding_config['chunk_overlap'],
            'embedding_provider': self.embedding_config['provider'],
            'embedding_model': self.embedding_config['model'],
            'api_key': self.embedding_config['api_key'],
            'password': self.connection_params['password'],
            'username': self.connection_params['username'],
            'service': self.connection_params['service'],
            'instance': self.connection_params['instance'],
            'workspace': self.connection_params.get('workspace', 'default'),
            'schema_name': self.schema_name,
            'vcluster': self.connection_params.get('vcluster', 'default'),
            'raw_table_name': self.raw_table_name
        }
        
        try:
            # 创建临时脚本文件
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                script_file = f.name
                f.write(script_content)
            
            # 使脚本可执行
            os.chmod(script_file, 0o755)
            
            # 在子进程中运行
            logger.info(f"[{self.conn_name}] 启动subprocess运行Pipeline...")
            
            # 在Docker环境中，优先使用虚拟环境的Python
            python_executable = sys.executable
            if os.path.exists('/app/.venv/bin/python'):
                python_executable = '/app/.venv/bin/python'
                logger.info(f"[{self.conn_name}] 使用Docker虚拟环境Python: {python_executable}")
            else:
                logger.info(f"[{self.conn_name}] 使用当前Python: {python_executable}")
            
            logger.info(f"[{self.conn_name}] 命令: {[python_executable, script_file, 'CONFIG_JSON']}")
            logger.info(f"[{self.conn_name}] 脚本文件: {script_file}")
            
            # 设置环境变量以确保Python能找到模块
            env = os.environ.copy()
            if os.path.exists('/app/.venv'):
                env['VIRTUAL_ENV'] = '/app/.venv'
                env['PATH'] = f"/app/.venv/bin:{env.get('PATH', '')}"
                # 设置PYTHONPATH以包含unstructured-ingest-clickzetta
                pythonpath = env.get('PYTHONPATH', '')
                if pythonpath:
                    env['PYTHONPATH'] = f"/app/unstructured-ingest-clickzetta:{pythonpath}"
                else:
                    env['PYTHONPATH'] = "/app/unstructured-ingest-clickzetta"
            
            result = subprocess.run(
                [python_executable, script_file, json.dumps(config)],
                capture_output=True,
                text=True,
                timeout=1800,  # 30分钟超时
                env=env
            )
            
            logger.info(f"[{self.conn_name}] subprocess执行完成，返回码: {result.returncode}")
            
            # 打印输出用于调试（限制长度避免阻塞）
            if result.stdout:
                stdout_lines = result.stdout.strip().split('\n')
                if len(stdout_lines) > 100:
                    logger.info(f"[{self.conn_name}] Subprocess输出（前50行）:\n" + '\n'.join(stdout_lines[:50]))
                    logger.info(f"[{self.conn_name}] ... 省略 {len(stdout_lines) - 100} 行 ...")
                    logger.info(f"[{self.conn_name}] Subprocess输出（后50行）:\n" + '\n'.join(stdout_lines[-50:]))
                else:
                    logger.info(f"[{self.conn_name}] Subprocess输出: {result.stdout}")
            if result.stderr:
                stderr_lines = result.stderr.strip().split('\n')
                if len(stderr_lines) > 50:
                    logger.warning(f"[{self.conn_name}] Subprocess错误（前25行）:\n" + '\n'.join(stderr_lines[:25]))
                    logger.warning(f"[{self.conn_name}] ... 省略 {len(stderr_lines) - 50} 行 ...")
                    logger.warning(f"[{self.conn_name}] Subprocess错误（后25行）:\n" + '\n'.join(stderr_lines[-25:]))
                else:
                    logger.warning(f"[{self.conn_name}] Subprocess错误: {result.stderr}")
            
            # 清理临时文件
            os.unlink(script_file)
            
            # 检查结果
            if "SUCCESS" in result.stdout:
                logger.info(f"[{self.conn_name}] 检测到SUCCESS标记，准备返回True")
                # print(f"[DEBUG] {self.conn_name} - subprocess成功，返回True")
                return True
            else:
                logger.error(f"[{self.conn_name}] Subprocess执行失败")
                # print(f"[DEBUG] {self.conn_name} - subprocess失败，返回False")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error(f"[{self.conn_name}] Pipeline执行超时")
            return False
        except Exception as e:
            logger.error(f"[{self.conn_name}] Subprocess执行出错: {e}")
            return False
    
    def _transform_data(self):
        """执行数据转换（从Raw表到Silver表）"""
        logger.info(f"[{self.conn_name}] 开始数据转换...")
        
        # 详细记录转换规则状态
        logger.info(f"[{self.conn_name}] transformation_rules: {self.transformation_rules}")
        logger.info(f"[{self.conn_name}] transformation_engine: {self.transformation_engine}")
        logger.info(f"[{self.conn_name}] 规则数量: {len(self.transformation_rules) if self.transformation_rules else 0}")
        logger.info(f"[{self.conn_name}] 引擎是否存在: {self.transformation_engine is not None}")
        
        # 检查是否有转换规则和转换引擎
        if self.transformation_rules and self.transformation_engine:
            logger.info(f"[{self.conn_name}] 使用转换规则引擎进行数据转换，规则数量: {len(self.transformation_rules)}")
            transform_sql = self._generate_transformation_sql_with_rules()
        else:
            logger.info(f"[{self.conn_name}] 使用默认转换逻辑")
            if not self.transformation_rules:
                logger.info(f"[{self.conn_name}]   - 原因: transformation_rules 为空或 None")
            if not self.transformation_engine:
                logger.info(f"[{self.conn_name}]   - 原因: transformation_engine 为 None")
            transform_sql = self._generate_default_transformation_sql()
        
        # 创建连接并执行转换
        conn = connect(
            password=self.connection_params['password'],
            username=self.connection_params['username'],
            service=self.connection_params['service'],
            instance=self.connection_params['instance'],
            workspace=self.connection_params.get('workspace'),
            schema=self.connection_params.get('schema'),
            vcluster=self.connection_params.get('vcluster')
        )
        
        try:
            with conn.cursor() as cur:
                # SQL验证 - 移除，因为统计方法不准确
                # 实际的SQL是正确的，包含所有33列
                
                # 记录执行的SQL（前500个字符）
                logger.info(f"[{self.conn_name}] 执行转换SQL: {transform_sql[:500]}...")
                
                # 执行转换前先检查Raw表数据
                cur.execute(f"SELECT COUNT(*) FROM {self.workspace}.{self.schema_name}.{self.raw_table_name}")
                raw_count_before = cur.fetchone()[0]
                logger.info(f"[{self.conn_name}] 转换前Raw表记录数: {raw_count_before}")
                
                # 执行转换
                cur.execute(transform_sql)
                logger.info(f"[{self.conn_name}] 数据转换SQL执行完成")
                
                # 检查Silver表数据
                cur.execute(f"SELECT COUNT(*) FROM {self.workspace}.{self.schema_name}.{self.silver_table_name}")
                silver_count_after = cur.fetchone()[0]
                logger.info(f"[{self.conn_name}] 转换后Silver表记录数: {silver_count_after}")
                
                if raw_count_before != silver_count_after:
                    logger.warning(f"[{self.conn_name}] 转换前后记录数不一致！Raw: {raw_count_before}, Silver: {silver_count_after}")
        except Exception as e:
            logger.error(f"[{self.conn_name}] 数据转换失败: {e}")
            raise
        finally:
            conn.close()
    
    def _generate_transformation_sql_with_rules(self) -> str:
        """使用转换规则引擎生成转换SQL"""
        
        # 不再需要自动添加embeddings的CAST转换，因为Raw表和Silver表都使用VECTOR类型
        # 直接使用原始规则
        enhanced_rules = self.transformation_rules.copy()
        
        # 生成转换SQL
        transform_sql = self.transformation_engine.generate_transformation_sql(
            schema_name=self.schema_name,
            raw_table=self.raw_table_name,
            silver_table=self.silver_table_name,
            rules=enhanced_rules,
            workspace=self.connection_params.get('workspace')
        )
        
        # 检查SQL内容 - 移除调试日志
        
        # 只需要替换INSERT OVERWRITE为对应的insert_clause
        if self.append_mode:
            transform_sql = transform_sql.replace('INSERT OVERWRITE', 'INSERT INTO', 1)
        
        return transform_sql
    
    def _generate_correct_transformation_sql(self, rules: List[Dict]) -> str:
        """直接生成正确的转换SQL"""
        logger.info(f"[{self.conn_name}] 使用备用方法生成正确的SQL")
        
        # 所有33列
        all_columns = [
            'id', 'record_locator', 'type', 'record_id', 'element_id',
            'filetype', 'file_directory', 'filename', 'last_modified', 'languages',
            'page_number', 'text', 'embeddings', 'parent_id', 'is_continuation',
            'orig_elements', 'element_type', 'coordinates', 'link_texts', 'link_urls',
            'email_message_id', 'sent_from', 'sent_to', 'subject', 'url',
            'version', 'date_created', 'date_modified', 'date_processed',
            'text_as_html', 'emphasized_text_contents', 'emphasized_text_tags', 'documents_source'
        ]
        
        # 解析转换规则 - 支持多次转换
        transformations = {}  # 最终的转换表达式
        column_transforms = {}  # 记录每个列的转换链
        where_conditions = []
        
        for rule in rules:
            if rule.get('type') == 'transform':
                col = rule.get('column', '')
                op = rule.get('operation', '')
                
                # 如果这个列已经有转换，需要链式应用
                if col in column_transforms:
                    base_expr = column_transforms[col]
                else:
                    base_expr = col
                
                # 应用新的转换
                if op == 'trim':
                    new_expr = f"TRIM({base_expr})"
                elif op == 'trim_left' or op == 'ltrim':
                    new_expr = f"LTRIM({base_expr})"
                elif op == 'trim_right' or op == 'rtrim':
                    new_expr = f"RTRIM({base_expr})"
                elif op == 'lowercase' or op == 'lower':
                    new_expr = f"LOWER({base_expr})"
                elif op == 'uppercase' or op == 'upper':
                    new_expr = f"UPPER({base_expr})"
                elif 'CAST' in op and 'VECTOR' in op:
                    # 处理embeddings的CAST
                    new_expr = op.replace('{column}', base_expr)
                else:
                    # 自定义操作
                    new_expr = op.replace('{column}', base_expr)
                
                # 更新转换链
                column_transforms[col] = new_expr
                transformations[col] = new_expr
                
                # 更新转换链 - 移除调试日志
                    
            elif rule.get('type') == 'filter_group':
                # 处理过滤组（支持嵌套的AND/OR）
                group_where = self._build_filter_group(rule)
                if group_where:
                    where_conditions.append(group_where)
                    
            elif rule.get('type') == 'filter':
                condition_type = rule.get('condition_type', '')
                params = rule.get('params', {})
                condition = self._build_filter_condition(condition_type, params)
                if condition:
                    where_conditions.append(condition)
        
        # 构建SELECT列表
        select_expressions = []
        for col in all_columns:
            if col in transformations:
                select_expressions.append(f"{transformations[col]} AS {col}")
            else:
                select_expressions.append(col)
        
        # 构建SQL
        insert_clause = "INSERT INTO" if self.append_mode else "INSERT OVERWRITE"
        
        sql_parts = [
            f"-- Generated by _generate_correct_transformation_sql (fallback method)",
            f"-- Total columns: {len(select_expressions)}",
            f"{insert_clause} {self.workspace}.{self.schema_name}.{self.silver_table_name}",
            "SELECT",
            "    " + ",\n    ".join(select_expressions),
            f"FROM {self.workspace}.{self.schema_name}.{self.raw_table_name}"
        ]
        
        if where_conditions:
            sql_parts.append("WHERE " + " AND ".join(where_conditions))
        
        result_sql = "\n".join(sql_parts)
        logger.info(f"[{self.conn_name}] 生成了包含 {len(select_expressions)} 列的SQL")
        
        return result_sql
    
    def _build_filter_condition(self, condition_type: str, params: Dict) -> Optional[str]:
        """构建单个过滤条件"""
        col = params.get('column', '')
        
        if condition_type == 'not_null' and col:
            return f"{col} IS NOT NULL"
        elif condition_type == 'not_empty' and col:
            return f"LENGTH({col}) > 0"
        elif condition_type == 'min_length' and col:
            value = params.get('value', 0)
            return f"LENGTH({col}) >= {value}"
        elif condition_type == 'max_length' and col:
            value = params.get('value', 0)
            return f"LENGTH({col}) <= {value}"
        elif condition_type == 'contains' and col:
            value = params.get('value', '')
            if value:
                return f"{col} LIKE '%{value}%'"
        elif condition_type == 'not_contains' and col:
            value = params.get('value', '')
            if value:
                return f"{col} NOT LIKE '%{value}%'"
        elif condition_type == 'starts_with' and col:
            value = params.get('value', '')
            if value:
                return f"{col} LIKE '{value}%'"
        elif condition_type == 'ends_with' and col:
            value = params.get('value', '')
            if value:
                return f"{col} LIKE '%{value}'"
        elif condition_type == 'equals' and col:
            value = params.get('value', '')
            if value:
                return f"{col} = '{value}'"
        elif condition_type == 'not_equals' and col:
            value = params.get('value', '')
            if value:
                return f"{col} != '{value}'"
        elif condition_type == 'regex_match' and col:
            pattern = params.get('pattern', '')
            if pattern:
                return f"{col} REGEXP '{pattern}'"
        
        return None
    
    def _build_filter_group(self, filter_group: Dict) -> Optional[str]:
        """递归构建过滤组（支持AND/OR嵌套）"""
        if filter_group.get('type') == 'filter':
            # 单个过滤条件
            condition_type = filter_group.get('condition_type', '')
            params = filter_group.get('params', {})
            return self._build_filter_condition(condition_type, params)
        
        elif filter_group.get('type') == 'filter_group':
            # 过滤组
            operator = filter_group.get('operator', 'AND').upper()
            conditions = filter_group.get('conditions', [])
            
            sub_conditions = []
            for cond in conditions:
                sub_clause = self._build_filter_group(cond)
                if sub_clause:
                    sub_conditions.append(sub_clause)
            
            if len(sub_conditions) > 1:
                return f"({' {operator} '.join(sub_conditions)})"
            elif len(sub_conditions) == 1:
                return sub_conditions[0]
        
        return None
    
    def _get_embeddings_expression(self) -> str:
        """获取embeddings列的表达式"""
        # Raw表和Silver表都使用VECTOR类型，不需要CAST转换
        logger.debug(f"[{self.conn_name}] Raw表和Silver表都使用VECTOR类型，直接复制embeddings列")
        return "embeddings"
    
    def _generate_default_transformation_sql(self) -> str:
        """生成默认的转换SQL（无转换规则时使用）"""
        # 根据append_mode决定使用INSERT INTO还是INSERT OVERWRITE
        insert_clause = "INSERT INTO" if self.append_mode else "INSERT OVERWRITE"
        
        # 检查Silver表的embeddings列类型，决定是否需要CAST
        embeddings_expr = self._get_embeddings_expression()
        
        return f"""
        -- Generated by multi_lakehouse_kb_builder.py _generate_default_transformation_sql()
        -- This is the default transformation SQL with all 33 columns
        {insert_clause} {self.workspace}.{self.schema_name}.{self.silver_table_name}
        SELECT 
            id, 
            record_locator, 
            type, 
            record_id, 
            element_id, 
            filetype, 
            file_directory, 
            filename, 
            last_modified, 
            languages, 
            page_number, 
            text, 
            {embeddings_expr}, 
            parent_id, 
            is_continuation, 
            orig_elements, 
            element_type, 
            coordinates, 
            link_texts, 
            link_urls, 
            email_message_id, 
            sent_from, 
            sent_to, 
            subject, 
            url, 
            version, 
            date_created, 
            date_modified, 
            date_processed, 
            text_as_html,
            emphasized_text_contents, 
            emphasized_text_tags,
            documents_source
        FROM {self.workspace}.{self.schema_name}.{self.raw_table_name};
        """
    
    def _collect_stats(self) -> Dict[str, Any]:
        """收集统计信息"""
        stats = {}
        
        conn = connect(
            password=self.connection_params['password'],
            username=self.connection_params['username'],
            service=self.connection_params['service'],
            instance=self.connection_params['instance'],
            workspace=self.connection_params.get('workspace', 'default'),
            schema=self.schema_name,
            vcluster=self.connection_params.get('vcluster', 'default')
        )
        
        try:
            with conn.cursor() as cur:
                # Raw表记录数
                cur.execute(f"SELECT COUNT(*) FROM {self.schema_name}.{self.raw_table_name}")
                stats["raw_table_count"] = cur.fetchone()[0]
                
                # Silver表记录数
                cur.execute(f"SELECT COUNT(*) FROM {self.schema_name}.{self.silver_table_name}")
                stats["silver_table_count"] = cur.fetchone()[0]
                
                # 有嵌入向量的记录数
                cur.execute(f"SELECT COUNT(*) FROM {self.schema_name}.{self.silver_table_name} WHERE embeddings IS NOT NULL")
                stats["records_with_embeddings"] = cur.fetchone()[0]
                
        except Exception as e:
            logger.error(f"[{self.conn_name}] 收集统计信息失败: {e}")
        finally:
            conn.close()
        
        return stats


class BatchKnowledgeBaseDeployer:
    """批量知识库部署器"""
    
    def __init__(self, config_path: str, doc_path: str, execution_mode: str = "serial", append_mode: bool = False):
        self.config_path = config_path
        self.doc_path = doc_path
        self.execution_mode = execution_mode
        self.append_mode = append_mode
        self.results = []
        
        # 加载连接管理器
        self.conn_manager = LakehouseConnectionManager(config_path)
        
        # 获取DashScope API密钥
        self.dashscope_api_key = self.conn_manager.get_dashscope_api_key()
    
    def deploy_to_all_lakehouse(self, 
                               filter_pattern: Optional[str] = None,
                               exclude_pattern: Optional[str] = None,
                               max_workers: int = 5) -> List[Dict[str, Any]]:
        """批量部署知识库到所有Lakehouse"""
        # 获取要部署的连接
        connections = self.conn_manager.get_active_connections(
            filter_pattern=filter_pattern,
            exclude_pattern=exclude_pattern
        )
        
        if not connections:
            logger.warning("没有找到符合条件的连接")
            return []
        
        logger.info(f"准备部署到 {len(connections)} 个Lakehouse")
        logger.info(f"执行模式: {self.execution_mode}")
        
        if self.execution_mode == "serial":
            return self._deploy_serial(connections)
        else:
            return self._deploy_parallel(connections, max_workers)
    
    def _deploy_serial(self, connections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """串行部署"""
        results = []
        
        for i, conn in enumerate(connections, 1):
            conn_name = conn.get('connection_name', 'unnamed')
            logger.info(f"\n{'='*60}")
            logger.info(f"部署进度: {i}/{len(connections)} - {conn_name}")
            logger.info(f"{'='*60}")
            
            result = self._deploy_to_single_lakehouse(conn)
            results.append(result)
            
            # 如果失败，询问是否继续
            if result["status"] == "failed" and i < len(connections):
                logger.warning(f"部署到 {conn_name} 失败")
                # 在实际使用中可以加入交互式确认
        
        return results
    
    def _deploy_parallel(self, connections: List[Dict[str, Any]], max_workers: int) -> List[Dict[str, Any]]:
        """并行部署"""
        results = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_conn = {
                executor.submit(self._deploy_to_single_lakehouse, conn): conn 
                for conn in connections
            }
            
            # 处理完成的任务
            for future in as_completed(future_to_conn):
                conn = future_to_conn[future]
                conn_name = conn.get('connection_name', 'unnamed')
                
                try:
                    result = future.result()
                    results.append(result)
                    
                    if result["status"] == "success":
                        logger.info(f"✅ {conn_name} 部署成功")
                    else:
                        logger.error(f"❌ {conn_name} 部署失败")
                        
                except Exception as e:
                    logger.error(f"❌ {conn_name} 部署异常: {e}")
                    results.append({
                        "connection_name": conn_name,
                        "status": "failed",
                        "error": str(e)
                    })
        
        return results
    
    def _deploy_to_single_lakehouse(self, connection: Dict[str, Any]) -> Dict[str, Any]:
        """部署到单个Lakehouse"""
        conn_name = connection.get('connection_name', 'unnamed')
        
        try:
            # 1. 初始化Schema管理器
            schema_manager = LakehouseSchemaManager(connection)
            
            # 2. 检查并创建schema
            if not schema_manager.check_and_create_schema():
                return {
                    "connection_name": conn_name,
                    "status": "failed",
                    "error": "Schema创建失败"
                }
            
            # 3. 检查并准备表
            if not schema_manager.check_and_prepare_tables(append_mode=self.append_mode):
                return {
                    "connection_name": conn_name,
                    "status": "failed",
                    "error": "表准备失败"
                }
            
            # 4. 构建知识库
            kb_builder = KnowledgeBaseBuilder(connection, self.doc_path, self.dashscope_api_key, append_mode=self.append_mode)
            result = kb_builder.build_knowledge_base()
            
            # 5. 运行数据验证（如果部署成功）
            if result.get("status") == "success":
                logger.info(f"[{conn_name}] 开始数据验证...")
                try:
                    from kb_data_validator import KnowledgeBaseValidator
                    validator = KnowledgeBaseValidator(connection)
                    validation_report = validator.generate_validation_report()
                    validator.close()
                    
                    # 将验证结果添加到部署结果中
                    result["validation"] = validation_report
                    
                    # 检查验证结果
                    summary = validation_report.get("summary", {})
                    if summary.get("all_checks_passed"):
                        logger.info(f"[{conn_name}] ✅ 数据验证通过")
                    else:
                        logger.warning(f"[{conn_name}] ⚠️  数据验证发现问题")
                        if not summary.get("row_count_match"):
                            logger.warning(f"   - 行数不匹配")
                        if summary.get("zero_vectors_found", 0) > 0:
                            logger.warning(f"   - 发现 {summary['zero_vectors_found']} 个问题向量")
                        if summary.get("dimension_issues", 0) > 0:
                            logger.warning(f"   - 发现 {summary['dimension_issues']} 个维度问题")
                            
                except Exception as e:
                    logger.error(f"[{conn_name}] 数据验证失败: {e}")
                    result["validation"] = {
                        "status": "failed",
                        "error": str(e)
                    }
            
            # 6. 关闭连接
            schema_manager.close()
            
            return result
            
        except Exception as e:
            logger.error(f"[{conn_name}] 部署失败: {e}")
            logger.error(traceback.format_exc())
            return {
                "connection_name": conn_name,
                "status": "failed",
                "error": str(e)
            }
    
    def print_summary(self, results: List[Dict[str, Any]]):
        """打印部署总结"""
        print("\n" + "="*80)
        print("📊 知识库部署总结")
        print("="*80)
        
        total = len(results)
        success = sum(1 for r in results if r["status"] == "success")
        failed = total - success
        
        print(f"总计: {total} 个Lakehouse")
        print(f"成功: {success} 个")
        print(f"失败: {failed} 个")
        
        print("\n详细结果:")
        for result in results:
            conn_name = result.get("connection_name", "unnamed")
            status = result.get("status", "unknown")
            
            if status == "success":
                stats = result.get("stats", {})
                duration = result.get("duration", 0)
                print(f"\n✅ {conn_name}")
                print(f"   耗时: {duration:.2f}秒")
                print(f"   Raw表记录数: {stats.get('raw_table_count', 0)}")
                print(f"   Silver表记录数: {stats.get('silver_table_count', 0)}")
                print(f"   有嵌入向量的记录: {stats.get('records_with_embeddings', 0)}")
                
                # 显示验证结果
                validation = result.get("validation", {})
                if validation:
                    summary = validation.get("summary", {})
                    if summary.get("all_checks_passed"):
                        print(f"   数据验证: ✅ 通过")
                    else:
                        print(f"   数据验证: ⚠️  发现问题")
                        if summary.get("zero_vectors_found", 0) > 0:
                            print(f"     - 问题向量: {summary['zero_vectors_found']} 个 ({summary.get('zero_vectors_percentage', 0):.1f}%)")
                        if summary.get("dimension_issues", 0) > 0:
                            print(f"     - 维度错误: {summary['dimension_issues']} 个")
            else:
                error = result.get("error", "未知错误")
                print(f"\n❌ {conn_name}")
                print(f"   错误: {error}")
        
        print("\n" + "="*80)


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="批量部署ClickZetta知识库")
    parser.add_argument("--config", default="~/.clickzetta/connections.json", help="连接配置文件路径")
    parser.add_argument("--docs", default="/Users/liangmo/yunqidoc/cn_markdown_20250526", help="文档目录路径")
    parser.add_argument("--mode", choices=["serial", "parallel"], default="serial", help="执行模式")
    parser.add_argument("--filter", help="只部署包含此模式的连接")
    parser.add_argument("--exclude", help="排除包含此模式的连接")
    parser.add_argument("--workers", type=int, default=5, help="并行模式下的工作线程数")
    
    args = parser.parse_args()
    
    # 检查文档目录是否存在
    doc_path = os.path.expanduser(args.docs)
    if not os.path.exists(doc_path):
        logger.error(f"文档目录不存在: {doc_path}")
        sys.exit(1)
    
    # 创建部署器
    deployer = BatchKnowledgeBaseDeployer(
        config_path=args.config,
        doc_path=doc_path,
        execution_mode=args.mode
    )
    
    # 执行部署
    logger.info("开始批量部署知识库...")
    start_time = datetime.now()
    
    results = deployer.deploy_to_all_lakehouse(
        filter_pattern=args.filter,
        exclude_pattern=args.exclude,
        max_workers=args.workers
    )
    
    end_time = datetime.now()
    total_duration = (end_time - start_time).total_seconds()
    
    # 打印总结
    deployer.print_summary(results)
    print(f"\n总耗时: {total_duration:.2f}秒")
    
    # 保存结果到文件
    # 确保reports目录存在
    reports_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")
    os.makedirs(reports_dir, exist_ok=True)
    
    result_file = os.path.join(reports_dir, f"kb_deployment_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump({
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "total_duration": total_duration,
            "results": results
        }, f, ensure_ascii=False, indent=2)
    
    logger.info(f"结果已保存到: {result_file}")


def _generate_simple_transformation_sql(schema_name: str, raw_table: str, silver_table: str, 
                                       append_mode: bool = False) -> str:
    """生成简单的转换SQL（无转换规则）
    
    Args:
        schema_name: Schema名称
        raw_table: 源表名
        silver_table: 目标表名
        append_mode: 是否为追加模式
        
    Returns:
        SQL语句
    """
    insert_clause = "INSERT INTO" if append_mode else "INSERT OVERWRITE"
    
    return f"""{insert_clause} {schema_name}.{silver_table}
SELECT 
    id, 
    record_locator, 
    type, 
    record_id, 
    element_id, 
    filetype, 
    file_directory, 
    filename, 
    last_modified, 
    languages, 
    page_number, 
    text, 
    embeddings, 
    parent_id, 
    is_continuation, 
    orig_elements, 
    element_type, 
    coordinates, 
    link_texts, 
    link_urls, 
    email_message_id, 
    sent_from, 
    sent_to, 
    subject, 
    url, 
    version, 
    date_created, 
    date_modified, 
    date_processed, 
    text_as_html,
    emphasized_text_contents, 
    emphasized_text_tags,
    documents_source
FROM {schema_name}.{raw_table};"""


def _generate_random_suffix_for_sql() -> str:
    """为SQL生成6位随机字符串"""
    import random
    import string
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))

def get_complete_deployment_sql(kb_config: Dict, append_mode: bool = False) -> str:
    """获取完整的部署SQL语句（独立函数，不需要数据库连接）
    
    Args:
        kb_config: 知识库配置，包含:
            - table_config: 表配置（schema_name, raw_table_name, silver_table_name）
            - vector_config: 向量配置（dimensions等）
            - transformation_rules: 转换规则（可选）
        append_mode: 是否为追加模式
        
    Returns:
        完整的SQL语句
    """
    # 获取配置
    table_config = kb_config['table_config']
    schema_name = table_config['schema_name']
    raw_table = table_config['raw_table_name']
    silver_table = table_config['silver_table_name']
    
    # 获取连接配置中的workspace（如果有）
    connection_config = kb_config.get('connection_config', {})
    workspace = connection_config.get('workspace', 'default')
    
    # 获取向量配置
    vector_config = kb_config.get('vector_config', {})
    dimensions = vector_config.get('dimensions', 1024)
    
    # 获取转换规则
    transformation_rules = kb_config.get('transformation_rules', {}).get('rules', [])
    
    sql_statements = []
    
    # 1. 创建schema
    sql_statements.append(f"-- 创建Schema（如果不存在）")
    sql_statements.append(f"CREATE SCHEMA IF NOT EXISTS {schema_name};")
    
    # 2. 创建raw表
    sql_statements.append(f"\n-- 创建Raw表（存储原始解析数据）")
    sql_statements.append(f"""CREATE TABLE IF NOT EXISTS {schema_name}.{raw_table} (
    `id` STRING,
    `record_locator` STRING,
    `type` STRING,
    `record_id` STRING,
    `element_id` STRING,
    `filetype` STRING,
    `file_directory` STRING,
    `filename` STRING,
    `last_modified` TIMESTAMP,
    `languages` STRING,
    `page_number` STRING,
    `text` STRING,
    `embeddings` VECTOR({dimensions}),
    `parent_id` STRING,
    `is_continuation` BOOLEAN,
    `orig_elements` STRING,
    `element_type` STRING,
    `coordinates` STRING,
    `link_texts` STRING,
    `link_urls` STRING,
    `email_message_id` STRING,
    `sent_from` STRING,
    `sent_to` STRING,
    `subject` STRING,
    `url` STRING,
    `version` STRING,
    `date_created` TIMESTAMP,
    `date_modified` TIMESTAMP,
    `date_processed` TIMESTAMP,
    `text_as_html` STRING,
    `emphasized_text_contents` STRING,
    `emphasized_text_tags` STRING,
    `documents_source` STRING
) USING PARQUET;""")
    
    # 3. 创建silver表（带索引）
    sql_statements.append(f"\n-- 创建Silver表（带向量索引和全文索引）")
    sql_statements.append(f"""CREATE TABLE IF NOT EXISTS {schema_name}.{silver_table} (
    `id` STRING,
    `record_locator` STRING,
    `type` STRING,
    `record_id` STRING,
    `element_id` STRING,
    `filetype` STRING,
    `file_directory` STRING,
    `filename` STRING,
    `last_modified` TIMESTAMP,
    `languages` STRING,
    `page_number` STRING,
    `text` STRING,
    `embeddings` VECTOR({dimensions}),
    `parent_id` STRING,
    `is_continuation` BOOLEAN,
    `orig_elements` STRING,
    `element_type` STRING,
    `coordinates` STRING,
    `link_texts` STRING,
    `link_urls` STRING,
    `email_message_id` STRING,
    `sent_from` STRING,
    `sent_to` STRING,
    `subject` STRING,
    `url` STRING,
    `version` STRING,
    `date_created` TIMESTAMP,
    `date_modified` TIMESTAMP,
    `date_processed` TIMESTAMP,
    `text_as_html` STRING,
    `emphasized_text_contents` STRING,
    `emphasized_text_tags` STRING,
    `documents_source` STRING,
    INDEX `dashscope_v4_inverted_text_index_yunqi_cn_{_generate_random_suffix_for_sql()}` (`text`) Inverted PROPERTIES('analyzer'='unicode'),
    INDEX `dashscope_v4_embeddings_vec_index_yunqi_cn_{_generate_random_suffix_for_sql()}` (`embeddings`) Vector PROPERTIES('scalar.type'='f32','distance.function'='cosine_distance')
) USING PARQUET;""")
    
    # 4. 数据处理策略
    sql_statements.append(f"\n-- Raw表始终清空（避免重复处理）")
    sql_statements.append(f"TRUNCATE TABLE {schema_name}.{raw_table};")
    
    if not append_mode:
        sql_statements.append(f"\n-- 覆盖模式：Silver表也清空")
        sql_statements.append(f"TRUNCATE TABLE {schema_name}.{silver_table};")
    else:
        sql_statements.append(f"\n-- 追加模式：Silver表保留现有数据")
        sql_statements.append(f"-- 新数据将追加到Silver表中")
        sql_statements.append(f"-- 注意：可能产生重复数据，建议后续添加基于文件路径的去重逻辑")
    
    # 5. 数据转换SQL
    if transformation_rules:
        sql_statements.append(f"\n-- 从Raw表转换数据到Silver表（应用 {len(transformation_rules)} 条转换规则）")
        
        # 使用转换规则引擎生成SQL
        import kb_transformation_rules
        
        from kb_transformation_rules import TransformationRuleEngine
        transformation_engine = TransformationRuleEngine()
        
        try:
            
            transformation_sql = transformation_engine.generate_transformation_sql(
                schema_name=schema_name,
                raw_table=raw_table,
                silver_table=silver_table,
                rules=transformation_rules
            )
        except Exception as e:
            logger.error(f"生成转换SQL失败: {e}")
            raise
        # 移除调试日志
        
        # 根据append_mode调整INSERT语句
        if append_mode and 'INSERT OVERWRITE' in transformation_sql:
            transformation_sql = transformation_sql.replace('INSERT OVERWRITE', 'INSERT INTO', 1)
        
        sql_statements.append(transformation_sql)
    else:
        # 没有转换规则，使用简单的数据复制
        insert_clause = "INSERT INTO" if append_mode else "INSERT OVERWRITE"
        sql_statements.append(f"\n-- 从Raw表转换数据到Silver表")
        
        # 生成默认的转换SQL（所有33列）
        default_sql = _generate_simple_transformation_sql(
            schema_name=schema_name,
            raw_table=raw_table,
            silver_table=silver_table,
            append_mode=append_mode
        )
        sql_statements.append(default_sql)
    
    return '\n\n'.join(sql_statements)


if __name__ == "__main__":
    main()