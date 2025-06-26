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
import json
import logging
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback
import time

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
        self.schema_name = "clickzetta_doc_kb"
        self.raw_table_name = "dashscope_v4_1024_2048_20250611_yunqi_raw_elements"
        self.silver_table_name = "dashscope_v4_1024_2048_20250611_yunqi_elements"
        self.embeddings_dimensions = 1024
        
        # 连接标识
        self.conn_name = connection_params.get('connection_name', 'unnamed')
        
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
                    return cur.fetchall()
                return [['OPERATION SUCCEED']]
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
            check_sql = f"SHOW TABLES IN {self.schema_name} LIKE '{table_name}'"
            results = self.execute_sql(check_sql)
            return any(table_name in str(row) for row in results)
        except Exception as e:
            logger.error(f"[{self.conn_name}] 检查表 {table_name} 失败: {e}")
            return False
    
    def get_table_ddl(self) -> Tuple[str, str]:
        """获取Raw表和Silver表的DDL"""
        raw_table_ddl = f"""
        CREATE TABLE IF NOT EXISTS {self.schema_name}.{self.raw_table_name} (
            id STRING,
            record_locator STRING,
            type STRING,
            record_id STRING,
            element_id STRING,
            filetype STRING,
            file_directory STRING,
            filename STRING,
            last_modified TIMESTAMP,
            languages STRING,
            page_number STRING,
            text STRING,
            embeddings VECTOR({self.embeddings_dimensions}),
            parent_id STRING,
            is_continuation BOOLEAN,
            orig_elements STRING,
            element_type STRING,
            coordinates STRING,
            link_texts STRING,
            link_urls STRING,
            email_message_id STRING,
            sent_from STRING,
            sent_to STRING,
            subject STRING,
            url STRING,
            version STRING,
            date_created TIMESTAMP,
            date_modified TIMESTAMP,
            date_processed TIMESTAMP,
            text_as_html STRING,
            emphasized_text_contents STRING,
            emphasized_text_tags STRING,
            documents_original_source STRING
        );
        """
        
        silver_table_ddl = f"""
        CREATE TABLE IF NOT EXISTS {self.schema_name}.{self.silver_table_name} (
            id STRING,
            record_locator STRING,
            type STRING,
            record_id STRING,
            element_id STRING,
            filetype STRING,
            file_directory STRING,
            filename STRING,
            last_modified TIMESTAMP,
            languages STRING,
            page_number STRING,
            text STRING,
            embeddings VECTOR({self.embeddings_dimensions}),
            parent_id STRING,
            is_continuation BOOLEAN,
            orig_elements STRING,
            element_type STRING,
            coordinates STRING,
            link_texts STRING,
            link_urls STRING,
            email_message_id STRING,
            sent_from STRING,
            sent_to STRING,
            subject STRING,
            url STRING,
            version STRING,
            date_created TIMESTAMP,
            date_modified TIMESTAMP,
            date_processed TIMESTAMP,
            text_as_html STRING,
            emphasized_text_contents STRING,
            emphasized_text_tags STRING,
            documents_source STRING,
            INDEX dashscope_v4_inverted_text_index_yunqi_cn (text) INVERTED PROPERTIES('analyzer'='unicode'),
            INDEX dashscope_v4_embeddings_vec_index_yunqi_cn(embeddings) USING vector properties (
                "scalar.type" = "f32",
                "distance.function" = "cosine_distance")
        );
        """
        
        return raw_table_ddl, silver_table_ddl
    
    def check_and_prepare_tables(self) -> bool:
        """检查表并清空数据（不删除表）"""
        try:
            raw_ddl, silver_ddl = self.get_table_ddl()
            
            # 处理Raw表
            if self.check_table_exists(self.raw_table_name):
                logger.info(f"[{self.conn_name}] Raw表存在，清空数据...")
                truncate_sql = f"TRUNCATE TABLE {self.schema_name}.{self.raw_table_name}"
                self.execute_sql(truncate_sql)
                logger.info(f"[{self.conn_name}] Raw表数据已清空")
            else:
                logger.info(f"[{self.conn_name}] Raw表不存在，创建中...")
                self.execute_sql(raw_ddl)
                logger.info(f"[{self.conn_name}] Raw表创建成功")
            
            # 处理Silver表
            if self.check_table_exists(self.silver_table_name):
                logger.info(f"[{self.conn_name}] Silver表存在，清空数据...")
                truncate_sql = f"TRUNCATE TABLE {self.schema_name}.{self.silver_table_name}"
                self.execute_sql(truncate_sql)
                logger.info(f"[{self.conn_name}] Silver表数据已清空")
            else:
                logger.info(f"[{self.conn_name}] Silver表不存在，创建中...")
                self.execute_sql(silver_ddl)
                logger.info(f"[{self.conn_name}] Silver表创建成功")
            
            return True
            
        except Exception as e:
            logger.error(f"[{self.conn_name}] 准备表失败: {e}")
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
    
    def __init__(self, connection_params: Dict[str, Any], doc_path: str, api_key: Optional[str] = None):
        self.connection_params = connection_params
        self.doc_path = doc_path
        self.conn_name = connection_params.get('connection_name', 'unnamed')
        
        # 嵌入配置
        self.embedding_config = {
            "provider": "dashscope",
            "model": "text-embedding-v4",
            "api_key": api_key or os.getenv("DASHSCOPE_API_KEY", "sk-7d178531cbd14ce6bba2d16fe3948239"),
            "dimensions": 1024,
            "chunk_size": 2048,
            "chunk_overlap": 512
        }
        
        # 表名
        self.schema_name = "clickzetta_doc_kb"
        self.raw_table_name = "dashscope_v4_1024_2048_20250611_yunqi_raw_elements"
        self.silver_table_name = "dashscope_v4_1024_2048_20250611_yunqi_elements"
    
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
            
            # 1. 创建Pipeline
            pipeline = self._create_pipeline()
            
            # 2. 运行Pipeline（处理文档并写入Raw表）
            logger.info(f"[{self.conn_name}] 运行Pipeline处理文档...")
            pipeline.run()
            
            # 3. 执行数据转换（从Raw表到Silver表）
            logger.info(f"[{self.conn_name}] 执行数据转换...")
            self._transform_data()
            
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
    
    def _create_pipeline(self) -> Pipeline:
        """创建处理Pipeline"""
        return Pipeline.from_configs(
            context=ProcessorConfig(
                verbose=False,
                tqdm=False,
                num_processes=2,  # 使用较少的进程数以确保稳定性
            ),
            
            indexer_config=LocalIndexerConfig(
                input_path=self.doc_path,
                file_glob="**/*",
                recursive=True
            ),
            downloader_config=LocalDownloaderConfig(),
            source_connection_config=LocalConnectionConfig(),
            
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
                chunk_max_characters=self.embedding_config["chunk_size"],
                chunk_overlap=self.embedding_config["chunk_overlap"],
                chunk_combine_text_under_n_chars=200,
            ),
            
            embedder_config=EmbedderConfig(
                embedding_provider=self.embedding_config["provider"],
                embedding_model_name=self.embedding_config["model"],
                embedding_api_key=self.embedding_config["api_key"],
            ),
            
            destination_connection_config=ClickzettaConnectionConfig(
                access_config=ClickzettaAccessConfig(
                    password=self.connection_params['password']
                ),
                username=self.connection_params['username'],
                service=self.connection_params['service'],
                instance=self.connection_params['instance'],
                workspace=self.connection_params.get('workspace', 'default'),
                schema=self.schema_name,
                vcluster=self.connection_params.get('vcluster', 'default'),
            ),
            stager_config=ClickzettaUploadStagerConfig(),
            uploader_config=ClickzettaUploaderConfig(
                table_name=self.raw_table_name,
                documents_original_source="https://yunqi.tech/documents"
            ),
        )
    
    def _transform_data(self):
        """执行数据转换（从Raw表到Silver表）"""
        transform_sql = f"""
        INSERT OVERWRITE {self.schema_name}.{self.silver_table_name}
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
            CAST(embeddings AS VECTOR({self.embedding_config['dimensions']})) AS embeddings, 
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
            "https://yunqi.tech/documents" as documents_source
        FROM {self.schema_name}.{self.raw_table_name};
        """
        
        # 创建连接并执行转换
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
                cur.execute(transform_sql)
                logger.info(f"[{self.conn_name}] 数据转换完成")
        finally:
            conn.close()
    
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
    
    def __init__(self, config_path: str, doc_path: str, execution_mode: str = "serial"):
        self.config_path = config_path
        self.doc_path = doc_path
        self.execution_mode = execution_mode
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
            if not schema_manager.check_and_prepare_tables():
                return {
                    "connection_name": conn_name,
                    "status": "failed",
                    "error": "表准备失败"
                }
            
            # 4. 构建知识库
            kb_builder = KnowledgeBaseBuilder(connection, self.doc_path, self.dashscope_api_key)
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


if __name__ == "__main__":
    main()