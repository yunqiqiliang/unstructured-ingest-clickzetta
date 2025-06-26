#!/usr/bin/env python3
"""
知识库知识管理器
提供知识的增删改查功能
"""

import os
import sys
import json
import logging
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import uuid
from clickzetta.connector import connect
import dashscope
from dashscope import TextEmbedding

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from multi_lakehouse_kb_builder import LakehouseConnectionManager

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class KnowledgeEntry:
    """知识条目数据结构"""
    
    def __init__(self, 
                 text: str,
                 source: str = "UserInput",
                 languages: List[str] = None,
                 filetype: str = "text",
                 metadata: Dict[str, Any] = None):
        self.id = str(uuid.uuid4())
        self.text = text
        self.source = source
        self.languages = languages or ["zh-cn"]
        self.filetype = filetype
        self.metadata = metadata or {}
        self.created_at = datetime.now()
        
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "text": self.text,
            "source": self.source,
            "languages": self.languages,
            "filetype": self.filetype,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat()
        }


class KnowledgeManager:
    """知识管理器 - 单个Lakehouse"""
    
    def __init__(self, connection_params: Dict[str, Any]):
        self.connection_params = connection_params
        self.conn_name = connection_params.get('connection_name', 'unnamed')
        self.schema_name = "clickzetta_doc_kb"
        self.silver_table_name = "dashscope_v4_1024_2048_20250611_yunqi_elements"
        self.embeddings_dimensions = 1024
        self.embedding_model_name = "text-embedding-v4"
        self.conn = None
        
        # 获取DashScope API密钥
        self.dashscope_api_key = self._get_dashscope_api_key()
        if self.dashscope_api_key:
            dashscope.api_key = self.dashscope_api_key
        
    def _get_dashscope_api_key(self) -> Optional[str]:
        """获取DashScope API密钥"""
        # 尝试从连接管理器获取
        try:
            conn_manager = LakehouseConnectionManager()
            api_key = conn_manager.get_dashscope_api_key()
            if api_key:
                return api_key
        except:
            pass
        
        # 从环境变量获取
        return os.getenv("DASHSCOPE_API_KEY")
    
    def create_connection(self):
        """创建数据库连接"""
        try:
            self.conn = connect(
                password=self.connection_params['password'],
                username=self.connection_params['username'],
                service=self.connection_params['service'],
                instance=self.connection_params['instance'],
                workspace=self.connection_params.get('workspace', 'default'),
                schema=self.schema_name,
                vcluster=self.connection_params.get('vcluster', 'default')
            )
            logger.info(f"[{self.conn_name}] 成功创建连接")
            return True
        except Exception as e:
            logger.error(f"[{self.conn_name}] 创建连接失败: {e}")
            return False
    
    def get_embedding(self, text: str) -> List[float]:
        """使用DashScope获取文本嵌入"""
        try:
            response = TextEmbedding.call(
                model=self.embedding_model_name,
                input=text
            )
            if response.status_code == 200:
                embedding = response.output['embeddings'][0]['embedding']
                if len(embedding) != self.embeddings_dimensions:
                    logger.warning(f"嵌入维度不匹配: 期望{self.embeddings_dimensions}, 实际{len(embedding)}")
                return embedding
            else:
                raise Exception(f"DashScope API错误: {response.message}")
        except Exception as e:
            logger.error(f"获取嵌入失败: {e}")
            return [0.0] * self.embeddings_dimensions
    
    def add_knowledge(self, entry: KnowledgeEntry) -> bool:
        """添加单条知识"""
        if not self.conn:
            if not self.create_connection():
                return False
        
        try:
            # 获取嵌入向量
            embedding = self.get_embedding(entry.text)
            embedding_str = str(embedding)
            
            # 转义文本中的单引号
            escaped_text = entry.text.replace("'", "''")
            
            # 构建插入SQL
            sql = f"""
            INSERT INTO {self.schema_name}.{self.silver_table_name} (
                id, type, record_id, element_id, filetype, 
                last_modified, languages, text, embeddings, 
                date_created, date_modified, date_processed,
                documents_source
            ) VALUES (
                '{entry.id}', 
                '{entry.source}', 
                '{entry.id}', 
                '{entry.id}', 
                '{entry.filetype}',
                CURRENT_TIMESTAMP, 
                '{json.dumps(entry.languages)}',
                '{escaped_text}',
                CAST('{embedding_str}' AS vector(float,{self.embeddings_dimensions})), 
                CURRENT_TIMESTAMP, 
                CURRENT_TIMESTAMP, 
                CURRENT_TIMESTAMP,
                'UserAdded'
            );
            """
            
            with self.conn.cursor() as cur:
                cur.execute(sql)
                
            logger.info(f"[{self.conn_name}] 成功添加知识: {entry.id}")
            return True
            
        except Exception as e:
            logger.error(f"[{self.conn_name}] 添加知识失败: {e}")
            return False
    
    def add_batch_knowledge(self, entries: List[KnowledgeEntry]) -> Dict[str, Any]:
        """批量添加知识"""
        results = {
            "total": len(entries),
            "success": 0,
            "failed": 0,
            "failed_entries": []
        }
        
        for entry in entries:
            if self.add_knowledge(entry):
                results["success"] += 1
            else:
                results["failed"] += 1
                results["failed_entries"].append(entry.id)
        
        return results
    
    def search_knowledge(self, 
                        query: str = None, 
                        source: str = None,
                        limit: int = 10) -> List[Dict[str, Any]]:
        """搜索知识"""
        if not self.conn:
            if not self.create_connection():
                return []
        
        try:
            where_conditions = []
            
            # 按来源筛选
            if source:
                where_conditions.append(f"type = '{source}'")
            
            # 文本搜索
            if query:
                escaped_query = query.replace("'", "''")
                where_conditions.append(f"text LIKE '%{escaped_query}%'")
            
            where_clause = " AND ".join(where_conditions) if where_conditions else "1=1"
            
            sql = f"""
            SELECT 
                id,
                type as source,
                text,
                filetype,
                languages,
                date_created,
                documents_source
            FROM {self.schema_name}.{self.silver_table_name}
            WHERE {where_clause}
            ORDER BY date_created DESC
            LIMIT {limit}
            """
            
            with self.conn.cursor() as cur:
                cur.execute(sql)
                results = cur.fetchall()
                columns = [desc[0] for desc in cur.description]
                
            knowledge_list = []
            for row in results:
                knowledge_dict = dict(zip(columns, row))
                knowledge_list.append(knowledge_dict)
                
            return knowledge_list
            
        except Exception as e:
            logger.error(f"[{self.conn_name}] 搜索知识失败: {e}")
            return []
    
    def delete_knowledge(self, knowledge_id: str) -> bool:
        """删除单条知识"""
        if not self.conn:
            if not self.create_connection():
                return False
        
        try:
            sql = f"""
            DELETE FROM {self.schema_name}.{self.silver_table_name}
            WHERE id = '{knowledge_id}'
            """
            
            with self.conn.cursor() as cur:
                cur.execute(sql)
                
            logger.info(f"[{self.conn_name}] 成功删除知识: {knowledge_id}")
            return True
            
        except Exception as e:
            logger.error(f"[{self.conn_name}] 删除知识失败: {e}")
            return False
    
    def delete_batch_knowledge(self, knowledge_ids: List[str]) -> Dict[str, Any]:
        """批量删除知识"""
        results = {
            "total": len(knowledge_ids),
            "success": 0,
            "failed": 0,
            "failed_ids": []
        }
        
        for kid in knowledge_ids:
            if self.delete_knowledge(kid):
                results["success"] += 1
            else:
                results["failed"] += 1
                results["failed_ids"].append(kid)
        
        return results
    
    def delete_by_source(self, source: str) -> int:
        """按来源删除知识"""
        if not self.conn:
            if not self.create_connection():
                return 0
        
        try:
            # 先查询数量
            count_sql = f"""
            SELECT COUNT(*) FROM {self.schema_name}.{self.silver_table_name}
            WHERE type = '{source}'
            """
            
            with self.conn.cursor() as cur:
                cur.execute(count_sql)
                count = cur.fetchone()[0]
                
                if count > 0:
                    # 执行删除
                    delete_sql = f"""
                    DELETE FROM {self.schema_name}.{self.silver_table_name}
                    WHERE type = '{source}'
                    """
                    cur.execute(delete_sql)
                    
            logger.info(f"[{self.conn_name}] 成功删除来源为'{source}'的{count}条知识")
            return count
            
        except Exception as e:
            logger.error(f"[{self.conn_name}] 按来源删除知识失败: {e}")
            return 0
    
    def update_knowledge(self, knowledge_id: str, new_text: str) -> bool:
        """更新知识内容"""
        if not self.conn:
            if not self.create_connection():
                return False
        
        try:
            # 获取新的嵌入向量
            embedding = self.get_embedding(new_text)
            embedding_str = str(embedding)
            
            # 转义文本
            escaped_text = new_text.replace("'", "''")
            
            sql = f"""
            UPDATE {self.schema_name}.{self.silver_table_name}
            SET 
                text = '{escaped_text}',
                embeddings = CAST('{embedding_str}' AS vector(float,{self.embeddings_dimensions})),
                date_modified = CURRENT_TIMESTAMP
            WHERE id = '{knowledge_id}'
            """
            
            with self.conn.cursor() as cur:
                cur.execute(sql)
                
            logger.info(f"[{self.conn_name}] 成功更新知识: {knowledge_id}")
            return True
            
        except Exception as e:
            logger.error(f"[{self.conn_name}] 更新知识失败: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取知识库统计信息"""
        if not self.conn:
            if not self.create_connection():
                return {}
        
        try:
            # 统计各来源的知识数量
            sql = f"""
            SELECT 
                type as source,
                COUNT(*) as count
            FROM {self.schema_name}.{self.silver_table_name}
            GROUP BY type
            ORDER BY count DESC
            """
            
            with self.conn.cursor() as cur:
                cur.execute(sql)
                results = cur.fetchall()
                
            stats = {
                "total": sum(row[1] for row in results),
                "by_source": {row[0]: row[1] for row in results}
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"[{self.conn_name}] 获取统计信息失败: {e}")
            return {}
    
    def close(self):
        """关闭连接"""
        if self.conn:
            try:
                self.conn.close()
                logger.info(f"[{self.conn_name}] 连接已关闭")
            except:
                pass


class BatchKnowledgeManager:
    """批量知识管理器 - 多个Lakehouse"""
    
    def __init__(self, config_path: str = "~/.clickzetta/connections.json"):
        self.conn_manager = LakehouseConnectionManager(config_path)
        self.connections = self.conn_manager.connections
        
    def add_to_all_lakehouse(self, entries: List[KnowledgeEntry]) -> List[Dict[str, Any]]:
        """添加知识到所有Lakehouse"""
        results = []
        
        logger.info(f"准备向 {len(self.connections)} 个Lakehouse添加 {len(entries)} 条知识")
        
        for conn in self.connections:
            conn_name = conn.get('connection_name', 'unnamed')
            logger.info(f"\n处理: {conn_name}")
            
            try:
                manager = KnowledgeManager(conn)
                result = manager.add_batch_knowledge(entries)
                manager.close()
                
                results.append({
                    "connection_name": conn_name,
                    "status": "success",
                    "result": result
                })
                
            except Exception as e:
                logger.error(f"{conn_name} 处理失败: {e}")
                results.append({
                    "connection_name": conn_name,
                    "status": "failed",
                    "error": str(e)
                })
        
        return results
    
    def delete_from_all_lakehouse(self, knowledge_ids: List[str]) -> List[Dict[str, Any]]:
        """从所有Lakehouse删除知识"""
        results = []
        
        logger.info(f"准备从 {len(self.connections)} 个Lakehouse删除 {len(knowledge_ids)} 条知识")
        
        for conn in self.connections:
            conn_name = conn.get('connection_name', 'unnamed')
            logger.info(f"\n处理: {conn_name}")
            
            try:
                manager = KnowledgeManager(conn)
                result = manager.delete_batch_knowledge(knowledge_ids)
                manager.close()
                
                results.append({
                    "connection_name": conn_name,
                    "status": "success",
                    "result": result
                })
                
            except Exception as e:
                logger.error(f"{conn_name} 处理失败: {e}")
                results.append({
                    "connection_name": conn_name,
                    "status": "failed",
                    "error": str(e)
                })
        
        return results
    
    def search_across_lakehouse(self, query: str = None, source: str = None) -> Dict[str, List[Dict[str, Any]]]:
        """在所有Lakehouse中搜索知识"""
        all_results = {}
        
        for conn in self.connections:
            conn_name = conn.get('connection_name', 'unnamed')
            
            try:
                manager = KnowledgeManager(conn)
                results = manager.search_knowledge(query=query, source=source)
                manager.close()
                
                all_results[conn_name] = results
                
            except Exception as e:
                logger.error(f"{conn_name} 搜索失败: {e}")
                all_results[conn_name] = []
        
        return all_results
    
    def get_all_statistics(self) -> Dict[str, Dict[str, Any]]:
        """获取所有Lakehouse的统计信息"""
        all_stats = {}
        
        for conn in self.connections:
            conn_name = conn.get('connection_name', 'unnamed')
            
            try:
                manager = KnowledgeManager(conn)
                stats = manager.get_statistics()
                manager.close()
                
                all_stats[conn_name] = stats
                
            except Exception as e:
                logger.error(f"{conn_name} 获取统计失败: {e}")
                all_stats[conn_name] = {"error": str(e)}
        
        return all_stats


def load_knowledge_from_file(file_path: str) -> List[KnowledgeEntry]:
    """从文件加载知识条目"""
    entries = []
    file_ext = os.path.splitext(file_path)[1].lower()
    
    try:
        if file_ext == '.json':
            # JSON格式
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # 支持单条或多条
            if isinstance(data, list):
                items = data
            else:
                items = [data]
                
            for item in items:
                entry = KnowledgeEntry(
                    text=item.get('text', ''),
                    source=item.get('source', 'UserInput'),
                    languages=item.get('languages', ['zh-cn']),
                    filetype=item.get('filetype', 'text'),
                    metadata=item.get('metadata', {})
                )
                entries.append(entry)
                
        elif file_ext == '.csv':
            # CSV格式
            df = pd.read_csv(file_path)
            
            for _, row in df.iterrows():
                entry = KnowledgeEntry(
                    text=str(row.get('text', '')),
                    source=str(row.get('source', 'UserInput')),
                    languages=eval(str(row.get('languages', '["zh-cn"]'))) if 'languages' in row else ['zh-cn']
                )
                entries.append(entry)
                
        elif file_ext in ['.txt', '.md']:
            # 文本文件，每行一条知识
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
            for line in lines:
                line = line.strip()
                if line:  # 跳过空行
                    entry = KnowledgeEntry(text=line)
                    entries.append(entry)
                    
        else:
            raise ValueError(f"不支持的文件格式: {file_ext}")
            
    except Exception as e:
        logger.error(f"加载文件失败: {e}")
        raise
        
    return entries


def main():
    """主函数 - 测试知识管理功能"""
    # 测试添加知识
    test_entries = [
        KnowledgeEntry(
            text="ClickZetta是云器、Singdata的技术品牌。云器是ClickZetta的中国商业公司，Singdata是ClickZetta的海外商业公司。",
            source="CompanyInfo"
        ),
        KnowledgeEntry(
            text="ClickZetta Lakehouse支持结构化、半结构化和非结构化数据的统一存储和分析。",
            source="ProductInfo"
        ),
        KnowledgeEntry(
            text="使用VECTOR类型存储向量数据，支持余弦距离、欧氏距离等多种相似度计算方法。",
            source="TechnicalDoc"
        )
    ]
    
    # 创建批量管理器
    batch_manager = BatchKnowledgeManager()
    
    # 添加知识
    logger.info("测试添加知识...")
    results = batch_manager.add_to_all_lakehouse(test_entries)
    
    # 打印结果
    for result in results:
        conn_name = result['connection_name']
        if result['status'] == 'success':
            res = result['result']
            logger.info(f"{conn_name}: 成功添加 {res['success']}/{res['total']} 条知识")
        else:
            logger.error(f"{conn_name}: 失败 - {result['error']}")
    
    # 搜索知识
    logger.info("\n测试搜索知识...")
    search_results = batch_manager.search_across_lakehouse(query="ClickZetta")
    
    for conn_name, results in search_results.items():
        logger.info(f"{conn_name}: 找到 {len(results)} 条相关知识")
        for r in results[:2]:  # 只显示前2条
            logger.info(f"  - {r.get('text', '')[:50]}...")
    
    # 获取统计
    logger.info("\n获取统计信息...")
    stats = batch_manager.get_all_statistics()
    
    for conn_name, stat in stats.items():
        if 'error' not in stat:
            logger.info(f"{conn_name}: 总计 {stat.get('total', 0)} 条知识")
            for source, count in stat.get('by_source', {}).items():
                logger.info(f"  - {source}: {count} 条")


if __name__ == "__main__":
    main()