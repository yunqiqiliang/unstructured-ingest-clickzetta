#!/usr/bin/env python3
"""
知识库知识添加器
用于向已部署的ClickZetta知识库中添加自定义知识条目
"""

import os
import sys
import json
import csv
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import uuid
import pandas as pd
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


@dataclass
class KnowledgeEntry:
    """知识条目数据结构"""
    text: str
    source: str = "UserInput"
    filetype: str = "text"
    languages: List[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.languages is None:
            self.languages = ["zh-cn"]
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "text": self.text,
            "source": self.source,
            "filetype": self.filetype,
            "languages": self.languages,
            "metadata": self.metadata
        }


class KnowledgeAdder:
    """知识添加器"""
    
    def __init__(self, connection_params: Dict[str, Any], dashscope_api_key: str):
        self.connection_params = connection_params
        self.conn_name = connection_params.get('connection_name', 'unnamed')
        self.dashscope_api_key = dashscope_api_key
        self.schema_name = "clickzetta_doc_kb"
        self.silver_table_name = "dashscope_v4_1024_2048_20250611_yunqi_elements"
        self.embedding_model = "text-embedding-v4"
        self.embedding_dimensions = 1024
        self.conn = None
        
        # 设置DashScope API
        dashscope.api_key = self.dashscope_api_key
        
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
                model=self.embedding_model,
                input=text
            )
            if response.status_code == 200:
                embedding = response.output['embeddings'][0]['embedding']
                if len(embedding) != self.embedding_dimensions:
                    logger.warning(f"嵌入维度不匹配: 期望{self.embedding_dimensions}, 实际{len(embedding)}")
                return embedding
            else:
                raise Exception(f"DashScope API错误: {response.message}")
        except Exception as e:
            logger.error(f"获取嵌入失败: {e}")
            return [0.0] * self.embedding_dimensions
    
    def add_single_entry(self, entry: KnowledgeEntry) -> bool:
        """添加单条知识"""
        if not self.conn and not self.create_connection():
            return False
        
        try:
            # 获取嵌入向量
            embedding = self.get_embedding(entry.text)
            
            # 准备SQL语句
            sql = f"""
            INSERT INTO {self.schema_name}.{self.silver_table_name} (
                id, type, record_id, element_id, filetype, 
                last_modified, languages, text, embeddings, 
                date_created, date_modified, date_processed,
                documents_source
            ) VALUES (
                '{str(uuid.uuid4())}', 
                '{entry.source}', 
                '{str(uuid.uuid4())}', 
                '{str(uuid.uuid4())}', 
                '{entry.filetype}',
                CURRENT_TIMESTAMP, 
                '{json.dumps(entry.languages)}',
                '{entry.text.replace("'", "''")}',
                CAST('{embedding}' AS vector(float, {self.embedding_dimensions})), 
                CURRENT_TIMESTAMP, 
                CURRENT_TIMESTAMP, 
                CURRENT_TIMESTAMP,
                'UserKnowledge'
            );
            """
            
            # 执行SQL
            with self.conn.cursor() as cur:
                cur.execute(sql)
            
            logger.info(f"[{self.conn_name}] 成功添加知识条目")
            return True
            
        except Exception as e:
            logger.error(f"[{self.conn_name}] 添加知识失败: {e}")
            return False
    
    def add_batch_entries(self, entries: List[KnowledgeEntry]) -> Tuple[int, int]:
        """批量添加知识条目"""
        if not self.conn and not self.create_connection():
            return 0, len(entries)
        
        success_count = 0
        failed_count = 0
        
        logger.info(f"[{self.conn_name}] 开始批量添加 {len(entries)} 条知识")
        
        for i, entry in enumerate(entries, 1):
            logger.info(f"[{self.conn_name}] 处理第 {i}/{len(entries)} 条")
            if self.add_single_entry(entry):
                success_count += 1
            else:
                failed_count += 1
        
        logger.info(f"[{self.conn_name}] 批量添加完成: 成功{success_count}, 失败{failed_count}")
        return success_count, failed_count
    
    def add_from_json_file(self, json_file_path: str) -> Tuple[int, int]:
        """从JSON文件添加知识"""
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            entries = []
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, str):
                        entries.append(KnowledgeEntry(text=item))
                    elif isinstance(item, dict):
                        entries.append(KnowledgeEntry(**item))
            elif isinstance(data, dict):
                # 单个条目
                entries.append(KnowledgeEntry(**data))
            
            return self.add_batch_entries(entries)
            
        except Exception as e:
            logger.error(f"读取JSON文件失败: {e}")
            return 0, 0
    
    def add_from_csv_file(self, csv_file_path: str, text_column: str = "text") -> Tuple[int, int]:
        """从CSV文件添加知识"""
        try:
            df = pd.read_csv(csv_file_path)
            
            if text_column not in df.columns:
                logger.error(f"CSV文件中没有找到列: {text_column}")
                return 0, 0
            
            entries = []
            for _, row in df.iterrows():
                entry_data = {"text": row[text_column]}
                
                # 尝试读取其他字段
                if "source" in df.columns:
                    entry_data["source"] = row["source"]
                if "languages" in df.columns:
                    try:
                        entry_data["languages"] = json.loads(row["languages"])
                    except:
                        entry_data["languages"] = [row["languages"]]
                
                entries.append(KnowledgeEntry(**entry_data))
            
            return self.add_batch_entries(entries)
            
        except Exception as e:
            logger.error(f"读取CSV文件失败: {e}")
            return 0, 0
    
    def add_from_markdown_file(self, md_file_path: str) -> Tuple[int, int]:
        """从Markdown文件添加知识（按段落分割）"""
        try:
            with open(md_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 按段落分割
            paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
            
            entries = []
            for para in paragraphs:
                # 跳过太短的段落
                if len(para) < 20:
                    continue
                
                entries.append(KnowledgeEntry(
                    text=para,
                    source="MarkdownImport",
                    filetype="markdown"
                ))
            
            return self.add_batch_entries(entries)
            
        except Exception as e:
            logger.error(f"读取Markdown文件失败: {e}")
            return 0, 0
    
    def close(self):
        """关闭连接"""
        if self.conn:
            try:
                self.conn.close()
                logger.info(f"[{self.conn_name}] 连接已关闭")
            except:
                pass


class BatchKnowledgeAdder:
    """批量知识添加器（多Lakehouse）"""
    
    def __init__(self, config_path: str = "~/.clickzetta/connections.json"):
        self.conn_manager = LakehouseConnectionManager(config_path)
        self.dashscope_key = self.conn_manager.get_dashscope_api_key()
        
        if not self.dashscope_key:
            logger.warning("未配置DashScope API密钥，尝试使用环境变量")
            self.dashscope_key = os.getenv("DASHSCOPE_API_KEY")
    
    def add_to_all_lakehouse(self, entries: List[KnowledgeEntry], 
                           filter_pattern: Optional[str] = None,
                           exclude_pattern: Optional[str] = None) -> List[Dict[str, Any]]:
        """添加知识到所有Lakehouse"""
        results = []
        
        # 获取活跃连接
        connections = self.conn_manager.get_active_connections(
            filter_pattern=filter_pattern,
            exclude_pattern=exclude_pattern
        )
        
        logger.info(f"准备向 {len(connections)} 个Lakehouse添加 {len(entries)} 条知识")
        
        for i, conn in enumerate(connections, 1):
            conn_name = conn.get('connection_name', 'unnamed')
            logger.info(f"\n[{i}/{len(connections)}] 处理 {conn_name}")
            
            try:
                adder = KnowledgeAdder(conn, self.dashscope_key)
                success, failed = adder.add_batch_entries(entries)
                adder.close()
                
                results.append({
                    "connection_name": conn_name,
                    "status": "success",
                    "total_entries": len(entries),
                    "success_count": success,
                    "failed_count": failed
                })
                
            except Exception as e:
                logger.error(f"[{conn_name}] 处理失败: {e}")
                results.append({
                    "connection_name": conn_name,
                    "status": "failed",
                    "error": str(e)
                })
        
        return results
    
    def print_summary(self, results: List[Dict[str, Any]]):
        """打印添加总结"""
        print("\n" + "="*60)
        print("📊 知识添加总结")
        print("="*60)
        
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
                total_entries = result.get("total_entries", 0)
                success_count = result.get("success_count", 0)
                failed_count = result.get("failed_count", 0)
                print(f"\n✅ {conn_name}")
                print(f"   总条目: {total_entries}")
                print(f"   成功: {success_count}")
                print(f"   失败: {failed_count}")
            else:
                error = result.get("error", "未知错误")
                print(f"\n❌ {conn_name}")
                print(f"   错误: {error}")
        
        print("\n" + "="*60)


def create_sample_knowledge_file():
    """创建示例知识文件"""
    sample_json = [
        {
            "text": "ClickZetta、云器、Singdata之间的关系：ClickZetta是云器、Singdata的技术品牌。云器是ClickZetta的中国商业公司，Singdata是ClickZetta的海外商业公司。",
            "source": "CompanyInfo",
            "languages": ["zh-cn"]
        },
        {
            "text": "ClickZetta Lakehouse支持的数据类型包括：INT、BIGINT、FLOAT、DOUBLE、DECIMAL、STRING、DATE、TIMESTAMP、BOOLEAN、ARRAY、MAP、STRUCT等。",
            "source": "TechnicalDoc",
            "languages": ["zh-cn"]
        },
        {
            "text": "创建向量索引的语法：CREATE INDEX index_name ON table_name(column_name) USING vector PROPERTIES ('scalar.type' = 'f32', 'distance.function' = 'cosine_distance')",
            "source": "SQLReference",
            "languages": ["zh-cn"]
        }
    ]
    
    # 保存JSON示例
    json_file = "sample_knowledge.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(sample_json, f, ensure_ascii=False, indent=2)
    
    # 保存CSV示例
    csv_file = "sample_knowledge.csv"
    df = pd.DataFrame([
        {"text": "TRUNCATE TABLE语句用于清空表中的所有数据，但保留表结构。", "source": "SQLReference"},
        {"text": "ClickZetta支持实时数据同步，可以通过RTSync功能实现。", "source": "Feature"},
        {"text": "向量搜索使用cosine_distance函数计算相似度。", "source": "VectorSearch"}
    ])
    df.to_csv(csv_file, index=False, encoding='utf-8')
    
    print(f"✅ 创建了示例文件:")
    print(f"  - {json_file}")
    print(f"  - {csv_file}")
    
    return json_file, csv_file


def main():
    """主函数 - 用于测试"""
    import argparse
    
    parser = argparse.ArgumentParser(description="向ClickZetta知识库添加自定义知识")
    parser.add_argument("--config", default="~/.clickzetta/connections.json", help="连接配置文件路径")
    parser.add_argument("--text", help="要添加的知识文本")
    parser.add_argument("--file", help="知识文件路径（支持json/csv/md）")
    parser.add_argument("--filter", help="只添加到包含此模式的连接")
    parser.add_argument("--create-sample", action="store_true", help="创建示例知识文件")
    
    args = parser.parse_args()
    
    if args.create_sample:
        create_sample_knowledge_file()
        return
    
    # 创建批量添加器
    batch_adder = BatchKnowledgeAdder(config_path=args.config)
    
    entries = []
    
    # 从命令行文本创建条目
    if args.text:
        entries.append(KnowledgeEntry(text=args.text))
    
    # 从文件读取
    elif args.file:
        if args.file.endswith('.json'):
            # 使用单个连接测试
            connections = batch_adder.conn_manager.get_active_connections()
            if connections:
                adder = KnowledgeAdder(connections[0], batch_adder.dashscope_key)
                success, failed = adder.add_from_json_file(args.file)
                print(f"添加结果: 成功{success}, 失败{failed}")
                adder.close()
        # 其他文件类型...
    
    else:
        print("请提供 --text 或 --file 参数")
        return
    
    # 批量添加
    if entries:
        results = batch_adder.add_to_all_lakehouse(entries, filter_pattern=args.filter)
        batch_adder.print_summary(results)


if __name__ == "__main__":
    main()