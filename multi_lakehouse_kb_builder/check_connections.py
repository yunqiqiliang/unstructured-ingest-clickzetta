#!/usr/bin/env python3
"""
检查和诊断Lakehouse连接
"""

import os
import sys
import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from clickzetta.connector import connect

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from multi_lakehouse_kb_builder import LakehouseConnectionManager
from utils import display_connections, format_duration, format_number

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class ConnectionChecker:
    """连接检查器"""
    
    def __init__(self, config_path: str = "~/.clickzetta/connections.json"):
        self.conn_manager = LakehouseConnectionManager(config_path)
        self.connections = self.conn_manager.connections
        self.dashscope_key = self.conn_manager.get_dashscope_api_key()
    
    def test_connection(self, conn_config: Dict[str, Any]) -> Dict[str, Any]:
        """测试单个连接"""
        conn_name = conn_config.get('connection_name', 'unnamed')
        result = {
            "connection_name": conn_name,
            "status": "pending",
            "response_time": None,
            "error": None,
            "details": {}
        }
        
        start_time = datetime.now()
        
        try:
            # 尝试建立连接
            conn = connect(
                password=conn_config['password'],
                username=conn_config['username'],
                service=conn_config['service'],
                instance=conn_config['instance'],
                workspace=conn_config.get('workspace', 'default'),
                schema=conn_config.get('schema', 'default'),
                vcluster=conn_config.get('vcluster', 'default')
            )
            
            # 执行简单查询测试连接
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                cur.fetchone()
            
            # 获取版本信息（如果可用）
            try:
                with conn.cursor() as cur:
                    cur.execute("SELECT VERSION()")
                    version = cur.fetchone()[0]
                    result["details"]["version"] = version
            except:
                pass
            
            # 获取当前用户
            try:
                with conn.cursor() as cur:
                    cur.execute("SELECT CURRENT_USER()")
                    current_user = cur.fetchone()[0]
                    result["details"]["current_user"] = current_user
            except:
                pass
            
            conn.close()
            
            result["status"] = "success"
            
        except Exception as e:
            result["status"] = "failed"
            result["error"] = str(e)
        
        end_time = datetime.now()
        result["response_time"] = (end_time - start_time).total_seconds()
        
        return result
    
    def check_all_connections(self) -> List[Dict[str, Any]]:
        """检查所有连接"""
        results = []
        
        print(f"\n🔍 检查 {len(self.connections)} 个连接...")
        print("="*60)
        
        for i, conn in enumerate(self.connections, 1):
            conn_name = conn.get('connection_name', 'unnamed')
            print(f"\n[{i}/{len(self.connections)}] 测试 {conn_name}...", end='', flush=True)
            
            result = self.test_connection(conn)
            results.append(result)
            
            if result["status"] == "success":
                print(f" ✅ 成功 ({result['response_time']:.2f}秒)")
            else:
                print(f" ❌ 失败")
                if result["error"]:
                    print(f"     错误: {result['error']}")
        
        return results
    
    def print_summary(self, results: List[Dict[str, Any]]):
        """打印检查总结"""
        print("\n" + "="*60)
        print("📊 连接检查总结")
        print("="*60)
        
        success_count = sum(1 for r in results if r["status"] == "success")
        failed_count = len(results) - success_count
        
        print(f"总连接数: {len(results)}")
        print(f"成功: {success_count}")
        print(f"失败: {failed_count}")
        
        # 显示成功的连接
        if success_count > 0:
            print("\n✅ 可用连接:")
            for result in results:
                if result["status"] == "success":
                    print(f"  - {result['connection_name']} (响应时间: {result['response_time']:.2f}秒)")
                    if result["details"]:
                        for key, value in result["details"].items():
                            print(f"    {key}: {value}")
        
        # 显示失败的连接
        if failed_count > 0:
            print("\n❌ 不可用连接:")
            for result in results:
                if result["status"] == "failed":
                    print(f"  - {result['connection_name']}")
                    if result["error"]:
                        print(f"    错误: {result['error']}")
        
        # 显示配置信息
        print(f"\n🔑 DashScope API密钥: {'已配置' if self.dashscope_key else '未配置'}")
        if self.dashscope_key:
            masked_key = self.dashscope_key[:10] + "..." if len(self.dashscope_key) > 10 else "***"
            print(f"   密钥: {masked_key}")


class KnowledgeBaseHealthChecker:
    """知识库健康检查器"""
    
    def __init__(self, config_path: str = "~/.clickzetta/connections.json"):
        self.conn_manager = LakehouseConnectionManager(config_path)
        self.connections = self.conn_manager.connections
        self.schema_name = "clickzetta_doc_kb"
        self.raw_table_name = "dashscope_v4_1024_2048_20250611_yunqi_raw_elements"
        self.silver_table_name = "dashscope_v4_1024_2048_20250611_yunqi_elements"
    
    def check_kb_health(self, conn_config: Dict[str, Any]) -> Dict[str, Any]:
        """检查单个Lakehouse的知识库健康状态"""
        conn_name = conn_config.get('connection_name', 'unnamed')
        result = {
            "connection_name": conn_name,
            "status": "pending",
            "has_schema": False,
            "has_raw_table": False,
            "has_silver_table": False,
            "raw_table_stats": {},
            "silver_table_stats": {},
            "vector_stats": {},
            "health_score": 0,
            "issues": [],
            "error": None
        }
        
        try:
            # 创建连接
            conn = connect(
                password=conn_config['password'],
                username=conn_config['username'],
                service=conn_config['service'],
                instance=conn_config['instance'],
                workspace=conn_config.get('workspace', 'default'),
                schema=conn_config.get('schema', 'default'),
                vcluster=conn_config.get('vcluster', 'default')
            )
            
            with conn.cursor() as cur:
                # 检查schema是否存在
                cur.execute(f"SHOW SCHEMAS LIKE '{self.schema_name}'")
                if cur.fetchall():
                    result["has_schema"] = True
                else:
                    result["issues"].append("知识库Schema不存在")
                    result["status"] = "no_kb"
                    conn.close()
                    return result
                
                # 切换到知识库schema
                cur.execute(f"USE {self.schema_name}")
                
                # 检查Raw表
                cur.execute(f"SHOW TABLES LIKE '{self.raw_table_name}'")
                if cur.fetchall():
                    result["has_raw_table"] = True
                    
                    # 获取Raw表统计
                    cur.execute(f"SELECT COUNT(*) FROM {self.raw_table_name}")
                    result["raw_table_stats"]["row_count"] = cur.fetchone()[0]
                    
                    if result["raw_table_stats"]["row_count"] == 0:
                        result["issues"].append("Raw表为空")
                
                # 检查Silver表
                cur.execute(f"SHOW TABLES LIKE '{self.silver_table_name}'")
                if cur.fetchall():
                    result["has_silver_table"] = True
                    
                    # 获取Silver表统计
                    cur.execute(f"SELECT COUNT(*) FROM {self.silver_table_name}")
                    result["silver_table_stats"]["row_count"] = cur.fetchone()[0]
                    
                    if result["silver_table_stats"]["row_count"] == 0:
                        result["issues"].append("Silver表为空")
                    else:
                        # 获取向量统计
                        cur.execute(f"""
                            SELECT 
                                COUNT(*) as total_vectors,
                                COUNT(CASE WHEN embeddings IS NOT NULL THEN 1 END) as non_null_vectors
                            FROM {self.silver_table_name}
                        """)
                        stats = cur.fetchone()
                        result["vector_stats"]["total_records"] = stats[0]
                        result["vector_stats"]["records_with_vectors"] = stats[1]
                        result["vector_stats"]["missing_vectors"] = stats[0] - stats[1]
                        
                        # 采样检查向量质量
                        try:
                            cur.execute(f"""
                                SELECT embeddings 
                                FROM {self.silver_table_name} 
                                WHERE embeddings IS NOT NULL 
                                LIMIT 100
                            """)
                            
                            sample_vectors = cur.fetchall()
                            if sample_vectors:
                                # 分析向量维度
                                dimensions = []
                                zero_vectors = 0
                                
                                for (embedding,) in sample_vectors:
                                    try:
                                        if isinstance(embedding, str):
                                            vec_list = json.loads(embedding)
                                        else:
                                            vec_list = list(embedding)
                                        
                                        dimensions.append(len(vec_list))
                                        
                                        # 检查是否为零向量
                                        zero_count = sum(1 for x in vec_list if float(x) == 0.0)
                                        if zero_count > len(vec_list) * 0.5:
                                            zero_vectors += 1
                                    except:
                                        continue
                                
                                if dimensions:
                                    result["vector_stats"]["sample_dimensions"] = list(set(dimensions))
                                    result["vector_stats"]["dimension_consistent"] = len(set(dimensions)) == 1
                                    result["vector_stats"]["zero_vectors_in_sample"] = zero_vectors
                                    
                                    if not result["vector_stats"]["dimension_consistent"]:
                                        result["issues"].append("向量维度不一致")
                                    if zero_vectors > 0:
                                        result["issues"].append(f"发现{zero_vectors}个问题向量")
                        except:
                            pass
                
                # 检查索引
                try:
                    cur.execute(f"SHOW INDEX FROM {self.silver_table_name}")
                    indexes = cur.fetchall()
                    result["silver_table_stats"]["index_count"] = len(indexes)
                    result["silver_table_stats"]["indexes"] = [idx[0] for idx in indexes]
                except:
                    pass
            
            conn.close()
            
            # 计算健康分数
            health_score = 100
            if not result["has_schema"]:
                health_score = 0
            else:
                if not result["has_raw_table"]:
                    health_score -= 20
                if not result["has_silver_table"]:
                    health_score -= 20
                if result["raw_table_stats"].get("row_count", 0) == 0:
                    health_score -= 20
                if result["silver_table_stats"].get("row_count", 0) == 0:
                    health_score -= 20
                if result["vector_stats"].get("missing_vectors", 0) > 0:
                    health_score -= 10
                if not result["vector_stats"].get("dimension_consistent", True):
                    health_score -= 10
                if result["vector_stats"].get("zero_vectors_in_sample", 0) > 0:
                    health_score -= 10
            
            result["health_score"] = max(0, health_score)
            result["status"] = "success"
            
        except Exception as e:
            result["status"] = "error"
            result["error"] = str(e)
            result["health_score"] = 0
        
        return result
    
    def check_all_kb_health(self) -> List[Dict[str, Any]]:
        """检查所有连接的知识库健康状态"""
        results = []
        
        print(f"\n🏥 检查 {len(self.connections)} 个Lakehouse的知识库健康状态...")
        print("="*60)
        
        for i, conn in enumerate(self.connections, 1):
            conn_name = conn.get('connection_name', 'unnamed')
            print(f"\n[{i}/{len(self.connections)}] 检查 {conn_name}...", end='', flush=True)
            
            result = self.check_kb_health(conn)
            results.append(result)
            
            if result["status"] == "success":
                score = result["health_score"]
                if score >= 80:
                    print(f" ✅ 健康 (分数: {score})")
                elif score >= 60:
                    print(f" ⚠️  一般 (分数: {score})")
                else:
                    print(f" ❌ 不健康 (分数: {score})")
            elif result["status"] == "no_kb":
                print(f" ⚪ 无知识库")
            else:
                print(f" ❌ 检查失败")
        
        return results
    
    def print_health_summary(self, results: List[Dict[str, Any]]):
        """打印健康检查总结"""
        print("\n" + "="*60)
        print("📊 知识库健康状态总结")
        print("="*60)
        
        # 统计
        total = len(results)
        healthy = sum(1 for r in results if r["health_score"] >= 80)
        warning = sum(1 for r in results if 60 <= r["health_score"] < 80)
        unhealthy = sum(1 for r in results if 0 < r["health_score"] < 60)
        no_kb = sum(1 for r in results if r["status"] == "no_kb")
        error = sum(1 for r in results if r["status"] == "error")
        
        print(f"总计: {total} 个Lakehouse")
        print(f"✅ 健康: {healthy}")
        print(f"⚠️  一般: {warning}")
        print(f"❌ 不健康: {unhealthy}")
        print(f"⚪ 无知识库: {no_kb}")
        print(f"❌ 检查失败: {error}")
        
        # 详细信息
        for result in results:
            print(f"\n{'='*40}")
            print(f"📌 {result['connection_name']}")
            
            if result["status"] == "no_kb":
                print("   状态: 未部署知识库")
                continue
            elif result["status"] == "error":
                print(f"   状态: 检查失败 - {result['error']}")
                continue
            
            score = result["health_score"]
            if score >= 80:
                status = "✅ 健康"
            elif score >= 60:
                status = "⚠️  一般"
            else:
                status = "❌ 不健康"
            
            print(f"   状态: {status} (健康分数: {score}/100)")
            
            # 基础信息
            print(f"   Schema: {'✅' if result['has_schema'] else '❌'} {self.schema_name}")
            print(f"   Raw表: {'✅' if result['has_raw_table'] else '❌'} " + 
                  (f"{format_number(result['raw_table_stats'].get('row_count', 0))} 行" if result['has_raw_table'] else ""))
            print(f"   Silver表: {'✅' if result['has_silver_table'] else '❌'} " +
                  (f"{format_number(result['silver_table_stats'].get('row_count', 0))} 行" if result['has_silver_table'] else ""))
            
            # 向量信息
            if result["vector_stats"]:
                vs = result["vector_stats"]
                print(f"   向量统计:")
                print(f"     - 总记录: {format_number(vs.get('total_records', 0))}")
                print(f"     - 有向量: {format_number(vs.get('records_with_vectors', 0))}")
                if vs.get('missing_vectors', 0) > 0:
                    print(f"     - 缺失向量: {vs.get('missing_vectors', 0)}")
                if 'sample_dimensions' in vs:
                    print(f"     - 向量维度: {vs['sample_dimensions']}")
                if 'zero_vectors_in_sample' in vs and vs['zero_vectors_in_sample'] > 0:
                    print(f"     - 问题向量: {vs['zero_vectors_in_sample']} (采样100个)")
            
            # 索引信息
            if result["silver_table_stats"].get("indexes"):
                print(f"   索引: {len(result['silver_table_stats']['indexes'])} 个")
                for idx in result["silver_table_stats"]["indexes"]:
                    print(f"     - {idx}")
            
            # 问题列表
            if result["issues"]:
                print(f"   发现的问题:")
                for issue in result["issues"]:
                    print(f"     ⚠️  {issue}")
        
        print("\n" + "="*60)


def main():
    """主函数"""
    print("🔍 ClickZetta 连接和知识库检查工具")
    print("="*50)
    
    print("\n请选择检查类型:")
    print("1. 检查所有连接的可用性")
    print("2. 检查所有知识库的健康状态")
    print("3. 两项都检查")
    
    choice = input("\n请输入选择(1-3): ").strip()
    
    if choice in ["1", "3"]:
        print("\n" + "="*60)
        print("🔌 连接可用性检查")
        print("="*60)
        
        checker = ConnectionChecker()
        results = checker.check_all_connections()
        checker.print_summary(results)
        
        # 保存结果
        # 确保reports目录存在
        reports_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")
        os.makedirs(reports_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(reports_dir, f"connection_check_{timestamp}.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                "check_time": datetime.now().isoformat(),
                "check_type": "connection_availability",
                "results": results
            }, f, ensure_ascii=False, indent=2)
        print(f"\n检查结果已保存到: {output_file}")
    
    if choice in ["2", "3"]:
        print("\n" + "="*60)
        print("🏥 知识库健康状态检查")
        print("="*60)
        
        kb_checker = KnowledgeBaseHealthChecker()
        kb_results = kb_checker.check_all_kb_health()
        kb_checker.print_health_summary(kb_results)
        
        # 保存结果
        # 确保reports目录存在
        reports_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")
        os.makedirs(reports_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(reports_dir, f"kb_health_check_{timestamp}.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                "check_time": datetime.now().isoformat(),
                "check_type": "knowledge_base_health",
                "results": kb_results
            }, f, ensure_ascii=False, indent=2)
        print(f"\n健康检查结果已保存到: {output_file}")


if __name__ == "__main__":
    main()