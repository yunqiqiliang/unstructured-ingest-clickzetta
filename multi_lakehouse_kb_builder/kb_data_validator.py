#!/usr/bin/env python3
"""
知识库数据验证器
验证部署后的数据质量，包括：
1. 数据行数验证
2. 嵌入向量质量检查（全零向量检测）
3. 向量维度一致性验证
"""

import os
import sys
import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from clickzetta.connector import connect

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class KnowledgeBaseValidator:
    """知识库数据验证器"""
    
    def __init__(self, connection_params: Dict[str, Any]):
        self.connection_params = connection_params
        self.conn_name = connection_params.get('connection_name', 'unnamed')
        self.schema_name = "clickzetta_doc_kb"
        self.raw_table_name = "dashscope_v4_1024_2048_20250611_yunqi_raw_elements"
        self.silver_table_name = "dashscope_v4_1024_2048_20250611_yunqi_elements"
        self.expected_dimensions = 1024
        self.conn = None
        
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
    
    def execute_sql(self, sql: str) -> List[Any]:
        """执行SQL查询"""
        if not self.conn:
            if not self.create_connection():
                raise Exception("无法创建数据库连接")
        
        try:
            with self.conn.cursor() as cur:
                cur.execute(sql)
                return cur.fetchall()
        except Exception as e:
            logger.error(f"[{self.conn_name}] SQL执行失败: {e}")
            raise
    
    def validate_row_counts(self) -> Dict[str, Any]:
        """验证表的行数"""
        logger.info(f"[{self.conn_name}] 开始验证数据行数...")
        
        result = {
            "raw_table_count": 0,
            "silver_table_count": 0,
            "count_match": False,
            "status": "pending"
        }
        
        try:
            # 获取Raw表行数
            raw_count_sql = f"SELECT COUNT(*) FROM {self.schema_name}.{self.raw_table_name}"
            raw_count = self.execute_sql(raw_count_sql)[0][0]
            result["raw_table_count"] = raw_count
            
            # 获取Silver表行数
            silver_count_sql = f"SELECT COUNT(*) FROM {self.schema_name}.{self.silver_table_name}"
            silver_count = self.execute_sql(silver_count_sql)[0][0]
            result["silver_table_count"] = silver_count
            
            # 检查行数是否匹配
            result["count_match"] = (raw_count == silver_count)
            result["status"] = "success"
            
            if result["count_match"]:
                logger.info(f"[{self.conn_name}] ✅ 行数验证通过: Raw表={raw_count}, Silver表={silver_count}")
            else:
                logger.warning(f"[{self.conn_name}] ⚠️  行数不匹配: Raw表={raw_count}, Silver表={silver_count}")
                
        except Exception as e:
            result["status"] = "failed"
            result["error"] = str(e)
            logger.error(f"[{self.conn_name}] 行数验证失败: {e}")
        
        return result
    
    def validate_embeddings_quality(self, sample_size: int = 1000) -> Dict[str, Any]:
        """验证嵌入向量质量（检测全零向量）"""
        logger.info(f"[{self.conn_name}] 开始验证嵌入向量质量...")
        
        result = {
            "total_checked": 0,
            "zero_vectors_count": 0,
            "zero_vectors_percentage": 0.0,
            "problematic_records": [],
            "status": "pending"
        }
        
        try:
            # 从Silver表中随机采样
            sample_sql = f"""
            SELECT 
                id,
                filename,
                LEFT(text, 100) as text_preview,
                embeddings,
                LENGTH(CAST(embeddings AS STRING)) as embedding_str_length
            FROM {self.schema_name}.{self.silver_table_name}
            WHERE embeddings IS NOT NULL
            ORDER BY RAND()
            LIMIT {sample_size}
            """
            
            records = self.execute_sql(sample_sql)
            result["total_checked"] = len(records)
            
            if result["total_checked"] == 0:
                logger.warning(f"[{self.conn_name}] 没有找到包含embeddings的记录")
                result["status"] = "no_data"
                return result
            
            # 分析每条记录
            for record in records:
                record_id, filename, text_preview, embeddings, _ = record
                
                try:
                    # 将embeddings转换为数组
                    if isinstance(embeddings, str):
                        # 如果是字符串格式，解析为列表
                        embedding_list = json.loads(embeddings)
                    else:
                        # 直接转换为列表
                        embedding_list = list(embeddings)
                    
                    # 转换为numpy数组进行分析
                    embedding_array = np.array(embedding_list, dtype=float)
                    
                    # 检查是否为全零向量
                    zero_count = np.sum(embedding_array == 0.0)
                    total_dims = len(embedding_array)
                    zero_percentage = (zero_count / total_dims) * 100
                    
                    # 如果超过50%为零，认为是问题向量
                    if zero_percentage > 50:
                        result["zero_vectors_count"] += 1
                        
                        # 记录问题向量的详细信息
                        problem_info = {
                            "id": record_id,
                            "filename": filename,
                            "text_preview": text_preview,
                            "zero_percentage": zero_percentage,
                            "zero_count": zero_count,
                            "total_dims": total_dims,
                            "non_zero_mean": float(np.mean(embedding_array[embedding_array != 0.0])) if zero_count < total_dims else 0.0,
                            "non_zero_std": float(np.std(embedding_array[embedding_array != 0.0])) if zero_count < total_dims else 0.0
                        }
                        
                        # 只记录前10个问题向量的详细信息
                        if len(result["problematic_records"]) < 10:
                            result["problematic_records"].append(problem_info)
                            
                except Exception as e:
                    logger.error(f"[{self.conn_name}] 分析记录 {record_id} 失败: {e}")
            
            # 计算问题向量百分比
            if result["total_checked"] > 0:
                result["zero_vectors_percentage"] = (result["zero_vectors_count"] / result["total_checked"]) * 100
            
            result["status"] = "success"
            
            # 输出结果
            if result["zero_vectors_count"] > 0:
                logger.warning(f"[{self.conn_name}] ⚠️  发现 {result['zero_vectors_count']} 个问题向量 ({result['zero_vectors_percentage']:.1f}%)")
            else:
                logger.info(f"[{self.conn_name}] ✅ 所有向量质量正常")
                
        except Exception as e:
            result["status"] = "failed"
            result["error"] = str(e)
            logger.error(f"[{self.conn_name}] 向量质量验证失败: {e}")
        
        return result
    
    def validate_embeddings_dimensions(self, sample_size: int = 500) -> Dict[str, Any]:
        """验证嵌入向量维度一致性"""
        logger.info(f"[{self.conn_name}] 开始验证嵌入向量维度...")
        
        result = {
            "expected_dimensions": self.expected_dimensions,
            "dimension_distribution": {},
            "all_dimensions_correct": True,
            "incorrect_dimensions_count": 0,
            "status": "pending"
        }
        
        try:
            # 直接采样并手动计算维度（不使用VECTOR_DIM函数）
            sample_sql = f"""
            SELECT 
                id,
                filename,
                embeddings
            FROM {self.schema_name}.{self.silver_table_name}
            WHERE embeddings IS NOT NULL
            ORDER BY RAND()
            LIMIT {sample_size}
            """
            
            raw_records = self.execute_sql(sample_sql)
            records = []
            
            for record in raw_records:
                record_id, filename, embeddings = record
                
                try:
                    # 计算维度
                    if isinstance(embeddings, str):
                        embedding_list = json.loads(embeddings)
                    else:
                        embedding_list = list(embeddings)
                    
                    dimension = len(embedding_list)
                    records.append((record_id, filename, embeddings, dimension))
                except:
                    continue
            
            # 统计维度分布
            for record in records:
                if len(record) >= 4:
                    dimension = record[3]
                    if dimension in result["dimension_distribution"]:
                        result["dimension_distribution"][dimension] += 1
                    else:
                        result["dimension_distribution"][dimension] = 1
                    
                    if dimension != self.expected_dimensions:
                        result["incorrect_dimensions_count"] += 1
            
            # 检查是否所有维度都正确
            result["all_dimensions_correct"] = (result["incorrect_dimensions_count"] == 0)
            result["status"] = "success"
            
            # 输出结果
            if result["all_dimensions_correct"]:
                logger.info(f"[{self.conn_name}] ✅ 所有向量维度正确 ({self.expected_dimensions}维)")
            else:
                logger.warning(f"[{self.conn_name}] ⚠️  发现 {result['incorrect_dimensions_count']} 个维度不正确的向量")
                logger.warning(f"[{self.conn_name}] 维度分布: {result['dimension_distribution']}")
                
        except Exception as e:
            result["status"] = "failed"
            result["error"] = str(e)
            logger.error(f"[{self.conn_name}] 维度验证失败: {e}")
        
        return result
    
    def generate_validation_report(self) -> Dict[str, Any]:
        """生成完整的验证报告"""
        logger.info(f"[{self.conn_name}] 开始生成验证报告...")
        
        report = {
            "connection_name": self.conn_name,
            "validation_time": datetime.now().isoformat(),
            "validations": {}
        }
        
        # 1. 验证行数
        report["validations"]["row_counts"] = self.validate_row_counts()
        
        # 2. 验证向量质量
        report["validations"]["embeddings_quality"] = self.validate_embeddings_quality()
        
        # 3. 验证向量维度
        report["validations"]["embeddings_dimensions"] = self.validate_embeddings_dimensions()
        
        # 生成总体状态
        all_success = all(
            v.get("status") == "success" 
            for v in report["validations"].values()
        )
        
        report["overall_status"] = "success" if all_success else "has_issues"
        
        # 生成总结
        report["summary"] = self._generate_summary(report["validations"])
        
        return report
    
    def _generate_summary(self, validations: Dict[str, Any]) -> Dict[str, Any]:
        """生成验证总结"""
        summary = {
            "total_rows": validations["row_counts"].get("silver_table_count", 0),
            "row_count_match": validations["row_counts"].get("count_match", False),
            "zero_vectors_found": validations["embeddings_quality"].get("zero_vectors_count", 0),
            "zero_vectors_percentage": validations["embeddings_quality"].get("zero_vectors_percentage", 0.0),
            "dimension_issues": validations["embeddings_dimensions"].get("incorrect_dimensions_count", 0),
            "all_checks_passed": True
        }
        
        # 检查是否所有检查都通过
        if not summary["row_count_match"]:
            summary["all_checks_passed"] = False
        if summary["zero_vectors_found"] > 0:
            summary["all_checks_passed"] = False
        if summary["dimension_issues"] > 0:
            summary["all_checks_passed"] = False
        
        return summary
    
    def close(self):
        """关闭连接"""
        if self.conn:
            try:
                self.conn.close()
                logger.info(f"[{self.conn_name}] 连接已关闭")
            except:
                pass


class BatchKnowledgeBaseValidator:
    """批量知识库验证器"""
    
    def __init__(self, config_path: str = "~/.clickzetta/connections.json"):
        self.config_path = os.path.expanduser(config_path)
        self.connections = self._load_connections()
        
    def _load_connections(self) -> List[Dict[str, Any]]:
        """加载连接配置"""
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            return config.get('connections', [])
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            return []
    
    def validate_all_deployments(self, 
                               filter_pattern: Optional[str] = None,
                               exclude_pattern: Optional[str] = None) -> List[Dict[str, Any]]:
        """验证所有部署的知识库"""
        results = []
        
        # 筛选连接
        active_connections = []
        for conn in self.connections:
            conn_name = conn.get('connection_name', 'unnamed')
            
            if filter_pattern and filter_pattern not in conn_name:
                continue
            if exclude_pattern and exclude_pattern in conn_name:
                continue
            
            active_connections.append(conn)
        
        logger.info(f"准备验证 {len(active_connections)} 个Lakehouse的知识库数据")
        
        # 逐个验证
        for i, conn in enumerate(active_connections, 1):
            conn_name = conn.get('connection_name', 'unnamed')
            logger.info(f"\n{'='*60}")
            logger.info(f"验证进度: {i}/{len(active_connections)} - {conn_name}")
            logger.info(f"{'='*60}")
            
            try:
                validator = KnowledgeBaseValidator(conn)
                report = validator.generate_validation_report()
                validator.close()
                
                results.append(report)
                
                # 输出简要结果
                summary = report.get("summary", {})
                if summary.get("all_checks_passed"):
                    logger.info(f"✅ {conn_name} 所有验证通过")
                else:
                    logger.warning(f"⚠️  {conn_name} 发现问题")
                    
            except Exception as e:
                logger.error(f"❌ {conn_name} 验证失败: {e}")
                results.append({
                    "connection_name": conn_name,
                    "overall_status": "failed",
                    "error": str(e)
                })
        
        return results
    
    def print_validation_summary(self, results: List[Dict[str, Any]]):
        """打印验证总结"""
        print("\n" + "="*80)
        print("📊 知识库数据验证总结")
        print("="*80)
        
        total = len(results)
        success = 0
        has_issues = 0
        failed = 0
        
        for result in results:
            status = result.get("overall_status")
            if status == "success":
                success += 1
            elif status == "has_issues":
                has_issues += 1
            else:
                failed += 1
        
        print(f"总计验证: {total} 个Lakehouse")
        print(f"完全通过: {success} 个")
        print(f"存在问题: {has_issues} 个")
        print(f"验证失败: {failed} 个")
        
        print("\n详细结果:")
        for result in results:
            conn_name = result.get("connection_name", "unnamed")
            status = result.get("overall_status")
            
            if status == "failed":
                print(f"\n❌ {conn_name} - 验证失败")
                print(f"   错误: {result.get('error', '未知错误')}")
                continue
            
            summary = result.get("summary", {})
            print(f"\n{'✅' if status == 'success' else '⚠️ '} {conn_name}")
            print(f"   总行数: {summary.get('total_rows', 0)}")
            print(f"   行数匹配: {'是' if summary.get('row_count_match') else '否'}")
            print(f"   问题向量: {summary.get('zero_vectors_found', 0)} ({summary.get('zero_vectors_percentage', 0):.1f}%)")
            print(f"   维度问题: {summary.get('dimension_issues', 0)}")
            
            # 如果有问题向量，显示示例
            quality = result.get("validations", {}).get("embeddings_quality", {})
            problematic = quality.get("problematic_records", [])
            if problematic:
                print(f"   问题向量示例:")
                for i, record in enumerate(problematic[:3], 1):
                    print(f"     {i}. {record['filename']} - {record['zero_percentage']:.1f}% 零值")
        
        print("\n" + "="*80)
    
    def save_validation_results(self, results: List[Dict[str, Any]], output_file: Optional[str] = None):
        """保存验证结果到文件"""
        # 确保reports目录存在
        reports_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")
        os.makedirs(reports_dir, exist_ok=True)
        
        if not output_file:
            output_file = os.path.join(reports_dir, f"kb_validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        elif not os.path.isabs(output_file):
            # 如果不是绝对路径，放到reports目录下
            output_file = os.path.join(reports_dir, output_file)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                "validation_time": datetime.now().isoformat(),
                "total_validations": len(results),
                "results": results
            }, f, ensure_ascii=False, indent=2)
        
        logger.info(f"验证结果已保存到: {output_file}")
        return output_file


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="验证ClickZetta知识库数据质量")
    parser.add_argument("--config", default="~/.clickzetta/connections.json", help="连接配置文件路径")
    parser.add_argument("--filter", help="只验证包含此模式的连接")
    parser.add_argument("--exclude", help="排除包含此模式的连接")
    parser.add_argument("--output", help="输出文件路径")
    
    args = parser.parse_args()
    
    # 创建验证器
    validator = BatchKnowledgeBaseValidator(config_path=args.config)
    
    # 执行验证
    logger.info("开始批量验证知识库数据...")
    start_time = datetime.now()
    
    results = validator.validate_all_deployments(
        filter_pattern=args.filter,
        exclude_pattern=args.exclude
    )
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    # 打印总结
    validator.print_validation_summary(results)
    print(f"\n验证总耗时: {duration:.2f}秒")
    
    # 保存结果
    output_file = validator.save_validation_results(results, args.output)
    
    # 返回状态码
    all_success = all(r.get("overall_status") == "success" for r in results)
    return 0 if all_success else 1


if __name__ == "__main__":
    sys.exit(main())