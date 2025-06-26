#!/usr/bin/env python3
"""
简化的知识库数据验证脚本
用于验证已部署的知识库数据质量
"""

import os
import sys
import logging
from datetime import datetime

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kb_data_validator import BatchKnowledgeBaseValidator

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def validate_all():
    """验证所有Lakehouse的知识库数据"""
    validator = BatchKnowledgeBaseValidator()
    
    # 显示所有连接
    connections = validator.connections
    if not connections:
        print("\n❌ 没有找到任何连接配置")
        return
    
    print(f"\n找到 {len(connections)} 个Lakehouse连接:")
    for i, conn in enumerate(connections, 1):
        conn_name = conn.get('connection_name', 'unnamed')
        service = conn.get('service', 'N/A')
        print(f"{i}. {conn_name} ({service})")
    
    # 确认是否继续
    print(f"\n⚠️  即将验证所有 {len(connections)} 个Lakehouse的知识库数据")
    confirm = input("是否继续？(y/n): ").strip().lower()
    
    if confirm != 'y' and confirm != 'yes':
        print("已取消验证")
        return None
    
    print("\n开始验证...")
    results = validator.validate_all_deployments()
    validator.print_validation_summary(results)
    
    # 保存结果
    output_file = validator.save_validation_results(results)
    print(f"\n详细验证结果已保存到: {output_file}")
    
    return results


def validate_specific(connection_pattern):
    """验证特定的Lakehouse"""
    validator = BatchKnowledgeBaseValidator()
    
    # 获取匹配的连接
    matched_connections = []
    for conn in validator.connections:
        conn_name = conn.get('connection_name', 'unnamed')
        if connection_pattern in conn_name:
            matched_connections.append(conn)
    
    if not matched_connections:
        print(f"\n❌ 没有找到匹配 '{connection_pattern}' 的连接")
        return None
    
    # 显示匹配结果
    print(f"\n找到 {len(matched_connections)} 个匹配的连接:")
    for i, conn in enumerate(matched_connections, 1):
        conn_name = conn.get('connection_name', 'unnamed')
        service = conn.get('service', 'N/A')
        print(f"{i}. {conn_name} ({service})")
    
    # 确认是否继续
    print(f"\n⚠️  即将验证以上 {len(matched_connections)} 个Lakehouse的知识库数据")
    confirm = input("是否继续？(y/n): ").strip().lower()
    
    if confirm != 'y' and confirm != 'yes':
        print("已取消验证")
        return None
    
    print("\n开始验证...")
    results = validator.validate_all_deployments(filter_pattern=connection_pattern)
    validator.print_validation_summary(results)
    
    # 保存结果
    output_file = validator.save_validation_results(results)
    print(f"\n详细验证结果已保存到: {output_file}")
    
    return results


def main():
    """主函数"""
    print("🔍 ClickZetta 知识库数据验证工具")
    print("="*60)
    
    print("\n请选择验证模式:")
    print("1. 验证所有Lakehouse")
    print("2. 验证特定Lakehouse")
    print("3. 快速检查（只验证第一个连接）")
    
    choice = input("\n请输入选择(1-3): ").strip()
    
    if choice == "1":
        print("\n开始验证所有Lakehouse的知识库数据...")
        validate_all()
        
    elif choice == "2":
        pattern = input("请输入连接名称匹配模式: ").strip()
        if pattern:
            print(f"\n开始验证匹配 '{pattern}' 的Lakehouse...")
            validate_specific(pattern)
        else:
            print("未输入匹配模式")
            
    elif choice == "3":
        print("\n快速检查模式：只验证第一个连接...")
        validator = BatchKnowledgeBaseValidator()
        
        # 只获取第一个连接
        if validator.connections:
            test_conn = validator.connections[0]
            logger.info(f"验证: {test_conn.get('connection_name', 'unnamed')}")
            
            from kb_data_validator import KnowledgeBaseValidator
            kb_validator = KnowledgeBaseValidator(test_conn)
            report = kb_validator.generate_validation_report()
            kb_validator.close()
            
            # 打印详细结果
            print("\n" + "="*60)
            print("验证结果详情")
            print("="*60)
            
            # 行数验证
            row_counts = report["validations"]["row_counts"]
            print(f"\n📊 行数验证:")
            print(f"   Raw表: {row_counts['raw_table_count']}")
            print(f"   Silver表: {row_counts['silver_table_count']}")
            print(f"   匹配: {'✅ 是' if row_counts['count_match'] else '❌ 否'}")
            
            # 向量质量验证
            quality = report["validations"]["embeddings_quality"]
            print(f"\n🔍 向量质量:")
            print(f"   检查样本数: {quality['total_checked']}")
            print(f"   问题向量: {quality['zero_vectors_count']} ({quality['zero_vectors_percentage']:.1f}%)")
            
            if quality['problematic_records']:
                print(f"   问题向量示例:")
                for i, record in enumerate(quality['problematic_records'][:5], 1):
                    print(f"     {i}. {record['filename']}")
                    print(f"        零值比例: {record['zero_percentage']:.1f}%")
                    print(f"        文本预览: {record['text_preview'][:50]}...")
            
            # 维度验证
            dimensions = report["validations"]["embeddings_dimensions"]
            print(f"\n📐 向量维度:")
            print(f"   期望维度: {dimensions['expected_dimensions']}")
            print(f"   维度分布: {dimensions['dimension_distribution']}")
            print(f"   维度正确: {'✅ 是' if dimensions['all_dimensions_correct'] else '❌ 否'}")
            
            # 总结
            summary = report["summary"]
            print(f"\n📋 总结:")
            print(f"   所有检查通过: {'✅ 是' if summary['all_checks_passed'] else '❌ 否'}")
            
        else:
            print("没有找到可用的连接")
    
    else:
        print("无效的选择")


if __name__ == "__main__":
    main()