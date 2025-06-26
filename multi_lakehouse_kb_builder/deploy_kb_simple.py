#!/usr/bin/env python3
"""
简化的知识库部署脚本
快速部署知识库到所有或指定的Lakehouse
"""

import os
import sys
import logging
from datetime import datetime

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from multi_lakehouse_kb_builder import BatchKnowledgeBaseDeployer

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def deploy_to_all():
    """部署到所有Lakehouse"""
    deployer = BatchKnowledgeBaseDeployer(
        config_path="~/.clickzetta/connections.json",
        doc_path="/Users/liangmo/yunqidoc/cn_markdown_20250526",
        execution_mode="serial"  # 串行执行更稳定
    )
    
    results = deployer.deploy_to_all_lakehouse()
    deployer.print_summary(results)
    
    return results


def deploy_to_specific(connection_pattern):
    """部署到特定的Lakehouse"""
    deployer = BatchKnowledgeBaseDeployer(
        config_path="~/.clickzetta/connections.json",
        doc_path="/Users/liangmo/yunqidoc/cn_markdown_20250526",
        execution_mode="serial"
    )
    
    results = deployer.deploy_to_all_lakehouse(filter_pattern=connection_pattern)
    deployer.print_summary(results)
    
    return results


def deploy_parallel(max_workers=3):
    """并行部署到所有Lakehouse"""
    deployer = BatchKnowledgeBaseDeployer(
        config_path="~/.clickzetta/connections.json",
        doc_path="/Users/liangmo/yunqidoc/cn_markdown_20250526",
        execution_mode="parallel"
    )
    
    results = deployer.deploy_to_all_lakehouse(max_workers=max_workers)
    deployer.print_summary(results)
    
    return results


def main():
    """主函数"""
    print("🚀 ClickZetta 知识库批量部署工具")
    print("="*60)
    
    # 检查环境变量
    if not os.getenv("DASHSCOPE_API_KEY"):
        logger.warning("未设置DASHSCOPE_API_KEY环境变量，使用默认API密钥")
    
    # 选择部署模式
    print("\n请选择操作:")
    print("部署操作:")
    print("  1. 部署到所有Lakehouse（串行）")
    print("  2. 部署到所有Lakehouse（并行）")
    print("  3. 部署到特定Lakehouse")
    print("  4. 测试部署（只部署到第一个连接）")
    print("")
    print("检查操作:")
    print("  5. 检查所有连接的可用性")
    print("  6. 检查知识库健康状态")
    print("  7. 执行完整诊断（连接+健康）")
    print("")
    print("知识库管理:")
    print("  8. 管理知识库内容（增/删/查）")
    
    choice = input("\n请输入选择(1-8): ").strip()
    
    if choice == "1":
        # 先显示所有连接
        deployer = BatchKnowledgeBaseDeployer(
            config_path="~/.clickzetta/connections.json",
            doc_path="/Users/liangmo/yunqidoc/cn_markdown_20250526",
            execution_mode="serial"
        )
        
        connections = deployer.conn_manager.connections
        if not connections:
            print("\n❌ 没有找到任何连接配置")
            return
        
        print(f"\n找到 {len(connections)} 个Lakehouse连接:")
        for i, conn in enumerate(connections, 1):
            conn_name = conn.get('connection_name', 'unnamed')
            service = conn.get('service', 'N/A')
            print(f"{i}. {conn_name} ({service})")
        
        # 确认是否继续
        print(f"\n⚠️  即将串行部署到所有 {len(connections)} 个Lakehouse")
        confirm = input("是否继续？(y/n): ").strip().lower()
        
        if confirm == 'y' or confirm == 'yes':
            print("\n开始串行部署到所有Lakehouse...")
            deploy_to_all()
        else:
            print("已取消部署")
        
    elif choice == "2":
        # 先显示所有连接
        deployer = BatchKnowledgeBaseDeployer(
            config_path="~/.clickzetta/connections.json",
            doc_path="/Users/liangmo/yunqidoc/cn_markdown_20250526",
            execution_mode="parallel"
        )
        
        connections = deployer.conn_manager.connections
        if not connections:
            print("\n❌ 没有找到任何连接配置")
            return
        
        print(f"\n找到 {len(connections)} 个Lakehouse连接:")
        for i, conn in enumerate(connections, 1):
            conn_name = conn.get('connection_name', 'unnamed')
            service = conn.get('service', 'N/A')
            print(f"{i}. {conn_name} ({service})")
        
        workers = input("\n请输入并行工作线程数(默认3): ").strip()
        workers = int(workers) if workers else 3
        
        # 确认是否继续
        print(f"\n⚠️  即将并行部署到所有 {len(connections)} 个Lakehouse（使用 {workers} 个线程）")
        confirm = input("是否继续？(y/n): ").strip().lower()
        
        if confirm == 'y' or confirm == 'yes':
            print(f"\n开始并行部署到所有Lakehouse（{workers}个线程）...")
            deploy_parallel(workers)
        else:
            print("已取消部署")
        
    elif choice == "3":
        pattern = input("请输入连接名称匹配模式: ").strip()
        if pattern:
            # 先显示匹配的连接
            deployer = BatchKnowledgeBaseDeployer(
                config_path="~/.clickzetta/connections.json",
                doc_path="/Users/liangmo/yunqidoc/cn_markdown_20250526",
                execution_mode="serial"
            )
            
            # 获取匹配的连接
            matched_connections = []
            for conn in deployer.conn_manager.connections:
                conn_name = conn.get('connection_name', 'unnamed')
                if pattern in conn_name:
                    matched_connections.append(conn)
            
            if not matched_connections:
                print(f"\n❌ 没有找到匹配 '{pattern}' 的连接")
                return
            
            # 显示匹配结果
            print(f"\n找到 {len(matched_connections)} 个匹配的连接:")
            for i, conn in enumerate(matched_connections, 1):
                conn_name = conn.get('connection_name', 'unnamed')
                service = conn.get('service', 'N/A')
                instance = conn.get('instance', 'N/A')
                print(f"{i}. {conn_name}")
                print(f"   服务: {service}")
                print(f"   实例: {instance}")
            
            # 确认是否继续
            print(f"\n⚠️  即将部署到以上 {len(matched_connections)} 个Lakehouse")
            confirm = input("是否继续？(y/n): ").strip().lower()
            
            if confirm == 'y' or confirm == 'yes':
                print(f"\n开始部署到匹配 '{pattern}' 的Lakehouse...")
                deploy_to_specific(pattern)
            else:
                print("已取消部署")
        else:
            print("未输入匹配模式")
            
    elif choice == "4":
        print("\n测试模式：只部署到第一个连接...")
        deployer = BatchKnowledgeBaseDeployer(
            config_path="~/.clickzetta/connections.json",
            doc_path="/Users/liangmo/yunqidoc/cn_markdown_20250526",
            execution_mode="serial"
        )
        
        # 只获取第一个连接
        connections = deployer.conn_manager.get_active_connections()
        if connections:
            test_conn = connections[0]
            logger.info(f"测试部署到: {test_conn.get('connection_name', 'unnamed')}")
            result = deployer._deploy_to_single_lakehouse(test_conn)
            deployer.print_summary([result])
        else:
            logger.error("没有找到可用的连接")
    
    elif choice == "5":
        print("\n检查所有连接的可用性...")
        from check_connections import ConnectionChecker
        
        checker = ConnectionChecker()
        results = checker.check_all_connections()
        checker.print_summary(results)
        
        # 询问是否保存结果
        save = input("\n是否保存检查结果？(y/n): ").strip().lower()
        if save in ['y', 'yes']:
            # 确保reports目录存在
            reports_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")
            os.makedirs(reports_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(reports_dir, f"connection_check_{timestamp}.json")
            import json
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "check_time": datetime.now().isoformat(),
                    "check_type": "connection_availability",
                    "results": results
                }, f, ensure_ascii=False, indent=2)
            print(f"结果已保存到: {output_file}")
    
    elif choice == "6":
        print("\n检查知识库健康状态...")
        from check_connections import KnowledgeBaseHealthChecker
        
        kb_checker = KnowledgeBaseHealthChecker()
        kb_results = kb_checker.check_all_kb_health()
        kb_checker.print_health_summary(kb_results)
        
        # 询问是否保存结果
        save = input("\n是否保存健康检查结果？(y/n): ").strip().lower()
        if save in ['y', 'yes']:
            # 确保reports目录存在
            reports_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")
            os.makedirs(reports_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(reports_dir, f"kb_health_check_{timestamp}.json")
            import json
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "check_time": datetime.now().isoformat(),
                    "check_type": "knowledge_base_health",
                    "results": kb_results
                }, f, ensure_ascii=False, indent=2)
            print(f"结果已保存到: {output_file}")
    
    elif choice == "7":
        print("\n执行完整诊断...")
        from check_connections import ConnectionChecker, KnowledgeBaseHealthChecker
        
        # 连接检查
        print("\n" + "="*60)
        print("🔌 步骤1: 连接可用性检查")
        print("="*60)
        
        checker = ConnectionChecker()
        conn_results = checker.check_all_connections()
        checker.print_summary(conn_results)
        
        # 知识库健康检查
        print("\n" + "="*60)
        print("🏥 步骤2: 知识库健康状态检查")
        print("="*60)
        
        kb_checker = KnowledgeBaseHealthChecker()
        kb_results = kb_checker.check_all_kb_health()
        kb_checker.print_health_summary(kb_results)
        
        # 保存综合结果
        # 确保reports目录存在
        reports_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")
        os.makedirs(reports_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(reports_dir, f"full_diagnostic_{timestamp}.json")
        import json
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                "check_time": datetime.now().isoformat(),
                "check_type": "full_diagnostic",
                "connection_check": conn_results,
                "kb_health_check": kb_results
            }, f, ensure_ascii=False, indent=2)
        print(f"\n诊断结果已保存到: {output_file}")
    
    elif choice == "8":
        print("\n启动知识库管理工具...")
        from manage_knowledge_simple import main as manage_knowledge_main
        manage_knowledge_main()
        from add_knowledge_simple import main as add_knowledge_main
        add_knowledge_main()
    
    else:
        print("无效的选择")


if __name__ == "__main__":
    main()