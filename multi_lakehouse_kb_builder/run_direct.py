#!/usr/bin/env python3
"""
直接运行脚本 - 统一入口
"""

import sys
import os

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("ClickZetta 知识库部署系统")
        print("=" * 40)
        print("用法: python run_direct.py [命令]")
        print("\n可用命令:")
        print("  test       - 测试环境配置")
        print("  deploy     - 交互式部署")
        print("  validate   - 验证数据")
        print("  deploy-all - 部署到所有Lakehouse")
        print("  check      - 检查连接和知识库状态")
        print("  check-conn - 只检查连接")
        print("  check-kb   - 只检查知识库")
        print("  manage-kb  - 管理知识库内容（增/删/查）")
        print("  add-kb     - 添加自定义知识")
        print("\n示例:")
        print("  python run_direct.py deploy")
        return
    
    command = sys.argv[1]
    
    if command == "test":
        print("\n运行环境测试...")
        from test_kb_deployment import main as test_main
        test_main()
    
    elif command == "deploy":
        print("\n启动交互式部署...")
        from deploy_kb_simple import main as deploy_main
        deploy_main()
    
    elif command == "validate":
        print("\n验证知识库数据...")
        from validate_kb_simple import main as validate_main
        validate_main()
    
    elif command == "deploy-all":
        print("\n串行部署到所有Lakehouse...")
        # 修改sys.argv以传递正确的参数
        sys.argv = [sys.argv[0], "--mode", "serial"]
        from multi_lakehouse_kb_builder import main as builder_main
        builder_main()
    
    elif command == "check":
        print("\n检查连接和知识库状态...")
        from check_connections import main as check_main
        check_main()
    
    elif command == "check-conn":
        print("\n检查所有连接...")
        from check_connections import ConnectionChecker
        checker = ConnectionChecker()
        results = checker.check_all_connections()
        checker.print_summary(results)
    
    elif command == "check-kb":
        print("\n检查知识库健康状态...")
        from check_connections import KnowledgeBaseHealthChecker
        kb_checker = KnowledgeBaseHealthChecker()
        kb_results = kb_checker.check_all_kb_health()
        kb_checker.print_health_summary(kb_results)
    
    elif command == "manage-kb":
        print("\n启动知识库管理工具...")
        from manage_knowledge_simple import main as manage_kb_main
        manage_kb_main()
    
    else:
        print(f"未知命令: {command}")
        print("请使用 test, deploy, validate, deploy-all, check, check-conn, check-kb 或 manage-kb")


if __name__ == "__main__":
    main()