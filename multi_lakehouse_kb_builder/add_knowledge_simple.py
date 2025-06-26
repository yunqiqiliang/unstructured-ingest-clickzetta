#!/usr/bin/env python3
"""
简化的知识添加脚本
提供交互式界面来添加自定义知识到知识库
"""

import os
import sys
import json
import logging
from datetime import datetime
from typing import List

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kb_knowledge_adder import KnowledgeEntry, BatchKnowledgeAdder, create_sample_knowledge_file

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def input_knowledge_interactively() -> List[KnowledgeEntry]:
    """交互式输入知识条目"""
    entries = []
    
    print("\n📝 交互式知识输入")
    print("="*60)
    print("输入知识内容，每条知识输入完成后按回车")
    print("输入空行结束输入")
    print("="*60)
    
    count = 1
    while True:
        text = input(f"\n知识条目 {count}: ").strip()
        if not text:
            break
        
        # 询问来源类型
        source = input("来源类型 [UserInput/FAQ/CompanyInfo/TechnicalDoc] (默认UserInput): ").strip()
        if not source:
            source = "UserInput"
        
        entries.append(KnowledgeEntry(
            text=text,
            source=source,
            languages=["zh-cn"]
        ))
        
        count += 1
        print(f"✅ 已添加第 {count-1} 条知识")
    
    return entries


def load_knowledge_from_file(file_path: str) -> List[KnowledgeEntry]:
    """从文件加载知识条目"""
    entries = []
    
    try:
        if file_path.endswith('.json'):
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, str):
                        entries.append(KnowledgeEntry(text=item))
                    elif isinstance(item, dict):
                        entries.append(KnowledgeEntry(**item))
            elif isinstance(data, dict):
                entries.append(KnowledgeEntry(**data))
        
        elif file_path.endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            for line in lines:
                line = line.strip()
                if line:
                    entries.append(KnowledgeEntry(text=line))
        
        else:
            logger.error(f"不支持的文件格式: {file_path}")
    
    except Exception as e:
        logger.error(f"读取文件失败: {e}")
    
    return entries


def preview_entries(entries: List[KnowledgeEntry]):
    """预览知识条目"""
    print("\n📋 知识条目预览")
    print("="*80)
    
    for i, entry in enumerate(entries, 1):
        print(f"\n{i}. [{entry.source}]")
        # 限制显示长度
        text_preview = entry.text[:100] + "..." if len(entry.text) > 100 else entry.text
        print(f"   {text_preview}")
    
    print(f"\n总计: {len(entries)} 条知识")
    print("="*80)


def main():
    """主函数"""
    print("🚀 ClickZetta 知识库知识添加工具")
    print("="*60)
    
    # 创建批量添加器
    batch_adder = BatchKnowledgeAdder()
    
    # 检查DashScope API密钥
    if not batch_adder.dashscope_key:
        print("❌ 未配置DashScope API密钥")
        print("请在 ~/.clickzetta/connections.json 中配置或设置环境变量 DASHSCOPE_API_KEY")
        return
    
    # 选择操作模式
    print("\n请选择操作模式:")
    print("1. 交互式输入知识")
    print("2. 从文件导入知识")
    print("3. 创建示例知识文件")
    print("4. 添加预定义的业务知识")
    print("0. 返回上级菜单/退出")
    
    choice = input("\n请输入选择(0-4): ").strip()
    
    entries = []
    
    if choice == "0":
        print("\n👋 返回上级菜单...")
        return
    
    elif choice == "1":
        # 交互式输入
        entries = input_knowledge_interactively()
        
    elif choice == "2":
        # 从文件导入
        file_path = input("\n请输入文件路径: ").strip()
        if os.path.exists(file_path):
            entries = load_knowledge_from_file(file_path)
            if entries:
                preview_entries(entries)
            else:
                print("❌ 未能从文件中读取到知识条目")
                return
        else:
            print(f"❌ 文件不存在: {file_path}")
            return
    
    elif choice == "3":
        # 创建示例文件
        json_file, csv_file = create_sample_knowledge_file()
        print("\n你可以编辑这些文件，然后使用选项2导入")
        return
    
    elif choice == "4":
        # 预定义的业务知识
        entries = [
            KnowledgeEntry(
                text="ClickZetta、云器、Singdata之间的关系：ClickZetta是云器、Singdata的技术品牌。云器是ClickZetta的中国商业公司，Singdata是ClickZetta的海外商业公司。所以在不区分商业和技术的时候，三者是等同的。另外云器经常也会用到Yunqi的汉语拼音名称。",
                source="CompanyInfo"
            ),
            KnowledgeEntry(
                text="ClickZetta Lakehouse是一个云原生的数据仓库，支持结构化和非结构化数据的统一存储和分析。",
                source="ProductInfo"
            ),
            KnowledgeEntry(
                text="使用TRUNCATE TABLE语句可以快速清空表中的所有数据，但会保留表结构和索引。这比DELETE语句更高效。",
                source="BestPractice"
            ),
            KnowledgeEntry(
                text="向量搜索使用cosine_distance函数计算相似度，返回值越小表示越相似。通常配合ORDER BY和LIMIT使用。",
                source="VectorSearch"
            ),
            KnowledgeEntry(
                text="ClickZetta支持的向量索引类型包括IVF、HNSW等，可以通过CREATE INDEX语句创建。",
                source="IndexInfo"
            )
        ]
        preview_entries(entries)
    
    else:
        print("无效的选择")
        return
    
    # 如果有知识条目，询问部署目标
    if entries:
        print("\n请选择部署目标:")
        print("1. 部署到所有Lakehouse")
        print("2. 部署到特定Lakehouse（通过名称匹配）")
        print("0. 返回上级菜单")
        
        deploy_choice = input("\n请输入选择(0-2): ").strip()
        
        if deploy_choice == "0":
            print("\n👋 返回上级菜单...")
            return
            
        elif deploy_choice == "1":
            # 确认
            connections = batch_adder.conn_manager.connections
            print(f"\n⚠️  即将向 {len(connections)} 个Lakehouse添加 {len(entries)} 条知识")
            confirm = input("是否继续？(y/n): ").strip().lower()
            
            if confirm in ['y', 'yes']:
                print("\n开始添加知识...")
                results = batch_adder.add_to_all_lakehouse(entries)
                batch_adder.print_summary(results)
                
                # 保存结果
                save = input("\n是否保存结果？(y/n): ").strip().lower()
                if save in ['y', 'yes']:
                    # 确保reports目录存在
                    reports_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")
                    os.makedirs(reports_dir, exist_ok=True)
                    
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_file = os.path.join(reports_dir, f"knowledge_add_result_{timestamp}.json")
                    
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump({
                            "add_time": datetime.now().isoformat(),
                            "total_entries": len(entries),
                            "entries": [e.to_dict() for e in entries],
                            "results": results
                        }, f, ensure_ascii=False, indent=2)
                    
                    print(f"结果已保存到: {output_file}")
        
        elif deploy_choice == "2":
            pattern = input("请输入连接名称匹配模式: ").strip()
            if pattern:
                # 获取匹配的连接
                matched_connections = []
                for conn in batch_adder.conn_manager.connections:
                    conn_name = conn.get('connection_name', 'unnamed')
                    if pattern in conn_name:
                        matched_connections.append(conn)
                
                if not matched_connections:
                    print(f"❌ 没有找到匹配 '{pattern}' 的连接")
                    return
                
                # 显示匹配结果
                print(f"\n找到 {len(matched_connections)} 个匹配的连接:")
                for i, conn in enumerate(matched_connections, 1):
                    conn_name = conn.get('connection_name', 'unnamed')
                    print(f"{i}. {conn_name}")
                
                # 确认
                print(f"\n⚠️  即将向以上 {len(matched_connections)} 个Lakehouse添加 {len(entries)} 条知识")
                confirm = input("是否继续？(y/n): ").strip().lower()
                
                if confirm in ['y', 'yes']:
                    print("\n开始添加知识...")
                    results = batch_adder.add_to_all_lakehouse(entries, filter_pattern=pattern)
                    batch_adder.print_summary(results)
        
        else:
            print("❌ 无效的选择")


if __name__ == "__main__":
    main()