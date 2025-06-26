#!/usr/bin/env python3
"""
知识库知识管理 - 交互式界面
"""

import os
import sys
import json
import logging
from datetime import datetime
from typing import List, Dict, Any

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kb_knowledge_manager import (
    KnowledgeEntry, 
    KnowledgeManager, 
    BatchKnowledgeManager,
    load_knowledge_from_file
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def add_single_knowledge_interactive():
    """交互式添加单条知识"""
    print("\n📝 添加单条知识")
    print("="*60)
    
    # 输入知识内容
    print("请输入知识内容（支持多行，输入END结束）：")
    lines = []
    while True:
        line = input()
        if line.strip().upper() == 'END':
            break
        lines.append(line)
    
    text = '\n'.join(lines)
    if not text.strip():
        print("❌ 知识内容不能为空")
        return None
    
    # 输入来源
    source = input("\n知识来源（默认: UserInput）: ").strip() or "UserInput"
    
    # 输入语言
    languages_input = input("语言（默认: zh-cn，多个用逗号分隔）: ").strip()
    if languages_input:
        languages = [lang.strip() for lang in languages_input.split(',')]
    else:
        languages = ["zh-cn"]
    
    # 创建知识条目
    entry = KnowledgeEntry(
        text=text,
        source=source,
        languages=languages
    )
    
    print(f"\n✅ 知识条目已创建:")
    print(f"   ID: {entry.id}")
    print(f"   文本: {entry.text[:100]}{'...' if len(entry.text) > 100 else ''}")
    print(f"   来源: {entry.source}")
    print(f"   语言: {', '.join(entry.languages)}")
    
    return entry


def add_knowledge_from_file():
    """从文件添加知识"""
    print("\n📁 从文件添加知识")
    print("="*60)
    
    # 输入文件路径
    file_path = input("请输入文件路径: ").strip()
    
    if not os.path.exists(file_path):
        print(f"❌ 文件不存在: {file_path}")
        return None
    
    try:
        entries = load_knowledge_from_file(file_path)
        print(f"\n✅ 成功从文件加载 {len(entries)} 条知识")
        
        # 显示预览
        print("\n预览前3条知识:")
        for i, entry in enumerate(entries[:3], 1):
            print(f"{i}. {entry.text[:80]}{'...' if len(entry.text) > 80 else ''}")
            print(f"   来源: {entry.source}")
        
        if len(entries) > 3:
            print(f"... 还有 {len(entries) - 3} 条知识")
        
        return entries
        
    except Exception as e:
        print(f"❌ 加载文件失败: {e}")
        return None


def search_knowledge_interactive(batch_manager: BatchKnowledgeManager):
    """交互式搜索知识"""
    print("\n🔍 搜索知识")
    print("="*60)
    
    # 输入搜索条件
    query = input("搜索关键词（留空搜索所有）: ").strip()
    source = input("按来源筛选（留空不筛选）: ").strip()
    
    # 选择Lakehouse
    print("\n选择Lakehouse:")
    print("1. 搜索所有Lakehouse")
    print("2. 搜索特定Lakehouse")
    print("0. 返回上级菜单")
    
    choice = input("请选择(0-2): ").strip()
    
    if choice == "0":
        print("👋 返回主菜单...")
        return
        
    elif choice == "1":
        # 搜索所有
        print("\n搜索中...")
        results = batch_manager.search_across_lakehouse(
            query=query if query else None,
            source=source if source else None
        )
        
        # 显示结果
        total_found = sum(len(items) for items in results.values())
        print(f"\n找到 {total_found} 条匹配的知识")
        
        for conn_name, items in results.items():
            if items:
                print(f"\n📌 {conn_name} ({len(items)} 条):")
                for i, item in enumerate(items[:3], 1):  # 只显示前3条
                    print(f"{i}. ID: {item.get('id', 'N/A')}")
                    print(f"   文本: {item.get('text', '')[:100]}...")
                    print(f"   来源: {item.get('source', 'N/A')}")
                    print(f"   创建时间: {item.get('date_created', 'N/A')}")
                
                if len(items) > 3:
                    print(f"   ... 还有 {len(items) - 3} 条")
                    
    elif choice == "2":
        # 搜索特定Lakehouse
        print("\n可用的Lakehouse:")
        for i, conn in enumerate(batch_manager.connections, 1):
            print(f"{i}. {conn.get('connection_name', 'unnamed')}")
        
        idx = input("\n选择Lakehouse编号: ").strip()
        try:
            idx = int(idx) - 1
            if 0 <= idx < len(batch_manager.connections):
                conn = batch_manager.connections[idx]
                manager = KnowledgeManager(conn)
                
                print(f"\n搜索 {conn.get('connection_name', 'unnamed')}...")
                items = manager.search_knowledge(
                    query=query if query else None,
                    source=source if source else None,
                    limit=20
                )
                manager.close()
                
                print(f"\n找到 {len(items)} 条匹配的知识:")
                for i, item in enumerate(items, 1):
                    print(f"\n{i}. ID: {item.get('id', 'N/A')}")
                    print(f"   文本: {item.get('text', '')[:200]}...")
                    print(f"   来源: {item.get('source', 'N/A')}")
                    print(f"   创建时间: {item.get('date_created', 'N/A')}")
            else:
                print("❌ 无效的选择")
        except:
            print("❌ 无效的输入")


def delete_knowledge_interactive(batch_manager: BatchKnowledgeManager):
    """交互式删除知识"""
    print("\n🗑️  删除知识")
    print("="*60)
    
    print("删除方式:")
    print("1. 按ID删除")
    print("2. 按来源批量删除")
    print("3. 先搜索后删除")
    print("0. 返回上级菜单")
    
    choice = input("\n请选择(0-3): ").strip()
    
    if choice == "0":
        print("👋 返回主菜单...")
        return
        
    elif choice == "1":
        # 按ID删除
        ids_input = input("\n输入要删除的知识ID（多个用逗号分隔）: ").strip()
        if not ids_input:
            print("❌ 未输入ID")
            return
        
        knowledge_ids = [id.strip() for id in ids_input.split(',')]
        
        print(f"\n⚠️  即将删除 {len(knowledge_ids)} 条知识")
        confirm = input("确认删除？(y/n): ").strip().lower()
        
        if confirm in ['y', 'yes']:
            print("\n删除中...")
            results = batch_manager.delete_from_all_lakehouse(knowledge_ids)
            
            # 显示结果
            for result in results:
                conn_name = result['connection_name']
                if result['status'] == 'success':
                    res = result['result']
                    print(f"✅ {conn_name}: 成功删除 {res['success']}/{res['total']} 条")
                else:
                    print(f"❌ {conn_name}: 失败 - {result['error']}")
        else:
            print("已取消删除")
            
    elif choice == "2":
        # 按来源删除
        source = input("\n输入要删除的知识来源: ").strip()
        if not source:
            print("❌ 未输入来源")
            return
        
        # 先统计数量
        print("\n统计中...")
        stats = batch_manager.get_all_statistics()
        
        total_to_delete = 0
        for conn_name, stat in stats.items():
            if 'by_source' in stat and source in stat['by_source']:
                count = stat['by_source'][source]
                print(f"{conn_name}: {count} 条")
                total_to_delete += count
        
        if total_to_delete == 0:
            print(f"\n未找到来源为 '{source}' 的知识")
            return
        
        print(f"\n⚠️  即将删除所有来源为 '{source}' 的知识（共 {total_to_delete} 条）")
        confirm = input("确认删除？(y/n): ").strip().lower()
        
        if confirm in ['y', 'yes']:
            print("\n删除中...")
            for conn in batch_manager.connections:
                conn_name = conn.get('connection_name', 'unnamed')
                try:
                    manager = KnowledgeManager(conn)
                    count = manager.delete_by_source(source)
                    manager.close()
                    
                    if count > 0:
                        print(f"✅ {conn_name}: 删除了 {count} 条")
                except Exception as e:
                    print(f"❌ {conn_name}: 失败 - {e}")
        else:
            print("已取消删除")
            
    elif choice == "3":
        # 先搜索后删除
        print("\n先搜索要删除的知识...")
        # 这里可以调用搜索功能，然后让用户选择要删除的条目
        print("（功能开发中...）")


def show_statistics(batch_manager: BatchKnowledgeManager):
    """显示统计信息"""
    print("\n📊 知识库统计")
    print("="*60)
    
    stats = batch_manager.get_all_statistics()
    
    grand_total = 0
    source_totals = {}
    
    for conn_name, stat in stats.items():
        if 'error' in stat:
            print(f"\n❌ {conn_name}: 获取统计失败 - {stat['error']}")
        else:
            total = stat.get('total', 0)
            grand_total += total
            
            print(f"\n📌 {conn_name}:")
            print(f"   总计: {total} 条知识")
            
            by_source = stat.get('by_source', {})
            if by_source:
                print("   按来源分类:")
                for source, count in sorted(by_source.items(), key=lambda x: x[1], reverse=True):
                    print(f"     - {source}: {count} 条")
                    source_totals[source] = source_totals.get(source, 0) + count
    
    # 总体统计
    print(f"\n📊 总体统计:")
    print(f"   Lakehouse数量: {len(stats)}")
    print(f"   知识总数: {grand_total}")
    
    if source_totals:
        print("   按来源汇总:")
        for source, count in sorted(source_totals.items(), key=lambda x: x[1], reverse=True):
            print(f"     - {source}: {count} 条")


def main():
    """主函数"""
    print("🎯 ClickZetta 知识库知识管理工具")
    print("="*60)
    
    # 创建批量管理器
    batch_manager = BatchKnowledgeManager()
    
    if not batch_manager.connections:
        print("❌ 没有找到任何Lakehouse连接")
        return
    
    print(f"✅ 找到 {len(batch_manager.connections)} 个Lakehouse连接")
    
    while True:
        print("\n" + "="*60)
        print("请选择操作:")
        print("1. 添加单条知识")
        print("2. 从文件批量添加知识")
        print("3. 搜索知识")
        print("4. 删除知识")
        print("5. 查看统计信息")
        print("6. 创建示例知识文件")
        print("0. 退出系统")
        
        choice = input("\n请输入选择(0-6): ").strip()
        
        if choice == "0":
            print("\n👋 退出程序")
            break
            
        elif choice == "1":
            # 添加单条知识
            entry = add_single_knowledge_interactive()
            if entry:
                # 选择目标Lakehouse
                print("\n选择目标:")
                print("1. 添加到所有Lakehouse")
                print("2. 添加到特定Lakehouse")
                print("0. 返回上级菜单")
                
                target_choice = input("请选择(0-2): ").strip()
                
                if target_choice == "0":
                    print("👋 返回主菜单...")
                    continue
                    
                elif target_choice == "1":
                    print("\n添加中...")
                    results = batch_manager.add_to_all_lakehouse([entry])
                    
                    for result in results:
                        conn_name = result['connection_name']
                        if result['status'] == 'success':
                            res = result['result']
                            print(f"✅ {conn_name}: 成功")
                        else:
                            print(f"❌ {conn_name}: 失败 - {result['error']}")
                            
                elif target_choice == "2":
                    print("\n可用的Lakehouse:")
                    for i, conn in enumerate(batch_manager.connections, 1):
                        print(f"{i}. {conn.get('connection_name', 'unnamed')}")
                    
                    idx = input("\n选择Lakehouse编号: ").strip()
                    try:
                        idx = int(idx) - 1
                        if 0 <= idx < len(batch_manager.connections):
                            conn = batch_manager.connections[idx]
                            manager = KnowledgeManager(conn)
                            
                            if manager.add_knowledge(entry):
                                print(f"✅ 成功添加到 {conn.get('connection_name', 'unnamed')}")
                            else:
                                print("❌ 添加失败")
                            
                            manager.close()
                        else:
                            print("❌ 无效的选择")
                    except:
                        print("❌ 无效的输入")
                        
        elif choice == "2":
            # 从文件添加
            entries = add_knowledge_from_file()
            if entries:
                confirm = input(f"\n确认添加 {len(entries)} 条知识到所有Lakehouse？(y/n): ").strip().lower()
                
                if confirm in ['y', 'yes']:
                    print("\n添加中...")
                    results = batch_manager.add_to_all_lakehouse(entries)
                    
                    success_total = 0
                    for result in results:
                        conn_name = result['connection_name']
                        if result['status'] == 'success':
                            res = result['result']
                            success_total += res['success']
                            print(f"✅ {conn_name}: 成功添加 {res['success']}/{res['total']} 条")
                        else:
                            print(f"❌ {conn_name}: 失败 - {result['error']}")
                    
                    print(f"\n总计成功添加 {success_total} 条知识")
                else:
                    print("已取消添加")
                    
        elif choice == "3":
            # 搜索知识
            search_knowledge_interactive(batch_manager)
            
        elif choice == "4":
            # 删除知识
            delete_knowledge_interactive(batch_manager)
            
        elif choice == "5":
            # 查看统计
            show_statistics(batch_manager)
            
        elif choice == "6":
            # 创建示例文件
            print("\n📝 创建示例知识文件")
            
            # JSON示例
            json_example = [
                {
                    "text": "ClickZetta是云器科技的技术品牌，提供云原生数据仓库解决方案。",
                    "source": "CompanyInfo",
                    "languages": ["zh-cn"]
                },
                {
                    "text": "Lakehouse支持SQL查询、实时分析和机器学习等多种数据处理场景。",
                    "source": "ProductInfo",
                    "languages": ["zh-cn"]
                },
                {
                    "text": "使用CREATE INDEX语句创建索引，支持倒排索引和向量索引。",
                    "source": "TechnicalDoc",
                    "languages": ["zh-cn"]
                }
            ]
            
            # CSV示例
            csv_content = """text,source,languages
"ClickZetta提供高性能的向量搜索功能，支持余弦相似度计算。","TechnicalDoc","['zh-cn']"
"云器科技总部位于中国，Singdata是其海外品牌。","CompanyInfo","['zh-cn']"
"支持Python、Java、SQL等多种开发语言和接口。","ProductInfo","['zh-cn']"
"""
            
            # TXT示例
            txt_content = """ClickZetta Lakehouse是一个云原生的数据仓库平台。
支持结构化、半结构化和非结构化数据的统一管理。
提供高性能的向量搜索和相似度计算功能。
支持实时数据分析和批处理。
兼容标准SQL语法，易于使用和集成。"""
            
            # 确保示例目录存在
            examples_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "examples")
            os.makedirs(examples_dir, exist_ok=True)
            
            # 保存示例文件
            json_file = os.path.join(examples_dir, "knowledge_example.json")
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(json_example, f, ensure_ascii=False, indent=2)
            
            csv_file = os.path.join(examples_dir, "knowledge_example.csv")
            with open(csv_file, 'w', encoding='utf-8') as f:
                f.write(csv_content)
            
            txt_file = os.path.join(examples_dir, "knowledge_example.txt")
            with open(txt_file, 'w', encoding='utf-8') as f:
                f.write(txt_content)
            
            print(f"\n✅ 示例文件已创建:")
            print(f"   - {json_file}")
            print(f"   - {csv_file}")
            print(f"   - {txt_file}")
            
        else:
            print("❌ 无效的选择")


if __name__ == "__main__":
    main()