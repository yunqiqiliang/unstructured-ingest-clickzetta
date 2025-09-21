#!/usr/bin/env python3
"""
Fork配置设置脚本

在与上游同步后运行此脚本来恢复fork特定的配置
"""

import os
import sys
from pathlib import Path

def update_pyproject_toml():
    """更新pyproject.toml为fork特定配置"""
    pyproject_path = Path("pyproject.toml")

    if not pyproject_path.exists():
        print("❌ pyproject.toml 文件不存在")
        return False

    # 读取当前内容
    content = pyproject_path.read_text(encoding="utf-8")

    # 需要替换的配置项
    replacements = {
        'name = "unstructured_ingest"': 'name = "unstructured-ingest-clickzetta"',
        'description = "Local ETL data pipeline to get data RAG ready"':
            'description = "ClickZetta connector for Unstructured data pipeline - Enhanced ETL with SQL and Volume support"',
        'authors = [{name = "Unstructured Technologies", email = "devops@unstructuredai.io"}]':
            'authors = [{name = "ClickZetta Community", email = "yunqiqiliang@gmail.com"}]',
    }

    # 执行替换
    modified = False
    for old, new in replacements.items():
        if old in content and new not in content:
            content = content.replace(old, new)
            modified = True
            print(f"✅ 已更新: {old[:50]}...")

    # 添加额外的项目信息（如果不存在）
    additional_fields = [
        'keywords = ["clickzetta", "unstructured", "etl", "data-pipeline", "rag", "document-processing"]',
        'homepage = "https://github.com/yunqiqiliang/unstructured-ingest-clickzetta"',
        'repository = "https://github.com/yunqiqiliang/unstructured-ingest-clickzetta"',
        'documentation = "https://github.com/yunqiqiliang/unstructured-ingest-clickzetta#readme"',
    ]

    # 在 dynamic = [...] 之后添加额外字段
    if 'dynamic = ["version", "dependencies", "optional-dependencies"]' in content:
        for field in additional_fields:
            if field not in content:
                content = content.replace(
                    'dynamic = ["version", "dependencies", "optional-dependencies"]',
                    f'dynamic = ["version", "dependencies", "optional-dependencies"]\n{field}'
                )
                modified = True
                print(f"✅ 已添加: {field}")

    # 添加CLI命令别名
    if 'unstructured-ingest-clickzetta = "unstructured_ingest.main:main"' not in content:
        content = content.replace(
            'unstructured-ingest = "unstructured_ingest.main:main"',
            'unstructured-ingest = "unstructured_ingest.main:main"\nunstructured-ingest-clickzetta = "unstructured_ingest.main:main"'
        )
        modified = True
        print("✅ 已添加CLI命令别名")

    if modified:
        pyproject_path.write_text(content, encoding="utf-8")
        print("✅ pyproject.toml 配置已更新")
        return True
    else:
        print("ℹ️  pyproject.toml 已经是正确配置")
        return True

def check_env_example():
    """检查环境配置示例文件是否存在"""
    env_example = Path("env.example.txt")
    if env_example.exists():
        print("✅ env.example.txt 文件存在")
        return True
    else:
        print("❌ env.example.txt 文件缺失，请重新创建")
        return False

def check_gitattributes():
    """检查.gitattributes文件是否存在"""
    gitattributes = Path(".gitattributes")
    if gitattributes.exists():
        print("✅ .gitattributes 文件存在")
        return True
    else:
        print("❌ .gitattributes 文件缺失，请重新创建")
        return False

def main():
    """主函数"""
    print("🔧 设置Fork特定配置...")
    print("=" * 50)

    success = True

    # 更新pyproject.toml
    success &= update_pyproject_toml()

    # 检查其他关键文件
    success &= check_env_example()
    success &= check_gitattributes()

    print("=" * 50)
    if success:
        print("🎉 Fork配置设置完成！")
    else:
        print("❌ 部分配置设置失败，请检查上述错误信息")
        sys.exit(1)

if __name__ == "__main__":
    main()