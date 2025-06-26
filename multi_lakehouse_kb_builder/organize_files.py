#!/usr/bin/env python3
"""
组织日志和报告文件到子目录
"""

import os
import shutil
import glob
from datetime import datetime

def organize_files():
    """组织日志和报告文件"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 创建子目录
    logs_dir = os.path.join(base_dir, "logs")
    reports_dir = os.path.join(base_dir, "reports")
    
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    
    # 移动日志文件
    log_patterns = ["*.log", "kb_deployment_*.log"]
    moved_logs = 0
    
    for pattern in log_patterns:
        for log_file in glob.glob(os.path.join(base_dir, pattern)):
            if os.path.isfile(log_file):
                filename = os.path.basename(log_file)
                dest = os.path.join(logs_dir, filename)
                if not os.path.exists(dest):  # 避免覆盖
                    shutil.move(log_file, dest)
                    print(f"移动日志: {filename} -> logs/")
                    moved_logs += 1
    
    # 移动报告文件
    report_patterns = [
        "kb_deployment_result_*.json",
        "kb_validation_report_*.json",
        "connection_check_*.json",
        "kb_health_check_*.json",
        "full_diagnostic_*.json"
    ]
    moved_reports = 0
    
    for pattern in report_patterns:
        for report_file in glob.glob(os.path.join(base_dir, pattern)):
            if os.path.isfile(report_file):
                filename = os.path.basename(report_file)
                dest = os.path.join(reports_dir, filename)
                if not os.path.exists(dest):  # 避免覆盖
                    shutil.move(report_file, dest)
                    print(f"移动报告: {filename} -> reports/")
                    moved_reports += 1
    
    print(f"\n整理完成:")
    print(f"- 移动了 {moved_logs} 个日志文件到 logs/ 目录")
    print(f"- 移动了 {moved_reports} 个报告文件到 reports/ 目录")
    
    # 创建 .gitignore 文件
    gitignore_content = """# 日志文件
logs/
*.log

# 报告文件
reports/
*_report_*.json
*_result_*.json
*_check_*.json
*_diagnostic_*.json

# Python 缓存
__pycache__/
*.pyc
*.pyo
*.pyd
.Python

# 虚拟环境
.venv/
venv/
ENV/
env/

# IDE
.vscode/
.idea/
*.swp
*.swo

# 系统文件
.DS_Store
Thumbs.db
"""
    
    gitignore_path = os.path.join(base_dir, ".gitignore")
    if not os.path.exists(gitignore_path):
        with open(gitignore_path, 'w') as f:
            f.write(gitignore_content)
        print(f"\n创建了 .gitignore 文件")
    
    # 创建 README 文件说明目录结构
    readme_content = f"""# 目录结构说明

## logs/
存放所有运行日志文件：
- kb_deployment_*.log - 部署日志
- 其他运行日志

## reports/
存放所有生成的报告文件：
- kb_deployment_result_*.json - 部署结果报告
- kb_validation_report_*.json - 数据验证报告
- connection_check_*.json - 连接检查报告
- kb_health_check_*.json - 知识库健康检查报告
- full_diagnostic_*.json - 完整诊断报告

生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    dirs_readme = os.path.join(base_dir, "DIRECTORY_STRUCTURE.md")
    with open(dirs_readme, 'w') as f:
        f.write(readme_content)
    print(f"创建了 DIRECTORY_STRUCTURE.md 文件")


if __name__ == "__main__":
    organize_files()