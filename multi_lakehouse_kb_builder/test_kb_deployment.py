#!/usr/bin/env python3
"""
ClickZetta 知识库部署环境测试脚本
"""

import os
import sys
import json
from pathlib import Path

def print_colored(text, color="green"):
    """打印彩色文本"""
    colors = {
        "green": "\033[32m",
        "red": "\033[31m",
        "yellow": "\033[33m",
        "blue": "\033[34m",
        "reset": "\033[0m"
    }
    print(f"{colors.get(color, '')}{text}{colors['reset']}")

def test_python_environment():
    """测试Python环境"""
    print_colored("\n🐍 Python环境检查:", "blue")
    print(f"  Python版本: {sys.version}")
    print(f"  Python路径: {sys.executable}")
    print_colored("  ✅ Python环境正常", "green")

def test_dependencies():
    """测试依赖包"""
    print_colored("\n📦 依赖包检查:", "blue")

    # 检测当前使用的Python环境
    virtual_env = os.getenv('VIRTUAL_ENV')
    if virtual_env:
        env_info = f"当前环境: {Path(virtual_env).name}"
    else:
        env_info = f"当前环境: {Path(sys.executable).parent.parent.name}"

    print_colored(f"  🐍 {env_info}", "blue")

    required_packages = [
        ("pandas", "数据处理"),
        ("clickzetta", "ClickZetta连接器"),
        ("dashscope", "DashScope API"),
        ("json", "JSON处理 (内置)"),
        ("pathlib", "路径处理 (内置)")
    ]

    missing_packages = []

    for package, description in required_packages:
        try:
            # 对于clickzetta，实际的包名是clickzetta-connector-python
            if package == "clickzetta":
                try:
                    import clickzetta
                    print_colored(f"  ✅ {package} - {description}", "green")
                except ImportError:
                    # 尝试导入实际的包
                    try:
                        import clickzetta_connector_python
                        print_colored(f"  ✅ {package} - {description} (通过 clickzetta-connector-python)", "green")
                    except ImportError:
                        print_colored(f"  ❌ {package} - {description} (缺失)", "red")
                        missing_packages.append("clickzetta-connector-python")
            else:
                __import__(package)
                print_colored(f"  ✅ {package} - {description}", "green")
        except ImportError:
            print_colored(f"  ❌ {package} - {description} (缺失)", "red")
            missing_packages.append(package)

    if missing_packages:
        print_colored(f"\n⚠️  缺少依赖包: {', '.join(missing_packages)}", "yellow")
        print_colored(f"建议运行: pip install {' '.join(missing_packages)}", "yellow")
        return False

    return True

def test_config_file():
    """测试配置文件"""
    print_colored("\n⚙️  配置文件检查:", "blue")

    config_path = Path.home() / ".clickzetta" / "connections.json"

    if not config_path.exists():
        print_colored(f"  ❌ 配置文件不存在: {config_path}", "red")
        print_colored("  建议创建配置文件: ~/.clickzetta/connections.json", "yellow")
        return False

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        print_colored(f"  ✅ 配置文件存在: {config_path}", "green")

        if 'connections' in config:
            conn_count = len(config['connections'])
            print_colored(f"  ✅ 发现 {conn_count} 个连接配置", "green")
        else:
            print_colored("  ⚠️  配置文件中没有连接配置", "yellow")

        if 'system_config' in config:
            print_colored("  ✅ 系统配置存在", "green")
        else:
            print_colored("  ⚠️  系统配置不存在", "yellow")

        return True

    except json.JSONDecodeError as e:
        print_colored(f"  ❌ 配置文件格式错误: {e}", "red")
        return False
    except Exception as e:
        print_colored(f"  ❌ 读取配置文件失败: {e}", "red")
        return False

def test_environment_variables():
    """测试环境变量"""
    print_colored("\n🔑 环境变量检查:", "blue")

    env_vars = [
        ("DASHSCOPE_API_KEY", "DashScope API密钥"),
        ("LOCAL_FILE_INPUT_DIR", "文档目录路径 (可选)")
    ]

    for var_name, description in env_vars:
        value = os.getenv(var_name)
        if value:
            # 不显示完整的API密钥，只显示前几位
            if "API_KEY" in var_name:
                masked_value = f"{value[:8]}...{value[-4:]}" if len(value) > 12 else "***"
                print_colored(f"  ✅ {var_name} - {description}: {masked_value}", "green")
            else:
                print_colored(f"  ✅ {var_name} - {description}: {value}", "green")
        else:
            if var_name == "DASHSCOPE_API_KEY":
                print_colored(f"  ⚠️  {var_name} - {description} (未设置，可在配置文件中设置)", "yellow")
            else:
                print_colored(f"  ℹ️  {var_name} - {description} (未设置，将使用默认值)", "blue")

def test_document_directory():
    """测试文档目录"""
    print_colored("\n📁 文档目录检查:", "blue")

    # 检查环境变量中的路径
    doc_dir = os.getenv("LOCAL_FILE_INPUT_DIR")

    if doc_dir:
        doc_path = Path(doc_dir)
        if doc_path.exists():
            file_count = len(list(doc_path.glob("**/*")))
            print_colored(f"  ✅ 文档目录存在: {doc_path} ({file_count} 个文件)", "green")
            return True
        else:
            print_colored(f"  ❌ 文档目录不存在: {doc_path}", "red")
            return False
    else:
        # 检查默认路径
        default_paths = [
            Path.home() / "Documents",
            Path("./documents"),
            Path(".")
        ]

        for path in default_paths:
            if path.exists():
                print_colored(f"  ✅ 可用路径: {path}", "green")
                return True

        print_colored("  ⚠️  未设置文档目录路径，且默认路径不可用", "yellow")
        print_colored("  建议设置环境变量: export LOCAL_FILE_INPUT_DIR=/path/to/documents", "yellow")
        return False

def test_project_structure():
    """测试项目结构"""
    print_colored("\n📂 项目结构检查:", "blue")

    # 检测当前是否在 multi_lakehouse_kb_builder 目录中
    current_dir = Path.cwd()
    if current_dir.name == "multi_lakehouse_kb_builder":
        # 在子目录中，检查当前目录的文件
        required_files = [
            "multi_lakehouse_kb_builder.py",
            "deploy_kb_simple.py",
            "validate_kb_simple.py",
            "check_connections.py",
            "run.sh",
            "test_kb_deployment.py"
        ]
        check_dir = current_dir
    else:
        # 在项目根目录，检查子目录
        required_files = [
            "multi_lakehouse_kb_builder/multi_lakehouse_kb_builder.py",
            "multi_lakehouse_kb_builder/deploy_kb_simple.py",
            "multi_lakehouse_kb_builder/validate_kb_simple.py",
            "multi_lakehouse_kb_builder/check_connections.py",
            "multi_lakehouse_kb_builder/run.sh",
            "multi_lakehouse_kb_builder/test_kb_deployment.py"
        ]
        check_dir = current_dir

    all_exist = True
    for file_name in required_files:
        file_path = check_dir / file_name
        if file_path.exists():
            print_colored(f"  ✅ {file_name}", "green")
        else:
            print_colored(f"  ❌ {file_name} (缺失)", "red")
            all_exist = False

    return all_exist

def main():
    """主测试函数"""
    print_colored("🚀 ClickZetta 知识库部署环境测试", "blue")
    print_colored("=" * 50, "blue")

    test_results = []

    # 执行各项测试
    test_python_environment()

    test_results.append(("依赖包", test_dependencies()))
    test_results.append(("配置文件", test_config_file()))
    test_results.append(("项目结构", test_project_structure()))

    test_environment_variables()
    test_results.append(("文档目录", test_document_directory()))

    # 汇总结果
    print_colored("\n📊 测试结果汇总:", "blue")
    print_colored("=" * 30, "blue")

    passed = 0
    total = len(test_results)

    for test_name, result in test_results:
        if result:
            print_colored(f"  ✅ {test_name}: 通过", "green")
            passed += 1
        else:
            print_colored(f"  ❌ {test_name}: 失败", "red")

    print_colored(f"\n总计: {passed}/{total} 项测试通过", "blue")

    if passed == total:
        print_colored("🎉 所有测试通过！环境配置正确，可以开始部署。", "green")
        return True
    else:
        print_colored("⚠️  部分测试失败，请检查上述问题后重试。", "yellow")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)