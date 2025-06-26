#!/bin/bash
# ClickZetta 知识库部署启动脚本

# 设置颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 检查uv是否安装
if ! command -v uv &> /dev/null; then
    echo -e "${RED}错误: uv 未安装${NC}"
    echo "请先安装uv: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# 设置工作目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR/.."

# 检查是否已有.venv环境
if [ -d ".venv" ]; then
    echo -e "${GREEN}检测到已存在的虚拟环境 .venv${NC}"
    
    # 检查是否需要同步（通过检查uv.lock文件的修改时间）
    if [ -f "uv.lock" ] && [ -f ".venv/.uv-sync-marker" ]; then
        if [ "uv.lock" -nt ".venv/.uv-sync-marker" ]; then
            echo -e "${YELLOW}检测到依赖更新，同步环境...${NC}"
            uv sync
            touch ".venv/.uv-sync-marker"
        else
            echo -e "${GREEN}环境已是最新，跳过同步${NC}"
        fi
    else
        echo -e "${YELLOW}首次运行，同步环境...${NC}"
        uv sync
        touch ".venv/.uv-sync-marker"
    fi
else
    echo -e "${YELLOW}未检测到虚拟环境，创建并同步...${NC}"
    
    # 设置Python版本
    echo -e "${YELLOW}设置Python版本为3.11...${NC}"
    uv python pin 3.11
    
    # 同步环境
    uv sync
    touch ".venv/.uv-sync-marker"
fi

# 激活虚拟环境（可选，uv run会自动使用正确的环境）
echo -e "${GREEN}使用虚拟环境: .venv${NC}"

# 显示菜单
echo -e "\n${GREEN}ClickZetta 知识库批量部署系统${NC}"
echo "=================================="
echo "部署操作:"
echo "  1. 测试环境配置"
echo "  2. 简单部署（交互式）"
echo "  3. 串行部署到所有Lakehouse"
echo "  4. 并行部署到所有Lakehouse"
echo "  5. 部署到特定Lakehouse"
echo ""
echo "验证操作:"
echo "  6. 验证已部署的知识库数据"
echo "  7. 独立运行数据验证器"
echo ""
echo "检查操作:"
echo "  8. 检查连接和知识库状态"
echo ""
echo "知识库管理:"
echo "  9. 管理知识库内容（增/删/查）"
echo ""
echo "其他:"
echo "  10. 高级选项（命令行参数）"
echo "  0. 退出"
echo ""

read -p "请选择操作 [0-10]: " choice

case $choice in
    1)
        echo -e "\n${YELLOW}运行环境测试...${NC}"
        uv run python multi_lakehouse_kb_builder/test_kb_deployment.py
        ;;
    2)
        echo -e "\n${YELLOW}启动交互式部署...${NC}"
        uv run python multi_lakehouse_kb_builder/deploy_kb_simple.py
        ;;
    3)
        echo -e "\n${YELLOW}串行部署到所有Lakehouse...${NC}"
        uv run python multi_lakehouse_kb_builder/multi_lakehouse_kb_builder.py --mode serial
        ;;
    4)
        echo -e "\n${YELLOW}并行部署到所有Lakehouse...${NC}"
        read -p "请输入并行工作线程数 [默认3]: " workers
        workers=${workers:-3}
        uv run python multi_lakehouse_kb_builder/multi_lakehouse_kb_builder.py --mode parallel --workers $workers
        ;;
    5)
        echo -e "\n${YELLOW}部署到特定Lakehouse...${NC}"
        read -p "请输入连接名称匹配模式: " pattern
        if [ -n "$pattern" ]; then
            uv run python multi_lakehouse_kb_builder/multi_lakehouse_kb_builder.py --filter "$pattern"
        else
            echo -e "${RED}错误: 未提供匹配模式${NC}"
        fi
        ;;
    6)
        echo -e "\n${YELLOW}验证已部署的知识库数据...${NC}"
        uv run python multi_lakehouse_kb_builder/validate_kb_simple.py
        ;;
    7)
        echo -e "\n${YELLOW}独立运行数据验证器...${NC}"
        echo "用法: uv run python multi_lakehouse_kb_builder/kb_data_validator.py [选项]"
        echo ""
        echo "选项:"
        echo "  --filter PATTERN  只验证包含此模式的连接"
        echo "  --exclude PATTERN 排除包含此模式的连接"
        echo "  --output FILE     指定输出文件路径"
        echo ""
        read -p "输入验证选项（直接回车验证所有）: " options
        uv run python multi_lakehouse_kb_builder/kb_data_validator.py $options
        ;;
    8)
        echo -e "\n${YELLOW}检查连接和知识库状态...${NC}"
        uv run python multi_lakehouse_kb_builder/check_connections.py
        ;;
    9)
        echo -e "\n${YELLOW}启动知识库管理工具...${NC}"
        uv run python multi_lakehouse_kb_builder/manage_knowledge_simple.py
        ;;
    10)
        echo -e "\n${YELLOW}高级选项 - 命令行参数说明:${NC}"
        echo ""
        echo "uv run python multi_lakehouse_kb_builder/multi_lakehouse_kb_builder.py [选项]"
        echo ""
        echo "选项:"
        echo "  --config PATH     连接配置文件路径 (默认: ~/.clickzetta/connections.json)"
        echo "  --docs PATH       文档目录路径 (默认: /Users/liangmo/yunqidoc/cn_markdown_20250526)"
        echo "  --mode MODE       执行模式: serial 或 parallel (默认: serial)"
        echo "  --filter PATTERN  只部署包含此模式的连接"
        echo "  --exclude PATTERN 排除包含此模式的连接"
        echo "  --workers N       并行模式下的工作线程数 (默认: 5)"
        echo ""
        echo "示例:"
        echo "  uv run python multi_lakehouse_kb_builder/multi_lakehouse_kb_builder.py --mode parallel --workers 3"
        echo "  uv run python multi_lakehouse_kb_builder/multi_lakehouse_kb_builder.py --filter production --exclude test"
        echo ""
        read -p "按回车键返回主菜单..."
        exec "$0"
        ;;
    0)
        echo -e "${GREEN}退出程序${NC}"
        exit 0
        ;;
    *)
        echo -e "${RED}无效的选择${NC}"
        exit 1
        ;;
esac