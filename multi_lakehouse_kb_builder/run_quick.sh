#!/bin/bash
# 快速启动脚本 - 直接使用现有的.venv环境

# 设置颜色输出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# 设置工作目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR/.."

# 检查.venv是否存在
if [ ! -d ".venv" ]; then
    echo -e "${RED}错误: 未找到虚拟环境 .venv${NC}"
    echo "请先运行完整的 run.sh 初始化环境"
    exit 1
fi

# 直接使用Python运行（假设.venv已经设置好）
echo -e "${GREEN}使用现有虚拟环境 .venv${NC}"

# 根据参数选择运行模式
case "$1" in
    "test")
        echo -e "${YELLOW}运行环境测试...${NC}"
        .venv/bin/python multi_lakehouse_kb_builder/test_kb_deployment.py
        ;;
    "deploy")
        echo -e "${YELLOW}启动交互式部署...${NC}"
        .venv/bin/python multi_lakehouse_kb_builder/deploy_kb_simple.py
        ;;
    "validate")
        echo -e "${YELLOW}验证知识库数据...${NC}"
        .venv/bin/python multi_lakehouse_kb_builder/validate_kb_simple.py
        ;;
    "deploy-all")
        echo -e "${YELLOW}串行部署到所有Lakehouse...${NC}"
        .venv/bin/python multi_lakehouse_kb_builder/multi_lakehouse_kb_builder.py --mode serial
        ;;
    *)
        # 默认运行交互式菜单
        .venv/bin/python multi_lakehouse_kb_builder/deploy_kb_simple.py
        ;;
esac