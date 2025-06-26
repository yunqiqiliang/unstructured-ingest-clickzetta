#!/bin/bash
# 使用当前激活的Python环境运行

# 设置颜色输出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# 设置工作目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# 检测Python环境
if [ -n "$VIRTUAL_ENV" ]; then
    echo -e "${GREEN}使用当前激活的虚拟环境: $(basename $VIRTUAL_ENV)${NC}"
    PYTHON="python"
elif command -v python3 &> /dev/null; then
    echo -e "${YELLOW}未检测到虚拟环境，使用系统Python3${NC}"
    PYTHON="python3"
else
    echo -e "${RED}错误: 未找到Python环境${NC}"
    exit 1
fi

# 验证Python版本
PYTHON_VERSION=$($PYTHON -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo -e "Python版本: $PYTHON_VERSION"

# 检查必要的包
echo -e "\n${YELLOW}检查依赖包...${NC}"
for package in clickzetta dashscope pandas; do
    if $PYTHON -c "import $package" 2>/dev/null; then
        echo -e "  ✅ $package"
    else
        echo -e "  ❌ $package (缺失)"
    fi
done

# 根据参数选择运行模式
case "$1" in
    "test")
        echo -e "\n${YELLOW}运行环境测试...${NC}"
        $PYTHON "$SCRIPT_DIR/test_kb_deployment.py"
        ;;
    "deploy")
        echo -e "\n${YELLOW}启动交互式部署...${NC}"
        $PYTHON "$SCRIPT_DIR/deploy_kb_simple.py"
        ;;
    "validate")
        echo -e "\n${YELLOW}验证知识库数据...${NC}"
        $PYTHON "$SCRIPT_DIR/validate_kb_simple.py"
        ;;
    "deploy-all")
        echo -e "\n${YELLOW}串行部署到所有Lakehouse...${NC}"
        $PYTHON "$SCRIPT_DIR/multi_lakehouse_kb_builder.py" --mode serial
        ;;
    *)
        echo -e "\n${GREEN}ClickZetta 知识库部署系统${NC}"
        echo "=================================="
        echo "用法: $0 [命令]"
        echo ""
        echo "可用命令:"
        echo "  test       - 测试环境配置"
        echo "  deploy     - 交互式部署"
        echo "  validate   - 验证数据"
        echo "  deploy-all - 部署到所有Lakehouse"
        echo ""
        echo "不带参数将显示此帮助信息"
        ;;
esac