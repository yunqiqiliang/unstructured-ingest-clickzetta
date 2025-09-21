#!/bin/bash
# ClickZetta 知识库部署智能启动脚本
# 整合了原有的 run.sh, run_quick.sh, run_with_current_env.sh 功能

# 设置颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 设置工作目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR/.."

# 智能环境检测和选择
detect_environment() {
    echo -e "${BLUE}🔍 检测Python环境...${NC}"

    # 方案1: 检查当前激活的虚拟环境
    if [ -n "$VIRTUAL_ENV" ]; then
        ENV_TYPE="current"
        PYTHON_CMD="python"
        echo -e "${GREEN}✅ 检测到已激活的虚拟环境: $(basename $VIRTUAL_ENV)${NC}"
        return 0
    fi

    # 方案2: 检查项目本地 .venv 环境
    if [ -d ".venv" ]; then
        ENV_TYPE="local_venv"
        PYTHON_CMD=".venv/bin/python"
        echo -e "${GREEN}✅ 检测到本地虚拟环境: .venv${NC}"
        return 0
    fi

    # 方案3: 检查是否有 uv 可以管理环境
    if command -v uv &> /dev/null; then
        ENV_TYPE="uv_managed"
        echo -e "${GREEN}✅ 检测到 uv，将自动管理环境${NC}"
        return 0
    fi

    # 方案4: 使用系统 Python
    if command -v python3 &> /dev/null; then
        ENV_TYPE="system_python"
        PYTHON_CMD="python3"
        echo -e "${YELLOW}⚠️  使用系统 Python3（不推荐用于生产）${NC}"
        return 0
    fi

    echo -e "${RED}❌ 未找到可用的Python环境${NC}"
    return 1
}

# 环境初始化
init_environment() {
    case "$ENV_TYPE" in
        "current")
            echo -e "${GREEN}使用当前激活环境${NC}"
            ;;
        "local_venv")
            echo -e "${GREEN}使用本地 .venv 环境${NC}"
            ;;
        "uv_managed")
            echo -e "${YELLOW}初始化 uv 管理的环境...${NC}"
            if [ ! -d ".venv" ]; then
                echo -e "${YELLOW}设置Python版本为3.11...${NC}"
                uv python pin 3.11
                uv sync
                touch ".venv/.uv-sync-marker"
            else
                # 检查是否需要同步
                if [ -f "uv.lock" ] && [ -f ".venv/.uv-sync-marker" ]; then
                    if [ "uv.lock" -nt ".venv/.uv-sync-marker" ]; then
                        echo -e "${YELLOW}检测到依赖更新，同步环境...${NC}"
                        uv sync
                        touch ".venv/.uv-sync-marker"
                    fi
                else
                    uv sync
                    touch ".venv/.uv-sync-marker"
                fi
            fi
            PYTHON_CMD="uv run python"
            ;;
        "system_python")
            echo -e "${YELLOW}使用系统Python3${NC}"
            ;;
    esac
}

# 检查依赖
check_dependencies() {
    echo -e "\n${BLUE}📦 检查依赖包...${NC}"
    local missing_deps=()

    for package in clickzetta dashscope pandas; do
        if [ "$ENV_TYPE" = "uv_managed" ]; then
            if uv run python -c "import $package" 2>/dev/null; then
                echo -e "  ✅ $package"
            else
                echo -e "  ❌ $package (缺失)"
                missing_deps+=("$package")
            fi
        else
            if $PYTHON_CMD -c "import $package" 2>/dev/null; then
                echo -e "  ✅ $package"
            else
                echo -e "  ❌ $package (缺失)"
                missing_deps+=("$package")
            fi
        fi
    done

    if [ ${#missing_deps[@]} -gt 0 ]; then
        echo -e "\n${RED}缺少依赖包: ${missing_deps[*]}${NC}"
        echo -e "${YELLOW}建议运行: pip install ${missing_deps[*]}${NC}"
        return 1
    fi

    return 0
}

# 执行命令的包装函数
run_python() {
    local script="$1"
    shift  # 移除第一个参数，剩下的是脚本参数

    case "$ENV_TYPE" in
        "uv_managed")
            uv run python "$script" "$@"
            ;;
        *)
            $PYTHON_CMD "$script" "$@"
            ;;
    esac
}

# 显示交互式菜单
show_menu() {
    echo -e "\n${GREEN}ClickZetta 知识库批量部署系统${NC}"
    echo -e "${BLUE}环境: $ENV_TYPE${NC}"
    echo "=================================="
    echo "🚀 部署操作:"
    echo "  1. 测试环境配置"
    echo "  2. 交互式部署"
    echo "  3. 串行部署到所有Lakehouse"
    echo "  4. 并行部署到所有Lakehouse"
    echo "  5. 部署到特定Lakehouse"
    echo ""
    echo "🔍 验证操作:"
    echo "  6. 验证已部署的知识库"
    echo "  7. 运行数据验证器"
    echo ""
    echo "🏥 检查操作:"
    echo "  8. 检查连接和知识库状态"
    echo ""
    echo "📚 知识库管理:"
    echo "  9. 管理知识库内容"
    echo ""
    echo "⚙️  其他:"
    echo "  h. 显示帮助信息"
    echo "  0. 退出"
    echo ""
}

# 处理命令行参数模式
handle_command_mode() {
    local command="$1"

    case "$command" in
        "test")
            echo -e "\n${YELLOW}运行环境测试...${NC}"
            run_python "multi_lakehouse_kb_builder/test_kb_deployment.py"
            ;;
        "deploy")
            echo -e "\n${YELLOW}启动交互式部署...${NC}"
            run_python "multi_lakehouse_kb_builder/deploy_kb_simple.py"
            ;;
        "validate")
            echo -e "\n${YELLOW}验证知识库数据...${NC}"
            run_python "multi_lakehouse_kb_builder/validate_kb_simple.py"
            ;;
        "deploy-all")
            echo -e "\n${YELLOW}串行部署到所有Lakehouse...${NC}"
            run_python "multi_lakehouse_kb_builder/multi_lakehouse_kb_builder.py" --mode serial
            ;;
        "deploy-parallel")
            echo -e "\n${YELLOW}并行部署到所有Lakehouse...${NC}"
            run_python "multi_lakehouse_kb_builder/multi_lakehouse_kb_builder.py" --mode parallel
            ;;
        "check")
            echo -e "\n${YELLOW}检查连接和知识库状态...${NC}"
            run_python "multi_lakehouse_kb_builder/check_connections.py"
            ;;
        "manage")
            echo -e "\n${YELLOW}启动知识库管理工具...${NC}"
            run_python "multi_lakehouse_kb_builder/manage_knowledge_simple.py"
            ;;
        "help"|"-h"|"--help")
            show_help
            ;;
        *)
            echo -e "${RED}未知命令: $command${NC}"
            show_help
            return 1
            ;;
    esac
}

# 处理交互式菜单
handle_interactive_mode() {
    while true; do
        show_menu
        read -p "请选择操作 [0-9, h]: " choice

        case $choice in
            1)
                echo -e "\n${YELLOW}运行环境测试...${NC}"
                run_python "multi_lakehouse_kb_builder/test_kb_deployment.py"
                ;;
            2)
                echo -e "\n${YELLOW}启动交互式部署...${NC}"
                run_python "multi_lakehouse_kb_builder/deploy_kb_simple.py"
                ;;
            3)
                echo -e "\n${YELLOW}串行部署到所有Lakehouse...${NC}"
                run_python "multi_lakehouse_kb_builder/multi_lakehouse_kb_builder.py" --mode serial
                ;;
            4)
                echo -e "\n${YELLOW}并行部署到所有Lakehouse...${NC}"
                read -p "请输入并行工作线程数 [默认3]: " workers
                workers=${workers:-3}
                run_python "multi_lakehouse_kb_builder/multi_lakehouse_kb_builder.py" --mode parallel --workers $workers
                ;;
            5)
                echo -e "\n${YELLOW}部署到特定Lakehouse...${NC}"
                read -p "请输入连接名称匹配模式: " pattern
                if [ -n "$pattern" ]; then
                    run_python "multi_lakehouse_kb_builder/multi_lakehouse_kb_builder.py" --filter "$pattern"
                else
                    echo -e "${RED}错误: 未提供匹配模式${NC}"
                fi
                ;;
            6)
                echo -e "\n${YELLOW}验证已部署的知识库数据...${NC}"
                run_python "multi_lakehouse_kb_builder/validate_kb_simple.py"
                ;;
            7)
                echo -e "\n${YELLOW}运行数据验证器...${NC}"
                read -p "输入验证选项（直接回车验证所有）: " options
                run_python "multi_lakehouse_kb_builder/kb_data_validator.py" $options
                ;;
            8)
                echo -e "\n${YELLOW}检查连接和知识库状态...${NC}"
                run_python "multi_lakehouse_kb_builder/check_connections.py"
                ;;
            9)
                echo -e "\n${YELLOW}启动知识库管理工具...${NC}"
                run_python "multi_lakehouse_kb_builder/manage_knowledge_simple.py"
                ;;
            h)
                show_help
                ;;
            0)
                echo -e "${GREEN}退出程序${NC}"
                exit 0
                ;;
            *)
                echo -e "${RED}无效的选择${NC}"
                read -p "按回车键继续..."
                ;;
        esac

        echo ""
        read -p "按回车键继续..."
    done
}

# 显示帮助信息
show_help() {
    echo -e "\n${GREEN}ClickZetta 知识库部署系统 - 帮助${NC}"
    echo "=================================="
    echo ""
    echo "用法:"
    echo "  $0                    # 启动交互式菜单"
    echo "  $0 [命令]             # 直接执行命令"
    echo ""
    echo "可用命令:"
    echo "  test                  # 测试环境配置"
    echo "  deploy                # 交互式部署"
    echo "  validate              # 验证知识库数据"
    echo "  deploy-all            # 串行部署到所有Lakehouse"
    echo "  deploy-parallel       # 并行部署到所有Lakehouse"
    echo "  check                 # 检查连接和知识库状态"
    echo "  manage                # 管理知识库内容"
    echo "  help                  # 显示此帮助信息"
    echo ""
    echo "环境说明:"
    echo "  脚本会自动检测并使用最适合的Python环境："
    echo "  1. 当前激活的虚拟环境 (VIRTUAL_ENV)"
    echo "  2. 项目本地 .venv 环境"
    echo "  3. uv 管理的环境"
    echo "  4. 系统 Python3 (不推荐)"
    echo ""
    echo "示例:"
    echo "  $0 test               # 测试环境"
    echo "  $0 deploy             # 交互式部署"
    echo "  $0 deploy-all         # 部署到所有实例"
}

# 主函数
main() {
    echo -e "${GREEN}🚀 ClickZetta 知识库部署系统${NC}"
    echo -e "${BLUE}智能启动脚本 v2.0${NC}"
    echo ""

    # 检测环境
    if ! detect_environment; then
        echo -e "${RED}环境检测失败，请检查Python安装${NC}"
        exit 1
    fi

    # 初始化环境
    init_environment

    # 检查依赖
    if ! check_dependencies; then
        echo -e "\n${YELLOW}依赖检查失败，但可以继续运行（某些功能可能不可用）${NC}"
        read -p "是否继续？ [y/N]: " continue_anyway
        if [[ ! "$continue_anyway" =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi

    # 根据参数决定运行模式
    if [ $# -eq 0 ]; then
        # 无参数，启动交互式模式
        handle_interactive_mode
    else
        # 有参数，命令行模式
        handle_command_mode "$1"
    fi
}

# 运行主函数
main "$@"