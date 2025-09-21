#!/bin/bash
# 迁移脚本：清理冗余的启动脚本，使用新的智能启动器

# 设置颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

echo -e "${BLUE}🔧 ClickZetta 知识库部署系统 - 脚本整合迁移${NC}"
echo "=================================================="
echo ""

# 显示当前脚本状态
echo -e "${YELLOW}📊 当前启动脚本状态：${NC}"
echo ""

scripts=("run.sh" "run_quick.sh" "run_with_current_env.sh" "run_direct.py" "run_smart.sh")
for script in "${scripts[@]}"; do
    if [ -f "$script" ]; then
        size=$(stat -f%z "$script" 2>/dev/null || stat -c%s "$script" 2>/dev/null)
        echo -e "  ✅ $script (${size} bytes)"
    else
        echo -e "  ❌ $script (不存在)"
    fi
done

echo ""
echo -e "${YELLOW}📋 整合分析：${NC}"
echo "  • run_smart.sh    - 新的智能整合脚本 ⭐"
echo "  • run_direct.py   - 保留，命令行接口"
echo "  • run.sh          - 可以替换 (功能重叠 90%)"
echo "  • run_quick.sh    - 可以移除 (功能重叠 80%)"
echo "  • run_with_current_env.sh - 可以移除 (功能重叠 75%)"

echo ""
read -p "是否开始迁移？这将备份旧脚本并替换为新的智能脚本 [y/N]: " confirm

if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}迁移已取消${NC}"
    exit 0
fi

echo ""
echo -e "${BLUE}🗄️  开始迁移过程...${NC}"

# 创建备份目录
backup_dir="backup_scripts_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$backup_dir"
echo -e "${GREEN}创建备份目录: $backup_dir${NC}"

# 备份现有脚本
echo -e "\n${YELLOW}📦 备份现有脚本...${NC}"
for script in "run.sh" "run_quick.sh" "run_with_current_env.sh"; do
    if [ -f "$script" ]; then
        cp "$script" "$backup_dir/"
        echo -e "  ✅ 备份 $script"
    fi
done

# 替换主启动脚本
echo -e "\n${YELLOW}🔄 替换主启动脚本...${NC}"
if [ -f "run.sh" ]; then
    mv "run.sh" "$backup_dir/run.sh.backup"
    echo -e "  📁 原 run.sh 移动到备份目录"
fi

cp "run_smart.sh" "run.sh"
chmod +x "run.sh"
echo -e "  ✅ run_smart.sh 复制为新的 run.sh"

# 移除冗余脚本
echo -e "\n${YELLOW}🗑️  移除冗余脚本...${NC}"
for script in "run_quick.sh" "run_with_current_env.sh"; do
    if [ -f "$script" ]; then
        rm "$script"
        echo -e "  ✅ 移除 $script"
    fi
done

# 保留 run_direct.py (不冗余，有独特用途)
echo -e "  ℹ️  保留 run_direct.py (命令行接口)"

# 更新权限
echo -e "\n${YELLOW}🔐 更新文件权限...${NC}"
chmod +x run.sh
chmod +x run_direct.py

# 创建使用说明
cat > "SCRIPT_MIGRATION_NOTES.md" << 'EOF'
# 脚本迁移说明

## 迁移时间
$(date)

## 变更内容

### ✅ 新的启动脚本
- **`run.sh`** - 智能启动脚本，自动检测和适配环境
- **`run_direct.py`** - 命令行接口，适合脚本调用

### 📁 备份文件
旧的脚本已备份到：`backup_scripts_*/`

### ❌ 移除的冗余脚本
- `run_quick.sh` - 功能已整合到 `run.sh`
- `run_with_current_env.sh` - 功能已整合到 `run.sh`

## 新脚本功能

### 智能环境检测
新的 `run.sh` 自动检测以下环境（按优先级）：
1. 当前激活的虚拟环境
2. 项目本地 .venv 环境
3. uv 管理的环境
4. 系统 Python3

### 使用方法
```bash
# 交互式菜单
./run.sh

# 直接命令
./run.sh test
./run.sh deploy
./run.sh validate
./run.sh deploy-all
./run.sh check
./run.sh manage

# 命令行接口
python run_direct.py deploy
python run_direct.py check-conn
python run_direct.py manage-kb
```

## 回滚方法
如需回滚到旧版本：
```bash
cp backup_scripts_*/run.sh.backup run.sh
cp backup_scripts_*/run_quick.sh .
cp backup_scripts_*/run_with_current_env.sh .
```
EOF

echo -e "\n${GREEN}✅ 迁移完成！${NC}"
echo ""
echo -e "${BLUE}📋 迁移总结：${NC}"
echo -e "  ✅ 备份目录: $backup_dir"
echo -e "  ✅ 新启动脚本: run.sh (智能版本)"
echo -e "  ✅ 保留脚本: run_direct.py"
echo -e "  ✅ 迁移说明: SCRIPT_MIGRATION_NOTES.md"
echo ""
echo -e "${YELLOW}📖 使用新脚本：${NC}"
echo -e "  ./run.sh          # 交互式菜单"
echo -e "  ./run.sh deploy   # 直接部署"
echo -e "  ./run.sh help     # 查看帮助"
echo ""
echo -e "${GREEN}🎉 享受更简洁的启动体验！${NC}"