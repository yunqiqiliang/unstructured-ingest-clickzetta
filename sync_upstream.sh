#!/bin/bash

# =============================================================================
# 上游同步脚本
# =============================================================================
# 此脚本用于安全地与上游 unstructured/unstructured-ingest 仓库同步
# 同时保护本fork的特定配置不被覆盖

set -e  # 遇到错误时退出

echo "🔄 开始与上游仓库同步..."

# 检查是否在正确的目录
if [ ! -f "pyproject.toml" ]; then
    echo "❌ 错误：请在项目根目录运行此脚本"
    exit 1
fi

# 备份关键配置文件
echo "📦 备份Fork特定配置..."
BACKUP_DIR="$(mktemp -d)"
cp pyproject.toml "$BACKUP_DIR/"
cp env.example.txt "$BACKUP_DIR/" 2>/dev/null || echo "⚠️  env.example.txt 不存在"
cp .gitattributes "$BACKUP_DIR/" 2>/dev/null || echo "⚠️  .gitattributes 不存在"
cp README.md "$BACKUP_DIR/" 2>/dev/null || echo "⚠️  README.md 不存在"

echo "📁 备份目录: $BACKUP_DIR"

# 确保上游remote存在
if ! git remote | grep -q "upstream"; then
    echo "🔗 添加上游仓库..."
    git remote add upstream https://github.com/Unstructured-IO/unstructured-ingest.git
fi

# 获取上游更新
echo "⬇️  获取上游更新..."
git fetch upstream

# 合并上游主分支
echo "🔀 合并上游主分支..."
if ! git merge upstream/main --no-edit; then
    echo "❌ 合并冲突！请手动解决冲突后重新运行此脚本"
    echo "📁 备份文件位置: $BACKUP_DIR"
    exit 1
fi

# 恢复Fork特定配置
echo "🔧 恢复Fork特定配置..."

# 恢复备份的文件
cp "$BACKUP_DIR/pyproject.toml" ./ 2>/dev/null || echo "⚠️  无法恢复 pyproject.toml"
cp "$BACKUP_DIR/env.example.txt" ./ 2>/dev/null || echo "⚠️  无法恢复 env.example.txt"
cp "$BACKUP_DIR/.gitattributes" ./ 2>/dev/null || echo "⚠️  无法恢复 .gitattributes"
cp "$BACKUP_DIR/README.md" ./ 2>/dev/null || echo "⚠️  无法恢复 README.md"

# 运行配置脚本
if [ -f "setup_fork_config.py" ]; then
    echo "🔧 运行配置脚本..."
    python setup_fork_config.py
fi

# 检查是否有需要提交的更改
if ! git diff --quiet; then
    echo "📝 发现配置更改，创建提交..."
    git add .
    git commit -m "🔧 chore: 恢复fork特定配置 (上游同步后)"
    echo "✅ 配置恢复提交已创建"
else
    echo "ℹ️  无需额外提交"
fi

# 清理备份
rm -rf "$BACKUP_DIR"

echo "=" * 50
echo "🎉 上游同步完成！"
echo ""
echo "📋 后续步骤:"
echo "1. 检查代码是否正常工作"
echo "2. 运行测试: python -m pytest test/"
echo "3. 如需要，推送到远程: git push origin main"
echo ""