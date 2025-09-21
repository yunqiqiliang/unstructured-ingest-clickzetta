# Fork维护指南

本文档描述如何维护此 `unstructured-ingest-clickzetta` fork，特别是如何与上游仓库同步而不丢失定制化配置。

## 🎯 Fork特性

此fork包含以下定制化内容：

### 📦 包配置差异
- **包名**: `unstructured-ingest-clickzetta` (而非上游的 `unstructured_ingest`)
- **作者**: ClickZetta Community (而非 Unstructured Technologies)
- **描述**: ClickZetta专用连接器描述
- **CLI命令**: 添加了 `unstructured-ingest-clickzetta` 别名

### 🔧 新增文件
- `.env.example` - 环境变量配置模板
- `.gitattributes` - Git合并策略配置
- `setup_fork_config.py` - 配置恢复脚本
- `sync_upstream.sh` - 上游同步脚本
- `FORK_MAINTENANCE.md` - 本文档

## 🔄 与上游同步

### 自动同步（推荐）

使用提供的自动化脚本：

```bash
# 运行自动同步脚本
./sync_upstream.sh
```

此脚本会：
1. 备份fork特定配置
2. 获取并合并上游更新
3. 恢复fork配置
4. 自动提交更改

### 手动同步

如果需要手动操作：

```bash
# 1. 添加上游仓库（首次）
git remote add upstream https://github.com/Unstructured-IO/unstructured-ingest.git

# 2. 获取上游更新
git fetch upstream

# 3. 备份重要文件
cp pyproject.toml pyproject.toml.backup
cp .env.example .env.example.backup

# 4. 合并上游主分支
git merge upstream/main

# 5. 恢复配置
python setup_fork_config.py

# 6. 提交更改
git add .
git commit -m "🔧 chore: 与上游同步并恢复fork配置"
```

## 🛡️ 保护机制

### .gitattributes 保护

`.gitattributes` 文件配置了关键文件的合并策略：

```
pyproject.toml merge=ours        # 优先使用我们的版本
.env.example merge=ours          # 保护环境配置
README.md merge=ours             # 保护文档
```

### 配置恢复脚本

`setup_fork_config.py` 可以在同步后恢复正确的配置：

```bash
python setup_fork_config.py
```

## 🔍 验证同步结果

同步后检查以下项目：

```bash
# 1. 检查包名是否正确
grep 'name = "unstructured-ingest-clickzetta"' pyproject.toml

# 2. 检查作者信息
grep 'ClickZetta Community' pyproject.toml

# 3. 运行测试
python -m pytest test/integration/connectors/sql/test_clickzetta.py -v
python -m pytest test/integration/connectors/fsspec/test_clickzetta_volume.py -v

# 4. 验证CLI命令
python -c "import unstructured_ingest; print('✅ 导入成功')"
```

## ⚠️ 注意事项

1. **不要直接编辑 pyproject.toml**: 使用 `setup_fork_config.py` 脚本确保一致性
2. **同步前备份**: 重要配置文件在同步前务必备份
3. **测试验证**: 同步后务必运行完整测试套件
4. **版本管理**: 考虑在同步后打tag记录版本

## 🚀 发布流程

1. 确保与上游同步
2. 运行完整测试套件
3. 更新版本号
4. 构建并发布包

```bash
# 构建包
python -m build

# 发布到PyPI
python -m twine upload dist/*
```

## 📞 支持

如有问题，请：
1. 检查此文档
2. 运行 `python setup_fork_config.py` 恢复配置
3. 查看 GitHub Issues
4. 联系维护者