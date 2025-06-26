# ClickZetta 多Lakehouse知识库部署系统 Makefile

# 检测Python路径：优先使用当前激活的环境，否则使用.venv
ifdef VIRTUAL_ENV
    PYTHON := python
else ifneq (,$(wildcard .venv/bin/python))
    PYTHON := .venv/bin/python
else ifneq (,$(wildcard ../venv/bin/python))
    PYTHON := ../venv/bin/python
else
    PYTHON := python
endif

UV := uv

.PHONY: help setup test deploy validate deploy-all clean

help: ## 显示帮助信息
	@echo "ClickZetta 知识库部署系统命令:"
	@echo ""
	@awk 'BEGIN {FS = ":.*##"; printf "Usage:\n  make \033[36m<target>\033[0m\n\nTargets:\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

setup: ## 初始化环境（仅首次需要）
	@echo "设置Python 3.11环境..."
	$(UV) python pin 3.11
	$(UV) sync
	@touch .venv/.uv-sync-marker
	@echo "环境设置完成！"

test: ## 测试环境配置
	@$(PYTHON) multi_lakehouse_kb_builder/test_kb_deployment.py

deploy: ## 交互式部署知识库
	@$(PYTHON) multi_lakehouse_kb_builder/deploy_kb_simple.py

validate: ## 验证已部署的知识库数据
	@$(PYTHON) multi_lakehouse_kb_builder/validate_kb_simple.py

deploy-all: ## 串行部署到所有Lakehouse
	@$(PYTHON) multi_lakehouse_kb_builder/multi_lakehouse_kb_builder.py --mode serial

deploy-parallel: ## 并行部署到所有Lakehouse（3个线程）
	@$(PYTHON) multi_lakehouse_kb_builder/multi_lakehouse_kb_builder.py --mode parallel --workers 3

validate-all: ## 验证所有Lakehouse的数据
	@$(PYTHON) multi_lakehouse_kb_builder/kb_data_validator.py

quick-test: ## 快速测试（只测试第一个连接）
	@$(PYTHON) multi_lakehouse_kb_builder/quick_test.py

clean: ## 清理临时文件和日志
	@echo "清理临时文件..."
	@find . -name "*.pyc" -delete
	@find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	@find . -name "kb_deployment_*.log" -delete
	@find . -name "kb_deployment_result_*.json" -delete
	@find . -name "kb_validation_report_*.json" -delete
	@echo "清理完成！"

update-deps: ## 更新依赖
	@echo "更新依赖..."
	$(UV) sync
	@touch .venv/.uv-sync-marker

.DEFAULT_GOAL := help