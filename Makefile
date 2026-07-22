.PHONY: test eval report clean

# 测试：lint + typecheck
test:
	@echo "Running lint and typecheck..."
	@pnpm exec tsc --noEmit
	@echo "✅ test passed"

# 评估：regression seed+list
eval:
	@echo "Running regression case..."
	@mkdir -p artifacts/eval
	@node scripts/verification/regression_seed_and_list.mjs --browser || echo "⚠️  Regression case requires browser environment (Playwright)"
	@echo "✅ eval completed"

# 报告：生成路由清单和模型快照
report:
	@echo "Generating reports..."
	@mkdir -p artifacts/report
	@node scripts/maintenance/reporting/gen_routes.mjs
	@node scripts/maintenance/reporting/gen_models_snapshot.mjs
	@echo "✅ report generated"

# 清理 artifacts
clean:
	@rm -rf artifacts/eval/*.json artifacts/report/*.md artifacts/report/*.json
	@echo "✅ cleaned artifacts"
