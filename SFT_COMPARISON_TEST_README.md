# SFT Comparison Test Script

这个增强版的SFT测试脚本不仅测试微调模型生成的rules，还会对比测试不带rules的baseline性能，让你能够评估rules是否真正提升了SQL生成的性能。

## 🆕 新功能

### 双重测试模式
脚本现在会为每个测试样本执行两种测试：

1. **With Rules (带规则)**: 使用SFT模型生成的rules + DeepSeek生成SQL
2. **Without Rules (无规则)**: 仅使用DeepSeek生成SQL（baseline）

### 性能对比分析
- 实时显示两种方式的准确率
- 计算rules带来的性能提升
- 详细的统计报告

## 🔄 测试流程

```
对于每个测试样本:
├── 1. 加载数据库schema
├── 2. 使用SFT模型生成rules
├── 3A. 使用DeepSeek + rules生成SQL
├── 3B. 使用DeepSeek（无rules）生成SQL
├── 4A. 验证带rules的SQL
├── 4B. 验证无rules的SQL
└── 5. 对比两种方式的性能
```

## 📊 输出结果

### 实时显示
```
📊 Current Accuracies:
  With rules: 0.800 (4/5)
  Without rules: 0.600 (3/5)
  Improvement: +0.200
```

### 最终统计
```
📊 FINAL RESULTS:
  With Rules:
    Correct Answers: 80
    Failed Executions: 5
    Accuracy: 0.800 (80.0%)

  Without Rules:
    Correct Answers: 60
    Failed Executions: 8
    Accuracy: 0.600 (60.0%)

🎯 IMPROVEMENT:
    Rules provide +0.200 (+20.0%) accuracy improvement
    ✅ Rules help improve performance!
```

### JSON输出格式
```json
{
  "test_config": { ... },
  "statistics": {
    "total_questions": 100,
    "with_rules": {
      "correct_answers": 80,
      "failed_executions": 5,
      "accuracy": 0.800
    },
    "without_rules": {
      "correct_answers": 60,
      "failed_executions": 8,
      "accuracy": 0.600
    },
    "improvement": {
      "absolute": 0.200,
      "relative": 20.0,
      "description": "Rules provide +0.200 (+20.0%) accuracy improvement"
    }
  },
  "results": [
    {
      "sample_idx": 0,
      "question": "...",
      "ground_truth": "...",
      "db_id": "...",
      "generated_rules": "...",
      "sql_with_rules": "...",
      "sql_without_rules": "...",
      "result_with_rules": 1,
      "result_without_rules": 0,
      "error_message": ""
    }
  ]
}
```

## 🚀 使用方法

### 快速测试（推荐）
```bash
export DEEPSEEK_API_KEY='your_api_key_here'
./quick_comparison_test.sh
```

### 完整测试
```bash
export DEEPSEEK_API_KEY='your_api_key_here'
./run_sft_test.sh
```

### 自定义测试
```bash
python test_sft_results_api.py \
  --base_model /path/to/base/model \
  --adapter_path /path/to/adapter \
  --dataset /path/to/dataset.json \
  --db_root_path /path/to/databases \
  --num_samples 50 \
  --output_file comparison_results.json
```

## 📈 性能分析

### 关键指标
- **准确率对比**: 带rules vs 无rules的准确率差异
- **改进幅度**: 绝对和相对的性能提升
- **错误分析**: 两种方式的失败原因对比

### 结果解释
- **正改进**: rules帮助提升性能 ✅
- **负改进**: rules反而降低性能 ❌
- **无影响**: rules对性能无影响 ➖

## 🔧 配置选项

### 生成参数
- `--max_new_tokens`: rules生成的最大token数
- `--temperature`: 生成温度
- `--do_sample`: 是否启用采样

### 测试参数
- `--num_samples`: 测试样本数量
- `--output_file`: 结果输出文件
- `--deepseek_model`: DeepSeek模型选择

## 📋 文件说明

- `test_sft_results_api.py`: 主要的对比测试脚本
- `run_sft_test.sh`: 完整测试脚本（100个样本）
- `quick_comparison_test.sh`: 快速测试脚本（5个样本）
- `SFT_COMPARISON_TEST_README.md`: 本说明文档

## 🎯 使用场景

1. **模型评估**: 评估SFT模型生成的rules是否有效
2. **性能对比**: 量化rules对SQL生成性能的影响
3. **A/B测试**: 对比不同prompt策略的效果
4. **优化指导**: 为模型改进提供数据支持

## 💡 最佳实践

1. **先运行快速测试**: 使用5个样本快速验证设置
2. **逐步增加样本**: 从少量样本开始，逐步增加到完整测试
3. **分析失败案例**: 关注两种方式都失败或表现差异大的案例
4. **记录配置**: 保存测试配置以便复现结果

## 🔍 故障排除

1. **API连接问题**: 检查DEEPSEEK_API_KEY设置
2. **模型加载失败**: 验证模型路径和权限
3. **内存不足**: 减少num_samples或使用更小的模型
4. **结果异常**: 检查数据库文件和schema生成

这个增强版脚本让你能够全面评估SFT模型的效果，不仅看生成的rules质量，更重要的是验证这些rules是否真正提升了SQL生成的性能！
