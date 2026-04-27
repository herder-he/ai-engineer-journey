# Function Calling 实践

通过 Day 3-4 完成的 Function Calling 系列项目。

## 项目列表

### tool_basic.py
最简单的单工具调用,演示 Function Calling 基本机制。

### tool_multi.py  
多工具自主选择,LLM 根据问题决定调用哪个工具。

### agent_loop.py
循环式 Agent,支持多步推理(LLM 可以多次调用工具直到任务完成)。

### agent_with_search.py
接入 Tavily 真实搜索 API,加错误处理和工具熔断。

### research_assistant.py(主要项目)
**技术调研助手** —— 输入一个技术问题,自动搜索、综合、生成结构化报告。

使用示例:
\```python
run_agent("RAG 最近有什么新的优化方法?")
\```

## 我学到的核心机制

1. **Function Calling 的本质**:LLM 输出"调用意图",代码执行,结果再喂回 LLM
2. **Agent 循环**:决策 → 执行 → 反馈 → 再决策,直到 LLM 不调工具
3. **工具描述是关键**:用途、时机、参数格式三要素
4. **必备的兜底**:max_iterations、错误处理、连续失败熔断

## 踩过的坑
1. 对于最新当前时间的理解可能会当成模型的截止更新时间
  