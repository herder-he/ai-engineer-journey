"""
Day 6: 用 LLM-as-Judge 评估调研助手的输出质量
"""
import os
import sys
import json
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI

# 把 FunctionCalling 加到 path,以便 import research_assistant
sys.path.append(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'FunctionCalling')
)
from research_assistant import run_agent

load_dotenv()
judge_client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)


# ========== 测试集:固定的"标准问题" ==========
TEST_CASES = [
    {
        "id": "rag_advances",
        "question": "RAG 最近有什么新的优化方法?",
        "expected_topics": ["Agentic RAG", "GraphRAG", "Rerank", "Hybrid Search", "多模态"],
        "expected_sources": True,  # 必须有参考来源
    },
    {
        "id": "mcp_implementations",
        "question": "MCP 协议目前有哪些主流实现?",
        "expected_topics": ["Anthropic", "Claude", "服务器", "工具"],
        "expected_sources": True,
    },
    {
        "id": "llm_eval_2026",
        "question": "2026 年 LLM 评估有什么新方法?",
        "expected_topics": ["benchmark", "评估", "自动化"],
        "expected_sources": True,
    },
    {
        "id": "agent_frameworks",
        "question": "目前主流的 Agent 框架有哪些,各自特点?",
        "expected_topics": ["LangChain", "LangGraph", "AutoGen", "CrewAI"],
        "expected_sources": True,
    },
    {
        "id": "vector_db",
        "question": "向量数据库的选型该怎么考虑?",
        "expected_topics": ["Pinecone", "Qdrant", "Chroma", "性能", "成本"],
        "expected_sources": True,
    },
]


# ========== Judge Prompt:让 LLM 当裁判 ==========
JUDGE_PROMPT = """你是一个严格的技术内容质量评审员。请基于以下维度评估"调研助手"的输出。

【用户问题】
{question}

【期望涉及的话题】
{expected_topics}

【调研助手的回答】
{answer}

请从以下 5 个维度打分(每个维度 0-10 分),并给出简短理由:

1. **结构性**:是否有清晰的核心要点 / 详细说明 / 参考来源结构
2. **完整性**:是否涵盖了期望的话题(允许有自己的扩展)
3. **准确性**:内容是否符合事实,有没有明显幻觉(凭你的知识判断)
4. **时效性**:是否提到了 2025-2026 的近期信息
5. **来源可信度**:是否提供了可验证的参考链接

输出严格的 JSON 格式,不要任何额外文字:
{{
    "structure_score": <0-10>,
    "structure_reason": "<简短理由>",
    "completeness_score": <0-10>,
    "completeness_reason": "<简短理由>",
    "accuracy_score": <0-10>,
    "accuracy_reason": "<简短理由>",
    "timeliness_score": <0-10>,
    "timeliness_reason": "<简短理由>",
    "source_score": <0-10>,
    "source_reason": "<简短理由>",
    "overall_score": <0-10>,
    "overall_comment": "<总评一句话>"
}}
"""


def judge(question: str, expected_topics: list, answer: str) -> dict:
    """让 LLM 评估一个回答"""
    prompt = JUDGE_PROMPT.format(
        question=question,
        expected_topics=", ".join(expected_topics),
        answer=answer,
    )
    
    response = judge_client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,  # 评估必须低温度,要稳定
    )
    
    content = response.choices[0].message.content.strip()
    # 清理可能的 markdown 包裹
    content = content.replace("```json", "").replace("```", "").strip()
    
    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        return {"error": f"JSON 解析失败: {e}", "raw": content}


def run_evaluation():
    """跑全套评估"""
    print(f"\n{'='*70}")
    print(f"调研助手自动评估 | {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print('='*70)
    
    results = []
    
    for i, case in enumerate(TEST_CASES, 1):
        print(f"\n[{i}/{len(TEST_CASES)}] 测试: {case['id']}")
        print(f"问题: {case['question']}")
        
        # 1. 跑 Agent 拿回答
        print("  → 调研助手正在工作...")
        try:
            answer = run_agent(case['question'], verbose=False)
            if not answer:
                print("  ✗ Agent 返回空")
                continue
        except Exception as e:
            print(f"  ✗ Agent 异常: {e}")
            continue
        
        # 2. 让 Judge LLM 评分
        print("  → 评审中...")
        judgment = judge(case['question'], case['expected_topics'], answer)
        
        if "error" in judgment:
            print(f"  ✗ 评审失败: {judgment['error']}")
            continue
        
        # 3. 打印结果
        print(f"  ✓ 总分: {judgment['overall_score']}/10")
        print(f"     结构: {judgment['structure_score']} | 完整: {judgment['completeness_score']} | 准确: {judgment['accuracy_score']}")
        print(f"     时效: {judgment['timeliness_score']} | 来源: {judgment['source_score']}")
        print(f"     总评: {judgment['overall_comment']}")
        
        results.append({
            "case_id": case['id'],
            "question": case['question'],
            "answer_preview": answer[:200],
            "judgment": judgment,
        })
    
    # ========== 汇总报告 ==========
    print(f"\n\n{'='*70}")
    print("评估汇总")
    print('='*70)
    
    if not results:
        print("无有效结果")
        return
    
    avg_score = sum(r['judgment']['overall_score'] for r in results) / len(results)
    print(f"\n平均分: {avg_score:.2f}/10")
    print(f"\n各维度平均分:")
    for dim in ['structure', 'completeness', 'accuracy', 'timeliness', 'source']:
        avg = sum(r['judgment'][f'{dim}_score'] for r in results) / len(results)
        print(f"  {dim:<15s}: {avg:.2f}")
    
    # 保存详细报告到文件
    report_path = f"eval_report_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "avg_score": avg_score,
            "results": results,
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n详细报告已保存: {report_path}")


if __name__ == "__main__":
    run_evaluation()