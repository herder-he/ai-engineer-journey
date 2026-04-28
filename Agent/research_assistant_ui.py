"""Day 5: 调研助手 Web UI"""
import sys, os
import gradio as gr

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from FunctionCalling.research_assistant import run_agent

def query_handler(user_message, history):
    """接收用户输入,返回 Agent 回答"""
    if not user_message.strip():
        return "请输入调研问题"
    
    answer = run_agent(user_message, verbose=False)
    return answer if answer else "未能完成调研,请重试"


# 创建 UI
demo = gr.ChatInterface(
    fn=query_handler,
    title="技术调研助手",
    description="输入一个技术问题,我会自动搜索资料并生成结构化报告",
    examples=[
        "RAG 最近有什么新的优化方法?",
        "MCP 协议目前有哪些主流实现?",
        "2026 年 LLM 评估有什么新方法?",
    ],
)

if __name__ == "__main__":
    demo.launch()  # 浏览器自动打开 http://127.0.0.1:7860