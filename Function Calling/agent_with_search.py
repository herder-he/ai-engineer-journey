"""
Day 4: 给 Agent 加一个真实的 web_search 工具
"""
import os
import json
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
from tavily import TavilyClient

load_dotenv()

llm_client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


# ========== 工具实现 ==========
def get_current_time():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def calculate(expression: str):
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"计算错误: {e}"


def web_search(query: str, max_results: int = 5):
    """
    用 Tavily 搜索网络,返回精简的结果
    """
    try:
        response = tavily_client.search(
            query=query,
            max_results=max_results,
            search_depth="basic",  # basic 比 advanced 便宜
        )
        
        # 提取关键信息,不要把完整 JSON 塞给 LLM
        results = []
        for item in response.get("results", []):
            results.append({
                "title": item.get("title", ""),
                "content": item.get("content", "")[:300],  # 截断,避免 context 太长
                "url": item.get("url", ""),
            })
        
        return json.dumps(results, ensure_ascii=False)
    except Exception as e:
        return f"搜索失败: {e}"


# ========== 工具描述 ==========
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "获取当前的日期和时间。用于回答'现在几点'、'今天几号'等时间相关问题、用于关于今天和当前最新的时间获取。",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "计算数学表达式。仅用于纯数学计算,不要传入自然语言描述。",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "数学表达式,例如 '2+3*4'、'(100-25)/3'。只能包含数字和运算符。"
                    }
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "搜索互联网获取实时信息。"
                "用于:新闻、近期事件、最新数据、人物动态、产品信息等时效性强的查询。"
                "不要用于:已有明确答案的常识问题、纯数学计算、当前时间查询。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索关键词,3-6 个词最佳,使用核心名词,避免完整句子。"
                    }
                },
                "required": ["query"]
            }
        }
    }
]


TOOL_FUNCTIONS = {
    "get_current_time": get_current_time,
    "calculate": calculate,
    "web_search": web_search,
}


# ========== Agent 循环 ==========
def run_agent(user_message: str, max_iterations: int = 5, verbose: bool = True):
    """带 web_search 的 Agent"""
    if verbose:
        print(f"\n{'='*70}")
        print(f"【用户】{user_message}")
        print('='*70)
    
    messages = [{"role": "user", "content": user_message}]

    tool_fail_count = {}
    
    for iteration in range(max_iterations):
        if verbose:
            print(f"\n--- 第 {iteration+1} 轮 ---")
        
        response = llm_client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            tools=tools,
        )
        
        msg = response.choices[0].message
        
        # 没调工具,任务完成
        if not msg.tool_calls:
            if verbose:
                print(f"\n【最终回答】{msg.content}")
            return msg.content
        
        # 调了工具,执行
        messages.append(msg)
        
        for tool_call in msg.tool_calls:
            fname = tool_call.function.name
            fargs = json.loads(tool_call.function.arguments)
            
            if verbose:
                # web_search 的 query 全打印,其他工具简单打印
                print(f"  → 调用 {fname}({fargs})")
            
            try:
                if fname in TOOL_FUNCTIONS:
                    result = TOOL_FUNCTIONS[fname](**fargs)
                else:
                    result = f"未知工具: {fname}"
            except Exception as e:
                result = f"工具执行异常: {e}"
                tool_fail_count[fname] = tool_fail_count.get(fname, 0) + 1
            if tool_fail_count.get(fname, 0) >= 1:
                result += f"后续禁用工具：{fname}"
            
            if verbose:
                # 截断显示,避免 web_search 结果太长
                result_preview = result[:200] + "..." if len(result) > 200 else result
                print(f"  ← 返回 {result_preview}")
            
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": str(result),
            })
    
    if verbose:
        print(f"\n【超过最大轮数 {max_iterations},强制结束】")
    return None


# ========== 测试 ==========
if __name__ == "__main__":
    test_questions = [
        # 1. 应该用 web_search
        "DeepSeek 最近发布了什么新模型?",
        
        # 2. 应该用 web_search + calculate(多步)
        "现在比特币价格是多少美元?换算成人民币大约多少?(假设汇率 7.2)",
        
        # 3. 不该用任何工具
        "Python 的列表和字典有什么区别?",
        
        # 4. 应该用 get_current_time + web_search
        "今天有什么 AI 领域的重要新闻?",
    ]
    
    for q in test_questions:
        run_agent(q)
        print("\n" + "="*70 + "\n")