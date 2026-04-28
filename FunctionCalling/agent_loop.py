"""
真正的 Agent:循环调用工具,直到任务完成
"""
import os
import json
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

# ====== 工具定义(同上) ======
def get_current_time():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def calculate(expression: str):
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception as e:
        return f"计算错误: {e}"

def get_weather(city: str):
    fake_data = {"北京": "晴,15℃", "上海": "多云,18℃", "深圳": "雷阵雨,24℃"}
    return fake_data.get(city, f"暂无{city}的天气数据")

TOOL_FUNCTIONS = {
    "get_current_time": get_current_time,
    "calculate": calculate,
    "get_weather": get_weather,
}

# ============ 工具描述列表 ============
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "获取当前的日期和时间。当用户询问现在几点、今天日期、当前时间等问题时调用。",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "计算一个数学表达式的结果。支持加减乘除、幂运算等。当用户询问数学计算时调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "要计算的数学表达式,例如 '2+3*4' 或 '(10+5)/3'"
                    }
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询某个城市的天气情况。当用户询问天气时调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称,例如 '北京'、'上海'"
                    }
                },
                "required": ["city"]
            }
        }
    }
]


def run_agent(user_message: str, max_iterations: int = 5):
    """
    真正的 Agent 循环:
    - LLM 决定 → 执行工具 → 把结果给 LLM → 继续决定 → ...
    - 直到 LLM 不再调用工具(给出最终回答),或达到最大轮数
    """
    print(f"\n{'='*60}")
    print(f"用户: {user_message}")
    print('='*60)
    
    messages = [{"role": "user", "content": user_message}]
    
    for iteration in range(max_iterations):
        print(f"\n--- 第 {iteration+1} 轮 ---")
        
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            tools=tools,
        )
        
        assistant_msg = response.choices[0].message
        finish_reason = response.choices[0].finish_reason
        
        # 如果没调工具,说明 LLM 准备给最终答案了
        if not assistant_msg.tool_calls:
            print(f"\n【最终回答】{assistant_msg.content}")
            return assistant_msg.content
        
        # 调了工具,执行,继续循环
        messages.append(assistant_msg)
        
        for tool_call in assistant_msg.tool_calls:
            fname = tool_call.function.name
            fargs = json.loads(tool_call.function.arguments)
            print(f"  → 调用 {fname}({fargs})")
            
            if fname in TOOL_FUNCTIONS:
                result = TOOL_FUNCTIONS[fname](**fargs)
            else:
                result = f"未知工具: {fname}"
            
            print(f"  ← 返回 {result}")
            
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": str(result),
            })
    
    print("\n【超过最大轮数,强制结束】")
    return None


# ============ 测试 ============
if __name__ == "__main__":
    # 把 tools 定义补全(从 tool_multi.py 复制)
    
    # 多步推理测试
    run_agent("现在时间加上 5 小时,是几点?")
    
    # 更复杂的
    run_agent("北京和深圳哪个温度更高?差几度?")