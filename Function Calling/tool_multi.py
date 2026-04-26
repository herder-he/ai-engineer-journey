"""
多工具 Function Calling:LLM 自主选择调用哪个工具
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


# ============ 定义多个工具的实际函数 ============
def get_current_time():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def calculate(expression: str):
    """安全的数学计算"""
    try:
        # 注意:eval 在生产环境不安全,这里仅作教学
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"计算错误: {e}"

def get_weather(city: str):
    """模拟天气查询(真实场景下你会调用天气 API)"""
    fake_data = {
        "北京": "晴,15℃",
        "上海": "多云,18℃",
        "深圳": "雷阵雨,24℃",
    }
    return fake_data.get(city, f"暂无{city}的天气数据")


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


# ============ 工具名 → 实际函数的映射(为了方便统一调用) ============
TOOL_FUNCTIONS = {
    "get_current_time": get_current_time,
    "calculate": calculate,
    "get_weather": get_weather,
}


def execute_tool(tool_call):
    """执行一个 tool_call,返回结果字符串"""
    function_name = tool_call.function.name
    function_args = json.loads(tool_call.function.arguments)
    
    if function_name not in TOOL_FUNCTIONS:
        return f"未知工具: {function_name}"
    
    func = TOOL_FUNCTIONS[function_name]
    return func(**function_args)


def chat_with_tools(user_message: str):
    """完整的"用户提问 → 工具调用 → 最终回答"流程"""
    print(f"\n{'='*60}")
    print(f"用户: {user_message}")
    print('='*60)
    
    messages = [
        {"role": "user", "content": user_message}
    ]
    
    # 第1次调用
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        tools=tools,
    )
    
    assistant_msg = response.choices[0].message
    
    # 如果 LLM 决定调用工具
    if assistant_msg.tool_calls:
        messages.append(assistant_msg)
        
        for tool_call in assistant_msg.tool_calls:
            print(f"  → LLM 调用: {tool_call.function.name}({tool_call.function.arguments})")
            result = execute_tool(tool_call)
            print(f"  ← 工具返回: {result}")
            
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            })
        
        # 把工具结果发回 LLM,生成最终回答
        final = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            tools=tools,
        )
        print(f"\nAI: {final.choices[0].message.content}")
    else:
        # LLM 直接回答了,没用工具
        print(f"\nAI(没用工具): {assistant_msg.content}")


# ============ 测试不同问题 ============
if __name__ == "__main__":
    test_questions = [
        "现在几点了?",                    # 应该调 get_current_time
        "1024乘以56等于多少?",            # 应该调 calculate
        "北京今天天气怎么样?",             # 应该调 get_weather
        "深圳天气如何?",                  # 应该调 get_weather
        "你好,介绍一下你自己",            # 应该不调工具
    ]
    
    for q in test_questions:
        chat_with_tools(q)
    chat_with_tools("现在时间加上 5 小时,是几点?")