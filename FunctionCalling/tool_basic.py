"""
Function Calling 入门:让 DeepSeek 调用一个简单工具
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

# ============ 第1步:定义一个真实的 Python 函数 ============
def get_current_time():
    """获取当前时间"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ============ 第2步:用规定的格式描述这个工具给 LLM ============
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "获取当前的日期和时间。当用户询问现在几点、今天日期等时间相关问题时调用。",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    }
]
tools.clear()


# ============ 第3步:第一次调用 LLM,看它是否决定调用工具 ============
messages = [
    {"role": "user", "content": "现在几点了?"}
]

print("=== 第1次调用 LLM ===")
response = client.chat.completions.create(
    model="deepseek-chat",
    messages=messages,
    tools=tools,  # 关键:把工具列表传给 LLM
)

assistant_message = response.choices[0].message
print(f"finish_reason: {response.choices[0].finish_reason}")
print(f"content: {assistant_message.content}")
print(f"tool_calls: {assistant_message.tool_calls}")

# ============ 第4步:执行 LLM 想调用的工具 ============
if assistant_message.tool_calls:
    # 把 LLM 的 tool_call 消息加入对话历史(很重要!)
    messages.append(assistant_message)
    
    # 处理每个 tool_call(通常只有一个,但可能有多个)
    for tool_call in assistant_message.tool_calls:
        function_name = tool_call.function.name
        function_args = json.loads(tool_call.function.arguments)
        
        print(f"\nLLM 决定调用: {function_name}({function_args})")
        
        # 真正执行工具
        if function_name == "get_current_time":
            result = get_current_time()
        else:
            result = f"未知工具: {function_name}"
        
        print(f"工具返回: {result}")
        
        # 把工具结果加入对话历史(这一步格式很关键)
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": result,
        })

# ============ 第5步:把对话历史(含工具结果)再发给 LLM,让它生成最终回答 ============
print("\n=== 第2次调用 LLM(带着工具结果) ===")
final_response = client.chat.completions.create(
    model="deepseek-chat",
    messages=messages,
    tools=tools,
)

print(f"最终回答: {final_response.choices[0].message.content}")