import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "user", "content": "用一句话解释什么是大语言模型"}
    ]
)

print("=== 回答 ===")
print(response.choices[0].message.content)
print(f"\n输入token: {response.usage.prompt_tokens}")
print(f"输出token: {response.usage.completion_tokens}")
print(f"停止原因: {response.choices[0].finish_reason}")