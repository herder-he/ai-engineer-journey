import os
import time
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

def ask(model, prompt):
    start = time.time()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    elapsed = time.time() - start
    msg = response.choices[0].message
    reasoning = getattr(msg, 'reasoning_content', None)
    return {
        "content": msg.content,
        "reasoning": reasoning,
        "time": elapsed,
        "out_tokens": response.usage.completion_tokens,
    }

tasks = [
    ("简单事实", "中国首都是哪里?"),
    ("逻辑推理", "一个房间有3个开关控制外面3个灯泡,你只能进房间一次,如何判断哪个开关对应哪个灯?"),
]

for name, q in tasks:
    print(f"\n{'='*60}\n任务: {name}\n问题: {q}\n{'='*60}")
    
    for model in ["deepseek-chat", "deepseek-reasoner"]:
        print(f"\n--- {model} ---")
        r = ask(model, q)
        print(f"耗时: {r['time']:.1f}秒 | 输出token: {r['out_tokens']}")
        if r['reasoning']:
            print(f"[思考过程前150字] {r['reasoning'][:150]}...")
        print(f"[最终回答] {r['content'][:300]}")

"""
我的观察:

1. reasoner模型独有的reasoning_content是什么?
思考过程

2. 简单问题上,reasoner是否过度?
并没有，反而更聪明，精简了

3. 复杂问题上,reasoner思考过程的质量如何?
与最终答案关联性大，质量高

4. 我什么时候会选reasoner?什么时候选chat?
简单问题想省token想chat，想更精确精简不考虑token用量选reasoner
"""