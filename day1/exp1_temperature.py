import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

def ask(prompt, temperature):
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=150,
    )
    return response.choices[0].message.content

# 测试3种任务 × 3个temperature × 各2次
tasks = {
    "创意": "写一句关于秋天的开头,要意外感",
    "结构化": "中国四大名著是哪四本?只输出JSON数组,例如:[\"xxx\",\"xxx\"]",
    "事实": "光速是多少?",
}

for name, prompt in tasks.items():
    print(f"\n{'='*50}\n任务: {name}  |  Prompt: {prompt}\n{'='*50}")
    for temp in [0, 0.7, 1.5]:
        print(f"\n--- temperature={temp} ---")
        for i in range(2):
            print(f"[{i+1}] {ask(prompt, temp)}")

"""
我的观察:

1. 创意任务上,temp=0和temp=1.5的差距: temp1.5内容更丰富，多次拿到不一样的结果，更适合创作

2. 结构化任务上,高temperature会带来什么问题:在规则内引入更复杂的结构，强行加信息

3. 事实任务上,不同temperature的答案有差吗:结论没区别，主要temperature会增加一些相关的信息

4. 我以后选temperature的原则:需要创意创作时增加temperature，反之需要明确信息和结论时降低temperature

"""