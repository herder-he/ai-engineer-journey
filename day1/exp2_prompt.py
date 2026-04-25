import os
import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

def ask(prompt):
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return response.choices[0].message.content

review = """这个蓝牙耳机我用了一周,音质确实好,降噪也不错,通勤挺安静的。
但是电池续航拉胯,说是8小时实际只有3小时多。充电盒盖子不好关。
698元在这个价位算合理,快递很快第二天就到了。总体还行,推荐度7分。"""

# 四个版本,逐步加strong
prompts = {
    "A_最基础": f"分析这段评论:{review}",
    
    "B_加角色": f"""你是评论分析助手。请分析以下评论,提取:产品类型、优点、缺点、价格、评分、是否推荐。

评论:{review}""",
    
    "C_加格式": f"""你是评论分析助手。请分析以下评论。

评论:{review}

请以JSON格式输出,字段:product_type, pros(数组), cons(数组), price(数字), rating(1-10), recommended(布尔)。
只输出JSON,不要任何说明文字。""",
    
    "D_加示例": f"""你是评论分析助手。

示例:
评论:"这款键盘手感不错,RGB漂亮,但打字声音大。400块。推荐。"
输出:
{{
  "product_type": "键盘",
  "pros": ["手感好", "RGB漂亮"],
  "cons": ["打字声音大"],
  "price": 400,
  "rating": 7,
  "recommended": true
}}

现在分析:
{review}

只输出JSON,不要其他文字。""",
}

# 每个版本跑3次,测稳定性
for name, prompt in prompts.items():
    print(f"\n{'='*50}\n{name}\n{'='*50}")
    success = 0
    for i in range(3):
        result = ask(prompt)
        print(f"\n[{i+1}] {result[:200]}")
        # 测能不能解析为JSON
        try:
            clean = result.strip().replace("```json","").replace("```","").strip()
            json.loads(clean)
            success += 1
            print("   ✓ JSON可解析")
        except:
            print("   ✗ JSON失败")
    print(f"\n>> {name} JSON成功率: {success}/3")

# === 跑完写观察 ===
"""
我的观察:

1. A版本输出大概是什么样?(应该不是JSON)
相对比较乱，也分了几个点回复，但是比较口语化

2. 哪一步带来的提升最明显?(B→C还是C→D?)
B->C，数据格式化后非常精准和简约

3. D版本的"给示例"加上后,有什么质的变化?
没有实质的变化，给示例和格式得到的结果是一样的，給示例应该也是去分析出格式再处理的

4. 如果面试官问"prompt怎么迭代",我的回答是:
首先要明确助手的角色，并且给出明确的结果格式或者示例
"""