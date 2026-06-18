import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew
from crewai import LLM
import chromadb

load_dotenv()

# 初始化模型
llm = LLM(
    model="deepseek/deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY")
)

# ===================== 自定义记忆模块 =====================
# 向量库：存储超过10轮的历史摘要
chroma_client = chromadb.PersistentClient(path="./vector_memory")
vector_collection = chroma_client.get_or_create_collection(name="chat_summary_memory")

# 配置：保留最近 10 轮原始对话
MAX_RECENT_ROUND = 10
recent_chat = []  # 存放最近10轮 (问题,回答)
round_count = 0

# 对话摘要生成
def generate_summary(text: str) -> str:
    """调用大模型生成精简摘要"""
    prompt = f"请精简总结以下对话内容，控制在30字以内：{text}"
    res = llm.call(prompt)
    return res.strip()

# 语义检索历史摘要
def retrieve_history_summary(query: str, top_k=3) -> str:
    """根据问题检索向量库中的历史摘要"""
    if vector_collection.count() == 0:
        return ""
    res = vector_collection.query(query_texts=[query], n_results=top_k)
    summaries = res["documents"][0]
    return "\n历史对话摘要：" + "\n".join(summaries)

# ===================== CrewAI 业务逻辑 =====================
chat_agent = Agent(
    role="智能助手",
    goal="结合历史对话连贯回答问题",
    backstory="能结合近期对话和过往记忆作答",
    llm=llm,
    memory=False,
    long_term_memory=False,
    verbose=True
)

def run_chat(user_input: str):
    global round_count
    # 1. 检索长期向量记忆（历史摘要）
    history_summary = retrieve_history_summary(user_input)
    # 2. 拼接最近10轮原文 + 历史摘要
    context = history_summary + "\n=== 最近对话 ==\n"
    for q, a in recent_chat:
        context += f"用户：{q}\n助手：{a}\n"
    full_prompt = context + f"\n用户新问题：{user_input}"

    # 3. 修复：必须补充 expected_output
    task = Task(
        description=full_prompt,
        expected_output="贴合上下文，自然流畅地回答用户问题",
        agent=chat_agent
    )
    crew = Crew(agents=[chat_agent], tasks=[task], verbose=True)
    answer = crew.kickoff().raw

    # 4. 更新轮次 & 管理记忆
    round_count += 1
    recent_chat.append((user_input, answer))

    # 超过10轮：旧对话转摘要存入向量库
    if len(recent_chat) > MAX_RECENT_ROUND:
        oldest_q, oldest_a = recent_chat.pop(0)
        old_text = f"用户：{oldest_q} 助手：{oldest_a}"
        summary = generate_summary(old_text)
        # 写入向量库
        vector_collection.add(
            documents=[summary],
            ids=[f"round_{round_count}"]
        )
    return answer

# 模拟多轮对话测试
if __name__ == "__main__":
    test_questions = [
        "我叫小明，喜欢打篮球",
        "我今年22岁",
        "我平时也喜欢看书",
        "我的专业是计算机",
        "周末一般会户外运动",
        "每天都会学习新知识",
        "常用的编程语言是Python",
        "未来想做AI相关工作",
        "业余爱好还有听歌",
        "最近在学习CrewAI框架",
        "刚才我说了哪些个人信息？",
        "再回顾一下我之前的职业规划"
    ]

    for idx, ques in enumerate(test_questions, 1):
        print(f"\n--- 第{idx}轮对话 ---")
        ans = run_chat(ques)
        print(f"助手：{ans}")