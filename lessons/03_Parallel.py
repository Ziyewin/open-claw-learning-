import os
import asyncio
from typing import Optional
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnableParallel, RunnablePassthrough

# 加载环境变量（读取 DeepSeek API Key）
load_dotenv()

# --- 模型配置：替换为 DeepSeek ---
try:
    llm: Optional[ChatOpenAI] = ChatOpenAI(
        base_url="https://api.deepseek.com/v1",
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        model="deepseek-chat",
        temperature=0.7
    )
except Exception as e:
    print(f"模型初始化失败: {e}")
    llm = None

# --- 定义并行子链路（提示词全部改为中文） ---
# 内容总结链路
summarize_chain: Runnable = (
    ChatPromptTemplate.from_messages([
        ("system", "简洁地总结以下主题内容："),
        ("user", "{topic}"),
    ])
    | llm
    | StrOutputParser()
)

# 生成问题链路
questions_chain: Runnable = (
    ChatPromptTemplate.from_messages([
        ("system", "围绕以下主题，生成三个有意思的问题："),
        ("user", "{topic}"),
    ])
    | llm
    | StrOutputParser()
)

# 提取关键词链路
terms_chain: Runnable = (
    ChatPromptTemplate.from_messages([
        ("system", "从以下主题中提取5-10个关键词，使用英文逗号分隔："),
        ("user", "{topic}"),
    ])
    | llm
    | StrOutputParser()
)

# --- 构建并行执行 + 结果整合链路 ---
# 并行执行三个任务，同时透传原始主题
map_chain = RunnableParallel(
    {
        "summary": summarize_chain,
        "questions": questions_chain,
        "key_terms": terms_chain,
        "topic": RunnablePassthrough(),
    }
)

# 结果整合提示词（中文）
synthesis_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """根据以下信息进行整合，输出一份完整的解答：
        内容总结：{summary}
        相关问题：{questions}
        关键词：{key_terms}"""
    ),
    ("user", "原始主题：{topic}"),
])

# 组装完整链路
full_parallel_chain = map_chain | synthesis_prompt | llm | StrOutputParser()

# --- 异步运行链路 ---
async def run_parallel_example(topic: str) -> None:
    """异步执行并行处理链路，并打印最终整合结果"""
    if not llm:
        print("模型未初始化，无法执行任务")
        return

    print(f"\n--- 正在处理主题：'{topic}' ---")
    try:
        response = await full_parallel_chain.ainvoke(topic)
        print("\n--- 最终整合结果 ---")
        print(response)
    except Exception as e:
        print(f"\n执行链路时出错: {e}")

if __name__ == "__main__":
    # 中文测试主题
    test_topic = "人类太空探索发展史"
    asyncio.run(run_parallel_example(test_topic))