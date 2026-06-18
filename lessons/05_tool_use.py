import os
import getpass
import asyncio
import nest_asyncio
from dotenv import load_dotenv

# 替换为 DeepSeek 对接
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool as langchain_tool
from langchain.agents import create_tool_calling_agent, AgentExecutor

# 加载环境变量
load_dotenv()

# 输入 DeepSeek API Key
if not os.getenv("DEEPSEEK_API_KEY"):
    os.environ["DEEPSEEK_API_KEY"] = getpass.getpass("Enter your DeepSeek API key: ")

DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"

try:
    # 初始化 DeepSeek 模型（支持工具调用）
    llm = ChatOpenAI(
        model="deepseek-chat",
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url=DEEPSEEK_BASE_URL,
        temperature=0
    )
    print(f"✅ 模型初始化成功: deepseek-chat")
except Exception as e:
    print(f"🛑 模型初始化失败: {e}")
    llm = None

# --- 定义工具 ---
@langchain_tool
def search_information(query: str) -> str:
    """
    查询指定主题的事实信息，例如城市首都、天气、地理知识等。
    :param query: 需要查询的问题
    """
    print(f"\n--- 🛠️ 调用工具: search_information，查询内容: '{query}' ---")

    simulated_results = {
        "伦敦天气": "伦敦当前多云，气温 15°C。",
        "法国首都": "法国的首都是巴黎。",
        "地球人口": "地球总人口约 80 亿。",
        "最高山峰": "珠穆朗玛峰是世界海拔最高的山峰。",
        "default": f"「{query}」暂无匹配信息。"
    }
    result = simulated_results.get(query.lower(), simulated_results["default"])
    print(f"--- 工具返回结果: {result} ---")
    return result

tools = [search_information]

# --- 创建工具调用智能体 ---
if llm:
    agent_prompt = ChatPromptTemplate.from_messages([
        ("system", "你是乐于助人的智能助手，遇到需要查询的内容请主动调用工具。"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    agent = create_tool_calling_agent(llm, tools, agent_prompt)
    agent_executor = AgentExecutor(agent=agent, verbose=True, tools=tools)

async def run_agent_with_tool(query: str):
    """异步执行智能体查询"""
    print(f"\n--- 🏃 开始执行查询: '{query}' ---")
    try:
        response = await agent_executor.ainvoke({"input": query})
        print("\n--- ✅ 最终回答 ---")
        print(response["output"])
    except Exception as e:
        print(f"\n🛑 执行出错: {e}")

async def main():
    """并发执行多个查询"""
    tasks = [
        run_agent_with_tool("伦敦天气"),
        run_agent_with_tool("法国首都"),
        run_agent_with_tool("介绍一下狗狗。")
    ]
    await asyncio.gather(*tasks)

nest_asyncio.apply()
asyncio.run(main())