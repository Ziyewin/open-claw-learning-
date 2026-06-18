# deepseek_agent_example.py
import os
from dotenv import load_dotenv
import asyncio
import nest_asyncio
from typing import List
from langchain_openai import ChatOpenAI  # 使用 OpenAI 的接口调用 DeepSeek
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool as langchain_tool
from langchain.agents import create_tool_calling_agent, AgentExecutor

# ============================================================================
# 加载环境变量
# ============================================================================
load_dotenv()

# 读取 DeepSeek API Key（从 .env 文件）
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
if not DEEPSEEK_API_KEY:
    # 如果没有 .env 文件，则通过命令行输入
    DEEPSEEK_API_KEY = getpass.getpass("Enter your DeepSeek API key: ")

# ============================================================================
# 初始化 DeepSeek LLM
# ============================================================================
try:
    # DeepSeek 兼容 OpenAI 接口，使用 ChatOpenAI 类
    llm = ChatOpenAI(
        model="deepseek-chat",  # 或 "deepseek-reasoner"
        temperature=0,
        openai_api_key=DEEPSEEK_API_KEY,
        openai_api_base="https://api.deepseek.com",  # DeepSeek API 地址
    )
    print(f"✅ DeepSeek 模型初始化成功: {llm.model_name}")
except Exception as e:
    print(f"🛑 初始化 DeepSeek 模型失败: {e}")
    llm = None

# ============================================================================
# 定义工具 (Tool)
# ============================================================================
@langchain_tool
def search_information(query: str) -> str:
    """
    提供关于特定主题的事实信息。使用这个工具来查找类似
    '法国的首都是什么' 或 '伦敦的天气怎么样' 等问题的答案。
    """
    print(f"\n--- 🛠️ 工具被调用: search_information with query: '{query}' ---")

    # 模拟一个搜索工具，用字典存储预设结果
    simulated_results = {
        "weather in london": "伦敦目前的天气是多云，气温15°C。",
        "capital of france": "法国的首都是巴黎。",
        "population of earth": "地球的估计人口约为80亿人。",
        "tallest mountain": "珠穆朗玛峰是海平面以上的最高山峰。",
        "default": f"关于 '{query}' 的模拟搜索结果：未找到具体信息，但这个话题看起来很有趣。",
    }
    result = simulated_results.get(query.lower(), simulated_results["default"])
    print(f"--- 工具结果: {result} ---")
    return result

# 将工具放入列表
tools = [search_information]

# ============================================================================
# 创建工具调用 Agent (Tool-Calling Agent)
# ============================================================================
if llm:
    # 这个 prompt 模板需要一个 `agent_scratchpad` 占位符来记录 agent 的内部步骤
    agent_prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个有帮助的助手。"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    # 创建 agent，将 LLM、工具和 prompt 绑定在一起
    agent = create_tool_calling_agent(llm, tools, agent_prompt)

    # AgentExecutor 是运行时，负责调用 agent 并执行选定的工具
    agent_executor = AgentExecutor(
        agent=agent, 
        verbose=True,  # 设置为 True 可以看到详细的执行过程
        tools=tools
    )

# ============================================================================
# 运行 Agent 的异步函数
# ============================================================================
async def run_agent_with_tool(query: str):
    """使用查询调用 agent executor 并打印最终响应"""
    print(f"\n--- 🏃 运行 Agent，查询: '{query}' ---")
    try:
        response = await agent_executor.ainvoke({"input": query})
        print("\n--- ✅ 最终 Agent 响应 ---")
        print(response["output"])
    except Exception as e:
        print(f"\n🛑 Agent 执行过程中出现错误: {e}")

# ============================================================================
# 主函数
# ============================================================================
async def main():
    """并发运行所有 agent 查询"""
    tasks = [
        run_agent_with_tool("法国的首都是什么？"),
        run_agent_with_tool("伦敦的天气怎么样？"),
        run_agent_with_tool("告诉我一些关于狗的事情。"),  # 应该触发默认工具响应
    ]
    await asyncio.gather(*tasks)

# ============================================================================
# 程序入口
# ============================================================================
if __name__ == "__main__":
    # 应用 nest_asyncio 以支持在已有事件循环中运行
    nest_asyncio.apply()
    
    # 运行主函数
    asyncio.run(main())