import asyncio
import json
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from fastmcp import Client
from fastmcp.client.transports import StdioTransport

load_dotenv()

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1",
    temperature=0
)

SYSTEM_PROMPT = """
你拥有以下可用工具，请根据用户问题自主判断是否调用工具、调用哪一个。
工具列表：
1. count_char：统计文本字符数，入参：{"text": "内容"}
2. concat_str：拼接两个字符串，入参：{"a": "内容1", "b": "内容2"}
3. add_num：两个整数相加，入参：{"x": 数字1, "y": 数字2}

规则：
1. 需要使用工具时，严格按格式输出：TOOL|工具名|JSON参数
2. 不需要工具、可直接回答时，输出：ANSWER|你的回答内容
3. 不要额外解释、不要多余文字。
"""

async def auto_run():
    # ========= 关键修复：使用 Python 绝对路径 =========
    python_abs_path = "/opt/miniconda3/envs/agent/bin/python"
    transport = StdioTransport(
        command=python_abs_path,
        args=["mcp_server.py"]
    )

    async with Client(transport=transport) as mcp_client:
        user_questions = [
            "计算 35 + 65 等于多少？",
            "统计这句话有多少字符：大模型自动调用外部工具",
            "把「AI」和「MCP」拼接在一起",
            "今天天气不错，简单聊两句"
        ]

        for question in user_questions:
            print(f"\n===== 用户问题：{question} =====")
            resp = llm.invoke([
                ("system", SYSTEM_PROMPT),
                ("user", question)
            ])
            llm_output = resp.content.strip()
            print(f"LLM 决策输出：{llm_output}")

            if llm_output.startswith("TOOL|"):
                parts = llm_output.split("|")
                if len(parts) != 3:
                    print("格式错误")
                    continue
                _, tool_name, param_json_str = parts
                try:
                    params = json.loads(param_json_str)
                    tool_result = await mcp_client.call_tool(tool_name, params)
                    tool_content = tool_result.content[0].text
                    print(f"MCP 工具返回：{tool_content}")

                    final_prompt = (
                        f"用户问题：{question}\n"
                        f"工具执行结果：{tool_content}\n"
                        "请整理成通顺的口语回答"
                    )
                    final_resp = llm.invoke(final_prompt)
                    print(f"最终回答：{final_resp.content}")

                except Exception as e:
                    print(f"工具调用异常：{e}")

            elif llm_output.startswith("ANSWER|"):
                _, ans = llm_output.split("|", 1)
                print(f"最终回答：{ans}")

if __name__ == "__main__":
    asyncio.run(auto_run())