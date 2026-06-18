import os
import re
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()
llm = ChatOpenAI(
    model="deepseek-v4-flash",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1",
    temperature=0
)

# ---------------------- Agent定义 ----------------------
def agent_customer_service(user_input: str, report: str = None):
    """前端客服智能体，标签：customer_service"""
    if report is None:
        # 场景1：第一次接收用户提问，无数据报告
        sys_prompt = """
你是客服智能体，标签customer_service，强制规则：
1. 普通咨询（营业时间、售后政策）直接自然语言完整回答。
2. 用户要求计算数字、统计数据时，只输出一行调用指令，格式固定：
CALL_AGENT|data_analysis|用户完整需求和数字
禁止自己计算、禁止写任何解释文字，只能保留这一行指令。
"""
        msg = f"{sys_prompt}\n用户问题：{user_input}"
    else:
        # 场景2：已经拿到数据分析报告，必须整理回复，禁止调用其他Agent
        sys_prompt = """
你是客服智能体，现在已经收到数据分析智能体返回的完整统计报告。
强制规则：
1. 绝对不能输出CALL_AGENT调用指令，不需要再请求数据分析；
2. 结合报告内容，用友好通俗的中文整理结果回复用户；
3. 简洁易懂，不用复杂表格，口语化总结数据。
"""
        msg = f"{sys_prompt}\n【统计报告】{report}\n用户原始提问：{user_input}"

    res = llm.invoke(msg)
    return res.content.strip()

def agent_data_analysis(task_content: str):
    """数据分析智能体，标签：data_analysis"""
    prompt = f"""
你是专业数据分析智能体，标签data_analysis。
接收统计任务，计算总和、平均值、最大最小值，生成清晰数据报告。
任务：{task_content}
"""
    res = llm.invoke(prompt)
    return res.content.strip()

# Agent注册表
AGENT_REGISTRY = {
    "customer_service": agent_customer_service,
    "data_analysis": agent_data_analysis
}

# ---------------------- A2A调度（正则提取指令） ----------------------
def a2a_scheduler(user_query: str):
    agent_a_output = AGENT_REGISTRY["customer_service"](user_query)
    print(f"\n客服Agent原始输出：\n{agent_a_output}\n")

    # 正则匹配调用指令
    pattern = r"CALL_AGENT\|([^|]+)\|(.+)"
    match = re.search(pattern, agent_a_output)
    if match:
        target_tag = match.group(1)
        task_args = match.group(2)
        print(f"===== A2A通信：customer_service → {target_tag} =====")

        # 调用数据分析智能体
        target_agent = AGENT_REGISTRY[target_tag]
        analysis_result = target_agent(task_args)
        print(f"【{target_tag}】分析结果：\n{analysis_result}\n")

        # 传入report参数，触发二次整理逻辑
        final_text = AGENT_REGISTRY["customer_service"](user_query, report=analysis_result)
        return final_text
    else:
        return agent_a_output

# ---------------------- 测试入口 ----------------------
if __name__ == "__main__":
    print("===== 测试1：普通咨询，不触发A2A =====")
    ans1 = a2a_scheduler("售后几点上班？")
    print(f"最终回复：{ans1}\n")
    print("-" * 70)

    print("===== 测试2：营收统计，触发A2A跨智能体调用 =====")
    ans2 = a2a_scheduler("五天营收：1000,2000,1500,3000,2500，计算总和与平均")
    print(f"最终用户回复：{ans2}")