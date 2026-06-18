import os
from typing import Optional, Dict, Any
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.agents import AgentAction
from langchain_core.messages import SystemMessage

# 加载环境变量
load_dotenv()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

# 初始化DeepSeek大模型
llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com/v1",
    temperature=0.3
)

# 全局会话状态（对应ADK的state，存放客户信息）
session_state: Dict[str, Any] = {
    "customer_info": {
        "name": "张先生",
        "tier": "VIP高端会员",
        "support_history": ["去年显示器黑屏报修", "上月耳机充电故障"],
        "recent_purchases": ["4K显示器", "无线降噪耳机", "机械键盘"]
    }
}

# ---------------------- 三个业务工具（对应原版） ----------------------
@tool
def troubleshoot_issue(issue: str) -> dict:
    """
    自动故障排查工具：分析用户设备问题，输出自助排查步骤
    参数 issue：用户描述的设备故障
    """
    report = f"自助排查方案：针对【{issue}】\n1. 断电重启设备 5分钟\n2. 检查电源线/蓝牙连接\n3. 更新设备配套驱动"
    return {"status": "success", "report": report}

@tool
def create_ticket(issue_type: str, details: str) -> dict:
    """
    创建电子报修工单，自动留存故障记录
    参数 issue_type：故障分类（显示器/耳机/键盘等）
    参数 details：用户完整故障描述
    """
    ticket_id = f"工单-{1000 + len(session_state['customer_info']['support_history'])}"
    return {"status": "success", "ticket_id": ticket_id, "record": details}

@tool
def escalate_to_human(issue_type: str) -> dict:
    """
    【人机协同核心工具】自动转接真人人工专员处理复杂问题
    适用场景：自助排查无效、硬件损坏、批量设备故障
    参数 issue_type：故障分类
    """
    return {
        "status": "success",
        "message": f"已为您转接专属人工客服专员，当前排队2人，优先接待{session_state['customer_info']['tier']}客户",
        "human_queue": "人工客服接入队列"
    }

# 工具列表
tools = [troubleshoot_issue, create_ticket, escalate_to_human]

# ---------------------- 自定义回调钩子（对应原版personalization_callback） ----------------------
class CustomerInfoInjectCallback(BaseCallbackHandler):
    """回调：每次大模型请求前，自动注入客户个性化信息，实现千人千面"""
    def before_chat_model_invoke(self, serialized, raw_messages, **kwargs):
        customer_info = session_state.get("customer_info", {})
        if not customer_info:
            return raw_messages

        # 拼接客户个性化信息
        name = customer_info.get("name", "尊敬的顾客")
        tier = customer_info.get("tier", "普通客户")
        history = customer_info.get("support_history", [])
        purchases = customer_info.get("recent_purchases", [])

        personal_text = f"""
【客户专属信息，回复必须参考】
客户姓名：{name}
会员等级：{tier}
历史报修记录：{'; '.join(history)}
近期购买商品：{', '.join(purchases)}
回复时称呼客户姓名，结合历史故障记录，VIP客户优先引导人工通道
"""
        # 插入系统消息到最顶部
        personal_msg = SystemMessage(content=personal_text)
        raw_messages.insert(0, personal_msg)
        return raw_messages

# 实例化回调
customer_callback = CustomerInfoInjectCallback()

# ---------------------- 智能体提示词（中文业务指令，对齐原版Agent instruction） ----------------------
prompt = ChatPromptTemplate.from_messages([
    ("system", """
你是电子设备品牌的智能售后客服，遵循固定处理流程：
第一步：查看客户历史报修记录，回复时主动关联过往故障；
第二步：普通故障先调用troubleshoot_issue给出自助排查步骤；
第三步：自助方案无效时，调用create_ticket创建正式报修工单；
第四步：硬件损毁、反复维修、复杂疑难问题，调用escalate_to_human转接人工真人客服【人机协同】。

沟通要求：语气温和共情，理解设备故障带来的麻烦；VIP客户可主动建议转接人工，不用反复自助排查。
"""),
    ("user", "{input}"),
    ("placeholder", "{agent_scratchpad}")
])

# 创建工具调用智能体 + 执行器
agent = create_tool_calling_agent(llm, tools, prompt)
support_agent = AgentExecutor(agent=agent, tools=tools, verbose=True, callbacks=[customer_callback])

# ---------------------- 测试运行（中文人机协同案例） ----------------------
if __name__ == "__main__":
    print("===== 电子售后人机协同智能体（DeepSeek）=====\n")

    # 案例1：简单故障，自动排查即可解决
    print("【案例1 简单故障】用户输入：我的无线耳机充不进电")
    res1 = support_agent.invoke({"input": "我的无线耳机充不进电"})
    print(f"\n客服最终回复：{res1['output']}\n")
    print("-" * 80)

    # 案例2：复杂故障，自动升级人工客服（人机协同核心场景）
    print("【案例2 疑难故障，触发人工转接】用户输入：显示器去年修过一次，现在黑屏反复重启，自助操作全部无效")
    res2 = support_agent.invoke({"input": "显示器去年修过一次，现在黑屏反复重启，自助操作全部无效"})
    print(f"\n客服最终回复：{res2['output']}\n")