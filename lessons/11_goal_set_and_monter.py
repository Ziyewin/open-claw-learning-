# MIT License
# 原作者 Mahtab Syed，已适配 DeepSeek 模型 + 全中文提示词与示例
"""
实操代码示例 - 迭代优化智能代码生成Agent（第二版）
本示例演示「目标设定与迭代自检」智能体模式，基于 LangChain + DeepSeek API 实现：

功能目标：构建AI代码生成智能体，根据业务需求与预设优化目标迭代生成Python代码
1. 支持自定义业务需求（中文描述）
2. 支持多组代码质量目标（简洁、容错、覆盖边界用例等）
3. 调用DeepSeek大模型生成、迭代优化代码，最多迭代5轮
4. 由大模型自动评审代码，判断是否全部达成目标，达标则终止迭代
5. 最终生成带注释头部的规范Python文件并本地保存
"""

import os
import random
import re
from pathlib import Path

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv, find_dotenv

# 加载环境变量
_ = load_dotenv(find_dotenv())
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
if not DEEPSEEK_API_KEY:
    raise EnvironmentError("❌ 请在.env文件中配置 DEEPSEEK_API_KEY 密钥")

# 初始化 DeepSeek 大模型
print("📡 正在连接 DeepSeek 大模型...")
llm = ChatOpenAI(
    model="deepseek-chat",
    temperature=0.3,
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com/v1"
)


# ---------------------- 工具函数 ----------------------
def generate_prompt(
    use_case: str, goals: list[str], previous_code: str = "", feedback: str = ""
) -> str:
    print("📝 组装代码生成提示词...")
    base_prompt = f"""
你是专业Python开发AI助手，根据下面的业务需求编写Python代码：

业务需求：{use_case}

代码需要满足以下全部目标：
{chr(10).join(f"- {g.strip()}" for g in goals)}
"""
    if previous_code:
        print("🔄 载入上一轮代码用于迭代优化")
        base_prompt += f"\n上一轮生成代码：\n{previous_code}"
    if feedback:
        print("📋 载入评审反馈用于修改")
        base_prompt += f"\n上一轮代码评审意见：\n{feedback}\n"

    base_prompt += "\n仅输出完整可运行的Python代码，不要在代码块外增加任何解释、说明文字。"
    return base_prompt


def get_code_feedback(code: str, goals: list[str]) -> str:
    print("🔍 调用模型评审代码是否满足目标...")
    feedback_prompt = f"""
你是严谨的Python代码评审专家，根据给定目标审查下方代码。
评审标准：
{chr(10).join(f"- {g.strip()}" for g in goals)}

逐条分析代码是否达标，指出缺陷、可优化点，包括可读性、正确性、边界场景容错等。

待评审代码：
{code}
"""
    return llm.invoke(feedback_prompt)


def goals_met(feedback_text: str, goals: list[str]) -> bool:
    """
    让大模型判断当前代码是否全部满足预设目标，仅返回True/False
    """
    review_prompt = f"""
你是代码验收审核员。
验收目标列表：
{chr(10).join(f"- {g.strip()}" for g in goals)}

代码评审反馈内容：
\"\"\"
{feedback_text}
\"\"\"

根据反馈判断代码是否完全达成所有目标。
仅允许输出一个单词：True 或 False
"""
    response = llm.invoke(review_prompt).content.strip().lower()
    return response == "true"


def clean_code_block(code: str) -> str:
    """移除代码块标记 ```python / ```"""
    lines = code.strip().splitlines()
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def add_comment_header(code: str, use_case: str) -> str:
    """给最终代码增加头部注释说明业务需求"""
    header = f"# 功能说明：{use_case.strip()}\n"
    return header + "\n" + code


def to_snake_case(text: str) -> str:
    """转为下划线小写文件名格式"""
    text = re.sub(r"[^a-zA-Z0-9 ]", "", text)
    return re.sub(r"\s+", "_", text.strip().lower())


def save_code_to_file(code: str, use_case: str) -> str:
    print("💾 正在保存最终代码文件...")

    # 生成简短文件名
    summary_prompt = (
        f"将下面业务需求总结为不超过10个字符的英文短名，仅用于Python文件名，只输出名称：\n\n{use_case}"
    )
    raw_summary = llm.invoke(summary_prompt).content.strip()
    short_name = re.sub(r"[^a-zA-Z0-9_]", "", raw_summary.replace(" ", "_").lower())[:10]

    random_suffix = str(random.randint(1000, 9999))
    filename = f"{short_name}_{random_suffix}.py"
    filepath = Path.cwd() / filename

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(code)

    print(f"✅ 代码已保存至：{filepath}")
    return str(filepath)


# ---------------------- 主智能体逻辑 ----------------------
def run_code_agent(use_case: str, goals_input: str, max_iterations: int = 5) -> str:
    goals = [g.strip() for g in goals_input.split("，")]

    print(f"\n🎯 当前业务需求：{use_case}")
    print("🎯 代码质量目标：")
    for g in goals:
        print(f"  - {g}")

    previous_code = ""
    feedback = ""

    for i in range(max_iterations):
        print(f"\n=== 🔁 第 {i + 1} / {max_iterations} 轮迭代 ===")
        prompt = generate_prompt(
            use_case,
            goals,
            previous_code,
            feedback.content if hasattr(feedback, "content") else feedback,
        )

        print("🚧 正在生成代码...")
        code_response = llm.invoke(prompt)
        raw_code = code_response.content.strip()
        code = clean_code_block(raw_code)
        print("\n🧾 当前生成代码：\n" + "-" * 50 + f"\n{code}\n" + "-" * 50)

        print("\n📤 提交代码进行评审...")
        feedback = get_code_feedback(code, goals)
        feedback_text = feedback.content.strip()
        print("\n📥 评审意见：\n" + "-" * 50 + f"\n{feedback_text}\n" + "-" * 50)

        if goals_met(feedback_text, goals):
            print("✅ 模型判定所有目标已达成，终止迭代")
            break

        print("🛠️ 未全部达标，进入下一轮优化")
        previous_code = code

    final_code = add_comment_header(code, use_case)
    return save_code_to_file(final_code, use_case)


# ---------------------- 测试入口（中文示例） ----------------------
if __name__ == "__main__":
    print("\n🧠 欢迎使用 DeepSeek 迭代式代码生成智能体")

    # 示例1：计算数字二进制间隔
    use_case_input = "编写代码，计算一个正整数的二进制间隔长度"
    goals_input = "代码简洁易懂，功能无错误，覆盖各类边界数字，仅接收正整数输入，附带多组示例打印结果"
    run_code_agent(use_case_input, goals_input)

    # 示例2：统计目录所有嵌套文件数量
    # use_case_input = "编写代码统计当前目录及所有子文件夹内全部文件总数并打印"
    # goals_input = "代码简洁易懂，逻辑正确，处理各类异常路径，不做性能优化，不增加单元测试"
    # run_code_agent(use_case_input, goals_input)

    # 示例3：读取Word文档统计文字与字符数量
    # use_case_input = "接收命令行传入的doc/docx文件，读取文档并统计总字数、总字符数并打印"
    # goals_input = "代码简洁易懂，功能正确，处理文件不存在、损坏等边界异常"
    # run_code_agent(use_case_input, goals_input)