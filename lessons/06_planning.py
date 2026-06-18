import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from crewai import LLM

# 加载环境变量
load_dotenv()

# 原生指定 deepseek 服务商
llm = LLM(
    model="deepseek/deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY")
)

# 定义智能体
planner_writer_agent = Agent(
    role='文章规划与撰稿专员',
    goal='针对指定主题梳理写作框架，并撰写简洁易懂的内容总结',
    backstory=(
        '你是资深技术文案与内容策划师，擅长先制定清晰可行的写作大纲，'
        '再依据大纲完成正文创作，保证内容兼具专业性与可读性。'
    ),
    verbose=True,
    allow_delegation=False,
    llm=llm
)

# 任务主题
topic = "强化学习在人工智能领域的重要作用"

high_level_task = Task(
    description=(
        f"1. 围绕主题「{topic}」，以列表形式梳理写作大纲。\n"
        f"2. 按照大纲撰写总结内容，全文控制在200字左右。"
    ),
    expected_output=(
        "最终输出分为两个板块：\n\n"
        "### 写作大纲\n"
        "- 使用列表列出总结的核心要点\n\n"
        "### 内容总结\n"
        "- 结构完整、语言简练的主题总结"
    ),
    agent=planner_writer_agent
)

# 编排任务流程
crew = Crew(
    agents=[planner_writer_agent],
    tasks=[high_level_task],
    process=Process.sequential
)

# 执行
print("## 开始执行规划与写作任务 ##")
result = crew.kickoff()

print("\n\n---\n## 任务执行结果 ##\n---")
print(result)