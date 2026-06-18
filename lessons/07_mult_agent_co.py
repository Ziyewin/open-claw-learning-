import os
from re import T
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from crewai import LLM


def setup_environment():
    """加载环境变量并校验密钥"""
    load_dotenv()
    if not os.getenv("DEEPSEEK_API_KEY"):
        raise ValueError("DEEPSEEK_API_KEY 未找到，请在 .env 文件中配置")


def main():
    """运行多智能体完成 AI 趋势调研与博客撰写"""
    setup_environment()

    # 初始化 DeepSeek 模型（CrewAI 原生写法）
    llm = LLM(
        model="deepseek/deepseek-chat",
        api_key=os.getenv("DEEPSEEK_API_KEY")
    )

    # 定义智能体
    researcher = Agent(
        role='资深行业研究员',
        goal='调研并总结当下人工智能领域的前沿趋势',
        backstory="你是经验丰富的行业分析师，擅长提炼核心趋势、整合信息并输出条理清晰的总结内容。",
        verbose=True,
        allow_delegation=False,
        llm=llm
    )

    writer = Agent(
        role='技术内容撰稿人',
        goal='根据调研结果撰写通俗易懂、可读性强的博客文章',
        backstory="你擅长将复杂的技术内容转化为大众容易理解的文字，行文流畅、风格自然。",
        verbose=True,
        allow_delegation=False,
        llm=llm
    )

    # 定义任务
    research_task = Task(
        description="调研2024-2025年人工智能领域三大新兴趋势，重点分析实际应用场景与行业影响。",
        expected_output="三大AI趋势的详细总结，包含核心要点与应用价值说明。",
        agent=researcher,
    )

    writing_task = Task(
        description="基于上方调研结果，撰写一篇500字左右的博客文章，内容生动易懂，面向普通读者。",
        expected_output="一篇完整、通顺的AI趋势主题博客文章，字数约500字。",
        agent=writer,
        context=[research_task],
    )

    # 组建团队并串行执行任务
    blog_creation_crew = Crew(
        agents=[researcher, writer],
        tasks=[research_task, writing_task],
        process=Process.sequential,
        llm=llm,
        verbose=True
    )

    # 执行任务
    print("## 开始执行 AI 趋势调研与博客撰写任务 ##")
    try:
        result = blog_creation_crew.kickoff()
        print("\n------------------\n")
        print("## 最终输出结果 ##")
        print(result)
    except Exception as e:
        print(f"\n程序运行出错：{e}")


if __name__ == "__main__":
    main()