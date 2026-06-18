import os
import asyncio
from typing import Optional, Dict
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.memory import ConversationBufferMemory

## --- 0. DeepSeek环境配置 ---
load_dotenv()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
if not DEEPSEEK_API_KEY:
    raise EnvironmentError("请在.env文件中配置 DEEPSEEK_API_KEY")

llm = ChatOpenAI(
    temperature=0,
    model="deepseek-chat",
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com/v1"
)

## --- 1. 内存任务管理模块 ---
class Task(BaseModel):
    id: str
    description: str
    priority: Optional[str] = None
    assigned_to: Optional[str] = None

class SuperSimpleTaskManager:
    def __init__(self):
        self.tasks: Dict[str, Task] = {}
        self.next_task_id = 1

    def create_task(self, description: str) -> Task:
        task_id = f"TASK-{self.next_task_id:03d}"
        new_task = Task(id=task_id, description=description)
        self.tasks[task_id] = new_task
        self.next_task_id += 1
        print(f"DEBUG: 任务创建成功 - {task_id}: {description}")
        return new_task

    def update_task(self, task_id: str, **kwargs) -> Optional[Task]:
        task = self.tasks.get(task_id)
        if task:
            update_data = {k: v for k, v in kwargs.items() if v is not None}
            updated_task = task.model_copy(update=update_data)
            self.tasks[task_id] = updated_task
            print(f"DEBUG: 任务 {task_id} 已更新，更新内容：{update_data}")
            return updated_task
        print(f"DEBUG: 未找到ID为 {task_id} 的任务，更新失败")
        return None

    def list_all_tasks(self) -> str:
        if not self.tasks:
            return "当前系统暂无任何任务"
        task_strings = []
        for task in self.tasks.values():
            task_strings.append(
                f"任务ID: {task.id}，描述: '{task.description}'，"
                f"优先级: {task.priority if task.priority else '未设置'}，"
                f"负责人: {task.assigned_to if task.assigned_to else '未分配'}"
            )
        return "==== 当前全部任务列表 ====\n" + "\n".join(task_strings)

task_manager = SuperSimpleTaskManager()

## --- 2. 工具定义 ---
@tool
def create_new_task(description: str) -> str:
    """创建新项目任务，第一步必须调用此工具获取task_id
    Args:
        description: 任务详细文字描述
    """
    task = task_manager.create_task(description)
    return f"创建任务完成，任务编号{task.id}：{task.description}"

@tool
def assign_priority_to_task(task_id: str, priority: str) -> str:
    """给已有任务设置优先级，仅支持 P0 / P1 / P2
    Args:
        task_id: 任务编号，示例 TASK-001
        priority: 优先级，P0最高紧急、P1普通、P2低延后
    """
    if priority not in ["P0", "P1", "P2"]:
        return "优先级参数无效，仅可设置 P0 / P1 / P2"
    task = task_manager.update_task(task_id, priority=priority)
    return f"已为任务{task_id}设置优先级{priority}" if task else f"不存在编号为{task_id}的任务"

@tool
def assign_task_to_worker(task_id: str, worker_name: str) -> str:
    """将任务分配给指定工作人员
    Args:
        task_id: 任务编号，示例 TASK-001
        worker_name: 工作人员姓名
    """
    task = task_manager.update_task(task_id, assigned_to=worker_name)
    return f"已将任务{task_id}分配给{worker_name}" if task else f"不存在编号为{task_id}的任务"

@tool
def list_all_tasks() -> str:
    """查看系统全部任务、优先级、负责人"""
    return task_manager.list_all_tasks()

pm_tools = [create_new_task, assign_priority_to_task, assign_task_to_worker, list_all_tasks]

## --- 3. 修复prompt：增加 {agent_scratchpad} 占位符 ---
prompt = ChatPromptTemplate.from_messages([
    ("system", """你是专业项目管理智能体，严格按流程处理任务：
1. 新建任务必须先调用 create_new_task，拿到task_id再执行后续操作；
2. 紧急需求 → P0；普通需求默认P1；延后需求 → P2；未指定人员默认分配 Worker A；
3. 用户指定优先级/负责人，依次调用 assign_priority_to_task、assign_task_to_worker；
4. 全部操作完成后调用 list_all_tasks 汇总所有任务；
可选工作人员：Worker A、Worker B、Review Team
优先级定义：P0最高紧急，P1中等常规，P2最低延后
"""),
    ("placeholder", "{chat_history}"),
    ("human", "{input}"),
    # 关键补充：工具调用智能体必须的中间步骤存储占位符
    ("placeholder", "{agent_scratchpad}"),
])

# 记忆（警告不影响运行，只是版本提示）
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# 构建原生工具调用智能体
agent = create_tool_calling_agent(llm, pm_tools, prompt)

pm_agent_executor = AgentExecutor(
    agent=agent,
    tools=pm_tools,
    verbose=True,
    handle_parsing_errors=True,
    memory=memory
)

## --- 4. 异步测试 ---
async def run_simulation():
    print("===== 项目管理智能体模拟运行 ====")
    print("\n【用户需求】需要快速开发新登录系统，交给Worker B负责，任务紧急")
    await pm_agent_executor.ainvoke({"input": "创建任务：开发新版登录系统，需求紧急，分配给Worker B"})
    print("\n" + "-"*60 + "\n")

    print("【用户需求】需要审核官网营销文案")
    await pm_agent_executor.ainvoke({"input": "新建任务：审核营销网站全部文案内容"})
    print("\n===== 模拟执行完毕 ====")

if __name__ == "__main__":
    asyncio.run(run_simulation())