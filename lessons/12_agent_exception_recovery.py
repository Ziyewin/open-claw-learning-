import logging
import time
from typing import Optional, Dict, Any
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.tools import tool
import os
# ===================== 1. 日志配置（异常记录，用于问题追溯） =====================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ===================== 2. 加载环境变量 & 初始化DeepSeek模型 =====================
load_dotenv()
llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1",
    temperature=0
)

# 全局状态：流转工具执行结果、失败标记（对应原文state状态管理）
agent_state: Dict[str, Any] = {
    "main_tool_failed": False,   # 主工具是否失败
    "file_detail": "",           # 主工具结果：文件详细信息
    "file_basic": ""             # 备用工具结果：文件基础信息
}

# ===================== 3. 工具定义（主工具 + 备用降级工具） =====================
# 模拟主工具：读取文件详细信息（人为植入故障场景）
@tool
def get_file_detail(file_path: str) -> str:
    """
    【主工具】读取文件完整详情（大小、行数、内容摘要）
    模拟异常：随机触发文件损坏、路径错误、权限异常
    """
    try:
        # 模拟线上随机故障（演示异常场景，正式环境删除这段）
        import random
        fault_rate = 0.6  # 60%概率触发异常，测试恢复逻辑
        if random.random() < fault_rate:
            raise FileNotFoundError(f"文件 {file_path} 不存在或已损坏")

        # 正常业务逻辑（读取文件）
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        line_count = len(content.splitlines())
        file_size = len(content)
        return f"文件详情：路径={file_path}，行数={line_count}，字符数={file_size}，内容摘要={content[:20]}..."

    except Exception as e:
        logger.error(f"主工具执行异常：{str(e)}")
        agent_state["main_tool_failed"] = True
        raise e

# 模拟备用降级工具：仅获取文件名（主工具失效时启用）
@tool
def get_file_basic(file_path: str) -> str:
    """【备用工具】降级方案，仅提取文件名，保障基础可用"""
    try:
        file_name = file_path.split("/")[-1]
        return f"文件基础信息：文件名={file_name}"
    except Exception as e:
        logger.error(f"备用工具执行异常：{str(e)}")
        raise e

# ===================== 4. 异常重试封装（处理瞬时故障） =====================
def execute_with_retry(tool_func, args: str, max_retry: int = 2, delay: int = 1) -> Optional[str]:
    """
    为工具增加重试机制：针对网络、临时文件锁等瞬时异常
    :param tool_func: 工具函数
    :param args: 工具入参
    :param max_retry: 最大重试次数
    :param delay: 重试间隔(秒)
    """
    for retry_times in range(max_retry + 1):
        try:
            return tool_func.invoke(args)
        except Exception as e:
            if retry_times < max_retry:
                logger.warning(f"第{retry_times + 1}次执行失败，{delay}秒后重试...")
                time.sleep(delay)
                continue
            logger.error(f"重试{max_retry}次后仍失败")
            return None
    return None

# ===================== 5. 分层智能体（顺序执行，和原文SequentialAgent对齐） =====================
def primary_agent(file_path: str) -> None:
    """第一层智能体：优先执行主工具"""
    logger.info("===== 第一层：执行主工具（文件详情查询） =====")
    result = execute_with_retry(get_file_detail, file_path)
    if result:
        agent_state["file_detail"] = result

def fallback_agent(file_path: str) -> None:
    """第二层智能体：主工具失败时，触发降级备用工具"""
    logger.info("===== 第二层：检测状态，判断是否启用降级工具 =====")
    if agent_state["main_tool_failed"]:
        logger.info("检测到主工具故障，启动备用降级方案")
        result = execute_with_retry(get_file_basic, file_path)
        if result:
            agent_state["file_basic"] = result
    else:
        logger.info("主工具执行正常，无需启用备用工具")

def response_agent() -> str:
    """第三层智能体：统一整理结果，向用户输出（优雅兜底）"""
    logger.info("===== 第三层：汇总结果，生成最终回复 =====")
    if agent_state["file_detail"]:
        return f"✅ 查询成功（完整信息）：{agent_state['file_detail']}"
    elif agent_state["file_basic"]:
        return f"⚠️  文件详情查询失败，已启用降级服务：{agent_state['file_basic']}"
    else:
        return "❌ 所有服务均异常，暂时无法查询文件信息，请稍后重试"

# ===================== 6. 主流程入口 =====================
def run_file_agent(target_file: str):
    """串联全流程：主工具 → 降级判断 → 结果输出"""
    # 重置状态（每次请求独立）
    agent_state["main_tool_failed"] = False
    agent_state["file_detail"] = ""
    agent_state["file_basic"] = ""

    # 分层顺序执行（和原文SequentialAgent逻辑完全一致）
    primary_agent(target_file)
    fallback_agent(target_file)
    final_result = response_agent()

    # 打印最终结果
    print("\n【最终回复】")
    print(final_result)
    print("-" * 60)

if __name__ == "__main__":
    # 测试：使用当前目录下的自身代码文件（保证路径有效）
    test_file = "./12_agent_exception_recovery.py"
    # 多次运行，模拟多次异常场景
    for i in range(4):
        print(f"\n========== 第{i+1}轮测试 ==========")
        run_file_agent(test_file)