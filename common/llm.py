import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# 从项目根目录加载 .env
load_dotenv(Path(__file__).resolve().parent.parent / ".env")


def get_llm(temperature: float = 0) -> ChatOpenAI:
    """初始化 DeepSeek 模型（兼容 OpenAI 接口格式）。"""
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError(
            "未找到 DEEPSEEK_API_KEY，请在 .env 文件中配置你的 API Key"
        )

    return ChatOpenAI(
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
        api_key=api_key,
        model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        temperature=temperature,
    )
