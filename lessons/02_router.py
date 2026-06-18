import os
import warnings
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableBranch

# 忽略 SSL 无关警告
warnings.filterwarnings("ignore")

# 加载环境变量
load_dotenv()

# 1. 初始化 DeepSeek 模型
llm = ChatOpenAI(
    base_url="https://api.deepseek.com/v1",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    model_name="deepseek-chat",
    temperature=0
)

# 2. 三个业务处理函数
def book_service(text: str) -> str:
    return f"【预订服务】已处理请求：{text}"

def query_service(text: str) -> str:
    return f"【查询服务】已处理请求：{text}"

def unknown_service(text: str) -> str:
    return f"【未知服务】无法识别需求，请重新描述"

# 3. 路由提示词：让模型分类
route_prompt = ChatPromptTemplate.from_messages([
    ("system", "判断用户请求类型，只输出一个单词：book / query / unknown\n"
               "book=订机票/酒店，query=知识查询，unknown=其他"),
    ("user", "{content}")
])

# 4. 分类链路
route_chain = route_prompt | llm | StrOutputParser()

# 5. 分支全部用 lambda
branch = RunnableBranch(
    (lambda x: x["type"].strip() == "book", lambda x: book_service(x["content"])),
    (lambda x: x["type"].strip() == "query", lambda x: query_service(x["content"])),
    lambda x: unknown_service(x["content"])
)

# 6. 组装整条链路
full_chain = {
    "type": route_chain,
    "content": RunnablePassthrough()
} | branch

# 7. 测试
if __name__ == "__main__":
    test1 = "帮我订一张去上海的机票"
    test2 = "地球赤道周长是多少"
    test3 = "随便聊聊天吧"

    print(full_chain.invoke(test1))
    print("-" * 40)
    print(full_chain.invoke(test2))
    print("-" * 40)
    print(full_chain.invoke(test3))