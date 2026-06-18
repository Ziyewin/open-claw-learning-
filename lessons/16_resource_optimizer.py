# MIT License
# Copyright (c) 2025 Mahtab Syed
# 已适配 DeepSeek + LangChain，模拟搜索
import os
import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

# 加载环境变量
load_dotenv()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
if not DEEPSEEK_API_KEY:
    raise ValueError("请在.env文件中配置 DEEPSEEK_API_KEY")

# 初始化DeepSeek大模型（LangChain标准封装）
llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com/v1",
    temperature=0.1
)

# ===================== 1. 定义分类结构化输出格式 =====================
class QueryClassify(BaseModel):
    classification: str = Field(description="仅允许输出：simple / reasoning / internet_search")

# 分类提示词（全中文）
classify_prompt = ChatPromptTemplate.from_messages([
    ("system", """
你是用户问题分类器，只能返回下面三类其中一个，严格遵循规则：
分类选项仅三个：simple、reasoning、internet_search

判定规则：
1. simple：简单静态常识，无需复杂逻辑、无需实时网络资讯（如国家首都、固定基础常识）
2. reasoning：需要数学计算、逻辑推导、多步骤拆解、专业原理分析
3. internet_search：需要最新新闻、当年赛事、实时数据、模型训练截止时间之后的信息

输出要求：仅返回标准JSON，格式固定：{{"classification": "simple"}}
禁止额外解释、禁止多余文字。
"""),
    ("human", "待分类用户问题：{user_query}")
])

# 分类链：Prompt + LLM + JSON解析器
classify_chain = classify_prompt | llm | JsonOutputParser(pydantic_object=QueryClassify)

# ===================== 2. 本地模拟搜索工具（替代Google CSE，离线无密钥） =====================
def mock_search(query: str, num_results=1) -> list:
    """模拟互联网搜索，内置预设资讯数据，无需联网、无需第三方API"""
    # 模拟知识库
    mock_database = {
        "2026澳网开赛日期": [
            {
                "title": "2026澳大利亚网球公开赛赛程官方公告",
                "snippet": "2026年澳大利亚网球公开赛将于2026年1月12日正式开赛，为期14天，决赛在1月25日举行。",
                "link": "https://tennis.org/ausopen2026"
            }
        ],
        "量子计算对密码学的影响": [
            {
                "title": "后量子密码标准与量子计算安全风险",
                "snippet": "大型量子计算机可快速破解RSA、ECC传统加密算法，各国已启动后量子密码标准化，商用改造周期约5-10年。",
                "link": "https://crypto.org/post-quantum"
            }
        ],
        "澳大利亚首都是哪里": [
            {
                "title": "澳大利亚国家行政中心介绍",
                "snippet": "澳大利亚首都是堪培拉，并非悉尼或墨尔本，1913年正式定为联邦首都。",
                "link": "https://country-data.com/australia"
            }
        ]
    }
    # 模糊匹配检索
    match_data = None
    for keyword, res_list in mock_database.items():
        if keyword in query:
            match_data = res_list[:num_results]
            break
    if match_data:
        return match_data
    else:
        return [{"title": "无匹配资讯", "snippet": "未检索到相关实时信息", "link": ""}]

# ===================== 3. 多模型生成回答 =====================
def generate_answer(user_query: str, classify_type: str, search_data=None) -> tuple[str, str]:
    """根据分类选择不同模型策略，整合搜索上下文生成回答"""
    # DeepSeek分层模型策略模拟（对应原代码不同GPT模型分工）
    if classify_type == "simple":
        model_tag = "deepseek-chat-light"
        prompt_text = user_query
    elif classify_type == "reasoning":
        model_tag = "deepseek-reasoner"
        prompt_text = user_query
    elif classify_type == "internet_search":
        model_tag = "deepseek-chat"
        # 拼接搜索上下文
        if search_data:
            context_blocks = []
            for item in search_data:
                block = f"标题：{item['title']}\n摘要：{item['snippet']}\n来源链接：{item['link']}"
                context_blocks.append(block)
            search_context = "\n---\n".join(context_blocks)
        else:
            search_context = "暂无检索到的网络资料"
        prompt_text = f"""参考以下检索资讯回答用户问题：
【检索资料】
{search_context}
用户问题：{user_query}
请基于资料内容客观完整作答。
"""
    else:
        raise ValueError("分类类型非法")

    # 调用模型生成回答
    resp = llm.invoke(prompt_text)
    return resp.content.strip(), model_tag

# ===================== 4. 总路由调度入口 =====================
def route_handle_query(user_input: str) -> dict:
    # 第一步：问题分类
    classify_result = classify_chain.invoke({"user_query": user_input})
    category = classify_result["classification"]
    search_result = None

    # 第二步：如果需要检索，执行模拟搜索
    if category == "internet_search":
        search_result = mock_search(user_input, num_results=1)

    # 第三步：生成最终回答
    final_ans, used_model = generate_answer(user_input, category, search_result)

    return {
        "query_class": category,
        "used_model": used_model,
        "search_info": search_result,
        "answer": final_ans
    }

# ===================== 5. 中文测试案例 =====================
if __name__ == "__main__":
    # # 测试1：simple简单常识
    # test1 = "澳大利亚的首都是哪里？"
    # print("===== 测试1【简单常识 simple】 =====")
    # res1 = route_handle_query(test1)
    # print(f"问题分类：{res1['query_class']}")
    # print(f"使用模型：{res1['used_model']}")
    # print(f"回答：\n{res1['answer']}\n")
    # print("-" * 70)

    # # 测试2：reasoning深度推理
    # test2 = "请讲解量子计算对现代密码体系带来的冲击与应对方案"
    # print("===== 测试2【逻辑推理 reasoning】 =====")
    # res2 = route_handle_query(test2)
    # print(f"问题分类：{res2['query_class']}")
    # print(f"使用模型：{res2['used_model']}")
    # print(f"回答：\n{res2['answer']}\n")
    # print("-" * 70)

    # 测试3：internet_search实时资讯（需要检索）
    test3 = "2026澳网开赛日期是什么时候，请给出完整日期"
    print("===== 测试3【实时检索 internet_search】 =====")
    res3 = route_handle_query(test3)
    print(f"问题分类：{res3['query_class']}")
    print(f"使用模型：{res3['used_model']}")
    print(f"检索资料：{res3['search_info']}")
    print(f"回答：\n{res3['answer']}")