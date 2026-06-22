# Agent Learning

一套面向 AI Agent 工程实践的动手课程，涵盖从基础编排模式到多智能体协作、MCP、RAG、A2A 等前沿主题。所有示例使用 **DeepSeek** 作为 LLM 后端，基于 **LangChain** + **LangGraph** + **CrewAI** 生态实现。

## 目录

- [项目概览](#项目概览)
- [快速开始](#快速开始)
- [课程一览](#课程一览)
- [项目结构](#项目结构)
- [技术栈](#技术栈)
- [许可](#许可)

## 项目概览

AI Agent 正在从单一对话模型演进为能感知环境、使用工具、制定计划、协同多智能体的主动系统。这个仓库用 18 个渐进式课程，系统拆解 Agent 的核心构建块：

- **基础编排** — Prompt Chain、Router、Parallel、Reflection
- **能力扩展** — Tool Use、Planning、Memory、Adaptation
- **多智能体** — Multi-Agent Collaboration、Agent-to-Agent (A2A)
- **协议与集成** — MCP (Model Context Protocol)、Human-in-the-Loop
- **进阶主题** — RAG Hybrid Search、Reasoning Techniques、Resource Optimizer、Exception Recovery、Goal Monitoring

每节课都是一个可独立运行的 Python 脚本，附有清晰的注释和运行说明。

## 快速开始

### 1. 克隆并进入项目

```bash
git clone <repo-url>
cd agent_learning
```

### 2. 创建虚拟环境

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

> 推荐 Python 3.10+。

### 4. 配置 API Key

```bash
cp .env.example .env
```

编辑 `.env`，填入你的 DeepSeek API Key：

```env
DEEPSEEK_API_KEY=sk-your_key_here
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
DEEPSEEK_MODEL=deepseek-chat
```

> 所有课程统一通过 `common/llm.py` 读取环境变量初始化 LLM，修改一处即可全局生效。

### 5. 运行课程

```bash
python lessons/01_prompt_chain.py
```

## 课程一览

| # | 文件 | 主题 | 核心概念 |
|---|------|------|----------|
| 01 | `01_prompt_chain.py` | Prompt Chain | LCEL 串联、提取 + 转换流水线 |
| 02 | `02_router.py` | Router | 意图分类、动态路由分发 |
| 03 | `03_Parallel.py` | Parallel Execution | 并行调用、结果合并 |
| 04 | `04_Reflection.py` | Reflection | 自评与纠错、迭代优化 |
| 05 | `05_tool_use.py` | Tool Use | 函数调用、外部工具集成 |
| 06 | `06_planning.py` | Planning | 任务分解、子目标编排 |
| 07 | `07_mult_agent_co.py` | Multi-Agent Collaboration | 多角色分工、信息交换 |
| 08 | `08_memory.py` | Memory | 会话记忆、短期/长期记忆 |
| 09 | `09_adaptation.py` | Adaptation | 动态适配、上下文感知 |
| 10 | `10_mcp.py` | MCP (Model Context Protocol) | FastMCP Client、标准工具调用 |
| 11 | `11_goal_set_and_monter.py` | Goal Setting & Monitoring | 目标设定、进度监控 |
| 12 | `12_agent_exception_recovery.py` | Agent Exception Recovery | 异常处理、自动恢复 |
| 13 | `13_HITL.py` | Human-in-the-Loop | 人工介入、审批流程 |
| 14 | `14_rag_hybrid_search.py` | RAG Hybrid Search | BM25 + 向量检索 + RRF 融合 |
| 15 | `15_A2A.py` | Agent-to-Agent (A2A) | 客服 ↔ 数据分析跨 Agent 通信 |
| 16 | `16_resource_optimizer.py` | Resource Optimizer | 资源预算、成本优化 |
| 17 | `17_Reasoning_Techniques.py` | Reasoning Techniques | CoT、ToT、ReAct 等推理策略 |
| 18 | `18_prior.py` | Prioritized Task Management | 优先队列、多 Agent 任务分配 |

此外还有两个辅助文件：

- `mcp_server.py` — MCP Server 示例，配合第 10 课使用
- `data/` — DeepSeek 知识库文件，供第 14 课 RAG 检索使用

## 项目结构

```
agent_learning/
├── .env                  # API Key 等环境变量（已 gitignore）
├── .env.example          # 环境变量模板
├── requirements.txt      # Python 依赖
├── README.md
├── common/
│   ├── __init__.py
│   └── llm.py            # 统一的 LLM 初始化（DeepSeek）
├── data/                 # RAG 课程使用的知识库文件
│   ├── deepseek_intro.txt
│   ├── deepseek_api.txt
│   ├── deepseek_best_practices.txt
│   ├── deepseek_capabilities.txt
│   └── deepseek_pricing.txt
└── lessons/              # 18 节课程 + MCP Server 示例
    ├── 01_prompt_chain.py
    ├── ...
    └── 18_prior.py
```

## 技术栈

| 组件 | 用途 |
|------|------|
| **DeepSeek API** | LLM 后端（OpenAI 兼容接口） |
| **LangChain** | LCEL 编排、提示词模板、输出解析、工具调用 |
| **LangGraph** | Agent 状态图、Checkpoint 持久化 |
| **CrewAI** | 多智能体角色编排 |
| **FastMCP** | MCP 协议客户端与服务端 |
| **LangSmith** | 链路追踪与调试 |
| **ChromaDB** | 向量存储（第 14 课） |
| **rank-bm25 + jieba** | 关键词检索与中文分词（第 14 课） |

完整依赖见 [requirements.txt](requirements.txt)。

## 许可

本项目基于 MIT 协议开源。
