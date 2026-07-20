# CourseRAG

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF0F6F?logo=streamlit&logoColor=white)](https://streamlit.io/)

一个可本地运行的课程资料 RAG 问答系统。支持多格式资料上传、语义检索、引用溯源、回答导出与聊天历史导出。

## 功能

- 支持上传 PDF、DOCX、PPT、XLS、CSV、MD、TXT、JSON、HTML 等资料
- 自动解析并做章节/页码感知分块
- 基于本地向量库做语义检索
- 生成带引用来源的回答
- 支持单轮导出与全部聊天历史导出

## 技术栈

- Python 3.10+
- Streamlit
- ChromaDB
- ONNX MiniLM-L6-v2
- OpenAI 兼容接口

## 快速开始

推荐先建立虚拟环境：

```bash
git clone https://github.com/zhongrongjun1220-dev/courserag.git
cd courserag
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cp .env.example .env
```

编辑 `.env`：

```env
OPENAI_API_KEY=你的密钥
OPENAI_API_BASE=https://api.openai.com/v1
OPENAI_MODEL=gpt-5.5
```

启动项目：

```bash
streamlit run app.py
```

打开 `http://localhost:8501`，上传 `data/smart_factory_knowledge_base.md` 即可体验。

## 常见问题

**Q: 本地没有 GPU 能跑吗？**
A: 可以。embedding 模型采用 ONNX MiniLM-L6-v2，CPU 即可运行；生成回答依赖你配置的模型接口。

**Q: 回答质量不稳定怎么办？**
A: 先检查上传资料是否完整；可以调整 `app.py` 里的 `TOP_K`；也可以替换更强的大模型。若 Chroma 出现 `hnsw` 相关错误，删除 `storage/` 后重新上传资料即可。

**Q: 项目里为什么不用更重的 RAG 框架？**
A: 这里刻意保持最小依赖，方便讲清 RAG 每层职责，也更容易调试和扩展。

## 项目结构

```text
courserag/
├── app.py                 # Streamlit 主程序
├── loaders.py             # 多格式解析、分块与溯源信息构造
├── requirements.txt       # 运行依赖
├── .env.example           # 接口配置模板
├── .gitignore             # 忽略密钥、向量库、导出文件
├── data/                  # 演示资料
│   └── smart_factory_knowledge_base.md
├── storage/               # 本地 Chroma 持久化目录，不提交
└── exports/               # 导出文件目录，不提交
```

## 后续可优化方向

- 增加 reranker，减少“能找到但答不准”的问题
- 增加评测集，量化 recall / answer relevancy
- 支持多用户会话、鉴权、用量统计
- 将 chunk metadata 和 citation 设计成可配置 schema

## 许可证

MIT
