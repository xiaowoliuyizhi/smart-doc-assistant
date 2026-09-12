# 混合检索与 RRF 融合设计

## 目标

在不移除现有纯向量 KNN 检索的前提下，新增 BM25 + 向量混合检索，并使用 RRF（Reciprocal Rank Fusion）合并候选。系统以响应速度为优先级，同时保留独立检索模式以支持后续 Recall@k 和 MRR 对比。

## 范围

- 保留纯向量检索作为 `vector` 模式和评测基线。
- 新增 `hybrid` 模式：KNN Top-10 与 BM25 Top-10 进行 RRF 融合，返回最终 Top-4。
- 在前端明确区分“语义相似度”和“综合排名”。
- 在内部保留候选来源与排名，供离线评测使用。

不包括 rerank、在线 A/B 测试或额外中文分词依赖。

## 配置

```env
RETRIEVAL_MODE=vector
TOP_K=4
VECTOR_CANDIDATE_K=10
BM25_CANDIDATE_K=10
RRF_K=60
```

- `RETRIEVAL_MODE` 可取 `vector` 或 `hybrid`，非法值在启动时拒绝。
- `TOP_K` 是实际传给 LLM 与前端展示的最终片段数。
- 两个候选数量仅适用于 `hybrid` 模式。
- 修改模式后重启服务；聊天 API 不新增参数，前端无需修改调用方式。

## 数据流

### vector

```text
问题 → Embedding → KNN Top-4 → LLM 与参考资料
```

### hybrid

```text
问题 → Embedding → KNN Top-10 ──┐
                                ├→ 按 chunk ID 去重 → RRF → Top-4 → LLM 与参考资料
问题 ───────────→ BM25 Top-10 ─┘
```

两条 Redis 查询可并行执行；Embedding 仅由 KNN 支路使用。BM25 使用已有的 RediSearch `content` TEXT 字段，不增加外部数据库或索引迁移。

## 检索接口

`VectorStore` 提供：

- `search(query_vector, top_k)`：现有 KNN 检索；
- `search_text(query, top_k)`：新的 BM25 检索；
- 两者都返回包含 `doc_id`、`doc_title`、`chunk_index`、`content`、`heading_path` 和原始分数的片段结构。

`RetrievalService` 提供：

- `retrieve_vector(query, candidate_k)`：供聊天和评测调用；
- `retrieve_hybrid(query, candidate_k)`：供聊天和评测调用；
- `retrieve(query, top_k)`：读取 `RETRIEVAL_MODE` 的统一入口。

内部结果会带有 `retrieval_mode`、`source`（`vector`、`bm25`、`both`）与 `rank`，以供评测和正确展示；不改变现有聊天请求格式。

## BM25 查询安全

查询文本先规范化空白和控制字符，再转义 RediSearch 查询保留字符，确保用户输入不会改变查询结构。第一版不引入中文分词依赖；中文 BM25 效果通过后续评测集验证。若效果不足，再独立评估轻量分词方案。

## RRF 融合

唯一片段键为：

```text
{doc_id}:{chunk_index}
```

融合分数：

```text
score(d) = Σ 1 / (RRF_K + rank_i(d))
```

- 每条检索列表的排名从 1 开始；
- 同时被两条链路命中的片段累计两次贡献；
- 仅在一条链路命中的片段保留一次贡献；
- RRF 分数降序；同分时按更好的向量排名、再按 `doc_id` 与 `chunk_index` 稳定排序；
- 最终截取 `TOP_K`。

## 前端展示

- `vector`：显示 `语义相似度 83% · 《标题》 · 片段 3`；
- `hybrid`：显示 `综合排名 #1 · 《标题》 · 片段 3`。

RRF 值不显示为百分比，因为它不是余弦相似度。旧会话缺少模式或标题时维持兼容：标题回退为文档 ID，评分显示沿用旧行为。

## 错误处理

- `vector`：沿用当前 KNN 错误行为；
- `hybrid` 的 BM25 查询为空或失败：使用 KNN 候选继续回答，并记录错误；
- KNN 失败：不输出伪造检索结果，沿用当前请求失败行为；
- 文档标题读取缺失：回退为 `doc_id`。

## 测试矩阵

| 场景 | 预期 |
|---|---|
| `vector` 模式 | 仅调用 KNN，返回语义分数 |
| `hybrid` 模式 | KNN 与 BM25 候选按 RRF 正确排序 |
| 两路命中同一 chunk | 结果去重，来源标记为 `both` |
| BM25 独有命中 | 纳入融合候选 |
| BM25 空结果/异常 | 自动回退 KNN |
| 特殊字符查询 | 安全转义，不改变查询语法 |
| 前端 vector 资料 | 显示语义相似度、标题、片段编号 |
| 前端 hybrid 资料 | 显示综合排名、标题、片段编号 |
| 旧历史 | 可正常加载并使用回退展示 |
| Recall@k 脚本 | 可分别取得 vector 与 hybrid 的 Top-k 结果 |

## 验收标准

1. `.env` 切换检索模式后，聊天 API 无需变更即可运行。
2. `vector` 的现有结果与行为保持兼容。
3. `hybrid` 的 Top-4 由 RRF 排序产生，且不会出现重复 chunk。
4. 前端不会将 RRF 分数显示成百分比。
5. 全量自动化测试通过，新增测试覆盖上述关键分支。
