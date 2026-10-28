# Local Life Agent — 工具定义

本文件由 `local_life_agent/tools/` 下的工具实现自动整理，描述当前 Agent 可调用的全部工具、入参 schema 与返回结构。

## 概览

| 工具 | 用途 | 召回方式 | 必填参数 |
| --- | --- | --- | --- |
| `search_pois` | 结构化检索地点/商家候选，或解析地标坐标 | Qwen3 向量召回 + reranker 重排 | `query` |
| `search_notes` | 检索用户体验类证据（安静、服务、正宗、带娃、口碑等） | Qwen3 向量召回 + reranker 重排 | `query` |
| `get_poi_notes` | 按精确 `poiid` 拉取该 POI 的全部本地笔记 | 精确查询，无近似召回 | `poiid` |
| `search_nearby` | 在坐标 + 半径内检索 POI，Haversine 精确过滤距离 | 向量召回 + 空间索引过滤 + reranker | `query`, `loc`, `distance_m` |

运行时全局配置（`AgentConfig` 默认值）：

- `default_top_k = 10`，`max_top_k = 20`（所有工具的 `top_k` 上限统一受此约束；取 20 而非 30 是因为同轮两个工具调用时每个观察只有 16000 字符可见预算，30 条结果约 17471 字符会被截断，而 20 条约 11852 字符可完整送达）
- `default_nearby_distance_m = 3000`，`max_nearby_distance_m = 50000`
- 每轮最多 `max_tools_per_round = 2` 个调用，最多 `max_tool_rounds = 5` 轮，累计 `max_tool_calls = 10` 次
- 单次观测（observation）最大 `max_observation_chars = 32000` 字符

---

## 统一调用封装

所有工具经 `ToolRegistry.execute(name, arguments)` 调用，返回统一信封：

成功：

```json
{ "ok": true, "tool": "<name>", "data": { ... } }
```

失败：

```json
{ "ok": false, "tool": "<name>", "error": "<原因>", "error_type": "input | runtime" }
```

注册表的入参校验规则（对所有工具生效）：

1. `arguments` 必须是 JSON object。
2. schema 中 `required` 列出的字段缺失 → `error_type: input`，报「缺少必填参数」。
3. 所有工具 schema 均为 `additionalProperties: false`，传入未定义字段 → 报「出现未定义参数」。
4. `error_type: input` 表示参数校验失败；`error_type: runtime` 表示工具执行内部异常。

---

## search_pois

从 POI 结构化 RAG 库以 Qwen3 纯向量召回地点或商家候选，并经 Qwen3 reranker 重排。适合查名称、类别、地址、商圈、评分、人均、营业时间、设施标签，也可用于先解析命名地标的坐标。结果中的 `comment` 是 0/1 标志：`1` 表示该 POI 至少有一条笔记，可继续调用 `get_poi_notes`。

### 入参

| 参数 | 类型 | 必填 | 约束 | 说明 |
| --- | --- | --- | --- | --- |
| `query` | string | 是 | 非空，≤ 500 字符 | POI 召回文本或精确 `poiid` |
| `top_k` | integer | 否 | 1 ~ `max_top_k`(20)，默认 10 | 返回条数 |
| `category` | string | 否 | ≤ 100 字符 | 类别/类别路径包含过滤 |
| `district` | string | 否 | ≤ 100 字符 | 行政区包含过滤 |
| `business_area` | string | 否 | ≤ 100 字符 | 商圈包含过滤 |
| `min_rating` | number | 否 | 0 ~ 5，非负 | 最低评分 |
| `min_star` | number | 否 | 0 ~ 5，非负 | 最低星级 |
| `min_average_cost` | number | 否 | ≥ 0，且 ≤ `max_average_cost` | 最低人均 |
| `max_average_cost` | number | 否 | ≥ 0 | 最高人均 |
| `min_parking_spaces` | integer | 否 | ≥ 0 | 最少停车位数 |
| `has_notes` | boolean | 否 | — | 是否必须有非空笔记 |

### 返回 `data`

```json
{
  "query": "<回显查询>",
  "retrieval_mode": "<召回模式，如 vector_qwen3 / lexical_fts5_bm25>",
  "applied_filters": { "category": "...", "min_rating": 4.5 },
  "count": 10,
  "results": [
    { "poiid": "...", "comment": 1, "evidence_id": "poi:<poiid>", "...": "其余POI字段" }
  ]
}
```

每条结果自动附加 `evidence_id = "poi:" + poiid`，用于最终答案的 `evidence` 引用。

---

## search_notes

从全局笔记 RAG 库以 Qwen3 纯向量召回用户体验证据候选，并经 Qwen3 reranker 重排。适合「安静、服务、正宗、适合带娃、环境、口碑」等无法由 POI 结构字段直接证明的条件。返回 `nid`、`poiid`、正文和标签。

### 入参

| 参数 | 类型 | 必填 | 约束 | 说明 |
| --- | --- | --- | --- | --- |
| `query` | string | 是 | 非空，≤ 500 字符 | 笔记内容召回文本 |
| `category` | string | 否 | ≤ 100 字符 | 笔记关联 POI 类别路径包含过滤 |
| `top_k` | integer | 否 | 1 ~ `max_top_k`(20)，默认 10 | 返回条数 |

### 返回 `data`

```json
{
  "query": "<回显查询>",
  "retrieval_mode": "<召回模式>",
  "applied_filters": { "category": "..." },
  "per_poi_limit": 2,
  "count": 10,
  "results": [
    { "nid": "...", "poiid": "...", "evidence_id": "note:<nid>", "...": "正文/标签等" }
  ]
}
```

`per_poi_limit = 2`：同一 POI 在全局笔记召回中最多返回 2 条，避免单一 POI 霸占结果。每条结果附加 `evidence_id = "note:" + nid`。

---

## get_poi_notes

按精确 `poiid` 查询该 POI 的全部本地笔记，不做近似召回。通常仅在 POI 召回结果 `comment=1` 且需要核验体验性条件或撰写证据化理由时调用。`comment=0` 只表示本地无笔记。

### 入参

| 参数 | 类型 | 必填 | 约束 | 说明 |
| --- | --- | --- | --- | --- |
| `poiid` | string | 是 | 非空，≤ 256 字符 | POI 唯一 ID |
| `limit` | integer | 否 | 1 ~ 10，默认 10 | 返回笔记条数 |

### 返回 `data`

```json
{
  "poiid": "<回显>",
  "count": 3,
  "total_available": 5,
  "truncated": true,
  "notes": [
    { "nid": "...", "evidence_id": "note:<nid>", "...": "笔记字段" }
  ]
}
```

`total_available` 为该 POI 的笔记总数；`truncated = total_available > count`，用于提示是否被 `limit` 截断。每条笔记附加 `evidence_id = "note:" + nid`。

---

## search_nearby

在给定发起点坐标和距离半径内以 Qwen3 纯向量召回 POI 候选，并经 Qwen3 reranker 重排。使用空间索引和 Haversine 距离精确过滤，保证每条结果 `distance_m` 不超过入参 `distance_m`。必须同时传 `query`、`loc{lng,lat}` 和 `distance_m`，不能只把「附近」写进文本召回。

### 入参

| 参数 | 类型 | 必填 | 约束 | 说明 |
| --- | --- | --- | --- | --- |
| `query` | string | 是 | 非空，≤ 500 字符 | 要搜索的 POI 文本 |
| `loc` | object | 是 | 必须且仅含 `lng`、`lat`（number） | 发起点坐标；`lng ∈ [-180,180]`，`lat ∈ [-90,90]` |
| `distance_m` | number | 是 | 1 ~ `max_nearby_distance_m`(50000)，有限值 | 搜索半径（米） |
| `top_k` | integer | 否 | 1 ~ `max_top_k`(20)，默认 10 | 返回条数 |
| `category` | string | 否 | ≤ 100 字符 | 类别/类别路径包含过滤 |
| `district` | string | 否 | ≤ 100 字符 | 行政区包含过滤 |
| `business_area` | string | 否 | ≤ 100 字符 | 商圈包含过滤 |
| `min_rating` | number | 否 | 0 ~ 5，非负 | 最低评分 |
| `min_star` | number | 否 | 0 ~ 5，非负 | 最低星级 |
| `min_average_cost` | number | 否 | ≥ 0，且 ≤ `max_average_cost` | 最低人均 |
| `max_average_cost` | number | 否 | ≥ 0 | 最高人均 |
| `min_parking_spaces` | integer | 否 | ≥ 0 | 最少停车位数 |
| `has_notes` | boolean | 否 | — | 是否必须有非空笔记 |

坐标校验细节：`loc` 必须恰好包含 `lng` 和 `lat` 两个键（多或缺都报错）；`lng/lat/distance_m` 不接受布尔值，必须为有限数值；坐标顺序固定为 `{lng, lat}`，越界会提示坐标顺序错误。

### 返回 `data`

```json
{
  "query": "<回显查询>",
  "retrieval_mode": "<召回模式>",
  "origin": { "lng": 121.47, "lat": 31.23 },
  "distance_m": 3000,
  "applied_filters": { "category": "..." },
  "count": 10,
  "results": [
    { "poiid": "...", "distance_m": 850, "evidence_id": "poi:<poiid>", "...": "其余POI字段" }
  ]
}
```

每条结果附加 `evidence_id = "poi:" + poiid`，并带 `distance_m` 表示与发起点的距离。

---

## 证据 ID 约定

所有工具在结果条目上注入 `evidence_id`，供最终答案 `recommendations[].evidence` 引用：

- POI 类结果（`search_pois` / `search_nearby`）：`poi:<poiid>`
- 笔记类结果（`search_notes` / `get_poi_notes`）：`note:<nid>`

最终答案的 `evidence` 数组只能包含这些非空字符串 ID，且推荐结果的 `poiid` 不可重复、`rank` 从 1 连续递增（见 `protocol.validate_final_answer`）。
