# Cooling Airflow Noise Long Note

## 长噪声说明

本文是检索压力测试文档，刻意包含大量与 cooling airflow、supply air、reset、risk、alarm、policy、comfort violation 和 containment 相关的重复词。它用于验证检索器不能只因为正文很长或关键词很多就把噪声文档排在更精确的短目标文档前。

cooling airflow reset risk policy comfort violation containment alarm cooling airflow reset risk policy comfort violation containment alarm cooling airflow reset risk policy comfort violation containment alarm.

## 近义干扰

冷却系统的 airflow、setpoint、reset、alarm 和 policy 都可能出现在同一个问题中，但真实回答仍应依据更具体的 title、section、source_id 和上下文。对于 supply air reset risk，短目标文档通常比泛化噪声更有解释价值。对于 sensor drift alarm，异常诊断文档通常比泛化 airflow 文档更有解释价值。

## 边界

本文件不提供真实生产遥测，不提供控制动作，也不代表 BEAR 原生字段包含 PUE、湿度或 IT 负载。它只用于 RAG baseline 和 reranker 压力评测。
