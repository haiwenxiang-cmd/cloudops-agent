# Skyflo Reliability Enhanced

面向 Kubernetes 与 Helm 场景的云原生运维 Agent 平台，通过自然语言完成状态诊断、变更审批、工具执行、结果核验与流式交互。

## 核心能力

- 基于 LangGraph 组织多轮模型决策、工具调用与上下文状态。
- 通过 MCP 接入 Kubernetes、Helm 等运维工具，并按能力、角色和命名空间控制执行范围。
- 对变更操作提供人工审批、幂等标识、执行记录和未知结果核验。
- 将变更后的只读核验作为运行时约束，避免未经验证就返回成功。
- 支持任务停止、远程进程清理、SSE 事件重放与会话恢复。
- 提供 Helm Chart，可部署 Engine、MCP、UI、Controller、PostgreSQL 与 Redis。

## 技术栈

Python、FastAPI、LangGraph、LiteLLM、MCP、PostgreSQL、Redis、Next.js、Kubernetes、Helm、Docker

## 运行与测试

环境变量模板位于 `engine/.env.example`、`mcp/.env.example` 和 `ui/.env.example`。部署配置位于 `charts/skyflo`，Engine 与 MCP 的测试分别位于各自的 `tests` 目录。

## License

本项目基于 [Skyflo](https://github.com/skyflo-ai/skyflo) 开发，保留原项目版权与 Apache License 2.0 许可信息。
