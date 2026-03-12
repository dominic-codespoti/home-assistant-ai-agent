# AI Agent HA - Architecture & Development Guide (AGENTS.MD)

## 📌 Purpose of this Document
This file serves as the definitive architecture guide and rulebook for AI agents and developers working on the `ai_agent_ha` integration. 
**Goal:** This document MUST be kept in-sync whenever new features, providers, or core architectural changes are introduced to the repository. Before implementing changes, consult this guide to maintain consistency.

## 🏗️ Architecture Overview
The `ai_agent_ha` integration is a unified Home Assistant custom integration that routes natural language queries to various AI providers (OpenAI, Gemini, Anthropic, Alter, z.ai, Llama, Local, OpenRouter).

### Core Components (`custom_components/ai_agent_ha/`)
- **`__init__.py`**: Integration entry point. Handles setup, config entry validation, panel registration, and exposes HA services (`query`, `create_automation`, `create_dashboard`, etc.).
- **`agent.py`**: The brain of the operation. Contains:
  - **Provider Clients:** Classes inheriting from `BaseAIClient` (e.g., `OpenAIClient`, `GeminiClient`) that handle API-specific formatting and requests.
  - **`AiAgentHaAgent`:** The main orchestrator class that manages conversation history, constructs system prompts with HA context, parses AI responses (often evaluating them as JSON for function execution), and handles features like automation/dashboard creation.
- **`config_flow.py`**: Manages the UI-based setup process in Home Assistant, handling API keys and model selection for different providers.
- **`const.py`**: Central repository for all constants (`DOMAIN`, configuration keys, provider lists).
- **`services.yaml`**: Defines the services exposed to Home Assistant UI and Automations.
- **`dashboard_templates.py`**: Pre-defined JSON-like templates used when the AI is asked to generate new HA dashboards.
- **`mcp_server.py`**: Integration of the Model Context Protocol (MCP) server, allowing external clients to discover and interact with Home Assistant entities.
- **`discovery.py` & `domain_registry.py`**: Handles discovery and capability mapping of Home Assistant entities for the MCP interface.
- **`index_manager.py`**: Maintains a lightweight, unified state and structure index of the Home Assistant instance, primarily used by the MCP server.

### Frontend Components (`custom_components/ai_agent_ha/frontend/`)
- **`ai_agent_ha-panel.js`**: Web Component implementing the chat interface in HA.

## 📁 Where to Look
- **Adding a new AI Provider:** Updates are required in `const.py` (add to `AI_PROVIDERS`), `config_flow.py` (auth fields), `agent.py` (add a new `<Provider>Client` class and map it), and `services.yaml` (update provider selectors).
- **Modifying HA Context provided to AI:** Look in `agent.py` inside the methods building the system prompt (e.g., compiling states, devices, weather).
- **Custom Services:** Start from `__init__.py` to see how the service handles the payload, then follow the call to `agent.py` method executions.
- **Frontend Changes:** Modify `frontend/ai_agent_ha-panel.js`. Note the testing server in `tests/frontend/` for local UI iteration without full HA restarts.

## 📜 Development Rules & Patterns

### 1. Unified Interface for AI Providers
- Always extend `BaseAIClient` when adding a new LLM provider in `agent.py`.
- Ensure responses are heavily sanitized and try to return standardized text or JSON to be parsed reliably by the main agent.
- API keys MUST be handled securely (use `sanitize_for_logging` in `agent.py` when logging payloads).

### 2. Defensive Programming & HA Standards
- **Asynchronous Code:** Heavy I/O (API calls) MUST use `aiohttp.ClientSession` to avoid blocking Home Assistant's event loop.
- **Error Handling:** Use `_LOGGER.error` to catch and log provider exceptions gracefully without crashing HA.
- **Python 3.12+ Compatibility:** As required by HA 2025.1.x+. Ensure type hinting and modern Python idioms where possible.

### 3. Service Interactions
- Exposed services (e.g., `query`, `create_automation`) must accept a `provider` string parameter with a fallback logic in case the user's primary provider fails.
- Responses to the frontend are often fired via the HA event bus (`hass.bus.async_fire("ai_agent_ha_response", result)`).

## 🔄 Keeping this File In-Sync
When modifying the underlying architecture—such as adding a new tool the AI can use, writing a new frontend integration method, or altering the config flow schema— developers and AI assistants **must append or modify** the relevant sections of this `AGENTS.MD` document. This ensures that context is never lost and future agents understand the repository's current state immediately.
