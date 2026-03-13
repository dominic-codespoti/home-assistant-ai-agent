"""Constants for the AI Agent HA integration."""

DOMAIN = "ai_agent_ha"
CONF_API_KEY = "api_key"
CONF_WEATHER_ENTITY = "weather_entity"

# AI Provider configuration keys
CONF_LLAMA_TOKEN = "llama_token"  # nosec B105
CONF_OPENAI_TOKEN = "openai_token"  # nosec B105
CONF_GEMINI_TOKEN = "gemini_token"  # nosec B105
CONF_OPENROUTER_TOKEN = "openrouter_token"  # nosec B105
CONF_ANTHROPIC_TOKEN = "anthropic_token"  # nosec B105
CONF_ALTER_TOKEN = "alter_token"  # nosec B105
CONF_ZAI_TOKEN = "zai_token"  # nosec B105
CONF_LOCAL_URL = "local_url"
CONF_LOCAL_MODEL = "local_model"

# Available AI providers
AI_PROVIDERS = [
    "llama",
    "openai",
    "gemini",
    "openrouter",
    "anthropic",
    "alter",
    "zai",
    "local",
]

# AI Provider constants
CONF_MODELS = "models"

# Supported AI providers
DEFAULT_AI_PROVIDER = "openai"

# MCP Server Constants
SYSTEM_ENTRY_UNIQUE_ID = "ai_agent_ha_system_settings"
CONF_MCP_PORT = "mcp_port"
CONF_ALLOWED_IPS = "allowed_ips"
CONF_SEARCH_PROVIDER = "search_provider"
CONF_ENABLE_CUSTOM_TOOLS = "enable_custom_tools"
CONF_ENABLE_GAP_FILLING = "enable_gap_filling"
CONF_LMSTUDIO_URL = "lmstudio_url"
DEFAULT_LMSTUDIO_URL = "http://127.0.0.1:1234/v1"
DEFAULT_MCP_PORT = 8124
DEFAULT_ALLOWED_IPS = ""
DEFAULT_SEARCH_PROVIDER = "none"
DEFAULT_ENABLE_GAP_FILLING = False

MCP_SERVER_NAME = "ha-entity-discovery"
MCP_PROTOCOL_VERSION = "2024-11-05"
MAX_ENTITIES_PER_DISCOVERY = 50
MAX_DISCOVERY_RESULTS = 100
CONF_MAX_ENTITIES_PER_DISCOVERY = "max_entities_per_discovery"
DEFAULT_MAX_ENTITIES_PER_DISCOVERY = 50
