"""Base AI client abstraction."""

import logging

_LOGGER = logging.getLogger(__name__)


class BaseAIClient:
    """Base class for all AI provider clients."""

    async def get_response(self, messages, **kwargs):
        """Get a response from the AI provider.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
            **kwargs: Additional provider-specific parameters (e.g., tools).

        Returns:
            Response string from the AI provider.
        """
        raise NotImplementedError
