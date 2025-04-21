# utils/context_manager.py

import re
from agent_core.utils.logger import get_logger


class ContextManager:

    def __init__(self):
        self.context = {}
        self.logger = get_logger(self.__class__.__name__)

    def get_context(self):
        """Return the underlying dictionary (if needed)."""
        return self.context

    def get_context_dict(self):
        """Return the underlying dictionary (if needed)."""
        return self.context

    def get_context_by_key(self, key):
        """Return the underlying dictionary (if needed)."""
        return self.context.get(key)

    def get_context_str(self):
        return self.context_to_str()

    def clear_context(self):
        """Reset the context to an empty dict."""
        self.context = {}

    def add_context(self, key, value):
        self.context[key] = value
        self.logger.info(f"Add '{key}' into the context.")

    def remove_context(self, key):
        if key in self.context:
            del self.context[key]
            self.logger.info(f"Remove '{key}' from the context.")

    def context_to_str(self):
        """
        If the context is empty, return an empty string.
        Otherwise, wrap each key/value pair in <key></key> inside <context></context>.
        """
        context_str = ""
        if not self.context:
            return context_str
        for key, value in self.context.items():
            context_str += f"*{key}*\n{value}\n"
        return context_str

    def __repr__(self):
        """So print(context) shows the stored keys and values nicely."""
        return f"ContextManager({self.context})"

