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
    #
    # def context_to_str(self):
    #     """
    #     If the context is empty, return an empty string.
    #     Otherwise, wrap each key/value pair in <key></key> inside <context></context>.
    #     """
    #     if not self.context:
    #         return ""
    #     context_str = "<Context>\n"
    #     for key, value in self.context.items():
    #         context_str += f"<{key}>\n{value}\n</{key}>\n"
    #     context_str += "</Context>\n"
    #     return context_str

    def __repr__(self):
        """So print(context) shows the stored keys and values nicely."""
        return f"ContextManager({self.context})"

    # TODO: need test
    def cleanup_context(self, current_node_name, restart_node_name):
        input_text = self.context_to_str

        # Extract the content inside <Context>
        context_match = re.search(r"<Context>(.*?)</Context>", input_text, re.DOTALL)
        if not context_match:
            return input_text
        context_content = context_match.group(1)

        # Extract all nodes within the context
        nodes = re.findall(
            r"(<Previous Step [^>]+>.*?</Previous Step [^>]+>)",
            context_content,
            re.DOTALL,
        )

        # Determine the range of node Names to remove
        restart_char = restart_node_name.upper()
        current_char = current_node_name.upper()
        start_range = min(restart_char, current_char)
        end_range = max(restart_char, current_char)

        kept_nodes = []
        for node in nodes:
            # Extract the opening tag line
            opening_tag_line = node.split("\n")[0].strip()
            # Extract the name part after 'Previous Step '
            start = opening_tag_line.find("Previous Step ") + len("Previous Step ")
            end = opening_tag_line.find(">", start)
            name_part = opening_tag_line[start:end].strip()
            if not name_part:
                continue
            base_id = name_part[0].upper()

            # Check if base_id is outside the removal range
            if not (start_range <= base_id <= end_range):
                kept_nodes.append(node.strip())

        # Rebuild the context content with kept nodes
        new_context_content = "\n".join(kept_nodes)
        # Ensure proper formatting with newlines
        cleaned_text = f"<Context>\n{new_context_content}\n</Context>"
        print(cleaned_text)
        return cleaned_text

    def identify_context_key(self, context_str, current_node_id, restart_node_id):

        context_match = re.search(r"<Context>(.*?)</Context>", context_str, re.DOTALL)
        if not context_match:
            return []
        context_content = context_match.group(1)

        nodes = re.findall(
            r"(<Previous Step [^>]+>.*?</Previous Step [^>]+>)",
            context_content,
            re.DOTALL,
        )

        if restart_node_id:
            restart_char = restart_node_id.upper()
        else:
            restart_char = current_node_id.upper()
        current_char = current_node_id.upper()
        start_range = min(restart_char, current_char)
        end_range = max(restart_char, current_char)

        nodes_to_remove = []

        for node in nodes:
            opening_tag_line = node.split("\n")[0].strip()
            node_name = opening_tag_line[
                1:-1
            ].strip()  # Extract node name from the opening tag
            parts = node_name.split()
            if len(parts) < 3:
                continue
            base_id_part = parts[2]
            if not base_id_part:
                continue
            base_id = base_id_part[0].upper()
            if start_range <= base_id <= end_range:
                nodes_to_remove.append(node_name)
        return nodes_to_remove
