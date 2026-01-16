import json, os
from copy import deepcopy
from typing import Dict


class AppState:
    def __init__(self, path):
        self.path = path
        self.data = {}
        self._load()

    # ---------------------
    # Public API
    # ---------------------
    def get(self, key, default=None):
        parts = key.split(".")
        node = self.data

        for part in parts:
            if not isinstance(node, dict):
                return default
            node = node.get(part)
            if node is None:
                return default

        return node

    def set(self, key, value):
        parts = key.split(".")
        node = self.data

        for part in parts[:-1]:
            node = node.setdefault(part, {})

        node[parts[-1]] = value
        self._save()

    # ---------------------
    # Internal
    # ---------------------
    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, "r") as f:
                    self.data = json.load(f)
            except Exception:
                self.data = {}
        else:
            self.data = {}
            
    def _save(self):
        with open(self.path, "w") as f:
            json.dump(self.data, f, indent=2)
            

    
