#classes.py
import pandas as pd
from contextlib import contextmanager

class HDFStoreManager:
    """Manages HDFStore access across multiple datasets"""
    _stores = {}
    
    @classmethod
    @contextmanager
    def get_store(cls, path):
        if path not in cls._stores:
            cls._stores[path] = pd.HDFStore(path, mode="r")
        try:
            yield cls._stores[path]
        finally:
            pass
    
    @classmethod
    def close_all(cls):
        """Close all open stores"""
        for store in cls._stores.values():
            store.close()
        cls._stores.clear()