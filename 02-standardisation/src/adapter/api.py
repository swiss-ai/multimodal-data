from typing import Dict, Type

from src.adapter.base import BaseAdapter

_ADAPTER_REGISTRY: Dict[str, Type[BaseAdapter]] = {}


def register_adapter(name: str):
    """Decorator to register a user-defined adapter."""

    def decorator(cls):
        if not issubclass(cls, BaseAdapter):
            raise TypeError(f"{cls} must inherit from BaseAdapter")
        _ADAPTER_REGISTRY[name] = cls
        return cls

    return decorator


def get_adapter(name: str) -> Type[BaseAdapter]:
    return _ADAPTER_REGISTRY[name]
