from typing import Dict, Type

from src.adapter.base import BaseAdapter

_ADAPTER_REGISTRY: Dict[str, Type[BaseAdapter]] = {}


def register_adapter(name: str):
    """
    Decorator to register a user-defined adapter.
    """

    def _decorator(cls):
        if name in _ADAPTER_REGISTRY:
            raise ValueError(f"Adapter '{name}' is already registered")

        if not issubclass(cls, BaseAdapter):
            raise TypeError(f"Class {cls.__name__} must inherit from BaseAdapter")

        _ADAPTER_REGISTRY[name] = cls
        return cls

    return _decorator


def get_adapter(name: str) -> Type[BaseAdapter]:
    if name not in _ADAPTER_REGISTRY:
        raise KeyError(f"Adapter '{name}' is not registered")

    return _ADAPTER_REGISTRY[name]
