from typing import Dict, Type

from src.adapter.base import BaseAdapter

ADAPTER_REGISTRY: Dict[str, Type[BaseAdapter]] = {}


def register_adapter(name: str):
    """
    Decorator to register a user-defined adapter.
    """

    def decorator(cls):
        if name in ADAPTER_REGISTRY:
            raise ValueError(f"Adapter '{name}' is already registered")

        if not issubclass(cls, BaseAdapter):
            raise TypeError(f"Class {cls.__name__} must inherit from BaseAdapter")

        ADAPTER_REGISTRY[name] = cls
        return cls

    return decorator


def get_adapter(name: str) -> Type[BaseAdapter]:
    if name not in ADAPTER_REGISTRY:
        raise KeyError(f"Adapter '{name}' is not registered")

    return ADAPTER_REGISTRY[name]
