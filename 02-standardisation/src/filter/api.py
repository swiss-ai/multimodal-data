from typing import Dict, Type

from src.filter.base import BaseFilter

_FILTER_REGISTRY: Dict[str, Type[BaseFilter]] = {}


def register_filter(name: str):
    """
    Decorator to register a filter class into the global registry.
    """

    def _decorator(cls):
        if name in _FILTER_REGISTRY:
            raise ValueError(f"Filter '{name}' is already registered!")

        if not issubclass(cls, BaseFilter):
            raise TypeError(f"Class {cls.__name__} must inherit from BaseFilter")

        _FILTER_REGISTRY[name] = cls
        return cls

    return _decorator


def get_filter(name: str) -> Type[BaseFilter]:
    if name not in _FILTER_REGISTRY:
        raise KeyError(f"Filter '{name}' is not registered.")

    return _FILTER_REGISTRY[name]
