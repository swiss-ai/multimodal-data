from filters.deduplication import ImageDeduplication
from filters.resolution import ResolutionFilter

FILTER_REGISTRY: dict[str, type] = {
    "MinResolution": ResolutionFilter,
    "ImageDeduplication": ImageDeduplication,
}
