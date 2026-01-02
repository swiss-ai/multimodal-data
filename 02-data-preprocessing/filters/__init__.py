from filters.deduplication import ImageDeduplicationFilter
from filters.resolution import ImageResolutionFilter

filters = [
    ImageResolutionFilter,
    ImageDeduplicationFilter,
]

FILTER_REGISTRY: dict[str, type] = {
    filter.__name__.removesuffix("Filter"): filter for filter in filters
}
