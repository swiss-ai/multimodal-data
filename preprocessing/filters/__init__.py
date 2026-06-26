from filters.deduplication import ImageDeduplicationFilter
from filters.downsample import ImageDownsampleFilter
from filters.resolution import ImageResolutionFilter

filters = [
    ImageResolutionFilter,
    ImageDownsampleFilter,
    ImageDeduplicationFilter,
]

FILTER_REGISTRY: dict[str, type] = {
    filter.__name__.removesuffix("Filter"): filter for filter in filters
}
