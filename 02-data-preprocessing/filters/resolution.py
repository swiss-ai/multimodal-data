from pipeline import BaseFilter, ImageSample, ImageTextSample, Sample


class ImageResolutionFilter(BaseFilter):
    def __init__(
        self,
        min_width: int | None = None,
        min_height: int | None = None,
        max_width: int | None = None,
        max_height: int | None = None,
    ):
        self.min_width = min_width
        self.min_height = min_height
        self.max_width = max_width
        self.max_height = max_height

    def process_batch(self, samples: list[Sample]) -> list[Sample]:
        results = []
        for sample in samples:
            if not isinstance(sample, (ImageSample, ImageTextSample)):
                results.append(sample)
                continue
            if not hasattr(sample, "image"):
                results.append(sample)
                continue

            width, height = sample.image.size
            if (
                (self.min_width is not None and width < self.min_width)
                or (self.min_height is not None and height < self.min_height)
                or (self.max_width is not None and width > self.max_width)
                or (self.max_height is not None and height > self.max_height)
            ):
                continue
            results.append(sample)

        return results
