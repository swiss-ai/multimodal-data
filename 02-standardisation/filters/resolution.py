from pipeline import BaseFilter, ImageSample, Sample


class ResolutionFilter(BaseFilter):
    def __init__(
        self,
        min_width: int,
        min_height: int,
        max_width: int | None = None,
        max_height: int | None = None,
    ):
        self.min_width = min_width
        self.min_height = min_height
        self.max_width = max_width
        self.max_height = max_height

    def process_batch(self, samples: list[Sample]) -> list[bool]:
        results = []
        for sample in samples:
            if not isinstance(sample, (ImageSample)):
                results.append(True)
            else:
                w, h = sample.image.size
                results.append(w >= self.min_width and h >= self.min_height)
                if results[-1] and self.max_width and self.max_height:
                    results[-1] = w <= self.max_width and h <= self.max_height
        return results
