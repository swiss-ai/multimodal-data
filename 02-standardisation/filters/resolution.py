from pipeline import BaseFilter, ImageSample, Sample


class ResolutionFilter(BaseFilter):
    def __init__(self, min_width: int, min_height: int):
        self.min_width = min_width
        self.min_height = min_height

    def process_batch(self, samples: list[Sample]) -> list[bool]:
        results = []
        for sample in samples:
            if not isinstance(sample, (ImageSample)):
                results.append(True)
            else:
                w, h = sample.image.size
                results.append(w >= self.min_width and h >= self.min_height)
        return results
