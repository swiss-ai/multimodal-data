from pipeline import BaseFilter, ImageSample, ImageTextSample, Sample


class ResolutionFilter(BaseFilter):
    def __init__(self, min_width: int, min_height: int):
        self.min_width = min_width
        self.min_height = min_height

    def __call__(self, sample: Sample) -> bool:
        if not isinstance(sample, (ImageSample, ImageTextSample)):
            return True

        w, h = sample.image.size
        return w >= self.min_width and h >= self.min_height
