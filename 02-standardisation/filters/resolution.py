from src.base import BaseFilter
from src.schema import ImageSample, RawSample


class ResolutionFilter(BaseFilter):
    def __init__(self, min_width: int = 64, min_height: int = 64):
        self.min_width = min_width
        self.min_height = min_height

    def __call__(self, sample: RawSample):
        if not isinstance(sample, ImageSample):
            return True

        width, height = sample.image.size
        if width < self.min_width or height < self.min_height:
            return False

        return True
