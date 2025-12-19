from src.filter import BaseFilter
from src.schema import ImageTextSample, RawSample, SampleType


class ResolutionFilter(BaseFilter):
    def __init__(self, min_width: int = 64, min_height: int = 64):
        self.min_width = min_width
        self.min_height = min_height

    @property
    def sample_type(self):
        return SampleType.IMAGE_TEXT

    def __call__(self, sample: RawSample):
        if not isinstance(sample, ImageTextSample):
            return True

        width, height = sample.image.size
        if width < self.min_width or height < self.min_height:
            return False

        return True
