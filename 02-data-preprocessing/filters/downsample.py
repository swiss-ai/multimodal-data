from pipeline import BaseFilter, ImageSample, ImageTextSample, Sample


class ImageDownsampleFilter(BaseFilter):
    """Downsample images that exceed max_pixels."""

    def __init__(self, max_pixels: int, pil_resample: int):
        """
        max_pixels: Maximum total pixels width*height.
                    Images exceeding this are downsampled.
        pil_resample: Resampling method: 0 = NEAREST
                                         1 = LANCZOS
                                         2 = BILINEAR
                                         3 = BICUBIC
                                         4 = BOX
                                         5 = HAMMING
        """
        self.max_pixels = max_pixels
        self.resample = pil_resample

    def process_batch(self, samples: list[Sample]) -> list[Sample]:
        results = []
        for sample in samples:
            if not isinstance(sample, (ImageSample, ImageTextSample)):
                results.append(sample)
                continue

            width, height = sample.image.size
            current_pixels = width * height
            if current_pixels <= self.max_pixels:
                results.append(sample)
                continue

            # get scale factor
            scale = (self.max_pixels / current_pixels) ** 0.5
            new_width = int(width * scale)
            new_height = int(height * scale)

            # resize
            sample.image = sample.image.resize((new_width, new_height), self.resample)
            results.append(sample)

        return results
