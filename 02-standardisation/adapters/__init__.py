from adapters.medmax import MedMaxRawImageAdapter
from adapters.medtrinity_demo import MedtrinityDemoAdapter
from adapters.mmc4 import MMC4Adapter

ADAPTER_REGISTRY: dict[str, type] = {
    "MedtrinityDemo": MedtrinityDemoAdapter,
    "MedMaxRawImage": MedMaxRawImageAdapter,
    "MMC4": MMC4Adapter,
}
