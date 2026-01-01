from adapters.meditron import MeditronImageAdapter
from adapters.medmax import MedMaxImageAdapter
from adapters.medtrinity_demo import MedtrinityDemoAdapter
from adapters.mmc4 import MMC4Adapter

ADAPTER_REGISTRY: dict[str, type] = {
    "MedtrinityDemo": MedtrinityDemoAdapter,
    "MedMaxRawImage": MedMaxImageAdapter,
    "MMC4": MMC4Adapter,
    "Meditron": MeditronImageAdapter,
}
