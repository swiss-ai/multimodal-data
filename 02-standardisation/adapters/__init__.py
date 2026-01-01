from adapters.meditron import MeditronImageAdapter
from adapters.medmax import MedMaxImageAdapter
from adapters.medtrinity import MedTrinityFullAdapter
from adapters.medtrinity_demo import MedtrinityDemoAdapter
from adapters.mmc4 import MMC4Adapter

ADAPTER_REGISTRY: dict[str, type] = {
    "Meditron": MeditronImageAdapter,
    "MedMaxRawImage": MedMaxImageAdapter,
    "MedtrinityDemo": MedtrinityDemoAdapter,
    "MedtrinityFull": MedTrinityFullAdapter,
    "MMC4": MMC4Adapter,
}
