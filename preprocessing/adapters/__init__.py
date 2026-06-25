from adapters.meditron import MeditronAdapter
from adapters.medmax import MedMaxAdapter
from adapters.medtrinity import MedTrinityFullAdapter
from adapters.medtrinity_demo import MedTrinityDemoAdapter
from adapters.mmc4 import MMC4Adapter
from adapters.pmc_oa import PMCOAAdapter

adapters = [
    MeditronAdapter,
    MedMaxAdapter,
    MedTrinityDemoAdapter,
    MedTrinityFullAdapter,
    MMC4Adapter,
    PMCOAAdapter,
]

ADAPTER_REGISTRY: dict[str, type] = {
    adapter.__name__.removesuffix("Adapter"): adapter for adapter in adapters
}
