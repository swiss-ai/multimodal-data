from adapters.meditron import MeditronAdapter
from adapters.medmax import MedMaxAdapter
from adapters.medmnist import MedMNISTAdapter
from adapters.medtrinity import MedTrinityFullAdapter
from adapters.medtrinity_demo import MedTrinityDemoAdapter
from adapters.mmc4 import MMC4Adapter
from adapters.open_pmc_18m import OpenPMC18mAdapter
from adapters.pmc_oa import PMCOAAdapter
from adapters.scin import SCINAdapter
from adapters.slide import SLIDEAdapter
from adapters.uwf import UWFAdapter

adapters = [
    MeditronAdapter,
    MedMaxAdapter,
    MedTrinityDemoAdapter,
    MedTrinityFullAdapter,
    MMC4Adapter,
    PMCOAAdapter,
    OpenPMC18mAdapter,
    MedMNISTAdapter,
    SCINAdapter,
    SLIDEAdapter,
    UWFAdapter,
]

ADAPTER_REGISTRY: dict[str, type] = {
    adapter.__name__.removesuffix("Adapter"): adapter for adapter in adapters
}
