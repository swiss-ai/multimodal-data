from adapters.brain_tumor_mri import BrainTumorMRIAdapter
from adapters.covid_radiography import CovidRadiographyAdapter
from adapters.diabetic_retinopathy import DiabeticRetinopathyAdapter
from adapters.ebhi_seg import EBHISegAdapter
from adapters.holoassist import HoloAssistAdapter
from adapters.isic import ISICAdapter
from adapters.laion import LAIONAestheticsAdapter
from adapters.liver_ultrasound import LiverUltrasoundAdapter
from adapters.medicat import MediCaTAdapter
from adapters.meditron import MeditronAdapter
from adapters.medmax import MedMaxAdapter
from adapters.medmnist import MedMNISTAdapter
from adapters.medpix import MedPixAdapter
from adapters.medtrinity import MedTrinityFullAdapter
from adapters.medtrinity_demo import MedTrinityDemoAdapter
from adapters.mmc4 import MMC4Adapter
from adapters.multicare import MultiCaReAdapter
from adapters.nct_crc_he import NctCrcHeAdapter
from adapters.nih_chest_xray import NihChestXrayAdapter
from adapters.open_pmc_18m import OpenPMC18mAdapter
from adapters.pmc_oa import PMCOAAdapter
from adapters.rfmid2 import RFMiD2Adapter
from adapters.scin import SCINAdapter
from adapters.slide import SLIDEAdapter
from adapters.uwf import UWFAdapter

adapters = [
    MeditronAdapter,
    MedMaxAdapter,
    MedPixAdapter,
    MedTrinityDemoAdapter,
    MedTrinityFullAdapter,
    MMC4Adapter,
    MultiCaReAdapter,
    PMCOAAdapter,
    OpenPMC18mAdapter,
    MedMNISTAdapter,
    SCINAdapter,
    SLIDEAdapter,
    UWFAdapter,
    RFMiD2Adapter,
    ISICAdapter,
    HoloAssistAdapter,
    LAIONAestheticsAdapter,
    # apertus extension
    MediCaTAdapter,
    NihChestXrayAdapter,
    CovidRadiographyAdapter,
    DiabeticRetinopathyAdapter,
    BrainTumorMRIAdapter,
    EBHISegAdapter,
    LiverUltrasoundAdapter,
    NctCrcHeAdapter,
]

ADAPTER_REGISTRY: dict[str, type] = {
    adapter.__name__.removesuffix("Adapter"): adapter for adapter in adapters
}
