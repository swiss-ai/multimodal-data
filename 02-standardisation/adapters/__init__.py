from adapters.medtrinity_demo import MedtrinityDemoAdapter

ADAPTER_REGISTRY: dict[str, type] = {
    "MedtrinityDemo": MedtrinityDemoAdapter,
}
