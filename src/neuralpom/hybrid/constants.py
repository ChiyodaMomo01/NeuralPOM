"""Network input includes the ghost layer; corrections cover 15 physical layers."""
SUPERVISED_KB = 16
PHYSICAL_LEVELS = 15
INPUT_THREED_FIELDS = ("u", "ub", "v", "vb", "t", "s")
INPUT_TWOD_FIELDS = ("el", "ua", "uab", "va", "vab")
INPUT_PACK_ORDER = INPUT_THREED_FIELDS + INPUT_TWOD_FIELDS
INPUT_CHANNELS = len(INPUT_THREED_FIELDS) * SUPERVISED_KB + len(INPUT_TWOD_FIELDS)
CORRECTION_FIELDS = ("u", "v", "t", "s")
CORRECTION_CHANNELS = len(CORRECTION_FIELDS) * PHYSICAL_LEVELS
MODEL_CONTRACT_VERSION = "p0-fixed-endpoint-v1"
