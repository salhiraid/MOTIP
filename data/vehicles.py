# Vehicle tracking dataset (converted from CVAT video XML by tools/vehicles/prepare_vehicle_dataset.py).
# Same unified layout as every other converted dataset -- see data/unified_mot.py for the file semantics.
from .unified_mot import UnifiedMOT


class Vehicles(UnifiedMOT):
    SUB_DIR = "Vehicles"
