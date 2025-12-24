# agentic_offloading/core/network.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Tuple, Iterable


@dataclass
class Location:
    """
    Represents a compute/network location (l_i) as used in the paper.
    l = 0 corresponds to the IoT device (mandatory).
    type ∈ {'iot', 'edge', 'cloud'} is informational.
    """
    l: int
    type: str  # 'iot' | 'edge' | 'cloud'
    metadata: Dict = field(default_factory=dict)


class Environment:
    """
    Environment model (Section III-A):
      - DR(li, lj): Data Time Consumption (ms/byte)
      - DE(li):    Data Energy Consumption (mJ/byte)
      - VR(li):    Task Time Consumption (ms/cycle)
      - VE(li):    Task Energy Consumption (mJ/cycle)

    Notes:
      • l = 0 denotes the IoT device; li=0 means no offloading (local).
      • The network/compute model is defined by the four matrices {DR, DE, VR, VE}.
    """
    def __init__(
        self,
        locations: Dict[int, Location],
        DR_map: Dict[Tuple[int, int], float],
        DE_map: Dict[int, float],
        VR_map: Dict[int, float],
        VE_map: Dict[int, float],
    ) -> None:
        self.locations = locations
        self._DR = DR_map
        self._DE = DE_map
        self._VR = VR_map
        self._VE = VE_map
        self._validate()

    # -------------------- Paper-defined functions --------------------

    def DR(self, li: int, lj: int) -> float:
        """Data Time Consumption: time to move one byte from li to lj (ms/byte)."""
        if li == lj:
            return 0.0
        return self._DR.get((li, lj), float("inf"))

    def DE(self, l: int) -> float:
        """Data Energy Consumption at location l (mJ/byte)."""
        return self._DE[l]

    def VR(self, l: int) -> float:
        """Task Time Consumption at location l (ms/cycle)."""
        return self._VR[l]

    def VE(self, l: int) -> float:
        """Task Energy Consumption at location l (mJ/cycle)."""
        return self._VE[l]

    # -------------------- Helpers & validation --------------------

    def edge_locations(self) -> Iterable[int]:
        return (lid for lid, loc in self.locations.items() if loc.type == "edge")

    def cloud_locations(self) -> Iterable[int]:
        return (lid for lid, loc in self.locations.items() if loc.type == "cloud")

    @property
    def E(self) -> int:
        """Number of edge servers (E)."""
        return sum(1 for _ in self.edge_locations())

    @property
    def C(self) -> int:
        """Number of cloud servers (C)."""
        return sum(1 for _ in self.cloud_locations())

    def _validate(self) -> None:
        """Ensure IoT and parameter completeness."""
        # Must have l=0 as IoT
        if 0 not in self.locations or self.locations[0].type != "iot":
            raise ValueError("Environment must contain l=0 of type 'iot' (IoT device).")
        # Ensure all locations have DE, VR, VE defined
        missing = [l for l in self.locations if l not in self._DE or l not in self._VR or l not in self._VE]
        if missing:
            raise ValueError(f"Missing DE/VR/VE entries for locations: {missing}")

    # -------------------- Factory for experiments --------------------

    @classmethod
    def from_matrices(
        cls,
        types: Dict[int, str],
        DR_matrix: Dict[Tuple[int, int], float],
        DE_vector: Dict[int, float],
        VR_vector: Dict[int, float],
        VE_vector: Dict[int, float],
    ) -> "Environment":
        """
        Build an Environment directly from {DR, DE, VR, VE} definitions.

        All units matching the paper:
          DR(ms/byte), DE(mJ/byte), VR(ms/cycle), VE(mJ/cycle)
        """
        locations = {l: Location(l=l, type=types.get(l, "edge")) for l in types}
        return cls(
            locations=locations,
            DR_map=DR_matrix,
            DE_map=DE_vector,
            VR_map=VR_vector,
            VE_map=VE_vector,
        )
