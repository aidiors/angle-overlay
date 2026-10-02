from __future__ import annotations

import ctypes
from dataclasses import dataclass


@dataclass(frozen=True)
class Adapter:
    index: int
    name: str
    dedicated_memory: int
    vendor_id: int = 0

    @property
    def label(self) -> str:
        return f"{self.index}: {self.name}"


def hardware_adapters() -> list[Adapter]:
    # Preserve DXGI indices because DXcam filters out headless adapters
    from dxcam._libs.dxgi import DXGI_ADAPTER_DESC1
    from dxcam.util.io import enum_dxgi_adapters

    adapters = []
    for index, pointer in enumerate(enum_dxgi_adapters()):
        description = DXGI_ADAPTER_DESC1()
        pointer.GetDesc1(ctypes.byref(description))
        if description.Flags & 2:
            continue
        adapters.append(
            Adapter(
                index,
                description.Description,
                int(description.DedicatedVideoMemory),
                int(description.VendorId),
            )
        )
    return adapters


def preferred_adapter(adapters: list[Adapter]) -> Adapter:
    if not adapters:
        raise RuntimeError("No hardware display adapter found")
    return max(adapters, key=lambda adapter: (adapter.dedicated_memory, -adapter.index))
