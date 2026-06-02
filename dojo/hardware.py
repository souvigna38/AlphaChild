"""Hardware profiler — assigns lightweight / standard / gpu tier for track gating."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from dataclasses import asdict, dataclass

import psutil

# Minimum specs (bytes)
TIER_RAM = {
    "lightweight": 4 * 1024**3,
    "standard": 8 * 1024**3,
    "gpu": 16 * 1024**3,
}


@dataclass
class HardwareProfile:
    platform: str
    python: str
    ram_total_gb: float
    ram_available_gb: float
    cuda_available: bool
    vram_total_gb: float | None
    tier: str
    tier_reason: str

    def blob(self) -> str:
        """Stable string bound into HW-* proof."""
        vram = -1.0 if self.vram_total_gb is None else round(self.vram_total_gb, 2)
        return f"{self.ram_total_gb:.2f}|{vram}|{int(self.cuda_available)}"


def _cuda_vram_gb() -> float | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        props = torch.cuda.get_device_properties(0)
        return props.total_memory / (1024**3)
    except Exception:
        return None


def profile_hardware() -> HardwareProfile:
    vm = psutil.virtual_memory()
    ram_total = vm.total
    ram_avail = vm.available
    ram_total_gb = ram_total / (1024**3)
    ram_avail_gb = ram_avail / (1024**3)
    cuda = False
    vram_gb = _cuda_vram_gb()
    if vram_gb is not None:
        cuda = True

    tier = "lightweight"
    reason = "Default: board-game track (4GB+ RAM)."

    if ram_total >= TIER_RAM["gpu"] and cuda and vram_gb is not None and vram_gb >= 4.0:
        tier = "gpu"
        reason = "16GB+ RAM and 4GB+ VRAM — Atari / heavy MuZero OK."
    elif ram_total >= TIER_RAM["standard"]:
        tier = "standard"
        reason = "8GB+ RAM — LLM tiny models and DeepSeek notebooks OK (CPU)."
        if cuda:
            reason += f" CUDA present ({vram_gb:.1f} GB VRAM)."
    else:
        reason = (
            f"Only {ram_total_gb:.1f} GB RAM detected — locked to Track 1 (TicTacToe/ConnectFour) "
            "until you upgrade or use a cloud GPU."
        )

    return HardwareProfile(
        platform=platform.platform(),
        python=sys.version.split()[0],
        ram_total_gb=round(ram_total_gb, 2),
        ram_available_gb=round(ram_avail_gb, 2),
        cuda_available=cuda,
        vram_total_gb=round(vram_gb, 2) if vram_gb is not None else None,
        tier=tier,
        tier_reason=reason,
    )


def tier_rank(tier: str) -> int:
    return {"lightweight": 0, "standard": 1, "gpu": 2}.get(tier.lower(), 0)


def tier_allows(min_tier: str, student_tier: str) -> bool:
    return tier_rank(student_tier) >= tier_rank(min_tier)


def main() -> int:
    parser = argparse.ArgumentParser(description="Dojo hardware profiler")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    parser.add_argument(
        "--discord-user-id",
        type=str,
        default="",
        help="If set with DOJO_PROOF_SECRET, append signed HW proof line",
    )
    args = parser.parse_args()
    prof = profile_hardware()

    if args.json:
        print(json.dumps(asdict(prof), indent=2))
        return 0

    print("=== Agentic Dojo — Hardware Profile ===")
    print(f"Platform:     {prof.platform}")
    print(f"Python:       {prof.python}")
    print(f"RAM total:    {prof.ram_total_gb} GB (available {prof.ram_available_gb} GB)")
    print(f"CUDA:         {prof.cuda_available}")
    if prof.vram_total_gb is not None:
        print(f"VRAM:         {prof.vram_total_gb} GB")
    print(f"Assigned tier: {prof.tier.upper()}")
    print(f"Note: {prof.tier_reason}")

    if args.discord_user_id:
        import os

        from dojo.proof import make_hardware_proof

        secret = os.environ.get("DOJO_PROOF_SECRET", "")
        if not secret:
            print("\nSet DOJO_PROOF_SECRET to generate HW proof.", file=sys.stderr)
            return 1
        blob = prof.blob()
        proof = make_hardware_proof(
            args.discord_user_id,
            prof.tier,
            secret,
            profile_blob=blob,
        )
        print(f"\nPost these two lines in #verify-setup:\n{proof}\nprofile_blob:{blob}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
