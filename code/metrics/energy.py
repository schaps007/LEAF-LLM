"""Explicit energy backends; carbon is derived from measured energy at fixed intensity."""
import math
import time
from metrics.carbon import calculate_carbon


class HybridEnergyMonitor:
    def __init__(
        self,
        method="software",
        gpu_id=0,
        carbon_intensity=0.233,
    ):
        if method not in ("software", "hardware"):
            raise ValueError(f"Unknown measurement method: {method}")

        if not math.isfinite(carbon_intensity) or carbon_intensity < 0:
            raise ValueError(
                "Carbon intensity must be finite and nonnegative"
            )

        self.method = method
        self.carbon_intensity = carbon_intensity

        if method == "software":
            from codecarbon import OfflineEmissionsTracker

            # No geolocation requests. Tracker carbon is ignored in favor
            # of our fixed intensity.
            self.tracker = OfflineEmissionsTracker(
                country_iso_code="USA",
                project_name="llm_energy",
                measure_power_secs=1,
                save_to_file=False,
                log_level="error",
                save_to_api=False,
            )
        else:
            import pynvml

            self.nvml = pynvml
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
            self._get_gpu_energy()

    def _get_gpu_energy(self):
        return (
            self.nvml.nvmlDeviceGetTotalEnergyConsumption(self.handle)
            / 1000.0
        )

    def start(self):
        if self.method == "hardware":
            self.start_energy = self._get_gpu_energy()
        else:
            self.tracker.start()

        self.start_time = time.perf_counter()

    def stop(self):
        duration = time.perf_counter() - self.start_time

        if self.method == "hardware":
            energy_kwh = (
                self._get_gpu_energy() - self.start_energy
            ) / 3_600_000
        else:
            self.tracker.stop()
            energy_kwh = float(
                self.tracker.final_emissions_data.energy_consumed
            )

        if not math.isfinite(energy_kwh) or energy_kwh <= 0:
            raise RuntimeError(
                "Energy backend returned a nonpositive or invalid measurement"
            )

        return dict(
            energy_kwh=energy_kwh,
            energy_joules=energy_kwh * 3_600_000,
            carbon_kg=calculate_carbon(
                energy_kwh,
                self.carbon_intensity,
            ),
            carbon_intensity=self.carbon_intensity,
            duration_seconds=duration,
            avg_power_watts=(
                energy_kwh * 3_600_000 / duration
                if duration
                else None
            ),
            actual_measurement_method=self.method,
            energy_scope=(
                "gpu_only"
                if self.method == "hardware"
                else "codecarbon_machine_components"
            ),
        )