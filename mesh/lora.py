"""LoRa physical layer model (used when SimConfig.radio == "lora").

* Airtime: Semtech SX127x/SX126x formula (AN1200.13), explicit header, CRC on,
  low-data-rate optimisation when the symbol time exceeds 16 ms.
* Presets: Meshtastic modem presets (US915 defaults).
* Link budget: log-distance path loss (mountain/forest exponent), extra terrain
  loss, per-link log-normal shadowing (fixed for a seed), per-packet fading.
* Demodulation: a frame survives only if its instantaneous SNR is above the
  spreading factor's demodulation floor.
* Collisions: overlapping frames at a receiver collide unless one is at least
  `capture_db` stronger (capture effect).  Radios are half-duplex.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class LoRaPreset:
    name: str
    sf: int
    bw_hz: float
    cr: int            # 1..4  ->  coding rate 4/5 .. 4/8
    preamble: int = 16  # Meshtastic uses a 16-symbol preamble

    @property
    def t_sym(self) -> float:
        return (2 ** self.sf) / self.bw_hz

    @property
    def bitrate(self) -> float:
        return self.sf * self.bw_hz / (2 ** self.sf) * 4 / (4 + self.cr)

    @property
    def snr_floor_db(self) -> float:
        return SNR_FLOOR[self.sf]


# Semtech SX1262 datasheet demodulator SNR limits
SNR_FLOOR = {7: -7.5, 8: -10.0, 9: -12.5, 10: -15.0, 11: -17.5, 12: -20.0}

PRESETS = {
    "long_slow": LoRaPreset("LongSlow", 12, 125e3, 4),
    "long_fast": LoRaPreset("LongFast", 11, 250e3, 1),
    "medium_fast": LoRaPreset("MediumFast", 9, 250e3, 1),
    "short_fast": LoRaPreset("ShortFast", 7, 250e3, 1),
    "short_turbo": LoRaPreset("ShortTurbo", 7, 500e3, 1),
}


def airtime_s(preset: LoRaPreset, payload_bytes: int, crc: bool = True, explicit_header: bool = True) -> float:
    sf, cr = preset.sf, preset.cr
    de = 1 if preset.t_sym > 0.016 else 0
    h = 0 if explicit_header else 1
    num = 8 * payload_bytes - 4 * sf + 28 + (16 if crc else 0) - 20 * h
    n_payload = 8 + max(math.ceil(num / (4 * (sf - 2 * de))) * (cr + 4), 0)
    return (preset.preamble + 4.25 + n_payload) * preset.t_sym


@dataclass
class RadioEnv:
    freq_mhz: float = 915.0
    tx_dbm: float = 22.0           # SX1262 max
    ant_gain_dbi: float = 2.0      # each end (stubby handheld / small mast)
    path_loss_exp: float = 3.6     # forested mountain terrain, near-ground antennas
    terrain_db: float = 0.0        # extra obstruction loss (UI slider)
    shadow_sigma_db: float = 3.0   # per-link, fixed for a given seed (ridges, trees)
    fade_sigma_db: float = 3.0     # per-packet fast fading
    noise_figure_db: float = 6.0
    capture_db: float = 6.0
    km_per_unit: float = 1.8       # map units -> km

    def fspl_1m(self) -> float:
        return 20 * math.log10(self.freq_mhz) - 27.55  # free-space loss at 1 m (dB)

    def path_loss_db(self, d_km: float) -> float:
        d_m = max(d_km * 1000.0, 1.0)
        return self.fspl_1m() + 10 * self.path_loss_exp * math.log10(d_m)

    def noise_dbm(self, preset: LoRaPreset) -> float:
        return -174 + 10 * math.log10(preset.bw_hz) + self.noise_figure_db

    def mean_rssi(self, d_km: float, shadow_db: float = 0.0) -> float:
        return self.tx_dbm + 2 * self.ant_gain_dbi - self.path_loss_db(d_km) - self.terrain_db - shadow_db

    def mean_snr(self, preset: LoRaPreset, d_km: float, shadow_db: float = 0.0) -> float:
        return self.mean_rssi(d_km, shadow_db) - self.noise_dbm(preset)

    def per(self, preset: LoRaPreset, mean_snr: float) -> float:
        """Packet error rate from fading only (no collisions)."""
        margin = mean_snr - preset.snr_floor_db
        return 0.5 * math.erfc(margin / (self.fade_sigma_db * math.sqrt(2)))
