"""
pipeline de ingestao do dataset CWRU Bearing (12 kHz drive end + normal baseline) para CNN 1D,

- 4 clases: 0=normal, 1=inner race, 2=ball, 3=out race @6:00
- severidades 0.007", 0.014", 0.021"
- apenas canal _DE_time (acelerometro de drive end)
- janela de 1024 amostras, stride 512
- saida: numpy arrays
"""

from __future__ import annotations
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.io import loadmat

SAMPLING_RATE_HZ        = 12_000
DEFAULT_WINDOW_SIZE     = 1024
DEFAULT_STRIDE          = 512

CLASS_NORMAL = 0
CLASS_INNER_RACE = 1
CLASS_BALL = 2
CLASS_OUTER_RACE    = 3

CLASS_NAMES = {
    CLASS_NORMAL: "normal",
    CLASS_INNER_RACE: "inner_race",
    CLASS_BALL: "ball",
    CLASS_OUTER_RACE: "outer_race",
}