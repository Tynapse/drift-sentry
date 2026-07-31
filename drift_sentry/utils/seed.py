from __future__ import annotations

import os
import random


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import numpy as np  # type: ignore[import-not-found]
    except Exception:
        pass
    else:
        np.random.seed(seed)

    try:
        import torch  # type: ignore[import-not-found]
    except Exception:
        pass
    else:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
