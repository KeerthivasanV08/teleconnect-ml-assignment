from pathlib import Path
import matplotlib.pyplot as plt


def save_figure(filepath: str) -> None:
    """
    Save matplotlib figure safely.

    Parameters
    ----------
    filepath : str
        Output image path.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")