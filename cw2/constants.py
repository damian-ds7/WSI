from pathlib import Path

EVALUATION_LIMIT: int = 10000
TABLE_DIR: Path = Path(__file__).parent / "tables"
MAX_X: int = 100
DIMENSIONALITY: int = 10
SYMBOLS: dict[str, str] = {"sigma": "\u03c3", "mu": "\u03bc"}
