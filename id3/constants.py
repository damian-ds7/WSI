from pathlib import Path

CUR_DIR: Path = Path(__file__).parent
TABLE_DIR = CUR_DIR / "tables"
TABLE_DIR.mkdir(exist_ok=True)
TRIES: int = 100
