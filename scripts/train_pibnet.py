"""Train pibnet using the packaged implementation and prepared data."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from shared.cli import main

if __name__ == '__main__':
    main('pibnet')
