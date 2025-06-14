import sys
from pathlib import Path
root_path = Path("/home/olivieri/exp").resolve()
src_path = root_path / "src"
sys.path.append(f"{str(src_path)}")

def main() -> None:
    pass


if __name__ == '__main__':
    main()
