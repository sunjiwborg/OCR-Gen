import argparse
from tqdm import tqdm

COLOR = "\x1b[34m"
STYLE = "\x1b[1m"
RESET = "\x1b[0m"

def cli():
    banner = r"""
        ______  __     __  __    __  _______  __   __     ______  __   __  ______  ______  _______
       |  ____| \ \   / / |  \  |  ||__   __||  | |  |   |  ____||  | |  ||  __  ||  ____||__   __|
       | |____   \ \ / /  |   \ |  |   | |   |  |_|  |   | | ___ |  |_|  || |  | || |____    | |
       |____  |   \   /   |    \|  |   | |   |   _   |   | ||_  ||   _   || |  | ||____  |   | |
        ____| |    | |    |  |\    |   | |   |  | |  |   | |__| ||  | |  || |__| | ____| |   | |
       |______|    |_|    |__| \___|   |_|   |__| |__|   |______||__| |__||______||______|   |_|
    """
    parser = argparse.ArgumentParser(description=STYLE + COLOR + banner + RESET, formatter_class=argparse.RawDescriptionHelpFormatter)

    # Arguments
    parser.add_argument("-i", required=True, help="Input file")
    parser.add_argument("-o", required=True, help="Output images directory")
    parser.add_argument("-f", required=True, help="Fonts directory")
    parser.add_argument("-s", type=int, default=32, nargs="+", help="Font sizes")
    parser.add_argument("-m", type=int, default=10, help="Margin value around text")
    parser.add_argument("-tc", help="Text colors")

    args = parser.parse_args()

    return args
