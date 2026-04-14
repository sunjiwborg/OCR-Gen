import argparse
from generator import generator

def cli():
    parser = argparse.ArgumentParser(description="SynthGhost: Synthetic text Image Generator")

    # Arguments
    parser.add_argument("-i", required=True, help="Path to source file")
    parser.add_argument("-f", help="Path to fonts directory")
    parser.add_argument("-o", help="Path to generated images directory")
    parser.add_argument("-pt", type=int, nargs="+", default=16, help="Set font sizes")

    args = parser.parse_args()

    generator(args)

if __name__ == "__main__":
    cli()    

