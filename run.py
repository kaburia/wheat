# Parsing arguments to run 
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    # Add arguments for command line
    args = parser.parse_args()

    return args




def main():
    args = parse_args()

   

if __name__ == '__main__':
    main()

