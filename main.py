""""Send a CPLEX file to WML and solve it.

For now, just accepts a path and sends that, assuming that's a model"""
import argparse


def solve(path):
    """Solve the model.

    The model is sent as online data to WML.

    Args:
        path: pathname to the file to solve"""
    pass


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create an OPL .dat file from one or more CSV files')
    # Accept a number of CSV file names to read -> csvfiles
    parser.add_argument(metavar='model', dest='model',
                        help='Name of the model to solve')
    args = parser.parse_args()

    solve(args.model)
