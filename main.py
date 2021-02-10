""""Send a CPLEX file to WML and solve it.

For now, just accepts a path and sends that, assuming that's a model
The code is mostly based on https://dataplatform.cloud.ibm.com/exchange/public/entry/view/50fa9246181026cd7ae2a5bc7e4ac7bd?audience=wdp&context=cpdaas"""
import argparse
import logging
import os
from ibm_watson_machine_learning import APIClient


class Error(Exception):
    """Base class for all errors in this script"""
    pass


class InvalidCredentials(Error):
    """The WML credentials were not found, or incorrect"""
    pass


def read_wml_credentials_from_env():
    """Return a string of credentials suitable for WML from the environment

    Raises InvalidCredentials if anything is wrong."""
    var_name = 'WML_CREDENTIALS'
    logging.info(f'Looking for credentials in environment variable ${var_name}...')
    try:
        wml_cred_str = os.environ[var_name]
    except KeyError:
        print(f'Environment variable ${var_name} not found.')
        print(f'It should contain credentials as a Python dict of the form:')
        print("'{'apikey': '<apikey>', 'url': 'https://us-south.ml.cloud.ibm.com'}")
        raise InvalidCredentials

    return wml_cred_str


def read_wml_credentials_from_file(file):
    """Return the content of the file, assumed to be WML credentials"""
    logging.info(f'Looking for credentials in file \'{file}\'...')
    with open(file) as f:
        wml_cred_str = f.read()
    return wml_cred_str


def read_wml_credentials(args):
    if args.wml_cred_file is not None:
        wml_cred_str = read_wml_credentials_from_file(args.wml_cred_file)
    else:
        wml_cred_str = read_wml_credentials_from_env()
    logging.info(f'Found credential string.')

    wml_credentials = eval(wml_cred_str)

    assert type(wml_credentials) is dict
    assert 'apikey' in wml_credentials
    assert type(wml_credentials['apikey']) is str
    assert 'url' in wml_credentials
    assert type(wml_credentials['url']) is str
    logging.info(f'Credentials have the expected structure.')

    return wml_credentials


def solve(path, *,
          wml_credentials):
    """Solve the model.

    The model is sent as online data to WML.

    Args:
        path: pathname to the file to solve
        wml_credentials: credentials to use to connect to WML"""

    logging.info(f'Creating the connexion...')
    client = APIClient(wml_credentials)
    logging.info(f'Creating the connexion succeeded.  Client version is {client.version}')


def main():
    parser = argparse.ArgumentParser(description='Create an OPL .dat file from one or more CSV files')
    # Accept a number of CSV file names to read -> csvfiles
    parser.add_argument(metavar='model', dest='model',
                        help='Name of the model to solve')
    parser.add_argument('--wml-cred-file',
                        help='Name of the file from which to read WML credentials. If not specified, \
credentials are read from an environment variable')
    args = parser.parse_args()

    logging.basicConfig(force=True, format='%(asctime)s %(message)s', level=logging.INFO)

    wml_credentials = read_wml_credentials(args)
    solve(args.model, wml_credentials=wml_credentials)


if __name__ == '__main__':
    main()
