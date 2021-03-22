import logging
import os
import pprint

import ibm_boto3
from ibm_botocore.config import Config
from ibm_botocore.exceptions import ClientError

from dowmllib import DOWMLLib

COS_ENDPOINT = "https://s3.par01.cloud-object-storage.appdomain.cloud"
COS_BUCKET = 'dowml-client-bucket'


def test_cos_connection():
    ibm_boto3.turn_debug_on()

    path = 'examples/more/neos-4332810-sesia.mps.gz'
    basename = os.path.basename(path)

    cos_client = ibm_boto3.resource("s3",
                                    config=Config(signature_version="oauth"),
                                    endpoint_url=COS_ENDPOINT,
                                    )
    files = cos_client.Bucket(COS_BUCKET).objects.all()
    already_exists = False
    try:
        for file in files:
            print(f'Item: {file.key} ({file.size} bytes).')
            if file.key == basename:
                already_exists = True
    except ClientError as be:
        print(f'CLIENT ERROR: {be}')
    except Exception as e:
        print(f'Unable to retrieve contents: {e}')

    if not already_exists:
        print(f'Uploading {path} to {COS_BUCKET}/{basename}...')
        cos_client.Object(COS_BUCKET, basename).upload_file(path)
        print(f'Done.')
    else:
        print(f'The file already exists!')

NAME = 'foo.lp.gz'

def solve_through_cos_connection():
    lib = DOWMLLib('xavier-wml-dev.txt')
    client = lib._get_or_make_client()
    #print(lib._find_asset_id_by_name('connected-afiro.mps'))
    #lib.create_asset(NAME)
    #print(lib._find_asset_id_by_name(NAME))
    #lib.solve('gui-foo.sav')
    lib.solve('foo.sav')
    #pprint.pprint(lib._get_asset_details())
    pprint.pprint(client.data_assets.get_details(lib._find_asset_id_by_name('foo.sav')))
    #pprint.pprint(client.data_assets.get_details(lib._find_asset_id_by_name('gui-foo.sav')))


def main():
    logging.basicConfig(force=True, format='%(asctime)s %(message)s', level=logging.DEBUG)
    logging.getLogger('ibm_watson_machine_learning').setLevel(logging.DEBUG)
    logging.getLogger('urllib3').setLevel(logging.DEBUG)
    logging.getLogger('requests').setLevel(logging.DEBUG)
    logging.getLogger('swagger_client').setLevel(logging.DEBUG)
    logging.getLogger('ibm_botocore').setLevel(logging.DEBUG)
    logging.getLogger('ibm_boto3').setLevel(logging.DEBUG)

    test_cos_connection()
    #solve_through_cos_connection()


if __name__ == '__main__':
    main()
