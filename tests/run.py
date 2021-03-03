import ibm_boto3
from ibm_botocore.config import Config
from ibm_botocore.exceptions import ClientError

COS_ENDPOINT = "https://s3.par01.cloud-object-storage.appdomain.cloud"
COS_BUCKET = 'dowml-client-bucket'


def test_cos_connection():
    ibm_boto3.turn_debug_on()

    cos_client = ibm_boto3.resource("s3",
                                    config=Config(signature_version="oauth"),
                                    endpoint_url=COS_ENDPOINT,
                                    )
    files = cos_client.Bucket(COS_BUCKET).objects.all()
    try:
        for file in files:
            print(f'Item: {file.key} ({file.size} bytes).')
    except ClientError as be:
        print(f'CLIENT ERROR: {be}')
    except Exception as e:
        print(f'Unable to retrieve contents: {e}')


def main():
    test_cos_connection()


if __name__ == '__main__':
    main()
