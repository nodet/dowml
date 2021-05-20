# (C) Copyright IBM Corp. 2020.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

__all__ = [
    "discover_input_data",
    "get_data_asset_attachment",
    "prepare_cos_data_location",
    "prepare_interaction_props_for_cos",
    "prepare_payload_for_excel",
    "check_location",
    "try_import_pyarrow"
]

import json
import requests
from ibm_boto3 import client
from ibm_botocore.client import Config
from typing import Tuple, TYPE_CHECKING
import pkg_resources
from packaging import version
from warnings import warn
from subprocess import check_call
from sys import executable

from .errors import UnsupportedConnection, UnsupportedOutputConnection, WrongDatabaseSchemaOrTable, MissingFileName
from ibm_watson_machine_learning.utils.autoai.errors import CannotInstallLibrary

if TYPE_CHECKING:
    from ibm_watson_machine_learning import APIClient

BOTO_CONFIG_DOWNLOAD = Config(connect_timeout=60, read_timeout=60, retries={'total_max_attempts': 5})


def discover_input_data(data_location):
    cos = client(
        service_name='s3',
        aws_access_key_id=data_location["connection"]["access_key_id"],
        aws_secret_access_key=data_location["connection"]["secret_access_key"],
        endpoint_url=data_location["connection"]["endpoint_url"],
        verify=False,
        config=BOTO_CONFIG_DOWNLOAD
    )

    parent_folder = str(data_location["location"].get("path", '')).replace("./", "", 1)
    try:
        csv_keys = cos.list_objects_v2(
            Bucket=data_location["location"]["bucket"],
            FetchOwner=False,
            Prefix=parent_folder + "/features/part-")

        if 'Contents' in csv_keys:
            parts = []
            for obj in csv_keys['Contents']:
                parts.append(obj['Key'])
            return parts
    except Exception as e:
        pass

    return [data_location["location"]["path"]]


def get_data_asset_attachment(data_location: dict, wml_client: 'APIClient') -> dict:
    """Here we can download data asset attachment for further inspection."""
    # note: try to support new design with id in connection part, if cannot, try to use old one with location

    if 'href' in data_location:
        data_location['id'] = data_location['href'].split('/')[3].split('?')[0]

    data_asset = wml_client.data_assets.get_details(data_location['id'])

    try:
        attachment_id = str(data_asset['attachments'][0]['id'])

    except KeyError:
        attachment_id = data_asset['metadata']['attachment_id']

    attachment_url = f"{wml_client.service_instance._href_definitions.get_attachments_href(data_location['id'])}/{attachment_id}"

    if wml_client.ICP:
        attachment = requests.get(attachment_url, params=wml_client._params(), headers=wml_client._get_headers(),
                                  verify=False)

    else:
        attachment = requests.get(attachment_url, params=wml_client._params(), headers=wml_client._get_headers())

    attachment_content = attachment.content
    attachment_json = json.loads(attachment_content)

    return attachment_json


def prepare_cos_data_location(data_location, wml_client: 'APIClient', out=False):
    """To be able to interact with COS directly, we need to prepare credentials for COS client.
    This is needed for COS introspection eg. for OBM data parts.
    """
    attachment_json = {}

    if data_location['type'] == 'data_asset':
        attachment_json = get_data_asset_attachment(data_location, wml_client)
        connection_details = wml_client.connections.get_details(attachment_json['connection_id'])

    elif data_location['type'] == 'connection_asset':
        connection_details = wml_client.connections.get_details(data_location['connection']['id'])

    else:
        raise UnsupportedConnection(conn_type=data_location['type'])

    # Note: try to find out if we have connection_asset pointing out to S3 COS
    if 'secret_key' in connection_details['entity']['properties']:
        bucket = connection_details['entity']['properties'].get('bucket')
        path = data_location['location'].get('path')

        # training service did not specified which version is correct for now
        if path is None:
            path = data_location['location'].get('file_name')

        new_data_location = {
            "connection": {
                "access_key_id": connection_details['entity']['properties']['access_key'],
                "secret_access_key": connection_details['entity']['properties']['secret_key'],
                "endpoint_url": connection_details['entity']['properties']['url'],
            },
            "location": {
                "bucket": bucket if bucket is not None else data_location['location'].get('bucket'),
                "path": path if path is not None else attachment_json.get('connection_path')
            }
        }

    elif 'cos_hmac_keys' in connection_details['entity']['properties'].get('credentials', ''):
        creds = json.loads(connection_details['entity']['properties']['credentials'])

        bucket = connection_details['entity']['properties'].get('bucket')
        path = data_location['location'].get('path')

        # training service did not specified which version is correct for now
        if path is None:
            path = data_location['location'].get('file_name')

        new_data_location = {
            "connection": {
                "access_key_id": creds['cos_hmac_keys']['access_key_id'],
                "secret_access_key": creds['cos_hmac_keys']['secret_access_key'],
                "endpoint_url": connection_details['entity']['properties']['url'],
            },
            "location": {
                "bucket": bucket if bucket is not None else data_location['location'].get('bucket'),
                "path": path if path is not None else attachment_json.get('connection_path')
            }
        }

    else:
        if out:
            raise UnsupportedOutputConnection(
                connection_id=attachment_json['connection_id'] if 'connection_id' in attachment_json else
                data_location['connection']['id'],
                reason="Supported output types are: COS for cloud and File System for CP4D. "
                       "Make sure that connection has correct COS credentials specified.")

        else:
            raise UnsupportedConnection(
                conn_type=data_location['type'],
                reason="Make sure that connection has correct COS credentials specified.")

    return new_data_location


def prepare_interaction_props_for_cos(params: dict, input_key_file) -> dict:
    """If user specified properties for training dataset as sheet_name, delimiter etc. we need to
    pass them as interaction properties for Flight Service.
    """
    interaction_properties = {}

    encoding = params.get('encoding')
    if ".xls" in input_key_file or ".xlsx" in input_key_file:
        file_format = "excel"
        interaction_properties["sheet_name"] = str(params.get('excel_sheet', 0))
    else:
        if encoding is not None:
            interaction_properties["encoding"] = encoding
        input_file_separator = params.get('input_file_separator', ',')
        if input_file_separator != ",":
            file_format = "delimited"
            interaction_properties["field_delimiter"] = input_file_separator
        else:
            file_format = "csv"
    interaction_properties["file_format"] = file_format

    return interaction_properties


def prepare_payload_for_excel(cos_data_location: dict, cos_interaction_properties: dict,
                              params: dict, input_key_file: str) -> Tuple[dict, str]:
    """If we have to download excel file we need to specify sheet_name. In some cases when user specifies
    input data as data_asset, the path is appended with sheet_name already so we need to remove it and place
    into interaction properties.
    """
    # data asset set path with spread_sheet name, we need to remove it
    if cos_data_location['location']['path'] is not None:
        if '.xlsx' in cos_data_location['location']['path']:
            if (not cos_data_location['location']['path'].endswith('.xlsx') and
                    'excel_sheet' not in str(params)):
                cos_interaction_properties['sheet_name'] = (
                    cos_data_location['location']['path'].split('/')[-1])

            input_key_file = cos_data_location['location']['path'].split('.xlsx')[0] + '.xlsx'

        elif '.xls' in cos_data_location['location']['path']:
            if (not cos_data_location['location']['path'].endswith('.xls') and
                    'excel_sheet' not in str(params)):
                cos_interaction_properties['sheet_name'] = (
                    cos_data_location['location']['path'].split('/')[-1])

            input_key_file = cos_data_location['location']['path'].split('.xls')[0] + '.xls'

    return cos_interaction_properties, input_key_file


def check_location(data_location: dict, connection_type: str) -> None:
    if data_location['type'] == 'connection_asset':
        if connection_type == 'database':
            if 'schema_name' not in data_location['location'] and 'table_name' not in data_location['location']:
                raise WrongDatabaseSchemaOrTable(reason="Connection asset requires to have specified schema_name "
                                                        "and table_name for database connection.")

        elif connection_type == 'file':
            if 'file_name' not in data_location['location']:
                raise MissingFileName(reason="Connection asset requires to have specified a 'file_name'.")


def try_import_pyarrow():
    """
    Check if pyarrow package is installed in local environment, if not, just download and install it.
    """
    pyarrow_version = '3.0.0'
    try:
        installed_module_version = pkg_resources.get_distribution('pyarrow').version

        if version.parse(installed_module_version) < version.parse(pyarrow_version):
            warn(f"\"pyarrow\" package version is lower than {pyarrow_version}."
                 f"If you want to download data from Database Connection, you need to have pyarrow installed. "
                 f"If you are working on conda environment, you can try to install "
                 f"this package from conda-forge repository."
                 f"Installing version {pyarrow_version}")

            try:
                check_call([executable, "-m", "pip", "install", f"pyarrow=={pyarrow_version}"])

            except Exception as e:
                raise CannotInstallLibrary(value_name=e,
                                           reason="pyarrow failed to install. Please install it manually.")

    except pkg_resources.DistributionNotFound as e:
        warn(f"\"pyarrow\" is not installed. If you want to download data from Database Connection, "
             f"you need to have pyarrow installed. If you are working on conda environment, you can try to install "
             f"this package from conda-forge repository."
             f"Installing version {pyarrow_version}")

        try:
            check_call([executable, "-m", "pip", "install", f"pyarrow=={pyarrow_version}"])

        except Exception as e:
            raise CannotInstallLibrary(value_name=e,
                                       reason="pyarrow failed to install. Please install it manually.")

