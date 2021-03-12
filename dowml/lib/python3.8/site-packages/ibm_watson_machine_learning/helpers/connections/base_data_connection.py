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
    "BaseDataConnection"
]

import io
import json


from abc import ABC, abstractmethod
from typing import Union, Tuple, TYPE_CHECKING

import requests
from pandas import concat, DataFrame

from ibm_watson_machine_learning.utils.autoai.utils import try_load_dataset
from ibm_watson_machine_learning.utils.autoai.enums import DataConnectionTypes
from ibm_watson_machine_learning.utils.autoai.errors import CannotReadSavedRemoteDataBeforeFit
from ibm_watson_machine_learning.wml_client_error import ApiRequestFailure


if TYPE_CHECKING:
    from ibm_boto3 import resource


class BaseDataConnection(ABC):
    """
    Base class for  DataConnection.
    """

    def __init__(self):

        self.type = None
        self.connection = None
        self.location = None
        self.auto_pipeline_params = None
        self._wml_client = None
        self._run_id = None
        self._obm = None
        self._obm_cos_path = None
        self.id = None

    @abstractmethod
    def _to_dict(self) -> dict:
        """Convert DataConnection object to dictionary representation."""
        pass

    @classmethod
    @abstractmethod
    def _from_dict(cls, _dict: dict) -> 'BaseDataConnection':
        """Create a DataConnection object from dictionary."""
        pass

    @abstractmethod
    def read(self, with_holdout_split: bool = False) -> Union['DataFrame', Tuple['DataFrame', 'DataFrame']]:
        """Download dataset stored in remote data storage."""
        pass

    @abstractmethod
    def write(self, data: Union[str, 'DataFrame'], remote_name: str) -> None:
        """Upload file to a remote data storage."""
        pass

    def _fill_experiment_parameters(self, prediction_type: str, prediction_column: str, test_size: float,
                                    csv_separator: str = ',', excel_sheet: Union[str, int] = 0,
                                    encoding: str = 'utf-8') -> None:
        """
        To be able to recreate a holdout split, this method need to be called.
        """
        self.auto_pipeline_params = {
            'prediction_type': prediction_type,
            'prediction_column': prediction_column,
            'test_size': test_size,
            'csv_separator': csv_separator,
            'excel_sheet': excel_sheet,
            'encoding': encoding
        }

    def _download_obm_data_from_cos(self, cos_client: 'resource') -> 'DataFrame':
        """Download preprocessed OBM data. COS version."""

        # note: fetch all OBM file part names
        cos_summary = cos_client.Bucket(self.location.bucket).objects.filter(Prefix=self._obm_cos_path)
        file_names = [file_name.key for file_name in cos_summary]

        # note: if path does not exist, try to find in different one
        if not file_names:
            cos_summary = cos_client.Bucket(self.location.bucket).objects.filter(
                Prefix=self._obm_cos_path.split('./')[-1])
            file_names = [file_name.key for file_name in cos_summary]
            # --- end note
        # --- end note

        # TODO: this can be done simultaneously (multithreading / multiprocessing)
        # note: download all data parts and concatenate them into one output
        parts = []
        for file_name in file_names:
            file = cos_client.Object(self.location.bucket, file_name).get()
            buffer = io.BytesIO(file['Body'].read())
            parts.append(try_load_dataset(buffer=buffer))

        data = concat(parts)
        # --- end note
        return data

    def _download_obm_json(self) -> dict:
        """Download obm.json log."""
        if self._obm:
            if self._obm_cos_path:
                path = self._obm_cos_path.rsplit('features', 1)[0] + 'obm.json'
            else:
                path = f"{self.location.path}/{self._run_id}/data/obm/obm.json"
            data = self._download_json_file(path)
            return data
        else:
            raise Exception("OBM function called when not OBM scenario.")

    def _download_json_file(self, path) -> dict:
        """Download json file."""
        json_content = {}

        if self.type == DataConnectionTypes.S3:
            cos_client = self._init_cos_client()

            try:
                file = cos_client.Object(self.location.bucket, path).get()
                content = file["Body"].read()
                json_content = json.loads(content.decode('utf-8'))
            except Exception as cos_access_exception:
                raise ConnectionError(
                    f"Unable to access data object in cloud object storage with credentials supplied. "
                    f"Error: {cos_access_exception}")
        elif self.type == DataConnectionTypes.FS:
            json_url = f"{self._wml_client.wml_credentials['url']}/v2/asset_files/auto_ml/{path.split('/auto_ml/')[-1]}"

            with requests.get(json_url, params=self._wml_client._params(), headers=self._wml_client._get_headers(),
                              stream=True, verify=False) as file_response:
                if file_response.status_code != 200:
                    raise ApiRequestFailure(u'Failure during {}.'.format("downloading json"), file_response)

                downloaded_asset = file_response.content
                buffer = io.BytesIO(downloaded_asset)
                json_content = json.loads(buffer.getvalue().decode('utf-8'))
        else:
            raise NotImplementedError(f"Unsupported connection type: {self.type}")

        return json_content

    def _download_data_from_cos(self, cos_client: 'resource') -> 'DataFrame':
        """Download training data for this connection. COS version"""

        file = cos_client.Object(self.location.bucket,
                                 self.location.path).get()
        buffer = io.BytesIO(file['Body'].read())
        data = try_load_dataset(buffer=buffer,
                                sheet_name=self.auto_pipeline_params.get('excel_sheet', 0),
                                separator=self.auto_pipeline_params.get('csv_separator', ','),
                                encoding=self.auto_pipeline_params.get('encoding', 'utf-8')
                                )

        return data

    def _download_training_data_from_data_asset_storage(self) -> 'DataFrame':
        """Download training data for this connection. Data Storage."""

        # it means that it is on ICP env and it is before fit, so let's throw error
        if not self._wml_client:
            raise CannotReadSavedRemoteDataBeforeFit()

        # note: as we need to load a data into the memory,
        # we are using pure requests and helpers from the WML client
        asset_id = self.location.href.split('?')[0].split('/')[-1]

        # note: download data asset details
        asset_response = requests.get(self._wml_client.data_assets._href_definitions.get_data_asset_href(asset_id),
                                      params=self._wml_client._params(),
                                      headers=self._wml_client._get_headers(),
                                      verify=False)

        asset_details = self._wml_client.data_assets._handle_response(200, u'get assets', asset_response)

        # note: read the csv url
        if 'handle' in asset_details['attachments'][0]:
            attachment_url = asset_details['attachments'][0]['handle']['key']

            # note: make the whole url pointing out the csv
            artifact_content_url = (f"{self._wml_client.data_assets._href_definitions.get_wsd_model_attachment_href()}"
                                    f"{attachment_url}")

            # note: stream the whole CSV file
            csv_response = requests.get(artifact_content_url,
                                        params=self._wml_client._params(),
                                        headers=self._wml_client._get_headers(),
                                        stream=True,
                                        verify=False)

            if csv_response.status_code != 200:
                raise ApiRequestFailure(u'Failure during {}.'.format("downloading model"), csv_response)

            downloaded_asset = csv_response.content

            # note: read the csv/xlsx file from the memory directly into the pandas DataFrame
            buffer = io.BytesIO(downloaded_asset)
            data = try_load_dataset(buffer=buffer,
                                    sheet_name=self.auto_pipeline_params.get('excel_sheet', 0),
                                    separator=self.auto_pipeline_params.get('csv_separator', ','),
                                    encoding=self.auto_pipeline_params.get('encoding', 'utf-8')
                                    )

            return data
        else:
            # NFS scenario
            connection_id = asset_details['attachments'][0]['connection_id']
            connection_path = asset_details['attachments'][0]['connection_path']

            return self._download_data_from_nfs_connection_using_id_and_path(connection_id, connection_path)

    def _download_training_data_from_file_system(self) -> 'DataFrame':
        """Download training data for this connection. File system version."""

        try:
            url = f"{self._wml_client.wml_credentials['url']}/v2/asset_files/{self.location.path.split('/assets/')[-1]}"
            # note: stream the whole CSV file
            csv_response = requests.get(url,
                                        params=self._wml_client._params(),
                                        headers=self._wml_client._get_headers(),
                                        stream=True,
                                        verify=False)

            if csv_response.status_code != 200:
                raise ApiRequestFailure(u'Failure during {}.'.format("downloading model"), csv_response)

            downloaded_asset = csv_response.content
            # note: read the csv/xlsx file from the memory directly into the pandas DataFrame
            buffer = io.BytesIO(downloaded_asset)
            data = try_load_dataset(buffer=buffer,
                                    sheet_name=self.auto_pipeline_params.get('excel_sheet', 0),
                                    separator=self.auto_pipeline_params.get('csv_separator', ','),
                                    encoding=self.auto_pipeline_params.get('encoding', 'utf-8')
                                    )
        except (ApiRequestFailure, AttributeError):
            with open(self.location.path, 'rb') as data_buffer:
                data = try_load_dataset(buffer=data_buffer,
                                        sheet_name=self.auto_pipeline_params.get('excel_sheet', 0),
                                        separator=self.auto_pipeline_params.get('csv_separator', ','),
                                        encoding=self.auto_pipeline_params.get('encoding', 'utf-8')
                                        )

        return data

    def _download_obm_data_from_file_system(self) -> 'DataFrame':
        """Download preprocessed OBM data. FS version."""

        # note: fetch all OBM file part names
        url = f"{self._wml_client.wml_credentials['url']}/v2/asset_files/auto_ml/{self.location.path.split('/auto_ml/')[-1]}/{self._run_id}/data/obm/features"
        params = self._wml_client._params()
        params['flat'] = "true"

        response = requests.get(url,
                                params=params,
                                headers=self._wml_client._get_headers(),
                                verify=False)

        if response.status_code != 200:
            raise ApiRequestFailure(u'Failure during {}.'.format("getting files information"), response)

        file_names = [e['path'].split('/')[-1] for e in response.json()['resources'] if e['type'] == 'file' and e['path'].split('/')[-1].startswith('part')]

        # TODO: this can be done simultaneously (multithreading / multiprocessing)
        # note: download all data parts and concatenate them into one output
        parts = []
        for file_name in file_names:
            csv_response = requests.get(url + '/' + file_name,
                                        params=self._wml_client._params(),
                                        headers=self._wml_client._get_headers(),
                                        stream=True,
                                        verify=False)

            if csv_response.status_code != 200:
                raise ApiRequestFailure(u'Failure during {}.'.format("downloading model"), csv_response)

            downloaded_asset = csv_response.content
            # note: read the csv/xlsx file from the memory directly into the pandas DataFrame
            buffer = io.BytesIO(downloaded_asset)
            parts.append(try_load_dataset(buffer=buffer))

        data = concat(parts)
        # --- end note
        return data

    def _download_data_from_nfs_connection(self) -> 'DataFrame':
        """Download training data for this connection. NFS."""

        # note: as we need to load a data into the memory,
        # we are using pure requests and helpers from the WML client
        data_path = self.location.path
        connection_id = self.connection.asset_id

        return self._download_data_from_nfs_connection_using_id_and_path(connection_id, data_path)

    def _download_data_from_nfs_connection_using_id_and_path(self, connection_id, connection_path) -> 'DataFrame':
        """Download training data for this connection. NFS."""

        # it means that it is on ICP env and it is before fit, so let's throw error
        if not self._wml_client:
            raise CannotReadSavedRemoteDataBeforeFit()

        # note: download data asset details
        href = self._wml_client.connections._href_definitions.get_connection_by_id_href(connection_id) + '/assets'
        params = self._wml_client._params().copy()
        params['path'] = connection_path
        params['fetch'] = 'metadata'
        nfs_response = requests.get(href,
                                    params=params,
                                    headers=self._wml_client._get_headers(),
                                    verify=False)

        asset_details = self._wml_client.data_assets._handle_response(200, u'get nfs asset', nfs_response)

        # note: read the column names of dataset
        dataset_columns = [column['name'] for column in asset_details['fields']]

        # note: stream the whole CSV file
        params['fetch'] = 'data'
        csv_response = requests.get(href,
                                    params=params,
                                    headers=self._wml_client._get_headers(),
                                    verify=False)

        if csv_response.status_code != 200:
            raise ApiRequestFailure(u'Failure during {}.'.format("downloading model"), csv_response)

        data = DataFrame(csv_response.json()['data'], columns=dataset_columns)

        return data
