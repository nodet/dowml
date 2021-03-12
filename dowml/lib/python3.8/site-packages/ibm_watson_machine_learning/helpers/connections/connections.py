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
    "DataConnection",
    "S3Connection",
    "S3Location",
    "FSLocation",
    "AssetLocation",
    "CP4DAssetLocation",
    "WMLSAssetLocation",
    "WSDAssetLocation",
    "CloudAssetLocation",
    "DeploymentOutputAssetLocation",
    "NFSConnection",
    "NFSLocation"
]

import io
import os
import uuid
import copy
from copy import deepcopy
from typing import Union, Tuple, List, TYPE_CHECKING, Optional
from warnings import warn

from ibm_boto3 import resource
from ibm_botocore.client import ClientError
from pandas import DataFrame
import requests

from ibm_watson_machine_learning.utils.autoai.enums import PredictionType, DataConnectionTypes
from ibm_watson_machine_learning.utils.autoai.errors import (
    MissingAutoPipelinesParameters, UseWMLClient, MissingCOSStudioConnection, MissingProjectLib,
    HoldoutSplitNotSupported, InvalidCOSCredentials, MissingLocalAsset, InvalidIdType, NotWSDEnvironment,
    NotExistingCOSResource
)
from ibm_watson_machine_learning.utils.autoai.watson_studio import get_project
from ibm_watson_machine_learning.wml_client_error import MissingValue, ApiRequestFailure
from .base_connection import BaseConnection
from .base_data_connection import BaseDataConnection
from .base_location import BaseLocation

if TYPE_CHECKING:
    from ibm_watson_machine_learning.workspace import WorkSpace


class DataConnection(BaseDataConnection):
    """
    Data Storage Connection class needed for WML training metadata (input data).

    Parameters
    ----------
    connection: Union[S3Connection], required
        connection parameters of specific type

    location: Union[S3Location, FSLocation, AssetLocation],
        required location parameters of specific type

    data_join_node_name: Union[str, List[str]], optional
        Names for node(s). If no value provided, data file name will be used as node name. If str will be passed,
        it will became node name. If multiple names will be passed, several nodes will have the same data connection
        (used for excel files with multiple sheets).
    """

    def __init__(self,
                 location: Union['S3Location',
                                 'FSLocation',
                                 'AssetLocation',
                                 'CP4DAssetLocation',
                                 'WMLSAssetLocation',
                                 'WSDAssetLocation',
                                 'CloudAssetLocation',
                                 'NFSLocation',
                                 'DeploymentOutputAssetLocation'],
                 connection: Optional[Union['S3Connection', 'NFSConnection']] = None,
                 data_join_node_name: Union[str, List[str]] = None):
        super().__init__()

        self.connection = connection
        self.location = location

        if isinstance(connection, S3Connection):
            self.type = DataConnectionTypes.S3

        elif isinstance(location, NFSLocation):
            self.type = DataConnectionTypes.CA
            if self.connection is not None:
                self.location.id = self.connection.asset_id

        elif isinstance(location, FSLocation):
            self.type = DataConnectionTypes.FS

        elif isinstance(location, (AssetLocation, CP4DAssetLocation, WMLSAssetLocation, CloudAssetLocation, WSDAssetLocation, DeploymentOutputAssetLocation)):
            self.type = DataConnectionTypes.DS

        self.auto_pipeline_params = {}  # note: needed parameters for recreation of autoai holdout split
        self._wml_client = None
        self._run_id = None
        self._obm = False
        self._obm_cos_path = None

        # note: make data connection id as a location path for OBM + KB
        if data_join_node_name is None:
            if self.type == DataConnectionTypes.S3:
                self.id = location.path

            else:
                self.id = None

        else:
            self.id = data_join_node_name
        # --- end note

    @classmethod
    def from_studio(cls, path: str) -> List['DataConnection']:
        """
        Create DataConnections from the credentials stored (connected) in Watson Studio. Only for COS.

        Parameters
        ----------
        path: str, required
            Path in COS bucket to the training dataset.

        Returns
        -------
        List with DataConnection objects.

        Example
        -------
        >>> data_connections = DataConnection.from_studio(path='iris_dataset.csv')
        """
        try:
            from project_lib import Project

        except ModuleNotFoundError:
            raise MissingProjectLib("Missing project_lib package.")

        else:
            data_connections = []
            for name, value in globals().items():
                if isinstance(value, Project):
                    connections = value.get_connections()

                    if connections:
                        for connection in connections:
                            asset_id = connection['asset_id']
                            connection_details = value.get_connection(asset_id)

                            if ('url' in connection_details and 'access_key' in connection_details and
                                    'secret_key' in connection_details and 'bucket' in connection_details):
                                data_connections.append(
                                    cls(connection=S3Connection(endpoint_url=connection_details['url'],
                                                                access_key_id=connection_details['access_key'],
                                                                secret_access_key=connection_details['secret_key']),
                                        location=S3Location(bucket=connection_details['bucket'],
                                                            path=path))
                                )

            if data_connections:
                return data_connections

            else:
                raise MissingCOSStudioConnection(
                    "There is no any COS Studio connection. "
                    "Please create a COS connection from the UI and insert "
                    "the cell with project API connection (Insert project token)")

    def _subdivide_connection(self):
        if type(self.id) is str or not self.id:
            return [self]
        else:
            def cpy(new_id):
                child = copy.copy(self)
                child.id = new_id
                return child

            return [cpy(id) for id in self.id]

    def _to_dict(self) -> dict:
        """
        Convert DataConnection object to dictionary representation.

        Returns
        -------
        Dictionary
        """

        if self.id and type(self.id) is list:
            raise InvalidIdType(list)

        _dict = {"type": self.type}

        # note: for OBM (id of DataConnection if an OBM node name)
        if self.id is not None:
            _dict['id'] = self.id
        # --- end note

        if self.connection is not None:
            _dict['connection'] = deepcopy(self.connection.to_dict())

        else:
            _dict['connection'] = {}

        _dict['location'] = deepcopy(self.location.to_dict())
        return _dict

    def __repr__(self):
        return str(self._to_dict())

    def __str__(self):
        return str(self._to_dict())

    @classmethod
    def _from_dict(cls, _dict: dict) -> 'DataConnection':
        """
        Create a DataConnection object from dictionary

        Parameters
        ----------
        _dict: dict, required
            A dictionary data structure with information about data connection reference.

        Returns
        -------
        DataConnection
        """
        if _dict['type'] == DataConnectionTypes.S3:
            data_connection: 'DataConnection' = cls(
                connection=S3Connection(
                    access_key_id=_dict['connection']['access_key_id'],
                    secret_access_key=_dict['connection']['secret_access_key'],
                    endpoint_url=_dict['connection']['endpoint_url']
                ),
                location=S3Location(
                    bucket=_dict['location']['bucket'],
                    path=_dict['location']['path']
                )
            )
        elif _dict['type'] == DataConnectionTypes.FS:
            data_connection: 'DataConnection' = cls(
                location=FSLocation._set_path(path=_dict['location']['path'])
            )
        elif _dict['type'] == DataConnectionTypes.CA:
            data_connection: 'DataConnection' = cls(
                connection=NFSConnection(asset_id=_dict['connection']['asset_id']),
                location=NFSLocation(path=_dict['location']['path'])
            )
        else:
            data_connection: 'DataConnection' = cls(
                location=AssetLocation._set_path(href=_dict['location']['href'])
            )

        if _dict.get('id'):
            data_connection.id = _dict['id']

        return data_connection

    def read(self,
             with_holdout_split: bool = False,
             csv_separator: str = ',',
             excel_sheet: Union[str, int] = 0,
             encoding: Optional[str] = 'utf-8') -> Union['DataFrame', Tuple['DataFrame', 'DataFrame']]:
        """
        Download dataset stored in remote data storage.

        Parameters
        ----------
        with_holdout_split: bool, optional
            If True, data will be split to train and holdout dataset as it was by AutoAI.

        csv_separator: str, optional
            Separator / delimiter for CSV file, default is ','

        excel_sheet: Union[str, int], optional
            Excel file sheet name to use, default is 0.

        encoding: str, optional
            Encoding type of the CSV

        Returns
        -------
        pandas.DataFrame contains dataset from remote data storage or Tuple[pandas.DataFrame, pandas.DataFrame]
            containing training data and holdout data from remote storage
            (only if only_holdout == True and auto_pipeline_params was passed)
        """

        if with_holdout_split:
            print("\"with_holdout_split\" option is depreciated.")
            warn("\"with_holdout_split\" option is depreciated.")

        from sklearn.model_selection import train_test_split

        if with_holdout_split and not self.auto_pipeline_params.get('prediction_type', False):
            raise MissingAutoPipelinesParameters(
                self.auto_pipeline_params,
                reason=f"To be able to recreate an original holdout split, you need to schedule a training job or "
                       f"if you are using historical runs, just call historical_optimizer.get_data_connections()")

        # note: allow to read data at any time
        elif (('csv_separator' not in self.auto_pipeline_params and 'excel_sheet' not in self.auto_pipeline_params and
               'encoding' not in self.auto_pipeline_params)
              or csv_separator != ',' or excel_sheet != 0 or encoding != 'utf-8'):
            self.auto_pipeline_params['csv_separator'] = csv_separator
            self.auto_pipeline_params['excel_sheet'] = excel_sheet
            self.auto_pipeline_params['encoding'] = encoding
        # --- end note

        data = DataFrame()
        if self.type == DataConnectionTypes.S3:
            cos_client = self._init_cos_client()

            try:
                if self._obm:
                    data = self._download_obm_data_from_cos(cos_client=cos_client)

                else:
                    data = self._download_data_from_cos(cos_client=cos_client)

            except Exception as cos_access_exception:
                raise ConnectionError(
                    f"Unable to access data object in cloud object storage with credentials supplied. "
                    f"Error: {cos_access_exception}")

        elif self.type == DataConnectionTypes.DS:

            if self._obm:
                raise NotImplementedError("Multiple files for CP4D / WML Server is not yet supported.")

            data = self._download_training_data_from_data_asset_storage()

        elif self.type == DataConnectionTypes.FS:

            if self._obm:
                data = self._download_obm_data_from_file_system()
            else:
                data = self._download_training_data_from_file_system()

        elif self.type == DataConnectionTypes.CA:
            data = self._download_data_from_nfs_connection()

        if isinstance(data, DataFrame) and 'Unnamed: 0' in data.columns.tolist():
            data.drop(['Unnamed: 0'], axis=1, inplace=True)

        # note: this code is depreciated, user information isa provided at the beginning.
        if with_holdout_split:
            if not isinstance(data, DataFrame):
                raise HoldoutSplitNotSupported(
                    None,
                    reason="SDK currently does not support a local holdout split with xlsx files without sheet_name "
                           "provided.")

            if not self._obm and self.auto_pipeline_params.get('data_join_graph', False):
                raise HoldoutSplitNotSupported(
                    None,
                    reason="Holdout split is not supported for not processed multiple input data. "
                           "Please use \"get_preprocessed_data().read(with_holdout_split=True)\"")

            if self.auto_pipeline_params.get('train_sample_rows_test_size'):
                pass

            # note: 'classification' check left for backward compatibility
            if (self.auto_pipeline_params['prediction_type'] == PredictionType.BINARY
                    or self.auto_pipeline_params['prediction_type'] == PredictionType.MULTICLASS
                    or self.auto_pipeline_params['prediction_type'] == 'classification'):
                x, x_holdout, y, y_holdout = train_test_split(
                    data.drop([self.auto_pipeline_params['prediction_column']], axis=1),
                    data[self.auto_pipeline_params['prediction_column']].values,
                    test_size=self.auto_pipeline_params['test_size'],
                    random_state=33,
                    stratify=data[self.auto_pipeline_params['prediction_column']].values)
            else:
                x, x_holdout, y, y_holdout = train_test_split(
                    data.drop([self.auto_pipeline_params['prediction_column']], axis=1),
                    data[self.auto_pipeline_params['prediction_column']].values,
                    test_size=self.auto_pipeline_params['test_size'],
                    random_state=33)

            data_train = DataFrame(data=x, columns=data.columns.tolist())
            data_train[self.auto_pipeline_params['prediction_column']] = y

            data_holdout = DataFrame(data=x_holdout, columns=data.columns.tolist())
            data_holdout[self.auto_pipeline_params['prediction_column']] = y_holdout

            return data_train, data_holdout
        # --- end note

        return data

    def write(self, data: Union[str, 'DataFrame'], remote_name: str) -> None:
        """
        Upload file to a remote data storage.

        Parameters
        ----------
        data: str, required
            Local path to the dataset or pandas.DataFrame with data.

        remote_name: str, required
            Name that dataset should be stored with in remote data storage.
        """
        if self.type == DataConnectionTypes.S3:
            cos_resource_client = self._init_cos_client()
            if isinstance(data, str):
                with open(data, "rb") as file_data:
                    cos_resource_client.Object(self.location.bucket, remote_name).upload_fileobj(
                        Fileobj=file_data)

            elif isinstance(data, DataFrame):
                # note: we are saving csv in memory as a file and stream it to the COS
                buffer = io.StringIO()
                data.to_csv(buffer, index=False)
                buffer.seek(0)

                with buffer as f:
                    cos_resource_client.Object(self.location.bucket, remote_name).upload_fileobj(
                        Fileobj=io.BytesIO(bytes(f.read().encode())))

            else:
                raise TypeError("data should be either of type \"str\" or \"pandas.DataFrame\"")

        elif self.type == DataConnectionTypes.DS:
            raise UseWMLClient('DataConnection.write()',
                               reason="If you want to upload any data to CP4D instance, "
                                      "firstly please get the WML client by calling "
                                      "\"client = WMLInstance().get_client()\" "
                                      "then call the method: \"client.data_asset.create()\"")

    def _init_cos_client(self) -> 'resource':
        """Initiate COS client for further usage."""
        from ibm_botocore.client import Config
        if hasattr(self.connection, 'auth_endpoint') and hasattr(self.connection, 'api_key'):
            cos_client = resource(
                service_name='s3',
                ibm_api_key_id=self.connection.api_key,
                ibm_auth_endpoint=self.connection.auth_endpoint,
                config=Config(signature_version="oauth"),
                endpoint_url=self.connection.endpoint_url
            )

        else:
            cos_client = resource(
                service_name='s3',
                endpoint_url=self.connection.endpoint_url,
                aws_access_key_id=self.connection.access_key_id,
                aws_secret_access_key=self.connection.secret_access_key
            )
        return cos_client

    def _validate_cos_resource(self):
        cos_client = self._init_cos_client()
        try:
            files = cos_client.Bucket(self.location.bucket).objects.all()
            next(x for x in files if x.key == self.location.path)
        except Exception as e:
            raise NotExistingCOSResource(self.location.bucket, self.location.path)


class S3Connection(BaseConnection):
    """
    Connection class to COS data storage in S3 format.

    Parameters
    ----------
    endpoint_url: str, required
        S3 data storage url (COS)

    access_key_id: str, optional
        access_key_id of the S3 connection (COS)

    secret_access_key: str, optional
        secret_access_key of the S3 connection (COS)

    api_key: str, optional
        API key of the S3 connection (COS)

    service_name: str, optional
        Service name of the S3 connection (COS)

    auth_endpoint: str, optional
        Authentication endpoint url of the S3 connection (COS)
    """

    def __init__(self, endpoint_url: str, access_key_id: str = None, secret_access_key: str = None,
                 api_key: str = None, service_name: str = None, auth_endpoint: str = None) -> None:

        if (access_key_id is None or secret_access_key is None) and (api_key is None or auth_endpoint is None):
            raise InvalidCOSCredentials(reason='You need to specify (access_key_id and secret_access_key) or'
                                               '(api_key and auth_endpoint)')

        if secret_access_key is not None:
            self.secret_access_key = secret_access_key

        if api_key is not None:
            self.api_key = api_key

        if service_name is not None:
            self.service_name = service_name

        if auth_endpoint is not None:
            self.auth_endpoint = auth_endpoint

        if access_key_id is not None:
            self.access_key_id = access_key_id

        if endpoint_url is not None:
            self.endpoint_url = endpoint_url


class S3Location(BaseLocation):
    """
    Connection class to COS data storage in S3 format.

    Parameters
    ----------
    bucket: str, required
        COS bucket name

    path: str, required
        COS data path in the bucket

    model_location: str, optional
        Path to the pipeline model in the COS.

    training_status: str, optional
        Path t the training status json in COS.
    """

    def __init__(self, bucket: str, path: str, **kwargs) -> None:
        self.bucket = bucket
        self.path = path

        if kwargs.get('model_location') is not None:
            self._model_location = kwargs['model_location']

        if kwargs.get('training_status') is not None:
            self._training_status = kwargs['training_status']

    def _get_file_size(self, cos_resource_client: 'resource') -> 'int':
        try:
            size = cos_resource_client.Object(self.bucket, self.path).content_length
        except ClientError:
            size = 0
        return size


class FSLocation(BaseLocation):
    """
    Connection class to File Storage in CP4D.
    """

    def __init__(self, path: Optional[str] = None) -> None:
        if path is None:
            self.path = "/{option}/{id}" + f"/assets/auto_ml/auto_ml.{uuid.uuid4()}/wml_data"

        else:
            self.path = path

    @classmethod
    def _set_path(cls, path: str) -> 'FSLocation':
        location = cls()
        location.path = path
        return location

    def _save_file_as_data_asset(self, workspace: 'WorkSpace') -> 'str':

        asset_name = self.path.split('/')[-1]
        if self.path:
            data_asset_details = workspace.wml_client.data_assets.create(asset_name, self.path)
            return workspace.wml_client.data_assets.get_uid(data_asset_details)
        else:
            raise MissingValue('path', reason="Incorrect initialization of class FSLocation")

    def _get_file_size(self, workspace: 'WorkSpace') -> 'int':
        # note if path is not file then returned size is 0
        try:
            # note: try to get file size from remote server
            url = f"{workspace.wml_client.wml_credentials['url']}/v2/asset_files/{self.path.split('/assets/')[-1]}"
            path_info_response = requests.head(url, headers=workspace.wml_client._get_headers(), params=workspace.wml_client._params(), verify=False)
            if path_info_response.status_code != 200:
                raise ApiRequestFailure(u"Failure during getting path details",path_info_response)
            path_info = path_info_response.headers
            if 'X-Asset-Files-Type' in path_info and path_info['X-Asset-Files-Type'] == 'file':
                size = path_info['X-Asset-Files-Size']
            else:
                size = 0
            # -- end note
        except (ApiRequestFailure, AttributeError):
            # note try get size of file from local fs
            size = os.stat(path=self.path).st_size if os.path.isfile(path=self.path) else 0
            # -- end note
        return size


class AssetLocation(BaseLocation):

    def __init__(self, asset_id: str) -> None:
        self._wsd = self._is_wsd()

        if self._wsd:
            self.href = f'/v2/assets/{asset_id}'
            self._asset_name = None
            self._asset_id = None
            self._local_asset_path = None
        else:
            self.href = f'/v2/assets/{asset_id}?' + '{option}' + '=' + '{id}'
            self.id = asset_id

    @classmethod
    def _is_wsd(cls):
        if os.environ.get('USER_ACCESS_TOKEN'):
            return False

        try:
            from project_lib import Project
            try:
                access = Project.access()
                return True
            except RuntimeError:
                pass
        except ModuleNotFoundError:
            pass

        return False

    @classmethod
    def _set_path(cls, href: str) -> 'AssetLocation':
        location = cls('.')
        location.href = href
        return location

    def _get_file_size(self, workspace: 'WorkSpace', *args) -> 'int':
        if self._wsd:
            return self._wsd_get_file_size()
        else:
            verify_request = False if workspace.wml_client.ICP else True
            asset_info_response = requests.get(workspace.wml_client.data_assets._href_definitions.get_data_asset_href(self.id),
                                                params= workspace.wml_client._params(),
                                                headers= workspace.wml_client._get_headers(), verify=verify_request)
            if asset_info_response.status_code != 200:
                raise ApiRequestFailure(u"Failure during getting asset details", asset_info_response)
            return asset_info_response.json()['metadata'].get('size')

    def _wsd_setup_local_asset_details(self) -> None:
        if not self._wsd:
            raise NotWSDEnvironment()

        # note: set local asset file from asset_id
        project = get_project()
        project_id = project.get_metadata()["metadata"]["guid"]

        local_assets = project.get_files()

        # note: reuse local asset_id when object is reused more times
        if self._asset_id is None:
            local_asset_id = self.href.split('/')[3].split('?space_id')[0]

        else:
            local_asset_id = self._asset_id
        # --- end note

        if local_asset_id not in str(local_assets):
            raise MissingLocalAsset(local_asset_id, reason="Provided asset_id cannot be found on WS Desktop.")

        else:
            for asset in local_assets:
                if asset['asset_id'] == local_asset_id:
                    asset_name = asset['name']
                    self._asset_name = asset_name
                    self._asset_id = local_asset_id

            local_asset_path = f"{os.path.abspath('.')}/{project_id}/assets/data_asset/{asset_name}"
            self._local_asset_path = local_asset_path

    def _wsd_move_asset_to_server(self, workspace: 'WorkSpace') -> None:
        if not self._wsd:
            raise NotWSDEnvironment()

        if not self._local_asset_path or self._asset_name or self._asset_id:
            self._wsd_setup_local_asset_details()

        remote_asset_details = workspace.wml_client.data_assets.create(self._asset_name, self._local_asset_path)
        self.href = remote_asset_details['metadata']['href']

    def _wsd_get_file_size(self) -> 'int':
        if not self._wsd:
            raise NotWSDEnvironment()

        if not self._local_asset_path or self._asset_name or self._asset_id:
            self._wsd_setup_local_asset_details()
        return os.stat(path=self._local_asset_path).st_size if os.path.isfile(path=self._local_asset_path) else 0

    @classmethod
    def list_wsd_assets(cls):
        if not cls._is_wsd():
            raise NotWSDEnvironment

        project = get_project()
        return project.get_files()

    def to_dict(self) -> dict:
        """Return a json dictionary representing this model."""
        _dict = vars(self).copy()

        del _dict['_wsd']

        if self._wsd:
            del _dict['_asset_name']
            del _dict['_asset_id']
            del _dict['_local_asset_path']

        return _dict


class NFSConnection(BaseConnection):
    """
    Connection class to file storage in CP4D of NFS format.

    Parameters
    ----------
    connection_id: str, required
        Connection ID from the project on CP4D
    """
    def __init__(self, asset_id: str):
        self.asset_id = asset_id


class NFSLocation(BaseLocation):
    """
    Location class to file storage in CP4D of NFS format.

    Parameters
    ----------
    path: str, required
        Data path form the project on CP4D.
    """
    def __init__(self, path: str):
        self.path = path
        self.id = None

    def _get_file_size(self, workspace: 'Workspace', *args) -> 'int':
        params = workspace.wml_client._params().copy()
        params['path'] = self.path
        params['detail'] = 'true'

        href = workspace.wml_client.connections._href_definitions.get_connection_by_id_href(self.id) + '/assets'
        asset_info_response = requests.get(href,
                                           params=params, headers=workspace.wml_client._get_headers(None), verify=False)
        if asset_info_response.status_code != 200:
            raise Exception(u"Failure during getting asset details", asset_info_response.json())
        return asset_info_response.json()['details']['file_size']


class CP4DAssetLocation(AssetLocation):
    """
    Connection class to data assets in CP4D.

    Parameters
    ----------
    asset_id: str, required
        Asset ID from the project on CP4D.
    """

    def __init__(self, asset_id: str) -> None:
        super().__init__(asset_id)
        warning_msg = ("Depreciation Warning: Class CP4DAssetLocation is no longer supported and will be removed."
                        "Use AssetLocation instead.")
        print(warning_msg)

    def _get_file_size(self, workspace: 'WorkSpace', *args) -> 'int':
        return super()._get_file_size(workspace)


class WMLSAssetLocation(AssetLocation):
    """
    Connection class to data assets in WML Server.

    Parameters
    ----------
    asset_id: str, required
        Asset ID of the file loaded on space in WML Server.
    """

    def __init__(self, asset_id: str) -> None:
        super().__init__(asset_id)
        warning_msg = ("Depreciation Warning: Class WMLSAssetLocation is no longer supported and will be removed."
                        "Use AssetLocation instead.")
        print(warning_msg)

    def _get_file_size(self, workspace: 'WorkSpace', *args) -> 'int':
        return  super()._get_file_size(workspace)


class CloudAssetLocation(AssetLocation):
    """
    Connection class to data assets as input data references to batch deployment job on Cloud.

    Parameters
    ----------
    asset_id: str, required
        Asset ID of the file loaded on space on Cloud.
    """
    def __init__(self, asset_id: str) -> None:
        super().__init__(asset_id)
        self.href = self.href
        warning_msg = ("Depreciation Warning: Class CloudAssetLocation is no longer supported and will be removed."
                        "Use AssetLocation instead.")
        print(warning_msg)

    def _get_file_size(self, workspace: 'WorkSpace', *args) -> 'int':
        return super()._get_file_size(workspace)


class WSDAssetLocation(BaseLocation):
    """
    Connection class to data assets in WS Desktop.

    Parameters
    ----------
    asset_id: str, required
        Asset ID from the project on WS Desktop.
    """

    def __init__(self, asset_id: str) -> None:
        self.href = f'/v2/assets/{asset_id}'
        self._asset_name = None
        self._asset_id = None
        self._local_asset_path = None

        warning_msg = ("Depreciation Warning: Class WSDAssetLocation is no longer supported and will be removed."
                       "Use AssetLocation instead.")
        print(warning_msg)

    @classmethod
    def list_assets(cls):
        project = get_project()
        return project.get_files()

    def _setup_local_asset_details(self) -> None:
        # note: set local asset file from asset_id
        project = get_project()
        project_id = project.get_metadata()["metadata"]["guid"]

        local_assets = project.get_files()

        # note: reuse local asset_id when object is reused more times
        if self._asset_id is None:
            local_asset_id = self.href.split('/')[3].split('?space_id')[0]

        else:
            local_asset_id = self._asset_id
        # --- end note

        if local_asset_id not in str(local_assets):
            raise MissingLocalAsset(local_asset_id, reason="Provided asset_id cannot be found on WS Desktop.")

        else:
            for asset in local_assets:
                if asset['asset_id'] == local_asset_id:
                    asset_name = asset['name']
                    self._asset_name = asset_name
                    self._asset_id = local_asset_id

            local_asset_path = f"{os.path.abspath('.')}/{project_id}/assets/data_asset/{asset_name}"
            self._local_asset_path = local_asset_path

    def _move_asset_to_server(self, workspace: 'WorkSpace') -> None:
        if not self._local_asset_path or self._asset_name or self._asset_id:
            self._setup_local_asset_details()

        remote_asset_details = workspace.wml_client.data_assets.create(self._asset_name, self._local_asset_path)
        self.href = remote_asset_details['metadata']['href']

    @classmethod
    def _set_path(cls, href: str) -> 'WSDAssetLocation':
        location = cls('.')
        location.href = href
        return location

    def to_dict(self) -> dict:
        """Return a json dictionary representing this model."""
        _dict = vars(self).copy()
        del _dict['_asset_name']
        del _dict['_asset_id']
        del _dict['_local_asset_path']

        return _dict

    def _get_file_size(self) -> 'int':
        if not self._local_asset_path or self._asset_name or self._asset_id:
            self._setup_local_asset_details()
        return os.stat(path=self._local_asset_path).st_size if os.path.isfile(path=self._local_asset_path) else 0


class DeploymentOutputAssetLocation(BaseLocation):
    """
    Connection class to data assets where output of batch deployment will be stored.

    Parameters
    ----------
    name: str, required
        name of .csv file which will be saved as data asset.
    description: str, optional
        description of the data asset
    """

    def __init__(self, name: str, description: str = "") -> None:
        self.name = name
        self.description = description

