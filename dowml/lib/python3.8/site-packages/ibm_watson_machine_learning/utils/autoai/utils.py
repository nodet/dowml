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
    'fetch_pipelines',
    'load_file_from_file_system',
    'load_file_from_file_system_nonautoai',
    'NextRunDetailsGenerator',
    'prepare_auto_ai_model_to_publish_normal_scenario',
    'prepare_auto_ai_model_to_publish_notebook_normal_scenario',
    'prepare_auto_ai_model_to_publish',
    'remove_file',
    'ProgressGenerator',
    'is_ipython',
    'try_import_lale',
    'try_load_dataset',
    'check_dependencies_versions',
    'try_import_autoai_libs',
    'try_import_tqdm',
    'try_import_xlrd',
    'try_import_graphviz',
    'prepare_cos_client',
    'create_model_download_link',
    'create_summary',
    'prepare_auto_ai_model_to_publish_notebook',
    'get_node_and_runtime_index',
    'download_experiment_details_from_file',
    'prepare_model_location_path',
    'download_wml_pipeline_details_from_file',
    'init_cos_client',
    'check_graphviz_binaries',
    'try_import_joblib',
    'get_sw_spec_and_type_based_on_sklearn',
    'validate_additional_params_for_optimizer'
]

import io
import json
import os
from contextlib import redirect_stdout
from functools import wraps
from subprocess import check_call
from sys import executable
from tarfile import open as open_tar
from typing import Dict, Union, Tuple, List, TYPE_CHECKING, Optional
from warnings import warn
from zipfile import ZipFile

import pkg_resources
import requests
from packaging import version

from .errors import (MissingPipeline, DataFormatNotSupported, LibraryNotCompatible,
                     CannotInstallLibrary, CannotDownloadTrainingDetails, CannotDownloadWMLPipelineDetails,
                     VisualizationFailed, AdditionalParameterIsUnexpected)

if TYPE_CHECKING:
    from io import BytesIO, BufferedIOBase
    from pandas import DataFrame
    from collections import OrderedDict
    from sklearn.pipeline import Pipeline
    from ibm_watson_machine_learning import APIClient
    from ibm_watson_machine_learning.helpers import DataConnection, S3Connection
    from ibm_boto3 import resource, client


def create_model_download_link(file_path: str):
    """
    Creates download link and shows it in the jupyter notebook

    Parameters
    ----------
    file_path: str, required
    """
    if is_ipython():
        from IPython.display import display
        from ibm_watson_machine_learning.utils import create_download_link
        display(create_download_link(file_path))


def fetch_pipelines(run_params: dict,
                    path: str,
                    wml_client: 'APIClient',
                    pipeline_name: str = None,
                    load_pipelines: bool = False,
                    store: bool = False) -> Union[None, Dict[str, 'Pipeline']]:
    """
    Helper function to download and load computed AutoAI pipelines (sklearn pipelines).

    Parameters
    ----------
    run_params: dict, required
        Fetched details of the run/fit.

    path: str, required
        Local system path indicates where to store downloaded pipelines.

    pipeline_name: str, optional
        Name of the pipeline to download, if not specified, all pipelines are downloaded.

    load_pipelines: bool, optional
        Indicator if we load and return downloaded piepelines.

    store: bool, optional
        Indicator to store pipelines in local filesystem

    wml_client: APIClient, required

    Returns
    -------
    List of sklearn Pipelines or None if load_pipelines is set to False.
    """

    def check_pipeline_nodes(pipeline: dict, request_json: dict, wml_client) -> None:
        """
        Automate check all pipeline nodes to find xgboost or lightgbm dependency.
        """
        xgboost_estimators = ['XGBClassifierEstimator', 'XGBRegressorEstimator']
        lightgbm_estimators = ['LGBMClassifierEstimator', 'LGBMRegressorEstimator']

        # note: check dependencies for estimators and other packages
        estimator_name = pipeline['context']['intermediate_model'].get('pipeline_nodes', [None])[-1]
        if estimator_name in xgboost_estimators:
            check_dependencies_versions(request_json, wml_client, 'xgboost')

        elif estimator_name in lightgbm_estimators:
            check_dependencies_versions(request_json, wml_client, 'lightgbm')

        else:
            check_dependencies_versions(request_json, wml_client, None)

        # TODO: When another package estimators will be available update above!
        # --- end note

    joblib = try_import_joblib()

    path = os.path.abspath(path)
    pipelines_names = []
    pipelines = {}

    if wml_client.ICP:
        model_paths = []

        # note: iterate over all computed pipelines
        for pipeline in run_params['entity']['status'].get('metrics', []):

            # note: fetch and create model paths from file system
            model_path = pipeline['context']['intermediate_model']['location']['model']
            # --- end note

            # note: populate available pipeline names
            if pipeline_name is None:  # checking all pipelines
                model_paths.append(model_path)
                pipelines_names.append(
                    f"Pipeline_{pipeline['context']['intermediate_model']['name'].split('P')[-1]}")

                # note: check dependencies for estimators
                request_json = download_request_json(run_params, pipelines_names[-1], wml_client)
                check_pipeline_nodes(pipeline, request_json, wml_client)

            # checking only chosen pipeline
            elif pipeline_name == f"Pipeline_{pipeline['context']['intermediate_model']['name'].split('P')[-1]}":
                model_paths.append(model_path)
                pipelines_names = [f"Pipeline_{pipeline['context']['intermediate_model']['name'].split('P')[-1]}"]

                # note: check dependencies for estimators and other packages
                request_json = download_request_json(run_params, pipelines_names[-1], wml_client)
                check_pipeline_nodes(pipeline, request_json, wml_client)

                break
            # --- end note

        if load_pipelines:
            # Disable printing to suppress warning from ai4ml
            with redirect_stdout(open(os.devnull, "w")):
                for model_path, pipeline_name in zip(model_paths, pipelines_names):
                    pipelines[pipeline_name] = joblib.load(load_file_from_file_system(wml_client=wml_client,
                                                                                      file_path=model_path))

        if store:
            for name, pipeline in pipelines.items():
                local_model_path = os.path.join(path, name)
                joblib.dump(pipeline, local_model_path)
                print(f"Selected pipeline stored under: {local_model_path}")

                # note: display download link to the model
                create_model_download_link(local_model_path)
                # --- end note

    else:
        from ibm_boto3 import client
        cos_client = client(
            service_name=run_params['entity']['results_reference']['type'],
            endpoint_url=run_params['entity']['results_reference']['connection']['endpoint_url'],
            aws_access_key_id=run_params['entity']['results_reference']['connection']['access_key_id'],
            aws_secret_access_key=run_params['entity']['results_reference']['connection']['secret_access_key']
        )
        buckets = []
        filenames = []
        keys = []

        for pipeline in run_params['entity']['status'].get('metrics', []):
            model_number = pipeline['context']['intermediate_model']['name'].split('P')[-1]
            model_phase = chose_model_output(model_number)

            if pipeline['context']['phase'] == model_phase:
                model_path = f"{pipeline['context']['intermediate_model']['location']['model']}"

                if pipeline_name is None:
                    buckets.append(run_params['entity']['results_reference']['location']['bucket'])
                    filenames.append(
                        f"{path}/Pipeline_{pipeline['context']['intermediate_model']['name'].split('P')[-1]}.pickle")
                    keys.append(model_path)
                    pipelines_names.append(
                        f"Pipeline_{pipeline['context']['intermediate_model']['name'].split('P')[-1]}")

                    # note: check dependencies for estimators and other packages
                    request_json = download_request_json(run_params, pipelines_names[-1], wml_client)
                    check_pipeline_nodes(pipeline, request_json, wml_client)

                elif pipeline_name == f"Pipeline_{pipeline['context']['intermediate_model']['name'].split('P')[-1]}":
                    buckets = [run_params['entity']['results_reference']['location']['bucket']]
                    filenames = [
                        f"{path}/Pipeline_{pipeline['context']['intermediate_model']['name'].split('P')[-1]}.pickle"]
                    keys = [model_path]
                    pipelines_names = [f"Pipeline_{pipeline['context']['intermediate_model']['name'].split('P')[-1]}"]

                    # note: check dependencies for estimators and other packages
                    request_json = download_request_json(run_params, pipelines_names[-1], wml_client)
                    check_pipeline_nodes(pipeline, request_json, wml_client)

                    break

        for bucket, filename, key, name in zip(buckets, filenames, keys, pipelines_names):
            cos_client.download_file(Bucket=bucket, Filename=filename, Key=key)
            if load_pipelines:

                # Disable printing to suppress warning from ai4ml
                with redirect_stdout(open(os.devnull, "w")):
                    pipelines[name] = joblib.load(filename)

                if not store:
                    if os.path.exists(filename):
                        os.remove(filename)

                else:
                    print(f"Selected pipeline stored under: {filename}")

                    # note: display download link to the model
                    create_model_download_link(filename)
                    # --- end note

    if load_pipelines and pipelines:
        return pipelines

    elif load_pipelines:
        raise MissingPipeline(
            pipeline_name if pipeline_name is not None else "global_output pipeline",
            reason="The name of the pipeline is incorrect or there are no pipelines computed.")


def load_file_from_file_system(wml_client: 'APIClient',
                               file_path: str,
                               stream: bool = True) -> 'io.BytesIO':
    """
    Load file into memory from the file system.

    Parameters
    ----------
    wml_client: APIClient, required
        WML v4 client.

    file_path: str, required
        Path in the file system of the file.

    stream: bool, optional
        Indicator to stream data content.

    Returns
    -------
    Sklearn Pipeline
    """
    # note: prepare the file path
    file_path = file_path.split('auto_ml/')[-1]

    if wml_client.default_project_id:
        file_path = f"{file_path}?project_id={wml_client.default_project_id}"

    else:
        file_path = f"{file_path}?space_id={wml_client.default_space_id}"
    # --- end note

    buffer = io.BytesIO()
    response_with_model = requests.get(
        url=f"{wml_client.data_assets._href_definitions.get_wsd_model_attachment_href()}auto_ml/{file_path}",
        headers=wml_client._get_headers(),
        stream=stream,
        verify=False)
    if stream:
        for data in response_with_model.iter_content():
            buffer.write(data)
    else:
        buffer.write(response_with_model.content)

    buffer.seek(0)

    return buffer


def load_file_from_file_system_nonautoai(wml_client: 'APIClient',
                               file_path: str,
                               stream: bool = True) -> 'io.BytesIO':
    """
    Load file into memory from the file system.

    Parameters
    ----------
    wml_client: APIClient, required
        WML v4 client.

    file_path: str, required
        Path in the file system of the file.

    stream: bool, optional
        Indicator to stream data content.

    Returns
    -------
        File content
    """
    # note: prepare the file path

    if wml_client.default_project_id:
        file_path = f"{file_path}?project_id={wml_client.default_project_id}"

    else:
        file_path = f"{file_path}?space_id={wml_client.default_space_id}"
    # --- end note

    buffer = io.BytesIO()

    response_with_model = requests.get(
        url=f"{wml_client.data_assets._href_definitions.get_wsd_model_attachment_href()}{file_path}",
        headers=wml_client._get_headers(),
        stream=stream,
        verify=False)

    if stream:
        for data in response_with_model.iter_content():
            buffer.write(data)
    else:
        buffer.write(response_with_model.content)

    buffer.seek(0)

    return buffer

class NextRunDetailsGenerator:
    """
    Generator class to produce next list of run details.

    Parameters
    ----------
    wml_client: APIClient, required
        WML Client Instance
    """

    def __init__(self, wml_client: 'APIClient', href: str) -> None:
        self.wml_client = wml_client
        self.next_href = href

    def __iter__(self):
        return self

    def __next__(self):
        if self.next_href is not None:
            response = requests.get(
                url=f"{self.wml_client.wml_credentials['url']}{self.next_href}",
                headers=self.wml_client._get_headers(),
                verify=not self.wml_client.ICP)
            details = response.json()
            self.next_href = details.get('next', {'href': None})['href']
            return details.get('resources', [])

        else:
            raise StopIteration


def preprocess_request_json(request_json: Dict, space_id: str) -> Dict:
    """Removes unused parts of request.json file got from autoai training.
    Allow to further store model in user space."""
    # note: if training was on project_id, change it to space_id as we can deploy only on space
    if 'project_id' in request_json:
        request_json.pop('project_id')

    request_json['space_id'] = space_id
    request_json.pop('pipeline')  # not needed for other space
    request_json.pop('training_data_references')  # not needed for other space
    # --- end note
    return request_json


def chose_model_output(model_number: str) -> str:
    """Chose correct path for particular model number"""
    model_number = int(model_number)
    hpo_c_numbers = (4, 8, 12, 16)
    cognito_numbers = (3, 7, 11, 15)
    hpo_d_numbers = (2, 6, 10, 14)
    pre_hpo_d_numbers = (1, 5, 9, 13)

    if model_number in pre_hpo_d_numbers:
        return 'pre_hpo_d_output'

    elif model_number in hpo_d_numbers:
        return 'hpo_d_output'

    elif model_number in cognito_numbers:
        return 'cognito_output'

    elif model_number in hpo_c_numbers:
        return 'hpo_c_output'

    else:
        return 'global_output'


def prepare_auto_ai_model_to_publish_notebook_normal_scenario(
        pipeline_model: Union['Pipeline', 'TrainablePipeline'],
        result_connection,
        cos_client,
        run_params: Dict,
        space_id: str) -> Union[Tuple[str, Dict[str, dict]]]:
    """
    Prepares autoai model to publish in Watson Studio via COS.
    Option only for auto-gen notebooks with correct result references on COS.

    Parameters
    ----------
    pipeline_model: Union['Pipeline', 'TrainablePipeline'], required
        model object to publish

    result_connection: DataConnection, required
        Connection object with COS credentials and all needed locations for jsons

    cos_client: ibm_boto3.resource, required
        initialized COS client

    run_params: dictionary, required
        Dictionary with training details

    space_id: str, required

    Returns
    -------
    String with path to the saved model and jsons in COS.
    """
    path = result_connection.location._model_location
    model_number = pipeline_model.split('_')[-1]
    run_id = path.split('/data/')[0].split('/')[-1]
    request_path = f"{path.split('/data/')[0]}/assets/{run_id}_P{model_number}_{chose_model_output(model_number)}/resources/wml_model/request.json"

    bucket = result_connection.location.bucket
    cos_client.meta.client.download_file(Bucket=bucket, Filename='request.json', Key=request_path)
    with open('request.json', 'r') as f:
        request_str = f.read()

    # note: only if there was 1 estimator during training
    if 'content_location' not in request_str:
        request_path = f"{path.split('/data/')[0]}/assets/{run_id}_P{model_number}_compose_model_type_output/resources/wml_model/request.json"
        cos_client.meta.client.download_file(Bucket=bucket, Filename='request.json', Key=request_path)
        with open('request.json', 'r') as f:
            request_str = f.read()

    request_json: Dict[str, dict] = json.loads(request_str)
    request_json['content_location']['connection'] = run_params['entity']['results_reference']['connection']
    request_json = preprocess_request_json(request_json, space_id)
    artifact_name = f"autoai_sdk{os.path.sep}{pipeline_model}.pickle"

    return artifact_name, request_json


# TODO: remove this function
def prepare_auto_ai_model_to_publish_notebook(pipeline_model: Union['Pipeline', 'TrainablePipeline'],
                                              result_connection,
                                              cos_client,
                                              obm: Optional[bool] = False) -> Union[Tuple[Dict[str, dict], str], str]:
    """
    Prepares autoai model to publish in Watson Studio via COS.
    Option only for auto-gen notebooks with correct result references on COS.

    Parameters
    ----------
    pipeline_model: Union['Pipeline', 'TrainablePipeline'], required
        model object to publish

    result_connection: DataConnection, required
        Connection object with COS credentials and all needed locations for jsons

    cos_client: ibm_boto3.resource, required
        initialized COS client

    obm: bool, optional
        Indicator if we need to extract OBM data

    Returns
    -------
    String with path to the saved model and jsons in COS.
    """
    joblib = try_import_joblib()

    artifact_type = ".gzip"

    artifact_name = f"artifact_auto_ai_model{artifact_type}"
    model_artifact_name = f"model_.tar.gz"
    wml_pipeline_definition_name = "pipeline-model.json"
    obm_model_name = "obm_model.zip"
    temp_model_name = '__temp_model.pickle'

    # note: path to the json describing the autoai POD specification
    path = result_connection.location._model_location.split('model.pickle')[0]
    pipeline_model_json_path = f"{path}pipeline-model.json"
    schema_path = f"{path}schema.json"

    bucket = result_connection.location.bucket

    # note: Check if we have OBM experiment and get paths for obm model and schema
    if obm:
        obm_model_path = f"{path.split('/data/')[0]}/data/obm/model.zip"
        schema_path = f"{path.split('/data/')[0]}/data/obm/schemas.json"
        cos_client.meta.client.download_file(Bucket=bucket, Filename=obm_model_name, Key=obm_model_path)

    # note: need to download model schema and wml pipeline definition json
    cos_client.meta.client.download_file(Bucket=bucket, Filename=wml_pipeline_definition_name,
                                         Key=pipeline_model_json_path)
    cos_client.meta.client.download_file(Bucket=bucket, Filename='schema.json', Key=schema_path)

    with open('schema.json', 'r') as f:
        schema_json = f.read()

    # note: update the schema, it has wrong field types
    schema_json = schema_json.replace('fieldType', 'type')
    # --- end note

    # note: saved passed model as pickle, for further tar.gz packaging
    joblib.dump(pipeline_model, temp_model_name)
    # --- end note

    # note: create a tar.gz file with model pickle, name it as 'model_run_id.tar.gz', model.pickle inside
    with open_tar(model_artifact_name, 'w:gz') as tar:
        tar.add(temp_model_name, arcname='model.pickle')

    remove_file(filename=temp_model_name)
    # --- end note

    # note: create final zip to publish on WML cloud v4 GA
    with ZipFile(artifact_name, 'w') as zip_file:
        if obm:
            # note: write order is important!
            zip_file.write(obm_model_name)
        zip_file.write(model_artifact_name)
        zip_file.write(wml_pipeline_definition_name)

    remove_file(filename=model_artifact_name)
    remove_file(filename=wml_pipeline_definition_name)
    if obm:
        remove_file(filename=obm_model_name)
    # --- end note

    return json.loads(schema_json), artifact_name


def prepare_auto_ai_model_to_publish_normal_scenario(
        pipeline_model: Union['Pipeline', 'TrainablePipeline'],
        run_params: dict,
        run_id: str,
        wml_client: 'APIClient',
        space_id: str) -> Union[Tuple[str, Dict[str, dict]]]:
    """
    Helper function to specify `content_location` statement for AutoAI models to store in repository.

    Parameters
    ----------
    pipeline_model: Union['Pipeline', 'TrainablePipeline'], required
        Model that will be prepared for an upload.

    run_params: dict, required
        Fetched details of the run/fit.

    run_id: str, required
        Fit/run ID associated with the model.

    wml_client: APIClient, required

    space_id: str, required

    Returns
    -------
    If cp4d: Dictionary with model schema and artifact name to upload, stored temporally in the user local file system.
    else: path name to the stored model in COS
    """

    request_json: Dict[str, dict] = download_request_json(run_params, pipeline_model, wml_client)
    # note: fill connection details
    request_json['content_location']['connection'] = run_params['entity']['results_reference']['connection']
    # note: if training was on project_id, change it to space_id as we can deploy only on space
    request_json = preprocess_request_json(request_json, space_id)
    artifact_name = f"autoai_sdk{os.path.sep}{pipeline_model}.pickle"

    return artifact_name, request_json


# TODO: remove this function
def prepare_auto_ai_model_to_publish(
        pipeline_model: Union['Pipeline', 'TrainablePipeline'],
        run_params: dict,
        run_id: str,
        wml_client: 'APIClient') -> Union[Tuple[Dict[str, dict], str], str]:
    """
    Helper function to download and load computed AutoAI pipelines (sklearn pipelines).
    Parameters
    ----------
    pipeline_model: Union['Pipeline', 'TrainablePipeline'], required
        Model that will be prepared for an upload.
    run_params: dict, required
        Fetched details of the run/fit.
    run_id: str, required
        Fit/run ID associated with the model.
    wml_client: APIClient, required
    Returns
    -------
    If cp4d: Dictionary with model schema and artifact name to upload, stored temporally in the user local file system.
    else: path name to the stored model in COS
    """

    joblib = try_import_joblib()

    artifact_type = ".tar.gz" if wml_client.ICP else ".gzip"

    artifact_name = f"artifact_auto_ai_model{artifact_type}"
    model_artifact_name = f"model_{run_id}.tar.gz"
    wml_pipeline_definition_name = "pipeline-model.json"
    obm_model_name = "obm_model.zip"
    temp_model_name = '__temp_model.pickle'

    # note: prepare file paths of pipeline-model and schema (COS / file system location)
    pipeline_info = run_params['entity']['status'].get('metrics')[-1]
    pipeline_model_path = f"{pipeline_info['context']['intermediate_model']['location']['pipeline_model']}"
    schema_path = f"{pipeline_info['context']['intermediate_model']['schema_location']}"
    obm_model_path = None
    # --- end note

    # note: Check if we have OBM experiment and get paths for obm model and schema
    if 'obm' in run_params['entity']['status'].get('feature_engineering_components', {}):
        obm_model_path = f"{pipeline_model_path.split('/data/')[0]}/data/obm/model.zip"
        schema_path = f"{pipeline_model_path.split('/data/')[0]}/data/obm/schemas.json"

    if wml_client.ICP:
        # note: downloading pipeline-model.json and schema.json from file system on CP4D
        schema_json = load_file_from_file_system(wml_client=wml_client, file_path=schema_path).read().decode()
        pipeline_model_json = load_file_from_file_system(wml_client=wml_client,
                                                         file_path=pipeline_model_path).read().decode()
        with open(wml_pipeline_definition_name, 'w') as f:
            f.write(pipeline_model_json)
        # --- end note

        # note: save obm model.zip locally
        if obm_model_path is not None:
            obm_model = load_file_from_file_system(wml_client=wml_client,
                                                   file_path=obm_model_path).read().decode()

            with open(obm_model_name, 'w') as f:
                f.write(obm_model)
        # --- end note

    else:
        cos_client = init_cos_client(run_params['entity']['results_reference']['connection'])
        bucket = run_params['entity']['results_reference']['location']['bucket']

        # note: need to download model schema and wml pipeline definition json
        cos_client.meta.client.download_file(Bucket=bucket, Filename=wml_pipeline_definition_name,
                                             Key=pipeline_model_path)
        cos_client.meta.client.download_file(Bucket=bucket, Filename='schema.json', Key=schema_path)

        with open('schema.json', 'r') as f:
            schema_json = f.read()

        # note: save obm model.zip locally
        if obm_model_path is not None:
            cos_client.meta.client.download_file(Bucket=bucket, Filename=obm_model_name, Key=obm_model_path)
        # --- end note

    # note: update the schema, it has wrong field types and missing id
    schema_json = schema_json.replace('fieldType', 'type')
    # --- end note

    # note: saved passed model as pickle, for further tar.gz packaging
    joblib.dump(pipeline_model, temp_model_name)
    # --- end note

    # note: create a tar.gz file with model pickle, name it as 'model_run_id.tar.gz', model.pickle inside
    with open_tar(model_artifact_name, 'w:gz') as tar:
        tar.add(temp_model_name, arcname='model.pickle')

    remove_file(filename=temp_model_name)
    # --- end note

    with ZipFile(artifact_name, 'w') as zip_file:
        if obm_model_path is not None:
            # note: write order is important!
            zip_file.write(obm_model_name)
        zip_file.write(model_artifact_name)
        zip_file.write(wml_pipeline_definition_name)

    remove_file(filename=model_artifact_name)
    remove_file(filename=wml_pipeline_definition_name)
    if obm_model_path is not None:
        remove_file(filename=obm_model_name)
    # --- end note

    return json.loads(schema_json), artifact_name


def modify_pipeline_model_json(data_location: str, model_path: str) -> None:
    """
    Change the location of KB model in pipeline-model.json

    Parameters
    ----------
    data_location: str, required
        pipeline-model.json data local path

    model_path: str, required
        Path to KB model stored in COS.
    """
    with open(data_location, 'r') as f:
        data = json.load(f)

    data['pipelines'][0]['nodes'][-1]['parameters']['output_model']['location'] = f"{model_path}model.pickle"

    with open(data_location, 'w') as f:
        f.write(json.dumps(data))


def init_cos_client(connection: dict) -> 'resource':
    """Initiate COS client for further usage."""
    from ibm_botocore.client import Config
    from ibm_boto3 import resource

    if connection.get('auth_endpoint') is not None and connection.get('api_key') is not None:
        cos_client = resource(
            service_name='s3',
            ibm_api_key_id=connection['api_key'],
            ibm_auth_endpoint=connection['auth_endpoint'],
            config=Config(signature_version="oauth"),
            endpoint_url=connection['endpoint_url']
        )

    else:
        cos_client = resource(
            service_name='s3',
            endpoint_url=connection['endpoint_url'],
            aws_access_key_id=connection['access_key_id'],
            aws_secret_access_key=connection['secret_access_key']
        )
    return cos_client


def remove_file(filename: str):
    """Helper function to clean user local storage from temporary package files."""
    if os.path.exists(filename):
        os.remove(filename)


class ProgressGenerator:
    def __init__(self):
        self.progress_messages = {
            "pre_hpo_d_output": 15,
            "hpo_d_output": 30,
            "cognito_output": 50,
            "hpo_c_output": 70,
            "compose_model_type_output": 80,
            "fold_output": 90,
            "global_output": 99
        }
        self.total = 100
        self.position = 0
        self.max_position = 5

    def get_progress(self, text):
        for i, e in enumerate(self.progress_messages):
            if e in text:
                pos = self.max_position
                self.max_position = max(self.max_position, self.progress_messages[e])
                if pos < self.max_position:
                    progress = pos - self.position
                    self.position = pos
                    return progress

        if self.position + 1 >= self.max_position:
            return 0
        else:
            self.position += 1
            return 1

    def get_total(self):
        return self.total


def is_ipython():
    """Check if code is running in the notebook."""
    try:
        name = get_ipython().__class__.__name__
        if name != 'ZMQInteractiveShell':
            return False
        else:
            return True

    except Exception:
        return False


def try_import_lale():
    """
    Check if lale package is installed in local environment, if not, just download and install it.
    """
    lale_version = '0.4.12'
    try:
        import lale
        from packaging import version
        if version.parse(lale.__version__) < version.parse(lale_version):
            warn(f"\"lale\" package version is to low."
                 f"Installing version >={lale_version}")

            try:
                check_call([executable, "-m", "pip", "install", f"lale>={lale_version}"])

            except Exception as e:
                raise CannotInstallLibrary(value_name=e,
                                           reason="lale failed to install. Please install it manually.")

    except AttributeError:
        warn(f"Cannot determine \"lale\" package version."
             f"Installing version >={lale_version}")

        try:
            check_call([executable, "-m", "pip", "install", f"lale>={lale_version}"])

        except Exception as e:
            raise CannotInstallLibrary(value_name=e,
                                       reason="lale failed to install. Please install it manually.")

    except ImportError:
        warn(f"\"lale\" package is not installed. "
             f"This is the needed dependency for pipeline model refinery, we will try to install it now...")

        try:
            check_call([executable, "-m", "pip", "install", f"lale>={lale_version}"])

        except Exception as e:
            raise CannotInstallLibrary(value_name=e,
                                       reason="lale failed to install. Please install it manually.")


def try_import_autoai_libs():
    """
    Check if autoai_libs package is installed in local environment, if not, just download and install it.
    """
    try:
        import autoai_libs

    except ImportError:
        warn(f"\"autoai_libs\" package is not installed. "
             f"This is the needed dependency for pipeline model refinery, we will try to install it now...")

        try:
            check_call([executable, "-m", "pip", "install", "autoai_libs>=1.11.0"])

        except Exception as e:
            raise CannotInstallLibrary(value_name=e,
                                       reason="autoai_libs>=1.11.0 failed to install. Please install it manually.")


def try_import_tqdm():
    """
    Check if tqdm package is installed in local environment, if not, just download and install it.
    """
    try:
        import tqdm

    except ImportError:
        warn(f"\"tqdm\" package is not installed. "
             f"This is the needed dependency for pipeline training, we will try to install it now...")

        try:
            check_call([executable, "-m", "pip", "install", "tqdm==4.43.0"])

        except Exception as e:
            raise CannotInstallLibrary(value_name=e,
                                       reason="tqdm==4.43.0 failed to install. Please install it manually.")


def try_import_xlrd():
    """
    Check if xlrd package is installed in local environment, if not, just download and install it.
    """
    try:
        import xlrd

    except ImportError:
        warn(f"\"xlrd\" package is not installed. "
             f"This is the needed dependency for loading dataset from xls files, we will try to install it now...")

        try:
            check_call([executable, "-m", "pip", "install", "xlrd==1.2.0"])

        except Exception as e:
            raise CannotInstallLibrary(value_name=e,
                                       reason="xlrd==1.2.0 failed to install. Please install it manually.")


def try_import_graphviz():
    """
    Check if graphviz package is installed in local environment, if not, just download and install it.
    """
    try:
        import graphviz

    except ImportError:
        warn(f"\"graphviz\" package is not installed. "
             f"This is the needed dependency for visualizing data join graph, we will try to install it now...")

        try:
            check_call([executable, "-m", "pip", "install", "graphviz==0.14"])

        except Exception as e:
            raise CannotInstallLibrary(value_name=e,
                                       reason="graphviz==0.14 failed to install. Please install it manually.")


def try_import_joblib():
    """
    Check if joblib is available from scikit-learn or externally and change 'load' method to inform the user about
    compatibility issues.
    """

    try:
        # note only up to scikit version 0.20.3
        from sklearn.externals import joblib

    except ImportError:
        # only for scikit 0.23.*
        import joblib

    return joblib


def try_load_dataset(
        buffer: Union['BytesIO', 'BufferedIOBase'],
        sheet_name: str = 0,
        separator: str = ',',
        encoding: Optional[str] = 'utf-8') -> Union['DataFrame', 'OrderedDict']:
    """
    Load data into a pandas DataFrame from BytesIO object.

    Parameters
    ----------
    buffer: Union['BytesIO', 'BufferedIOBase'], required
        Buffer with bytes data.

    sheet_name: str, optional
        Name of the xlsx sheet to read.

    separator: str, optional
        csv separator

    encoding: str, optional

    Returns
    -------
    DataFrame or OrderedDict
    """
    from pandas import read_csv, read_excel

    try:
        buffer.seek(0)
        data = read_csv(buffer, sep=separator, encoding=encoding)

    except Exception as e1:
        try:
            try_import_xlrd()
            buffer.seek(0)
            data = read_excel(buffer, sheet_name=sheet_name)

        except Exception as e2:
            raise DataFormatNotSupported(None, reason=f"Error1: {e1} Error2: {e2}")

    return data


def check_dependencies_versions(request_json: dict, wml_client, estimator_pkg: str) -> None:
    """
    Check packages installed versions and inform the user about needed ones.

    Parameters
    ----------
    request_json: dict, required
        Dictionary with request from training saved on user COS or CP4D fs.

    wml_client: APIClient, required
        Internal WML client used for sw spec requests.

    estimator_pkg: str, required
        Name of the estimator package to check with.
    """
    sw_spec_name = request_json.get('hybrid_pipeline_software_specs', [{'name': None}])[-1]['name']
    sw_spec_id = wml_client.software_specifications.get_id_by_name(sw_spec_name)
    sw_spec = wml_client.software_specifications.get_details(sw_spec_id)

    packages = sw_spec['entity']['software_specification']['software_configuration']['included_packages']
    packages_to_check = ['numpy', 'scikit-learn', 'autoai-libs', 'gensim', 'lale']

    if estimator_pkg is not None:
        packages_to_check.append(estimator_pkg)
        packages_to_check.append(f'py-{estimator_pkg}')

    errored_packages = []

    for package in packages:
        if package['name'] in packages_to_check:
            try:
                installed_module_version = pkg_resources.get_distribution(package['name']).version

                # workaround for autai-libs and numpy versions in SW spec
                if package['name'] == 'autoai-libs' or package['name'] == 'numpy':
                    if version.parse(installed_module_version) < version.parse(package['version']):
                        errored_packages.append(package)

                else:
                    if installed_module_version != package['version']:
                        errored_packages.append(package)

            except pkg_resources.DistributionNotFound as e:
                errored_packages.append(package)

        else:
            pass

    if errored_packages:
        raise LibraryNotCompatible(reason=f"Please check if you have installed correct versions "
                                          f"of the following packages: {errored_packages} "
                                          f"These packages are required to load ML model successfully "
                                          f"on your environment.")


def prepare_cos_client(
        training_data_references: List['DataConnection'] = None,
        training_result_reference: 'DataConnection' = None) -> Tuple[Union[List[Tuple['DataConnection', 'resource']]],
                                                                     Union[Tuple['DataConnection', 'resource'], None]]:
    """
    Create COS clients for training data and results.

    Parameters
    ----------
    training_data_references: List['DataConnection'], optional

    training_result_reference: 'DataConnection', optional

    Returns
    -------
    list of COS clients for training data , client for results
    """
    from ibm_watson_machine_learning.helpers import S3Connection
    from ibm_boto3 import resource
    from ibm_botocore.client import Config

    def differentiate_between_credentials(connection: 'S3Connection') -> 'resource':
        # note: we do not know which version of COS credentials user used during training
        if hasattr(connection, 'auth_endpoint') and hasattr(connection, 'api_key'):
            cos_client = resource(
                service_name='s3',
                ibm_api_key_id=connection.api_key,
                ibm_auth_endpoint=connection.auth_endpoint,
                config=Config(signature_version="oauth"),
                endpoint_url=connection.endpoint_url
            )

        else:
            cos_client = resource(
                service_name='s3',
                endpoint_url=connection.endpoint_url,
                aws_access_key_id=connection.access_key_id,
                aws_secret_access_key=connection.secret_access_key
            )
        # --- end note

        return cos_client

    cos_client_results = None
    data_cos_clients = []

    if training_result_reference is not None:
        if isinstance(training_result_reference.connection, S3Connection):
            cos_client_results = (training_result_reference,
                                  differentiate_between_credentials(connection=training_result_reference.connection))

    if training_data_references is not None:
        for reference in training_data_references:
            if isinstance(reference.connection, S3Connection):
                data_cos_clients.append((reference,
                                         differentiate_between_credentials(connection=reference.connection)))

    return data_cos_clients, cos_client_results


def create_summary(details: dict, scoring: str) -> 'DataFrame':
    """
    Creates summary in a form of a pandas.DataFrame of computed pipelines (should be used in remote and local scenario
    with COS).

    Parameters
    ----------
    details: dict, required
        Dictionary with all training data

    scoring: str, required
        scoring method

    Returns
    -------
    pandas.DataFrame with pipelines summary
    """
    from pandas import DataFrame

    columns = (['Pipeline Name', 'Number of enhancements', 'Estimator'] +
               [metric_name for metric_name in
                details['entity']['status'].get('metrics', [{}])[0].get('ml_metrics', {}).keys()])
    values = []

    for pipeline in details['entity']['status'].get('metrics', []):
        model_number = pipeline['context']['intermediate_model']['name'].split('P')[-1]
        model_phase = chose_model_output(model_number)

        if pipeline['context']['phase'] == model_phase:
            number_of_enhancements = len(pipeline['context']['intermediate_model']['composition_steps']) - 5

            # note: workaround when some pipelines have less or more metrics computed
            metrics = columns[3:]
            pipeline_metrics = [None] * len(metrics)
            for metric, value in pipeline['ml_metrics'].items():
                for i, metric_name in enumerate(metrics):
                    if metric_name == metric:
                        pipeline_metrics[i] = value
            # --- end note

            values.append(
                ([f"Pipeline_{pipeline['context']['intermediate_model']['name'].split('P')[-1]}"] +
                 [number_of_enhancements] +
                 [pipeline['context']['intermediate_model']['pipeline_nodes'][-1]] +
                 pipeline_metrics
                 ))

    pipelines = DataFrame(data=values, columns=columns)
    pipelines.drop_duplicates(subset="Pipeline Name", keep='first', inplace=True)
    pipelines.set_index('Pipeline Name', inplace=True)

    try:
        pipelines = pipelines.sort_values(
            by=[f"training_{scoring}"], ascending=False).rename(
            {
                f"training_{scoring}":
                    f"training_{scoring}_(optimized)"
            }, axis='columns')

    # note: sometimes backend will not return 'training_' prefix to the metric
    except KeyError:
        pass

    neg_columns = [col for col in pipelines if '_neg_' in col]
    pipelines[neg_columns] = -pipelines[neg_columns]
    pipelines = pipelines.rename(columns={col: col.replace('_neg_', '_') for col in neg_columns})

    return pipelines


def get_node_and_runtime_index(node_name: str, optimizer_config: dict) -> Tuple[int, int]:
    """Find node index from node name in experiment parameters."""
    node_number = None
    runtime_number = None

    for i, node in enumerate(optimizer_config['entity']['document']['pipelines'][0]['nodes']):
        if node_name == 'kb' and (node.get('id') == 'kb' or node.get('id') == 'automl'):
            node_number = i
            break

        elif node_name == 'obm' and node.get('id') == 'obm':
            node_number = i
            break

    for i, runtime in enumerate(optimizer_config['entity']['document']['runtimes']):
        if node_name == 'kb' and (runtime.get('id') == 'kb' or runtime.get('id') == 'automl' or
                                  runtime.get('id') == 'autoai'):
            runtime_number = i
            break

        elif node_name == 'obm' and runtime.get('id') == 'obm':
            runtime_number = i
            break

    return node_number, runtime_number


def download_experiment_details_from_file(result_client_and_connection: Tuple['DataConnection', 'resource']) -> dict:
    """Try to download training details from user COS."""

    try:
        file = result_client_and_connection[1].Object(
            result_client_and_connection[0].location.bucket,
            result_client_and_connection[0].location._training_status).get()

        details = json.loads(file['Body'].read())

    except Exception as e:
        raise CannotDownloadTrainingDetails('', reason=f"Error: {e}")

    return details


def download_wml_pipeline_details_from_file(result_client_and_connection: Tuple['DataConnection', 'resource']) -> dict:
    """Try to download wml pipeline details from user COS."""

    try:
        path = result_client_and_connection[0].location._model_location.split('model.pickle')[0]
        path = f"{path}pipeline-model.json"

        file = result_client_and_connection[1].Object(
            result_client_and_connection[0].location.bucket,
            path).get()

        details = json.loads(file['Body'].read())

    except Exception as e:
        raise CannotDownloadWMLPipelineDetails('', reason=f"Error: {e}")

    return details


def prepare_model_location_path(model_path: str) -> str:
    """
    To be able to get best pipeline after computation we need to change model_location string to global_output.
    """

    if "data/automl/" in model_path:
        path = model_path.split('data/automl/')[0]
        path = f"{path}data/automl/global_output/"

    else:
        path = model_path.split('data/kb/')[0]
        path = f"{path}data/kb/global_output/"

    return path


def check_graphviz_binaries(f):
    @wraps(f)
    def _f(*method_args, **method_kwargs):
        from graphviz.backend import ExecutableNotFound
        try:
            output = f(*method_args, **method_kwargs)

        except ExecutableNotFound as e:
            raise VisualizationFailed(
                reason=f"Cannot perform visualization with graphviz. Please make sure that you have Graphviz binaries "
                       f"installed in your system. Please follow this guide: https://www.graphviz.org/download/")

        return output

    return _f


def get_sw_spec_and_type_based_on_sklearn(client: 'APIClient', spec: str) -> Tuple[str, str]:
    """Based on user environment and pipeline sw spec, check sklearn version and find apropriate sw spec.

    Returns
    -------
    model_type, sw_spec
    """
    import sklearn

    if '0.20.' in sklearn.__version__ and 'autoai-kb_3.0-py3.6' == spec:
        sw_spec = client.software_specifications.get_id_by_name('autoai-kb_3.0-py3.6')
        model_type = 'scikit-learn_0.20'

    elif '0.20.' in sklearn.__version__ and 'autoai-kb_3.0-py3.6' != spec:
        raise LibraryNotCompatible(reason="Your version of scikit-learn is different then trained pipeline. "
                                          "Trained pipeline version: 0.23.* "
                                          "Your version: " + sklearn.__version__)

    elif '0.23.' in sklearn.__version__ and 'autoai-kb_3.1-py3.7' == spec:
        sw_spec = client.software_specifications.get_id_by_name('autoai-kb_3.1-py3.7')
        model_type = 'scikit-learn_0.23'

    elif '0.23.' in sklearn.__version__ and 'autoai-kb_3.1-py3.7' != spec:
        raise LibraryNotCompatible(reason="Your version of scikit-learn is different then trained pipeline. "
                                          "Trained pipeline version: 0.20.* "
                                          "Your version: " + sklearn.__version__)

    else:
        raise LibraryNotCompatible(reason="Your version of scikit-learn is not supported. Use one of [0.20.*, 0.23.*]")

    return model_type, sw_spec


def validate_additional_params_for_optimizer(params):
    expected_params = [
        'learning_type', 'positive_label', 'scorer_for_ranking', 'scorers', 'num_folds', 'random_state',
        'preprocessor_flag', 'preprocessor_hpo_flag', 'preprocessor_hpo_estimator', 'hpo_searcher', 'cv_num_folds',
        'hpo_d_iter_threshold', 'hpo_c_iter_threshold', 'max_initial_points', 'preprocess_transformer_chain',
        'daub_ensembles_flag', 'max_num_daub_ensembles', 'run_hpo_after_daub_flag', 'daub_include_only_estimators',
        'run_cognito_flag', 'cognito_ensembles_flag', 'max_num_cognito_ensembles', 'cognito_display_flag',
        'run_hpo_after_cognito_flag', 'cognito_kwargs', 'cognito_scorers', 'daub_adaptive_subsampling_max_mem_usage',
        'daub_adaptive_subsampling_used_mem_ratio_threshold', 'daub_kwargs', 'compute_feature_importances_flag',
        'compute_feature_importances_options', 'compute_feature_importances_pipeline_options', 'show_status_flag',
        'status_msg_handler', 'state_max_report_priority', 'msg_max_report_priority', 'cognito_pass_ptype',
        'hpo_timeout_in_seconds', 'cognito_use_feature_importances_flag', 'cognito_max_iterations',
        'cognito_max_search_level', 'cognito_transform_names', 'cognito_use_grasspile', 'cognito_subsample',
        'holdout_param', 'missing_values_reference_list', 'datetime_processing_flag',
        'datetime_delete_source_columns', 'datetime_processing_options', 'ensemble_pipelines_flag', 'ensemble_tags',
        'ensemble_comb_method', 'ensemble_selection_flag', 'ensemble_weighted_flag', 'ensemble_corr_sel_method',
        'ensemble_corr_termination_diff_threshold', 'ensemble_num_best_pipelines', 'ensemble_num_folds',
        'compute_pipeline_notebooks_flag', 'pipeline_ranking_metric', 'cpus_available', 'wml_status_msg_version',
        'float32_processing_flag', 'train_remove_missing_target_rows_flag', 'train_sample_rows_test_size',
        'train_sample_columns_index_list', 'preprocessor_cat_imp_strategy', 'preprocessor_cat_enc_encoding',
        'preprocessor_num_imp_strategy', 'preprocessor_num_scaler_use_scaler_flag', 'preprocessor_num_scaler_with_mean',
        'preprocessor_num_scaler_with_std', 'preprocessor_string_compress_type', 'FE_drop_unique_columns_flag',
        'FE_drop_constant_columns_flag', 'FE_add_frequency_columns_flag', 'FE_add_missing_indicator_columns_flag',
        'data_provenance', 'target_label_name', 'preprocessor_data_filename', 'cognito_data_filename',
        'holdout_roc_curve_max_size', 'holdout_reg_pred_obs_max_size', 'max_estimator_n_jobs',
        'enabled_feature_engineering_as_json', 'fairness_info']

    for k in params:
        if k not in expected_params:
            raise AdditionalParameterIsUnexpected(k)


def download_request_json(run_params: dict, model_name: str, wml_client) -> dict:
    run_id = run_params['metadata']['id']
    pipeline_info = run_params['entity']['status'].get('metrics')[-1]
    schema_path = f"{pipeline_info['context']['intermediate_model']['schema_location']}"
    model_number = model_name.split('_')[-1]
    request_path = f"{schema_path.split('/data/')[0]}/assets/{run_id}_P{model_number}_{chose_model_output(model_number)}/resources/wml_model/request.json"

    if wml_client.ICP:
        request_str = load_file_from_file_system(wml_client=wml_client, file_path=request_path).read().decode()
        # note: only if there was 1 estimator during training
        if 'content_location' not in request_str:
            request_path = f"{schema_path.split('/data/')[0]}/assets/{run_id}_P{model_number}_compose_model_type_output/resources/wml_model/request.json"
            request_str = load_file_from_file_system(wml_client=wml_client, file_path=request_path).read().decode()

    else:
        cos_client = init_cos_client(run_params['entity']['results_reference']['connection'])
        bucket = run_params['entity']['results_reference']['location']['bucket']
        cos_client.meta.client.download_file(Bucket=bucket, Filename='request.json', Key=request_path)
        with open('request.json', 'r') as f:
            request_str = f.read()

        # note: only if there was 1 estimator during training
        if 'content_location' not in request_str:
            request_path = f"{schema_path.split('/data/')[0]}/assets/{run_id}_P{model_number}_compose_model_type_output/resources/wml_model/request.json"
            cos_client.meta.client.download_file(Bucket=bucket, Filename='request.json', Key=request_path)
            with open('request.json', 'r') as f:
                request_str = f.read()

    request_json: Dict[str, dict] = json.loads(request_str)

    return request_json


