import base64
import logging
import os
import pprint
import re
import tempfile
from collections import namedtuple
from datetime import datetime
from time import sleep

import requests
from ibm_watson_machine_learning import APIClient


class Error(Exception):
    """Base class for all errors in this script"""
    pass


class InvalidCredentials(Error):
    """The WML credentials were not found, or incorrect"""
    pass


class SimilarNamesInJob(Error):
    """A job can't have two input files with the same name, irrespective of path"""
    pass


class ConnectionIdNotFound(Error):
    """Using Cloud Object Storage requires a connection, but none was found"""
    pass


#
# We patch APIClient._params with this function instead
#
the_filter = None
the_old_params = None


def new_params():
    global the_old_params
    global the_filter
    result = the_old_params()
    if the_filter:
        # Beware: the parameter list must not have spaces!
        result['include'] = the_filter
    return result


class DOWMLLib:
    """A Python client to run DO models on WML"""

    ENVIRONMENT_VARIABLE_NAME = 'DOWML_CREDENTIALS'
    SPACE_NAME = 'DOWMLClient-space'
    MODEL_NAME = 'DOWMLClient-model'
    MODEL_TYPES = ['cplex', 'cpo', 'opl', 'docplex']
    TSHIRT_SIZES = ['S', 'M', 'XL']
    DEPLOYMENT_NAME = 'DOWMLClient-deployment'
    JOB_END_SLEEP_DELAY = 2

    def __init__(self, wml_credentials_file=None):
        """Read and validate the WML credentials

        Args:
            wml_credentials_file: path to the file that contains the WML credentials.
            If None, they are read from the environment."""
        self._logger = logging.getLogger(self.__class__.__name__)

        if wml_credentials_file is not None:
            wml_cred_str = self._read_wml_credentials_from_file(wml_credentials_file)
        else:
            wml_cred_str = self._read_wml_credentials_from_env(self.ENVIRONMENT_VARIABLE_NAME)
        self._logger.debug(f'Found credential string.')

        wml_credentials = eval(wml_cred_str)
        assert type(wml_credentials) is dict
        assert 'apikey' in wml_credentials
        assert type(wml_credentials['apikey']) is str
        assert 'url' in wml_credentials
        assert type(wml_credentials['url']) is str
        # This string is the 'resource_instance_id' in the cos_credentials file
        assert 'cos_resource_crn' in wml_credentials
        assert type(wml_credentials['cos_resource_crn']) is str
        # This one is the CRN for the ML service to use. Open
        # https://cloud.ibm.com/resources, find the service you want to use,
        # click anywhere on the line except the name, and copy the CRN
        assert 'ml_instance_crn' in wml_credentials
        assert type(wml_credentials['ml_instance_crn']) is str
        self._logger.debug(f'Credentials have the expected structure.')

        # This one is not required to create the APIClient, but can be used
        # to get data through Cloud Object Storage or any other type of
        # external storage.
        if 'connection_id' in wml_credentials:
            self._logger.debug(f'And they contain a connection id.')

        self._wml_credentials = wml_credentials

        # We don't initialize the client at this time, because this is an
        # expensive operation.
        self._client = None
        self._space_id = None
        self.model_type = self.MODEL_TYPES[0]
        self.tshirt_size = self.TSHIRT_SIZES[0]
        self.timelimit = None
        self.inline = False
        self._data_connection = None
        self._type_from_deployment = {}
        self._size_from_deployment = {}

    def _create_client(self):
        """Create the Python APIClient instance"""
        assert self._client is None
        self._logger.debug(f'Creating the connexion...')
        client = APIClient(self._wml_credentials)
        self._logger.info(f'Creating the connexion succeeded.  Client version is {client.version}')
        return client

    def _get_or_make_client(self):
        if self._client is not None:
            return self._client
        self._client = self._create_client()
        assert self._client is not None
        self._get_space_id()
        return self._client

    def _read_wml_credentials_from_env(self, var_name):
        """Return a string of credentials suitable for WML from the environment

        Raises InvalidCredentials if anything is wrong."""
        self._logger.debug(f'Looking for credentials in environment variable {var_name}...')
        try:
            wml_cred_str = os.environ[var_name]
        except KeyError:
            print(f'Environment variable ${var_name} not found.')
            print(f'It should contain credentials as a Python dict of the form:')
            print("'{'apikey': '<apikey>', 'url': 'https://us-south.ml.cloud.ibm.com'}")
            raise InvalidCredentials

        return wml_cred_str

    def _read_wml_credentials_from_file(self, file):
        """Return the content of the file, assumed to be WML credentials"""
        self._logger.debug(f'Looking for credentials in file \'{file}\'...')
        with open(file) as f:
            wml_cred_str = f.read()
        return wml_cred_str

    def solve(self, paths):
        """Solve the model, return the job id

        The model is sent as online data to WML.

        :param paths: one or more pathname to the files to send, as a single
                      string, separated by space
        """
        self._get_or_make_client()

        deployment_id = self._get_deployment_id()
        self._logger.info(f'Deployment id: {deployment_id}')
        job_id = self.create_job(paths, deployment_id)
        self._logger.info(f'Job id: {job_id}')
        return job_id

    def get_log(self, job_id):
        """Extracts the CPLEX log from the job

        :param job_id: The id of the job to get the log from
        :return: The decoded log, or None
        """
        job_details = self.get_job_details(job_id, with_contents='log')
        for output_data in job_details['entity']['decision_optimization']['output_data']:
            if output_data['id'] == 'log.txt':
                output = output_data['content']
                output = self.decode_log(output)
                output = self.remove_empty_lines(output)
                return output
        return None

    def get_output(self, details):
        """"Extracts the outputs from the job

        :param details: The details of the job to get the output from
        :return: A list of outputs. Each output is a tuple (name, content)
        where the name is, well, the name of the output, and content is the
        decoded content, as bytes. We don't assume that the content is actually
        text.
        """
        try:
            outputs = details['entity']['decision_optimization']['output_data']
        except KeyError:
            self._logger.warning(f'No output structure available for this job')
            return []
        result = []
        for output_data in outputs:
            name = output_data['id']
            content = output_data['content']
            content = content.encode('UTF-8')
            content = base64.b64decode(content)
            result.append((name, content))
        return result

    def get_job_details(self, job_id, with_contents=None):
        """ Get the job details for the given job
        :param job_id: The id of the job to look for
        :param with_contents: if 'names', the details returned include
        the input and output files names. If 'full', the content of these files
        is included as well. If 'log', the content only includes the output files
        :return: The job details
        """
        client = self._get_or_make_client()
        self._logger.debug(f'Fetching output...')
        output_filter = None
        if not with_contents:
            output_filter = 'solve_parameters,solve_state,status'
        elif with_contents == 'log':
            output_filter = 'output_data'
        job_details = self.client_get_job_details(client, job_id, output_filter)
        self._logger.debug(f'Done.')
        if with_contents != 'full' and with_contents != 'log':
            self.filter_large_chunks_from_details(job_details)
        return job_details

    def client_get_job_details(self, client, job_id, with_filter=None):
        global the_filter
        global the_old_params
        the_filter = with_filter
        the_old_params = client._params
        client._params = new_params
        try:
            result = client.deployments.get_job_details(job_id)
        finally:
            client._params = the_old_params
        return result

    @staticmethod
    def filter_large_chunks_from_details(job_details):
        """Remove the large blobs (input/output) from the given job_details."""
        try:
            do = job_details['entity']['decision_optimization']
            for data in do.get('output_data', []):
                if 'content' in data:
                    data['content'] = '[not shown]'
            for data in do.get('input_data', []):
                if 'content' in data:
                    data['content'] = '[not shown]'
            if 'latest_engine_activity' in do['solve_state']:
                do['solve_state']['latest_engine_activity'] = ['[not shown]']
        except KeyError:
            # GH-1: This happens when the job failed
            pass

    def delete_job(self, job_id, hard=False):
        """ Delete the given job
        :param job_id: the job to be deleted
        :param hard: if False, cancel the job. If true, delete it completely
        """
        client = self._get_or_make_client()
        self._logger.debug(f'Deleting job {job_id}...')
        client.deployments.delete_job(job_id, hard)
        self._logger.debug(f'Done.')

    def decode_log(self, output):
        """ Decode the log from DO4WML

        :param output: A base-64 encoded text with empty lines
        :return: The decoded text, without empty lines
        """
        output = output.encode('UTF-8')
        output = base64.b64decode(output)
        output = output.decode('UTF-8')
        output = self.remove_empty_lines(output)
        return output

    @staticmethod
    def remove_empty_lines(output):
        """Remove empty lines from the log

        :param output: The text to process
        :return: The text, with no empty lines
        """
        output = '\n'.join([s for s in output.splitlines() if s])
        return output

    @staticmethod
    def _get_job_status_from_details(job_details):
        return job_details['entity']['decision_optimization']['status']['state']

    @staticmethod
    def _get_job_id_from_details(job_details):
        return job_details['metadata']['id']

    @staticmethod
    def _get_creation_time_from_details(job_details):
        created = job_details['metadata']['created_at']
        if created[-1] == 'Z':
            dt = datetime.fromisoformat(created[:-1])
            created = dt.isoformat(sep=' ', timespec='seconds')
        return created

    @staticmethod
    def _get_input_names_from_details(job_details):
        do = job_details['entity']['decision_optimization']
        inputs = do.get('input_data', [])
        names = [i['id'] for i in inputs]
        inputs = do.get('input_data_references', [])
        for i in inputs:
            if 'id' in i:
                names.append('*' + i['id'])
            else:
                names.append('Unknown')
        return names

    def wait_for_job_end(self, job_id, print_activity=False):
        """Wait for the job to finish, return its status and details as a tuple"""
        client = self._get_or_make_client()
        while True:
            job_details = self.client_get_job_details(client, job_id, with_filter='solve_state,status')
            do = job_details['entity']['decision_optimization']
            status = self._get_job_status_from_details(job_details)
            self._logger.info(f'Job status: {status}')
            if status in ['completed', 'failed', 'canceled']:
                break
            else:
                if print_activity:
                    # There may be a bit of log to look at
                    try:
                        activity = do['solve_state']['latest_engine_activity']
                        if activity:
                            # We are joining the lines in the activity with a CR,
                            # only to remove them if they were already included...
                            # FIXME: what a waste!
                            act = '\n'.join(activity)
                            act = self.remove_empty_lines(act)
                            print(act)
                    except KeyError:
                        # This must mean that no activity is available yet
                        pass
            sleep(self.JOB_END_SLEEP_DELAY)
        return status, job_details

    @staticmethod
    def get_file_as_data(path):
        """Returns the base-64 encoded content of a file"""
        with open(path, 'rb') as f:
            data = f.read()
        data = base64.b64encode(data)
        data = data.decode('UTF-8')
        return data

    def _get_type_from_details(self, job):
        try:
            deployment_id = job['entity']['deployment']['id']
            if deployment_id in self._type_from_deployment:
                return self._type_from_deployment[deployment_id]
            client = self._get_or_make_client()
            deployment = client.deployments.get_details(deployment_id)
            model_id = deployment['entity']['asset']['id']
            model = client.model_definitions.get_details(model_id)
            deployment_type = model['entity']['wml_model']['type']
            match = re.fullmatch(r"do-(....*)_[0-9.]*", deployment_type)
            if match:
                deployment_type = match.group(1)
            self._type_from_deployment[deployment_id] = deployment_type
            return deployment_type
        except KeyError:
            # Something changed. But let's not fail just for that
            self._logger.warning(f'Error while fetching type of a job!')
            return '?????'

    def _get_size_from_details(self, job):
        try:
            deployment_id = job['entity']['deployment']['id']
            if deployment_id in self._size_from_deployment:
                return self._size_from_deployment[deployment_id]
            client = self._get_or_make_client()
            deployment = client.deployments.get_details(deployment_id)
            size = deployment['entity']['hardware_spec']['name']
            self._size_from_deployment[deployment_id] = size
            return size
        except KeyError:
            # Something changed. But let's not fail just for that
            self._logger.warning(f'Error while fetching size of a job!')
            return '?'

    def get_jobs(self):
        """Return the list of tuples (id, status) for all jobs in the deployment"""
        client = self._get_or_make_client()
        self._logger.debug(f'Getting job details...')
        job_details = client.deployments.get_job_details()
        self._logger.debug(f'Done.')
        result = []
        for job in job_details['resources']:
            status = self._get_job_status_from_details(job)
            job_id = self._get_job_id_from_details(job)
            created = self._get_creation_time_from_details(job)
            names = self._get_input_names_from_details(job)
            deployment_type = self._get_type_from_details(job)
            size = self._get_size_from_details(job)
            JobTuple = namedtuple('Job', ['status', 'id', 'created', 'names', 'type', 'size'])
            j = JobTuple(status=status, id=job_id, created=created, names=names, type=deployment_type, size=size)
            result.append(j)
        return result

    def create_job(self, paths, deployment_id):
        """Create a deployment job (aka a run) and return its id"""
        client = self._get_or_make_client()
        cdd = client.deployments.DecisionOptimizationMetaNames
        cdd_inputdata = cdd.INPUT_DATA
        cdd_outputdata = cdd.OUTPUT_DATA
        if not self.inline:
            cdd_inputdata = cdd.INPUT_DATA_REFERENCES
            # cdd_outputdata = cdd.OUTPUT_DATA_REFERENCES
        solve_payload = {
            cdd.SOLVE_PARAMETERS: {
                'oaas.logAttachmentName': 'log.txt',
                'oaas.logTailEnabled': 'true',
                'oaas.includeInputData': 'false',
                'oaas.resultFormat': 'JSON'
            },
            cdd_inputdata: [],
            cdd_outputdata: [
                {'id': '.*\\.*'}
            ]
        }
        if self.timelimit:
            params = solve_payload[cdd.SOLVE_PARAMETERS]
            params['oaas.timeLimit'] = 1000 * self.timelimit
        # There may be more than one input file
        names = []
        for path in paths.split():
            basename = os.path.basename(path)
            if basename in names:
                raise SimilarNamesInJob(basename)
            names.append(basename)
            if self.inline:
                input_data = {
                    'id': basename,
                    'content': self.get_file_as_data(path)
                }
            else:
                data_asset_id = self._create_data_asset_and_upload_if_necessary(path)
                input_data = {
                    'id': basename,
                    "type": "data_asset",
                    "location": {
                        "href": "/v2/assets/" + data_asset_id + "?space_id=" + self._space_id
                    }
                }
            solve_payload[cdd_inputdata].append(input_data)
        self._logger.debug(f'Creating the job...')
        if self.inline:
            self._logger.debug(f'Data is inline. Let\'s not print the payload...')
        else:
            self._logger.debug(repr(solve_payload))
        job_details = client.deployments.create_job(deployment_id, solve_payload)
        self._logger.debug(f'Done. Getting its id...')
        job_id = client.deployments.get_job_uid(job_details)
        return job_id

    def _get_deployment_id(self):
        """Create deployment if doesn't exist already, return its id"""

        if not self._space_id:
            self._space_id = self._get_space_id()

        self._logger.debug(f'Getting deployments...')
        client = self._get_or_make_client()
        deployment_details = client.deployments.get_details()
        self._logger.debug(f'Done.')
        resources = deployment_details['resources']
        deployment_name = f'{self.DEPLOYMENT_NAME}-{self.model_type}-{self.tshirt_size}'
        self._logger.debug(f'Got the list. Looking for deployment named \'{deployment_name}\'')
        deployment_id = None
        for r in resources:
            if r['entity']['name'] == deployment_name:
                deployment_id = r['metadata']['id']
                self._logger.debug(f'Found it.')
                break
        if deployment_id is not None:
            return deployment_id

        self._logger.debug(f'This deployment doesn\'t exist yet. Creating it...')

        deployment_id = self._create_deployment(deployment_name)
        return deployment_id

    def _create_deployment(self, deployment_name):
        # We need a model to create a deployment
        model_id = self._get_model_id()
        # Create the deployment
        self._logger.debug(f'Creating the deployment itself...')
        client = self._get_or_make_client()
        cdc = client.deployments.ConfigurationMetaNames
        meta_props = {
            cdc.NAME: deployment_name,
            cdc.DESCRIPTION: "Deployment for the Solve on WML Python script",
            cdc.BATCH: {},
            cdc.HARDWARE_SPEC: {'name': self.tshirt_size, 'num_nodes': 2}
        }
        deployment = client.deployments.create(artifact_uid=model_id, meta_props=meta_props)
        self._logger.debug(f'Deployment created.')
        deployment_id = client.deployments.get_id(deployment)
        return deployment_id

    def _get_model_id(self):
        """Create an empty model if one doesn't exist, return its id"""
        self._logger.debug(f'Getting models...')
        client = self._get_or_make_client()
        details = client.repository.get_details()
        self._logger.debug(f'Done.')
        resources = details['models']['resources']
        model_name = f'{self.MODEL_NAME}-{self.model_type}'
        self._logger.debug(f'Got the list. Looking for model named \'{model_name}\'')
        model_id = None
        for r in resources:
            if r['metadata']['name'] == model_name:
                model_id = r['metadata']['id']
                self._logger.debug(f'Found it.')
                self._logger.debug(f'Model id: {model_id}')
                break
        if model_id is None:
            self._logger.debug(f'This model doesn\'t exist yet. Creating it...')
            model_id = self._create_model(model_name)
        return model_id

    def _create_model(self, model_name):
        client = self._get_or_make_client()
        cr = client.repository
        crm = cr.ModelMetaNames
        model_metadata = {
            crm.NAME: model_name,
            crm.DESCRIPTION: "Model for the solve-on-wml script",
            crm.TYPE: f'do-{self.model_type}_12.10',
            crm.SOFTWARE_SPEC_UID:
                client.software_specifications.get_id_by_name("do_12.10")
        }
        # We need an empty.zip file, because APIClient doesn't know better
        handle, path = tempfile.mkstemp(suffix='.zip', text=False)
        try:
            # This string is the result of converting the file
            # empty.zip in the repository using
            #   openssl base64 < empty.zip
            file_content = base64.b64decode('UEsFBgAAAAAAAAAAAAAAAAAAAAAAAA==')
            os.write(handle, file_content)
        finally:
            os.close(handle)
        try:
            model_details = cr.store_model(model=path,
                                           meta_props=model_metadata)
        finally:
            os.remove(path)
        self._logger.debug(f'Model created.')
        model_id = client.repository.get_model_id(model_details)
        self._logger.debug(f'Model id: {model_id}')
        return model_id

    def _get_space_id(self):
        """Find the Space to use, create it if it doesn't exist"""
        if self._space_id:
            return self._space_id

        client = self._get_or_make_client()
        self._logger.debug(f'Fetching existing spaces...')
        space_name = self.SPACE_NAME
        csc = client.spaces.ConfigurationMetaNames
        metadata = {
            csc.NAME: space_name,
            csc.DESCRIPTION: space_name + ' description',
            csc.STORAGE: {
                "type": "bmcos_object_storage",
                "resource_crn": self._wml_credentials['cos_resource_crn']
            },
            csc.COMPUTE: {
                "name": "existing_instance_id",
                "crn": self._wml_credentials['ml_instance_crn']
            }
        }

        space_details = client.spaces.get_details()
        resources = space_details['resources']
        self._logger.debug(f'Got the list. Looking for space named \'{space_name}\'')
        space_id = None
        for r in resources:
            if r['entity']['name'] == space_name:
                space_id = r['metadata']['id']
                self._logger.debug(f'Found it.')
                break
        if space_id is None:
            self._logger.debug(f'This space doesn\'t exist yet. Creating it...')
            # Create the space
            space = client.spaces.store(meta_props=metadata)
            self._logger.debug(f'Space created')
            space_id = client.spaces.get_id(space)
        self._logger.info(f'Space id: {space_id}')

        self._logger.debug(f'Setting default space...')
        self._client.set.default_space(space_id)
        self._logger.debug(f'Done.')

        self._space_id = space_id
        return space_id

    def _get_connection_details(self):
        """This function returns the list of all the connections in the space

        This function should have been in the wml Python client, but unfortunately,
        it's not.  At least, not as of version 1.0.53.
        C.f. https://github.ibm.com/NGP-TWC/ml-planning/issues/21577#issuecomment-28950762"""
        client = self._get_or_make_client()
        if client.WSD:
            header_param = client._get_headers(wsdconnection_api_req=True)
        else:
            header_param = client._get_headers()
        if not client.ICP_30 and not client.ICP and not client.WSD:
            response = requests.get(client.connections._href_definitions.get_connections_href(),
                                    params=client._params(),
                                    headers=header_param)
        else:
            response = requests.get(client.connections._href_definitions.get_connections_href(),
                                    params=client._params(),
                                    headers=header_param, verify=False)
        client.connections._handle_response(200, u'list datasource type', response)
        datasource_details = client.connections._handle_response(200, u'list datasource types', response)['resources']
        return datasource_details

    def _find_connection_to_use(self):
        """Returns information about the connection to use

        A Watson Studio data asset doesn't (or doesn't necessarily) contain the
        information necessary to actually get the data from the external service.
        It actually has the id of a 'connection'. That connection holds all the
        information (API key, secret key, etc).

        This function looks for a suitable connection in the WS space.  It can be
        either a connection the id of which was given in the DOWML credentials,
        or one that simply exists in the space and is named
        'DOWMLClient-connection'.

        The information from this connection is also used to check whether the
        model to solve already exists on Cloud Object Storage, and upload it there
        if that's not the case.

        This function returns a named tuple: 'connection_id', 'bucket_name',
        'endpoint_url'. """
        if self._data_connection:
            return self._data_connection
        client = self._get_or_make_client()
        connection_id = self._wml_credentials.get('connection_id', '')
        if connection_id:
            self._logger.debug(f'Found the connection id to use in the WML credentials.')
        else:
            name = 'DOWMLClient-connection'
            self._logger.debug(f'Looking for a connection named "{name}"...')
            connections = self._get_connection_details()
            for c in connections:
                if c['entity']['name'] == name:
                    connection_id = c['metadata']['asset_id']
                    self._logger.debug(f'Found one.')
        if not connection_id:
            self._logger.error(f'Could not find a Connection to get the data!')
            raise ConnectionIdNotFound
        self._logger.debug(f'Connection id: {connection_id}')
        connection = client.connections.get_details(connection_id)
        DataConnection = namedtuple('DataConnection', ['connection_id', 'bucket_name', 'endpoint_url'])
        self._data_connection = DataConnection(connection_id,
                                               connection['entity']['properties']['bucket'],
                                               connection['entity']['properties']['url'])
        self._logger.debug(f'Fetched its details.')
        return self._data_connection

    def _get_asset_details(self):
        """This function returns the list of all the data assets in the space

        This function should have been in the wml Python client, but unfortunately,
        it's not.  At least, not as of version 1.0.53.
        C.f. https://github.ibm.com/NGP-TWC/ml-planning/issues/21577#issuecomment-29056420"""
        client = self._get_or_make_client()
        href = client.data_assets._href_definitions.get_search_asset_href()
        data = {
                "query": "*:*"
        }
        if not client.data_assets._ICP and not client.WSD:
            response = requests.post(href,
                                     params=self._client._params(),
                                     headers=self._client._get_headers(),
                                     json=data)
        else:
            response = requests.post(href,
                                     params=self._client._params(),
                                     headers=self._client._get_headers(),
                                     json=data,
                                     verify=False)
        client.data_assets._handle_response(200, u'list assets', response)
        asset_details = client.data_assets._handle_response(200, u'list assets', response)["results"]
        return asset_details

    def _find_asset_id_by_name(self, name):
        """Looks for a data asset with the given name, returns its id, or None"""
        assets = self._get_asset_details()
        for asset in assets:
            metadata = asset['metadata']
            if metadata['name'] == name:
                return metadata['asset_id']
        return None

    def create_asset(self, path):
        """Create a data asset with the given name

        A Watson Studio data asset is an entity that mimicks a file.  It actually
        connects to some external service to fetch the data.  It does so using
        a Connection.  In this case, we create an asset that reads the object in
        Cloud Object Storage.  The connection gives us the information about
        the bucket to use and the endpoint to contact to read the object."""
        basename = os.path.basename(path)
        client = self._get_or_make_client()
        asset_details = client.data_assets.create(basename, path)
        return asset_details['metadata']['guid']

    def _create_data_asset_and_upload_if_necessary(self, path):
        """Create a data asset (and upload file) if it doesn't exist already"""
        basename = os.path.basename(path)
        self._logger.debug(f'Checking whether a connected data asset named \'{basename}\' already exists.')
        data_asset_id = self._find_asset_id_by_name(basename)
        if data_asset_id:
            self._logger.debug(f'Yes, with id {data_asset_id}.')
            return data_asset_id
        self._logger.debug(f'Creating the data asset.')
        data_asset_id = self.create_asset(path)
        self._logger.debug(f'Done.')
        return data_asset_id
