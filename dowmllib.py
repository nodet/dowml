import base64
import csv
import glob
import io
import logging
import os
import pprint
import re
import tempfile
from collections import namedtuple
from datetime import datetime
from functools import lru_cache
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


class NoCredentialsToCreateSpace(Error):
    """Need to create a space, but credentials are incomplete to allow that"""
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

    DOWML_PREFIX = 'dowml'
    ENVIRONMENT_VARIABLE_NAME = 'DOWML_CREDENTIALS'
    SPACE_NAME = f'{DOWML_PREFIX}-space'
    MODEL_NAME = f'{DOWML_PREFIX}-model'
    MODEL_TYPES = ['cplex', 'cpo', 'opl', 'docplex']
    DO_VERSION = '20.1'
    TSHIRT_SIZES = ['S', 'M', 'XL']
    DEPLOYMENT_NAME = f'{DOWML_PREFIX}-deployment'
    JOB_END_SLEEP_DELAY = 2

    # The keys in the credentials
    APIKEY = 'apikey'
    TOKEN = 'token'
    SPACE_ID = 'space_id'
    URL = 'url'
    COS_CRN = 'cos_resource_crn'
    ML_CRN = 'ml_instance_crn'

    def __init__(self, wml_credentials_file=None, space_id=None):
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
        assert (self.APIKEY in wml_credentials or self.TOKEN in wml_credentials)
        if self.APIKEY in wml_credentials:
            assert type(wml_credentials[self.APIKEY]) is str
        else:
            assert type(wml_credentials[self.TOKEN]) is str
        assert self.URL in wml_credentials
        assert type(wml_credentials[self.URL]) is str
        self._logger.debug(f'Credentials have the expected structure.')

        if self.SPACE_ID in wml_credentials:
            self._logger.debug(f'And they contain a space id.')

        # The space_id specified here takes precedence
        # over the one, if any, defined in the credentials
        if space_id:
            wml_credentials[self.SPACE_ID] = space_id

        self._wml_credentials = wml_credentials

        # We don't initialize the client at this time, because this is an
        # expensive operation.
        self._client = None
        self._space_id = None
        self.model_type = self.MODEL_TYPES[0]
        self.tshirt_size = self.TSHIRT_SIZES[0]
        self.do_version = self.DO_VERSION
        self.timelimit = None
        self.inline = False
        self._data_connection = None

    def _create_client(self):
        """Create the Python APIClient instance"""
        assert self._client is None
        self._logger.debug(f'Creating the WML client...')
        # http://ibm-wml-api-pyclient.mybluemix.net/#api-for-ibm-cloud
        client = APIClient(self._wml_credentials)
        self._logger.info(f'Creating the client succeeded.  Client version is {client.version}')
        return client

    def _get_or_make_client(self):
        if self._client is None:
            self._client = self._create_client()
        if self._space_id is None:
            self._space_id = self._get_space_id()
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
            print(f'{{\'{self.APIKEY}\': \'<apikey>\', \'{self.URL}\': \'https://us-south.ml.cloud.ibm.com\'}}')
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

        The model is sent as online data to WML (if 'inline yes') or is uploaded as a data asset
        to be reused later (default).

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
        """Extracts the engine log from the job

        :param job_id: The id of the job to get the log from
        :return: The decoded log, or None
        """
        job_details = self.get_job_details(job_id, with_contents='log')
        try:
            outputs = job_details['entity']['decision_optimization']['output_data']
        except KeyError:
            self._logger.warning(f'No output structure available for this job')
            return None
        for output_data in outputs:
            if output_data['id'] == 'log.txt':
                if 'content' not in output_data:
                    self._logger.error(f'Log without content for job {job_id}')
                    continue
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
            if 'content' in output_data:
                # What we have here is a regular file, encoded
                self._logger.debug(f'Found a regular file named {name}')
                content = output_data['content']
                content = content.encode('UTF-8')
                content = base64.b64decode(content)
                result.append((name, content))
            elif ('values' in output_data and
                  'fields' in output_data and
                  name.lower().endswith('.csv')):
                self._logger.debug(f'Found a CSV file named {name}')
                content = io.StringIO()
                writer = csv.writer(content)
                writer.writerow(output_data['fields'])
                for r in output_data['values']:
                    writer.writerow(r)
                result.append((name, content.getvalue().encode()))
            else:
                self._logger.warning(f'Found an unknown file named {name}')
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
                    # This is the case for regular files, such as the log
                    data['content'] = '[not shown]'
                elif 'values' in data:
                    # This is the case for CSV files
                    data['values'] = ['[not shown]']
            for data in do.get('input_data', []):
                if 'content' in data:
                    data['content'] = '[not shown]'
            if 'solve_state' in do and 'latest_engine_activity' in do['solve_state']:
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
            deployment = self._get_deployment_from_id(deployment_id)
            model_id = deployment['entity']['asset']['id']
            model = self._get_model_definition_from_id(model_id)
            deployment_type = model['entity']['wml_model']['type']
            match = re.fullmatch(r"do-(....*)_[0-9.]*", deployment_type)
            if match:
                deployment_type = match.group(1)
            return deployment_type
        except KeyError:
            # Something changed. But let's not fail just for that
            self._logger.warning(f'Error while fetching type of a job!')
            return '?????'

    def _get_version_from_details(self, job):
        try:
            deployment_id = job['entity']['deployment']['id']
            deployment = self._get_deployment_from_id(deployment_id)
            model_id = deployment['entity']['asset']['id']
            model = self._get_model_definition_from_id(model_id)
            deployment_type = model['entity']['wml_model']['type']
            match = re.fullmatch(r"do-....*_([0-9.]*)", deployment_type)
            version = '?????'
            if match:
                version = match.group(1)
            return version
        except KeyError:
            # Something changed. But let's not fail just for that
            self._logger.warning(f'Error while fetching version of a job!')
            return '?????'

    @lru_cache
    def _get_model_definition_from_id(self, model_id):
        client = self._get_or_make_client()
        model = client.model_definitions.get_details(model_id)
        return model

    @lru_cache
    def _get_deployment_from_id(self, deployment_id):
        client = self._get_or_make_client()
        deployment = client.deployments.get_details(deployment_id)
        return deployment

    def _get_size_from_details(self, job):
        try:
            deployment_id = job['entity']['deployment']['id']
            deployment = self._get_deployment_from_id(deployment_id)
            size = deployment['entity']['hardware_spec']['name']
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
        self._logger.debug(f'Getting information about deployments and models...')
        result = []
        for job in job_details['resources']:
            status = self._get_job_status_from_details(job)
            job_id = self._get_job_id_from_details(job)
            created = self._get_creation_time_from_details(job)
            names = self._get_input_names_from_details(job)
            deployment_type = self._get_type_from_details(job)
            version = self._get_version_from_details(job)
            size = self._get_size_from_details(job)
            JobTuple = namedtuple('Job', ['status', 'id', 'created', 'names', 'type', 'version', 'size'])
            j = JobTuple(status=status, id=job_id, created=created, names=names, type=deployment_type, version=version, size=size)
            result.append(j)
        self._logger.debug(f'Done.')
        return result

    def create_job(self, paths, deployment_id):
        """Create a deployment job (aka a run) and return its id"""
        client = self._get_or_make_client()
        cdd = client.deployments.DecisionOptimizationMetaNames
        # We always use inline output
        cdd_outputdata = cdd.OUTPUT_DATA
        # Assume we use inline data (i.e. content in the job request)
        cdd_inputdata = cdd.INPUT_DATA
        if not self.inline:
            # But if we don't want inline data, we have to submit
            # input references instead
            cdd_inputdata = cdd.INPUT_DATA_REFERENCES
        solve_payload = {
            cdd.SOLVE_PARAMETERS: {
                'oaas.logAttachmentName': 'log.txt',
                'oaas.logTailEnabled': 'true',
                'oaas.includeInputData': 'false',
                'oaas.resultsFormat': 'JSON'
            },
            cdd_inputdata: [],
            cdd_outputdata: [
                {'id': '.*\\.*'}
            ]
        }
        if self.timelimit:
            params = solve_payload[cdd.SOLVE_PARAMETERS]
            params['oaas.timeLimit'] = 1000 * self.timelimit
        self.create_inputs(paths, cdd_inputdata, solve_payload)
        self._logger.debug(f'Creating the job...')
        if self.inline:
            self._logger.debug(f'Data is inline. Let\'s not print the payload...')
        else:
            self._logger.debug(repr(solve_payload))
        job_details = client.deployments.create_job(deployment_id, solve_payload)
        self._logger.debug(f'Done. Getting its id...')
        job_id = client.deployments.get_job_uid(job_details)
        return job_id

    def create_inputs(self, paths, cdd_inputdata, solve_payload):
        # First deal with wildcards
        globbed = self.parse_paths(paths)
        # And let's now create the inputs from these files
        names = []
        for path in globbed:
            path, basename, force = self._get_file_spec(path)
            if basename in names:
                raise SimilarNamesInJob(basename)
            names.append(basename)
            if self.inline:
                input_data = {
                    'id': basename,
                    'content': self.get_file_as_data(path)
                }
            else:
                data_asset_id = self._create_data_asset_if_necessary(path, basename, force)
                input_data = {
                    'id': basename,
                    "type": "data_asset",
                    "location": {
                        "href": "/v2/assets/" + data_asset_id + "?space_id=" + self._space_id
                    }
                }
            solve_payload[cdd_inputdata].append(input_data)

    def parse_paths(self, paths):
        self._logger.debug(f'Parsing input list: {paths}')
        # There may be wildcards, so let's deal with them first
        globbed = []
        for path in paths.split():
            # Let's first get rid of the 'force' flag that glob
            # would not understand
            path, _, force = self._get_file_spec(path)
            files = glob.glob(path)
            if not files:
                # If the path doesn't actually match an existing file, this is
                # not necessarily an error: this name can refer to a data
                # asset that exists already. So let's keep it.
                files = [path]
            if force:
                # Put back the '+' in front
                files = [f'+{file}' for file in files]
            globbed += files
        self._logger.debug(f'Actual input list: {globbed}')
        return globbed

    def _get_file_spec(self, path):
        force = False
        if path[0] == '+':
            force = True
            path = path[1:]
        basename = os.path.basename(path)
        return path, basename, force

    def _get_deployment_id(self):
        """Create deployment if doesn't exist already, return its id"""

        if not self._space_id:
            self._space_id = self._get_space_id()

        self._logger.debug(f'Getting deployments...')
        client = self._get_or_make_client()
        deployment_details = client.deployments.get_details()
        self._logger.debug(f'Done.')
        resources = deployment_details['resources']
        deployment_name = f'{self.DEPLOYMENT_NAME}-{self.model_type}-{self.do_version}-{self.tshirt_size}'
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
        model_name = f'{self.MODEL_NAME}-{self.model_type}-{self.do_version}'
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
            crm.TYPE: f'do-{self.model_type}_{self.do_version}',
            crm.SOFTWARE_SPEC_UID:
                client.software_specifications.get_id_by_name(f'do_{self.do_version}')
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
        if self._space_id:
            return self._space_id

        if self.SPACE_ID in self._wml_credentials:
            space_id = self._wml_credentials[self.SPACE_ID]
            self._logger.debug(f'Using specified space \'{space_id}\'.')
        else:
            space_id = self._find_or_create_space()

        self._logger.debug(f'Setting default space...')
        self._client.set.default_space(space_id)
        self._logger.debug(f'Done.')
        return space_id

    def _find_or_create_space(self):
        """Find the Space to use, create it if it doesn't exist"""
        assert self._client
        client = self._client
        self._logger.debug(f'Fetching existing spaces...')
        space_details = client.spaces.get_details()
        resources = space_details['resources']
        self._logger.debug(f'Got the list. Looking for space named \'{self.SPACE_NAME}\'')
        space_id = None
        for r in resources:
            if r['entity']['name'] == self.SPACE_NAME:
                space_id = r['metadata']['id']
                self._logger.debug(f'Found it.')
                break
        if space_id is None:
            self._logger.debug(f'This space doesn\'t exist yet. Creating it...')
            # Prepare necessary information

            wml_credentials = self._wml_credentials
            if self.COS_CRN not in wml_credentials or self.ML_CRN not in wml_credentials:
                raise NoCredentialsToCreateSpace(f'WML credentials do not contain the information necessary '
                                                 f'to create a deployment space. \nMissing \'{self.COS_CRN}\' '
                                                 f'and/or \'{self.ML_CRN}\'.')
            assert type(wml_credentials[self.COS_CRN]) is str
            assert type(wml_credentials[self.ML_CRN]) is str

            csc = client.spaces.ConfigurationMetaNames
            metadata = {
                csc.NAME: self.SPACE_NAME,
                csc.DESCRIPTION: self.SPACE_NAME + ' description',
                csc.STORAGE: {
                    "type": "bmcos_object_storage",
                    "resource_crn": self._wml_credentials[self.COS_CRN]
                },
                csc.COMPUTE: {
                    "name": "existing_instance_id",
                    "crn": self._wml_credentials[self.ML_CRN]
                }
            }
            # Create the space
            space = client.spaces.store(meta_props=metadata)
            self._logger.debug(f'Space created')
            space_id = client.spaces.get_id(space)
        self._logger.info(f'Space id: {space_id}')
        return space_id

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

    def create_asset(self, path, basename):
        """Create a data asset with the given name

        A Watson Studio data asset is an entity that mimicks a file."""
        client = self._get_or_make_client()
        asset_details = client.data_assets.create(basename, path)
        return asset_details['metadata']['guid']

    def delete_asset(self, id):
        """Delete an existing asset. Return True if ok, False if not"""
        client = self._get_or_make_client()
        status = client.data_assets.delete(id)
        return status == "SUCCESS"

    def _create_data_asset_if_necessary(self, path, basename, force):
        """Create a data asset (and upload file) if it doesn't exist already (or force is True)."""
        asset_to_delete = None
        self._logger.info(f'Checking whether a data asset named \'{basename}\' already exists.')
        data_asset_id = self._find_asset_id_by_name(basename)
        if data_asset_id:
            self._logger.debug(f'Yes, with id {data_asset_id}.')
            if not force:
                return data_asset_id
            self._logger.debug(f'Creating new asset with local content.')
            asset_to_delete = data_asset_id
        else:
            self._logger.debug(f'No, creating the data asset.')
        data_asset_id = self.create_asset(path, basename)
        self._logger.debug(f'Done.')
        if asset_to_delete:
            self._logger.debug(f'Deleting the old data asset.')
            if self.delete_asset(asset_to_delete):
                self._logger.debug(f'Done.')
            else:
                self._logger.warning(f'Could not delete pre-existing asset')
        return data_asset_id
