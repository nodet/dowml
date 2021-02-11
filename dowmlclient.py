import base64
import logging
import os
import pprint
from collections import namedtuple
from time import sleep

from ibm_watson_machine_learning import APIClient


class Error(Exception):
    """Base class for all errors in this script"""
    pass


class InvalidCredentials(Error):
    """The WML credentials were not found, or incorrect"""
    pass


class DOWMLClient:
    """A Python client to run DO models on WML"""

    ENVIRONMENT_VARIABLE_NAME = 'WML_CREDENTIALS'
    SPACE_NAME = 'DOWMLClient-space'
    MODEL_NAME = 'DOWMLClient-model'
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
        self._logger.info(f'Found credential string.')

        wml_credentials = eval(wml_cred_str)
        assert type(wml_credentials) is dict
        assert 'apikey' in wml_credentials
        assert type(wml_credentials['apikey']) is str
        assert 'url' in wml_credentials
        assert type(wml_credentials['url']) is str
        self._logger.info(f'Credentials have the expected structure.')

        self._wml_credentials = wml_credentials

        # We don't initialize the client at this time, because this is an
        # expensive operation.
        self._client = None
        self._space_id = None

    def _get_or_make_client(self):
        if self._client is not None:
            return self._client
        self._create_connexion()
        assert self._client is not None
        self._get_space_id()
        return self._client

    @staticmethod
    def _read_wml_credentials_from_env(var_name):
        """Return a string of credentials suitable for WML from the environment

        Raises InvalidCredentials if anything is wrong."""
        logging.info(f'Looking for credentials in environment variable {var_name}...')
        try:
            wml_cred_str = os.environ[var_name]
        except KeyError:
            print(f'Environment variable ${var_name} not found.')
            print(f'It should contain credentials as a Python dict of the form:')
            print("'{'apikey': '<apikey>', 'url': 'https://us-south.ml.cloud.ibm.com'}")
            raise InvalidCredentials

        return wml_cred_str

    @staticmethod
    def _read_wml_credentials_from_file(file):
        """Return the content of the file, assumed to be WML credentials"""
        logging.info(f'Looking for credentials in file \'{file}\'...')
        with open(file) as f:
            wml_cred_str = f.read()
        return wml_cred_str

    def solve(self, path, wait_for_completion=True):
        """Solve the model, return the job id

        The model is sent as online data to WML.

        :param path: pathname to the file to solve
        :param wait_for_completion: whether to wait for the job to complete
        """
        self._create_connexion()
        self._get_space_id()

        deployment_id = self._get_deployment_id()
        self._logger.info(f'Deployment id: {deployment_id}')
        job_id = self.create_job(path, deployment_id)
        self._logger.info(f'Job id: {job_id}')
        if wait_for_completion:
            status, job_details = self.wait_for_job_end(job_id)
            print(f'Job {status}')
            print(self.get_log(job_id, job_details))
        return job_id

    def get_log(self, job_id, job_details=None):
        """Extracts the CPLEX log from the job

        :param job_id: The id of the job to get the log from
        :param job_details: If not None, this should be the job details
        previously downloaded for that job. If None, the job details will be
        fetched from WML
        :return: The decoded log, or None
        """
        if job_details is None:
            job_details = self.get_job_details(job_id, with_contents=True)
        for output_data in job_details['entity']['decision_optimization']['output_data']:
            if output_data['id'] == 'log.txt':
                output = output_data['content']
                output = self.decode_log(output)
                output = self.remove_empty_lines(output)
                return output
        return None

    def get_job_details(self, job_id, with_contents=False):
        ''' Get the job details for the given job
        :param job_id:
        :return: The job details
        '''
        client = self._get_or_make_client()
        self._logger.info(f'Fetching output...')
        job_details = client.deployments.get_job_details(job_id)
        self._logger.info(f'Done.')
        if with_contents:
            return job_details
        # Let's remove the largest, mostly useless, parts
        do = job_details['entity']['decision_optimization']
        for data in do['output_data']:
            data['content'] = '[not shown]'
        for data in do['input_data']:
            data['content'] = '[not shown]'
        do['solve_state']['latest_engine_activity'] = ['[not shown]']
        return job_details

    def delete_job(self, job_id, hard=False):
        ''' Delete the given job
        :param job_id: the job to be deleted
        :param hard: if False, cancel the job. If true, delete it completely
        '''
        client = self._get_or_make_client()
        self._logger.info(f'Deleting job...')
        job_details = client.deployments.delete_job(job_id, hard)
        self._logger.info(f'Done.')


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
        return job_details['metadata']['created_at']

    @staticmethod
    def _get_input_names_from_details(job_details):
        inputs = job_details['entity']['decision_optimization']['input_data']
        names = [input['id'] for input in inputs]
        return names

    def wait_for_job_end(self, job_id, print_activity=False):
        """Wait for the job to finish and return its status"""
        client = self._get_or_make_client()
        while True:
            job_details = client.deployments.get_job_details(job_id)
            do = job_details['entity']['decision_optimization']
            status = self._get_job_status_from_details(job_details)
            self._logger.info(f'Job status: {status}')
            if status in ['completed', 'failed', 'canceled']:
                break
            else:
                if print_activity:
                    # There may be a bit of log to look at
                    try:
                        # FIXME: Only print whatever was not printed before
                        activity = do['solve_state']['latest_engine_activity']
                        print(''.join(activity))
                    except KeyError:
                        # This must mean that no activity is available yet
                        pass
            sleep(self.JOB_END_SLEEP_DELAY)
        return status, job_details

    @staticmethod
    def get_file_as_data(path):
        with open(path, 'rb') as f:
            data = f.read()
        data = base64.b64encode(data)
        data = data.decode('UTF-8')
        return data

    def get_jobs(self):
        '''Return the list of tuples (id, status) for all jobs in the deployment'''
        client = self._get_or_make_client()
        job_details = client.deployments.get_job_details()
        result = []
        for job in job_details['resources']:
            status = self._get_job_status_from_details(job)
            id = self._get_job_id_from_details(job)
            created = self._get_creation_time_from_details(job)
            names = self._get_input_names_from_details(job)
            JobTuple = namedtuple('Job', ['status', 'id', 'created', 'names'])
            j = JobTuple(status=status, id=id, created=created, names=names)
            result.append(j)
        return result

    def create_job(self, path, deployment_id):
        client = self._get_or_make_client()
        cdd = client.deployments.DecisionOptimizationMetaNames
        solve_payload = {
            cdd.SOLVE_PARAMETERS: {
                'oaas.logAttachmentName': 'log.txt',
                'oaas.logTailEnabled': 'true',
                'oaas.includeInputData': 'false',
                'oaas.resultFormat': 'JSON'
            },
            cdd.INPUT_DATA: [
                {
                    'id': path,
                    'content': self.get_file_as_data(path)
                }
            ],
            cdd.OUTPUT_DATA: [
                {'id': '.*\.json'},
                {'id': '.*\.txt'}
            ]
        }
        self._logger.info(f'Creating the job...')
        job_details = client.deployments.create_job(deployment_id, solve_payload)
        self._logger.info(f'Done. Getting its id...')
        job_id = client.deployments.get_job_uid(job_details)
        return job_id

    def _get_deployment_id(self):
        """Create deployment if doesn't exist already, return its id"""

        if not self._space_id:
            self._space_id = self._get_space_id()

        logging.info(f'Getting deployments...')
        client = self._get_or_make_client()
        deployment_details = client.deployments.get_details()
        logging.info(f'Done.')
        resources = deployment_details['resources']
        deployment_name = self.DEPLOYMENT_NAME
        logging.info(f'Got the list. Looking for deployment named \'{deployment_name}\'')
        deployment_id = None
        for r in resources:
            if r['entity']['name'] == deployment_name:
                deployment_id = r['metadata']['id']
                logging.info(f'Found it.')
                break
        if deployment_id is not None:
            return deployment_id

        logging.info(f'This deployment doesn\'t exist yet. Creating it...')

        # We need a model to create a deployment
        model_id = self._get_model_id()

        # Create the deployment
        logging.info(f'Creating the deployment itself...')
        cdc = client.deployments.ConfigurationMetaNames
        meta_props = {
            cdc.NAME: deployment_name,
            cdc.DESCRIPTION: "Deployment for the Solve on WML Python script",
            cdc.BATCH: {},
            # FIXME: should be configurable
            cdc.HARDWARE_SPEC: {'name': 'S', 'nodes': 1}
        }
        deployment = client.deployments.create(artifact_uid=model_id, meta_props=meta_props)
        logging.info(f'Deployment created.')
        deployment_id = client.deployments.get_id(deployment)
        return deployment_id

    def _get_model_id(self):
        """Create an empty model"""
        client = self._get_or_make_client()
        model_name = self.MODEL_NAME
        crm = client.repository.ModelMetaNames
        model_metadata = {
            crm.NAME: model_name,
            crm.DESCRIPTION: "Model for the solve-on-wml script",
            # FIXME: Must default to latest version
            # FIXME: Should be configurable
            crm.TYPE: "do-cplex_12.10",
            # FIXME: should not be hard-coded
            crm.SOFTWARE_SPEC_UID:
                client.software_specifications.get_id_by_name("do_12.10")
        }
        logging.info(f'Creating the model...')
        model_details = client.repository.store_model(model='empty.zip',
                                                      meta_props=model_metadata)
        logging.info(f'Model created.')
        model_id = client.repository.get_model_id(model_details)
        logging.info(f'Model id: {model_id}')
        return model_id

    def _get_space_id(self):
        """Find the Space to use, create it if it doesn't exist"""
        if self._space_id:
            return self._space_id

        client = self._get_or_make_client()
        logging.info(f'Fetching existing spaces...')
        space_name = self.SPACE_NAME
        cos_resource_crn = ('crn:v1:bluemix:public:cloud-object-storage:global'
                            ':a/76260f9157016d38ed1b725fa796f7bc:'
                            '7df9ff41-d7db-4df7-9efa-b6fadcbb1228::')
        instance_crn = ('crn:v1:bluemix:public:pm-20:eu-de:a/'
                        '76260f9157016d38ed1b725fa796f7bc:'
                        '031c5823-a324-4f66-a585-c41a5734efe1::')
        csc = client.spaces.ConfigurationMetaNames
        metadata = {
            csc.NAME: space_name,
            csc.DESCRIPTION: space_name + ' description',
            csc.STORAGE: {
                "type": "bmcos_object_storage",
                "resource_crn": cos_resource_crn
            },
            csc.COMPUTE: {
                "name": "existing_instance_id",
                "crn": instance_crn
            }
        }

        space_details = client.spaces.get_details()
        resources = space_details['resources']
        logging.info(f'Got the list. Looking for space named \'{space_name}\'')
        space_id = None
        for r in resources:
            if r['entity']['name'] == space_name:
                space_id = r['metadata']['id']
                logging.info(f'Found it.')
                break
        if space_id is None:
            logging.info(f'This space doesn\'t exist yet. Creating it...')
            # Create the space
            space = client.spaces.store(meta_props=metadata)
            logging.info(f'Space created')
            space_id = client.spaces.get_id(space)
        logging.info(f'Space id: {space_id}')

        self._logger.info(f'Setting default space...')
        self._client.set.default_space(space_id)
        self._logger.info(f'Done.')

        self._space_id = space_id
        return space_id

    def _create_connexion(self):
        if self._client is not None:
            return
        logging.info(f'Creating the connexion...')
        self._client = APIClient(self._wml_credentials)
        logging.info(f'Creating the connexion succeeded.  Client version is {self._client.version}')
