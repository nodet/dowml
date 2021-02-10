""""Send a CPLEX file to WML and solve it.

For now, just accepts a path and sends that, assuming that's a model
The code is mostly based on
https://dataplatform.cloud.ibm.com/exchange/public/entry/view/50fa9246181026cd7ae2a5bc7e4ac7bd"""
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


class DOWMLClient:
    """A Python client to run DO models on WML"""

    ENVIRONMENT_VARIABLE_NAME = 'WML_CREDENTIALS'
    # FIXME: It must be possible to not repeat the class name here
    SPACE_NAME = 'DOWMLClient-space'
    MODEL_NAME = 'DOWMLClient-model'
    DEPLOYMENT_NAME = 'DOWMLClient-deployment'

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

    def solve(self, path):
        """Solve the model.

        The model is sent as online data to WML.

        Args:
            path: pathname to the file to solve"""
        self.create_connexion()
        space_id = self.get_space_id()

        self._logger.info(f'Setting default space...')
        self._client.set.default_space(space_id)
        self._logger.info(f'Done.')

        deployment_id = self.get_deployment_id()
        self._logger.info(f'Deployment id: {deployment_id}')

    def get_deployment_id(self):
        """Create deployment if doesn't exist already, return its id"""
        logging.info(f'Getting deployments...')
        client = self._client
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
        model_id = self.get_model_id()

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

    def get_model_id(self):
        """Find the (empty) model to be deployed, or create it"""
        client = self._client
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
                client.software_specifications.get_uid_by_name("do_12.10")
        }
        logging.info(f'Creating the model...')
        # FIXME: look for the model before creating one
        model_details = client.repository.store_model(model='empty.zip',
                                                      meta_props=model_metadata)
        logging.info(f'Model created.')
        model_id = client.repository.get_model_id(model_details)
        logging.info(f'Model id: {model_id}')
        return model_id

    def get_space_id(self):
        """Find the Space to use, create it if it doesn't exist"""
        client = self._client
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
        return space_id

    def create_connexion(self):
        if self._client is not None:
            return
        logging.info(f'Creating the connexion...')
        self._client = APIClient(self._wml_credentials)
        logging.info(f'Creating the connexion succeeded.  Client version is {self._client.version}')


def main():
    parser = argparse.ArgumentParser(description='Send a CPLEX model for solving on WML')
    parser.add_argument(metavar='model', dest='model',
                        help='Name of the model to solve')
    parser.add_argument('--wml-cred-file',
                        help='Name of the file from which to read WML credentials. If not specified, \
credentials are read from an environment variable')
    args = parser.parse_args()

    logging.basicConfig(force=True, format='%(asctime)s %(message)s', level=logging.INFO)

    client = DOWMLClient(args.wml_cred_file)
    client.solve(args.model)


if __name__ == '__main__':
    main()
