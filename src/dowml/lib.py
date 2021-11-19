#  --------------------------------------------------------------------------
#  Source file provided under Apache License, Version 2.0, January 2004,
#  http://www.apache.org/licenses/
#  (c) Copyright Xavier Nodet, 2021
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#  --------------------------------------------------------------------------

import ast
import base64
import csv
import glob
import io
import logging
import os
import re
import sys
import tempfile
import time
import urllib.parse
from collections import namedtuple
from contextlib import contextmanager
from datetime import datetime
from functools import lru_cache
from operator import attrgetter

import requests
from ibm_watson_machine_learning.wml_client_error import WMLClientError, ApiRequestFailure
from packaging import version

from ibm_watson_machine_learning import APIClient

# WML Python API version with a fixed Assets.download function
WML_HAS_FIXED_DOWNLOAD = "1000.0.0"

LOGNAME = 'log.txt'


class Error(Exception):
    """Base class for all errors in this script"""
    pass


class InvalidCredentials(Error):
    """The WML credentials were not found, or incorrect"""
    pass


class SimilarNamesInJob(Error):
    """A job can't have two input files with the same name, irrespective of path"""
    pass


class NoCredentialsToCreateSpace(Error):
    """Need to create a space, but credentials are incomplete to allow that"""
    pass


#
# The WML API offers a way to choose which entities are returned when asking
# for job details. Unfortunately, that parameter is not surfaced in the WML
# Python API, and we have to patch the code in order to send the value we want.
#
# APIClient._params() is the function that creates the parameters for the REST
# call.  We replace it with our own function '_new_params' that (1) calls the
# original function and adds the filter we want, if we want one.
#

# The function will not be called with any parameter. So the filter we want to
# use, if any, must be set in a global variable.
_the_filter = None
# Backup of the original function, so that we can call it and restore it later.
_the_old_params = None


def _new_params():
    """Our new function to build a parameter list for the REST call.

    Called by the instance of APIClient itself."""
    global _the_old_params
    global _the_filter
    # Use the original code and get its output
    # noinspection PyCallingNonCallable
    result = _the_old_params()
    # Add the filter, if one is required
    if _the_filter:
        result['include'] = _the_filter
    return result


def _get_file_spec(path):
    force = False
    if path[0] == '+':
        force = True
        path = path[1:]
    basename = os.path.basename(path)
    return path, basename, force


class _CredentialsProvider:
    """"Reads credentials for a DOWMLLib instance. Stores them as a
    'credentials' attribute."""

    ENVIRONMENT_VARIABLE_NAME = 'DOWML_CREDENTIALS'
    ENVIRONMENT_VARIABLE_NAME_FILE = 'DOWML_CREDENTIALS_FILE'
    # The keys in the credentials
    APIKEY = 'apikey'
    TOKEN = 'token'
    SPACE_ID = 'space_id'
    SPACE_NAME = 'space_name'
    URL = 'url'
    REGION = 'region'
    COS_CRN = 'cos_resource_crn'
    ML_CRN = 'ml_instance_crn'
    REGION_TO_URL = {
        'us-south': 'https://us-south.ml.cloud.ibm.com',
        'eu-de': 'https://eu-de.ml.cloud.ibm.com',
        'eu-gb': 'https://eu-gb.ml.cloud.ibm.com',
        'jp-tok': 'https://jp-tok.ml.cloud.ibm.com',
    }

    def __init__(self, wml_credentials_file=None, api_key=None, wml_credentials_str=None, url=None, region=None):
        self._logger = logging.getLogger(self.__class__.__name__)

        if wml_credentials_str is None:
            if wml_credentials_file is not None:
                wml_credentials_str = self._read_wml_credentials_from_file(wml_credentials_file)
            else:
                wml_credentials_str = self._read_wml_credentials_from_env()
            if wml_credentials_str:
                self._logger.debug('Found credential string.')
        self.credentials = self.check_credentials(wml_credentials_str, api_key=api_key, url=url, region=region)

    def usage(self):
        print(f'${self.ENVIRONMENT_VARIABLE_NAME} should contain credentials as a Python dict of the form:')
        print(f'  {{\'{self.APIKEY}\': \'<apikey>\', \'{self.URL}\': \'https://us-south.ml.cloud.ibm.com\'}}')
        print(f'Or set ${self.ENVIRONMENT_VARIABLE_NAME_FILE} to the path to a file containing the same information.')

    def check_credentials(self, wml_cred_str, api_key, url, region):
        wml_credentials = None
        if wml_cred_str is not None:
            assert type(wml_cred_str) is str
            if not wml_cred_str:
                raise InvalidCredentials('WML credentials must not be an empty string.')
            wml_credentials = ast.literal_eval(wml_cred_str)
            assert type(wml_credentials) is dict
        if not wml_credentials:
            # If we don't find a credentials string through the environment, we will
            # assume that the parameters are enough to build one.
            wml_credentials = {}
        if api_key:
            wml_credentials[self.APIKEY] = api_key
        if url:
            wml_credentials[self.URL] = url
        if region:
            wml_credentials[self.REGION] = region
        if self.APIKEY not in wml_credentials and self.TOKEN not in wml_credentials:
            raise InvalidCredentials('API key (or token) must be specified.')
        if self.URL not in wml_credentials and self.REGION not in wml_credentials:
            raise InvalidCredentials('URL or region must be specified (but not both).')
        if self.APIKEY in wml_credentials:
            assert type(wml_credentials[self.APIKEY]) is str
        else:
            assert type(wml_credentials[self.TOKEN]) is str
        if region:
            if url:
                raise InvalidCredentials(f"You must not specify both '{self.URL}' and '{self.REGION}'.")
            wml_credentials[self.REGION] = region
            # Setting a region must clear the URL, otherwise there will be an
            # ambiguity (and therefore an error) just below
            wml_credentials.pop(self.URL, None)
        if self.REGION in wml_credentials:
            region = wml_credentials[self.REGION]
            if self.URL in wml_credentials:
                raise InvalidCredentials(f"WML credentials must not have both '{self.URL}' and '{self.REGION}'.")
            try:
                wml_credentials[self.URL] = self.REGION_TO_URL[region]
            except KeyError:
                raise InvalidCredentials(f"Unknown region '{region}'.")
            wml_credentials.pop(self.REGION)
        if url:
            # The url specified takes priority over the one in the credentials, if any.
            wml_credentials[self.URL] = url
        assert self.URL in wml_credentials
        assert type(wml_credentials[self.URL]) is str
        url = wml_credentials[self.URL]
        if not url:
            raise InvalidCredentials('URL must not be an empty string.')
        if url[-1] == '/':
            self._logger.warning('URL should not have a \'/\' at the end.')
            wml_credentials[self.URL] = url[:-1]
        self._logger.debug('Credentials have the expected structure.')
        return wml_credentials

    def _read_wml_credentials_from_env(self):
        """Return a string of credentials suitable for WML from the environment

        Raises InvalidCredentials if anything is wrong."""
        var_name = self.ENVIRONMENT_VARIABLE_NAME
        var_file_name = self.ENVIRONMENT_VARIABLE_NAME_FILE
        self._logger.debug(f'Looking for credentials in environment variable {var_name}...')
        wml_cred_str = None
        if var_name in os.environ:
            wml_cred_str = os.environ[var_name]
        elif var_file_name in os.environ:
            self._logger.debug(f'Looking for credentials file name in environment variable {var_file_name}...')
            wml_cred_str = self._read_wml_credentials_from_file(os.environ[var_file_name])
        return wml_cred_str

    def _read_wml_credentials_from_file(self, file):
        """Return the content of the file, assumed to be WML credentials"""
        self._logger.debug(f'Looking for credentials in file \'{file}\'...')
        with open(file) as f:
            wml_cred_str = f.read()
        return wml_cred_str


def version_is_greater(current, minimum):
    """Compare two 'vv.nn.pp' versions.

    Return True is the current version string is greater or equal to the
minimum string.  Assumes that each string is of type vv.nn.pp, with vv, nn and
pp being integers."""
    return version.parse(current) >= version.parse(minimum)


@contextmanager
def suppress_stdout():
    """"Sometimes it's nicer to not get printed output from APIClient"""
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


class DOWMLLib:
    """A Python client to run DO models on WML"""

    DOWML_PREFIX = 'dowml'
    space_name = f'{DOWML_PREFIX}-space'
    MODEL_NAME = f'{DOWML_PREFIX}-model'
    MODEL_TYPES = ['cplex', 'cpo', 'opl', 'docplex']
    DO_VERSION = '20.1'
    TSHIRT_SIZES = ['S', 'M', 'XL']
    DEPLOYMENT_NAME = f'{DOWML_PREFIX}-deployment'

    def __init__(self, wml_credentials_file=None,
                 api_key=None,
                 space_id=None,
                 url=None,
                 region=None,
                 tz=datetime.utcnow().astimezone().tzinfo):
        f"""Read and validate the WML credentials

        Args:
            wml_credentials_file: path to the file that contains the WML credentials.
            If None, they are read from the environment.
            space_id: the id of the space that should be used. If specified, this
            replaces the one in the credentials.
            url: the URL for the Machine Learning service to use. If specified, this
            replaces the one in the credentials. url and region can't be both specified.
            region: An alternative way to specify the URL for the Machine Learning service
            to use. If specified, the URL will be deduced from the region. This
            replaces the one in the credentials. url and region can't be both specified.
            Possible values for the region are {list(_CredentialsProvider.REGION_TO_URL.keys())}
            tz: timezone to use to display time, defaults to Python's default timezone."""

        self.model_type = self.MODEL_TYPES[0]
        """Type of model to solve"""
        self.tshirt_size = self.TSHIRT_SIZES[0]
        """Tee-shirt size for the hardware to solve the model"""
        self.do_version = self.DO_VERSION
        """DO engines version to use"""
        self.timelimit = None
        """Time limit for the solve, in seconds"""
        self.inputs = 'assets'
        """Type of inputs that the created jobs should use"""
        self.outputs = 'inline'
        """Type of outputs that the created jobs should use"""

        self._logger = logging.getLogger(self.__class__.__name__)

        cred_provider = _CredentialsProvider(wml_credentials_file, api_key=api_key, url=url, region=region)
        wml_credentials = cred_provider.credentials

        # A space name in the credentials changes the default
        if cred_provider.SPACE_NAME in wml_credentials:
            self._logger.debug('They contain a space name.')
            self.space_name = wml_credentials[cred_provider.SPACE_NAME]

        if cred_provider.SPACE_ID in wml_credentials:
            self._logger.debug('And they contain a space id.')

        # The space_id specified here takes precedence
        # over the one, if any, defined in the credentials
        if space_id:
            wml_credentials[cred_provider.SPACE_ID] = space_id

        self._wml_credentials = wml_credentials

        # We don't initialize the client at this time, because this is an
        # expensive operation.
        self._client = None
        self._space_id = None
        self._data_connection = None
        self.tz = tz

    @property
    def url(self):
        """The URL for the WML service instance that the library is connected to."""
        return self._wml_credentials[_CredentialsProvider.URL]

    @property
    def space_id(self):
        """The id for the deployment space that the library is connected to."""
        return self._space_id

    @property
    def inline(self):
        self._logger.warning('Attribute \'inline\' is deprecated: use \'inputs\' instead.')
        return self.inputs == 'inline'

    @inline.setter
    def inline(self, value):
        self._logger.warning('Attribute \'inline\' is deprecated: use \'inputs\' instead.')
        self.inputs = 'inline' if value else 'assets'

    def _create_client(self):
        """Create the Python APIClient instance"""
        assert self._client is None
        self._logger.debug('Creating the WML client...')
        # http://ibm-wml-api-pyclient.mybluemix.net/#api-for-ibm-cloud
        client = APIClient(self._wml_credentials)
        self._logger.info(f'Creating the client succeeded.  Client version is {client.version}.')
        self._logger.info(f'Client uses URL {self._wml_credentials[_CredentialsProvider.URL]}.')
        return client

    def _set_default_space(self):
        space_id_key = _CredentialsProvider.SPACE_ID
        if space_id_key in self._wml_credentials:
            space_id = self._wml_credentials[space_id_key]
            self._logger.debug(f'Using specified space \'{space_id}\'.')
        else:
            space_id = self._find_or_create_space()

        self._logger.debug('Setting default space...')
        self._client.set.default_space(space_id)
        self._space_id = space_id
        self._logger.debug('Done.')

    def _get_or_make_client(self):
        if self._client is None:
            self._client = self._create_client()
            # The client is pretty much useless when it doesn't yet have a
            # default space. So let's set it immediately.
            self._set_default_space()
        # It would seem natural to assert that self._space_id is not None.
        # But this fails when we are in unit-tests and we just set _client to
        # a mock object from outside, without also setting the _space_id.
        return self._client

    def solve(self, paths):
        """Solve the model, return the job id

        The model is sent as online data to WML (if 'inline yes') or is uploaded as a data asset
        to be reused later (default).

        :param paths: one or more pathname to the files to send, as a single
                      string, separated by space
        :return: The id of the submitted job
        """
        self._get_or_make_client()
        # As _get_deployment_id caches its results, it may happen that what it returns is
        # invalid. For example if the user deleted the deployment after the last solve.
        # So if we get an error about the deployment not existing, we clear the cache
        # and retry once.
        first_try = True
        while True:
            deployment_id = self._get_deployment_id()
            self._logger.info(f'Deployment id: {deployment_id}')
            try:
                job_id = self.create_job(paths, deployment_id)
                self._logger.info(f'Job id: {job_id}')
                return job_id
            except ApiRequestFailure as e:
                if first_try and b'deployment_does_not_exist' in e.args[1].content:
                    self._logger.warning('Deployment was not found. Clearing the cache and retrying...')
                    self._get_deployment_id_with_params.cache_clear()
                    first_try = False
                else:
                    if not first_try:
                        self._logger.warning('Clearing the cache didn\'t help...')
                    raise

    def client_data_asset_download(self, asset_id, filename):
        self._logger.debug(f'Downloading asset {asset_id} in {filename}...')
        with suppress_stdout():
            name = self._client.data_assets.download(asset_id, filename)
            if version_is_greater(self._client.version, WML_HAS_FIXED_DOWNLOAD):
                filename = name
            # else the return value is useless when filename is an absolute path
        self._logger.debug(f'Done saving {filename}.')

    def get_log(self, job_id):
        """Extract the engine log from the job.

        :param job_id: The id of the job to get the log from
        :return: The decoded log, or None
        """

        def _get_asset_content(asset_id):
            if version_is_greater(self._client.version, WML_HAS_FIXED_DOWNLOAD):
                return self._client.data_assets.download(asset_id).decode('ascii')
            else:
                with tempfile.TemporaryDirectory() as temp_dir_name:
                    filename = os.path.join(temp_dir_name, f'{asset_id}-log.txt')
                    self.client_data_asset_download(asset_id, filename)
                    with open(filename) as f:
                        content = f.read()
                        return content

        def _get_log_from_output_references(references):
            self._logger.debug(f'Looking for {LOGNAME} in output_data_references...')
            for ref in references:
                if ref.get('type') != 'data_asset':
                    continue
                if 'id' not in ref:
                    self._logger.warning('Ignoring data asset with no id.')
                    continue
                if ref['id'] == LOGNAME:
                    self._logger.debug('Found it.')
                    try:
                        asset_id = ref['location']['id']
                    except KeyError:
                        self._logger.error('Log data asset has no location/id information.')
                        break
                    self._logger.debug(f'This is asset {asset_id}.')
                    return _get_asset_content(asset_id)
            return None

        def _get_log_from_outputs(outputs):
            self._logger.debug(f'Looking for {LOGNAME} in output_data...')
            for output_data in outputs:
                if output_data['id'] == LOGNAME:
                    if 'content' not in output_data:
                        self._logger.error(f'Log without content for job {job_id}.')
                        continue
                    self._logger.debug('Found it. Decoding it...')
                    output = output_data['content']
                    output = self.decode_log(output)
                    output = self.remove_empty_lines(output)
                    self._logger.debug('Decoded the log.')
                    return output
            return None

        job_details = self.get_job_details(job_id, with_contents='log')
        try:
            do = job_details['entity']['decision_optimization']
        except KeyError:
            self._logger.warning('No decision_optimization structure available for this job.')
            return None
        # When we have references in the job, the 'output_data' may be an empty list
        if 'output_data' in do and do['output_data']:
            return _get_log_from_outputs(do['output_data'])
        elif 'output_data_references' in do:
            return _get_log_from_output_references(do['output_data_references'])
        else:
            self._logger.warning('No output_data or output_data_references structure available for this job.')
            return None

    def _parse_asset_references(self, details, key):

        def find_id_in_href(loc):
            href = loc.get('href') if loc else None
            if not href:
                return None
            path = urllib.parse.urlparse(href).path
            s = re.search('/v2/assets/(.*)', path)
            if s:
                return s.group(1)
            self._logger.warning(f'Could not decode href for asset \'{name}\'.')
            return None

        def find_id_in_id(loc):
            return loc.get('id') if loc else None

        try:
            refs = details['entity']['decision_optimization'][key]
        except KeyError:
            self._logger.debug(f'No \'{key}\' structure available for this job.')
            return {}
        result = {}
        for ref in refs:
            asset_type = ref.get('type')
            if asset_type != 'data_asset':
                self._logger.debug(f'Ignoring asset of unknown type \'{asset_type}\'.')
                continue
            name = ref.get('id')
            if not name:
                self._logger.warning('Found a data asset with no name.')
                continue
            self._logger.debug(f'Found a data asset named {name}.')
            location = ref.get('location')
            asset_id = find_id_in_id(location) or find_id_in_href(location)
            if asset_id:
                result[name] = asset_id
            else:
                self._logger.warning(f'Could not find asset id for asset \'{name}\'.')
        return result

    def get_output_asset_ids(self, details):
        """"Extract the output data asset ids from the job details.

        :param details: The details of the job to get the output from
        :return: A dict of outputs. Keys are the names of the outputs,
        and the corresponding value for each key is the id of the asset.
        """
        return self._parse_asset_references(details, 'output_data_references')

    def get_input_asset_ids(self, details):
        """"Extract the input data asset ids from the job details.

        :param details: The details of the job to get the output from
        :return: A dict of inputs. Keys are the names of the inputs,
        and the corresponding value for each key is the id of the asset.
        """
        return self._parse_asset_references(details, 'input_data_references')

    def get_output(self, details, csv_as_dataframe=None, tabular_as_csv=False):
        """Deprecated. Use get_outputs instead"""
        return self.get_outputs(details, csv_as_dataframe, tabular_as_csv)

    def _extract_inline_files_from_details(self, details, key, tabular_as_csv):
        try:
            files = details['entity']['decision_optimization'][key]
        except KeyError:
            self._logger.debug(f'No \'{key}\' structure available for this job.')
            return {}
        result = {}
        for output_data in files:
            name = output_data['id']
            if 'content' in output_data:
                # What we have here is a regular file, encoded
                self._logger.debug(f'Found a regular file named {name}.')
                content = self._extract_regular_file(output_data)
                result[name] = content
            elif ('values' in output_data and
                  'fields' in output_data and
                  name.lower().endswith('.csv')):
                self._logger.debug(f'Found a CSV file named {name}.')
                content = self._extract_csv_file(output_data, tabular_as_csv)
                result[name] = content
            else:
                self._logger.warning(f'Found an unknown file named {name}.')
                content = output_data
                result[name] = content
        return result

    def get_outputs(self, details, csv_as_dataframe=None, tabular_as_csv=False):
        """"Extract the inline outputs from the job details.

        :param details: The details of the job to get the outputs from
        :param csv_as_dataframe: Whether the content of a CSV file should be
        returned as a Pandas DataFrame or not. Deprecated: use tabular_as_csv
        instead
        :param tabular_as_csv: Whether tabular outputs should be returned as
        CSV file content instead of Pandas dataframe
        :return: A dict of outputs, with the names of the assets as keys, and
        the content as value, as bytes. We don't assume that the content is
        actually text.
        """
        if csv_as_dataframe is not None:
            # We have a non-default value for this deprecated parameter
            # Let's first check that the replacement parameter is at its default
            # value
            assert(tabular_as_csv is False)
            # Now we can replace it with the correct value
            tabular_as_csv = not csv_as_dataframe
        return self._extract_inline_files_from_details(details, 'output_data', tabular_as_csv)

    def get_inputs(self, details, tabular_as_csv=False):
        """"Extract the inline inputs from the job details.

        :param details: The details of the job to get the inputs from
        :param tabular_as_csv: Whether tabular inputs should be returned as
        CSV file content instead of Pandas dataframe
        :return: A dict of inputs, with the names of the assets as keys, and
        the content as value, as bytes. We don't assume that the content is
        actually text.
        """
        return self._extract_inline_files_from_details(details, 'input_data', tabular_as_csv)

    @staticmethod
    def _extract_csv_file(output_data, tabular_as_csv):
        if tabular_as_csv:
            content = io.StringIO()
            writer = csv.writer(content)
            writer.writerow(output_data['fields'])
            for r in output_data['values']:
                writer.writerow(r)
            content = content.getvalue().encode()
        else:
            import pandas
            content = pandas.DataFrame(output_data['values'],
                                       columns=output_data['fields'])
        return content

    @staticmethod
    def _extract_regular_file(output_data):
        content = output_data['content']
        content = content.encode('UTF-8')
        content = base64.b64decode(content)
        return content

    def get_job_details(self, job_id, with_contents=None):
        """ Get the job details for the given job

        :param job_id: The id of the job to look for
        :param with_contents: if 'names', the details returned include
        the input and output files names. If 'full', the content of these files
        is included as well. If 'log', the content only includes the output files
        :return: The job details
        """
        client = self._get_or_make_client()
        self._logger.debug('Fetching output...')
        output_filter = None
        if not with_contents:
            output_filter = 'solve_parameters,solve_state,status'
        elif with_contents == 'log':
            output_filter = 'output_data,output_data_references'
        job_details = self.client_get_job_details(client, job_id, output_filter)
        self._logger.debug('Done.')
        if with_contents != 'full' and with_contents != 'log':
            self.filter_large_chunks_from_details(job_details)
        return job_details

    @staticmethod
    def client_get_job_details(client, job_id, with_filter=None):
        # The filter doesn't work correctly if it contains spaces
        assert(not with_filter or (with_filter.find(' ') == -1))
        if version_is_greater(client.version, "1.0.154"):
            result = client.deployments.get_job_details(job_id, with_filter)
        else:
            global _the_filter
            global _the_old_params
            # Save the filter in a global variable for our new function to find it
            _the_filter = with_filter
            # Save the pointer to the original code
            # noinspection PyProtectedMember
            _the_old_params = client._params
            # and replace it with our new function
            client._params = _new_params
            try:
                result = client.deployments.get_job_details(job_id)
            finally:
                # Put back the original code
                client._params = _the_old_params
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

    def _delete_data_assets(self, job_details):
        job_id = job_details['metadata']['id']
        try:
            odr = job_details['entity']['decision_optimization']['output_data_references']
        except KeyError:
            odr = []
        for output in odr:
            if output.get('type') != 'data_asset':
                continue
            if 'location' not in output:
                self._logger.error(f'Missing \'location\' in details for job {job_id}.')
            elif 'id' not in output['location']:
                self._logger.error(f'Missing \'location.id\' in details for job {job_id}.')
            else:
                data_asset_id = output['location']['id']
                self._logger.debug(f'Deleting data asset {data_asset_id}...')
                try:
                    self._client.data_assets.delete(data_asset_id)
                    self._logger.debug('Done.')
                except WMLClientError:
                    self._logger.error('Exception raised while trying to delete the asset.', exc_info=True)

    def _client_deployments_delete_job(self, job_id, hard, job_details):
        """Deletes the platform run, so that the deployment job is deleted as well.

        If only calling
           client.deployments.delete_job(job_id, hard)
        the 'run' of the 'platform job' on the Watson Studio side is left,
        and it will never be deleted.
        On the other hand, deleting the run on the WS side also deletes the
        deployment job on the WML side. So let's do that.
        """
        client = self._get_or_make_client()
        wml_url = self._wml_credentials['url']
        # We don't want to (try to) delete the WS run if we only cancel the job
        # Instead, we skip everything until calling deployments.delete_job(True)
        everything_ok_so_far = hard
        ws_url = client.PLATFORM_URLS_MAP.get(wml_url)
        if everything_ok_so_far and not ws_url:
            self._logger.error(f'Unknown Watson Studio URL for WML URL {wml_url}.')
            everything_ok_so_far = False
        if everything_ok_so_far:
            try:
                platform_job_id = job_details['entity']['platform_job']['job_id']
                platform_run_id = job_details['entity']['platform_job']['run_id']
                url = f'{ws_url}/v2/jobs/{platform_job_id}/runs/{platform_run_id}?space_id={self.space_id}'
            except KeyError:
                self._logger.error('Watson Studio job id or run id not found in WML job details.')
                everything_ok_so_far = False
        if everything_ok_so_far:
            # noinspection PyUnboundLocalVariable
            self._logger.debug(f'Trying to delete run {platform_run_id} of Watson Studio job {platform_job_id}...')
            # noinspection PyProtectedMember
            # noinspection PyUnboundLocalVariable
            r = requests.delete(url, headers={'Authorization': f'Bearer {client.service_instance._get_token()}',
                                              'Content-Type': 'application/json',
                                              'cache-control': 'no-cache'})
            if r.status_code != 204:
                self._logger.error(f'Error when trying to delete the Watson Studio run. {r.text}')
                everything_ok_so_far = False
        if not everything_ok_so_far:
            if hard:
                self._logger.error('Could not delete the Watson Studio run. Deleting the WML job deployment instead...')
            # else:
            #   we just wanted to cancel the job, so there's nothing to warn against
            client.deployments.delete_job(job_id, hard)

    def delete_job(self, job_id, hard=True):
        """Delete the given job.

        :param job_id: the job to be deleted
        :param hard: deprecated. Use cancel_job instead of passing False
        """
        if not hard:
            self.cancel_job(job_id)
            return
        self._logger.debug(f'Deleting data assets (if any) for job {job_id}...')
        job_details = self.get_job_details(job_id, with_contents='names')
        self._delete_data_assets(job_details)
        self._logger.debug(f'Done. Deleting job {job_id}...')
        self._client_deployments_delete_job(job_id, True, job_details)
        self._logger.debug('Done.')

    def cancel_job(self, job_id):
        """Cancel the given job.

        :param job_id: the job to be canceled
        """
        self._logger.debug(f'Canceling job {job_id}...')
        self._client_deployments_delete_job(job_id, False, None)
        self._logger.debug('Done.')

    def decode_log(self, output):
        """Decode the engine log coming from DO4WML.

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
        """Remove empty lines from the log.

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
    def _get_creation_time_from_details(job_details, tz):
        created = job_details['metadata']['created_at']
        if created[-1] == 'Z':
            # A suffix of Z is not understood by isoformat. Let's replace
            # it with one that's understood
            created = created[:-1] + '+00:00'
            dt = datetime.fromisoformat(created)
            # Transform to local time
            dt = dt.astimezone(tz)
            # Remove timezone information so that ...
            dt = dt.replace(tzinfo=None)
            # ... just naively prints local time
            created = dt.isoformat(sep=' ', timespec='seconds')
        else:
            # Let's not mess with what we don't know
            pass
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
        """Wait for the job to finish, return its status and details as a tuple.

        If print_activity is True, some information is printed in the console."""

        class ProgressiveDelay:
            def __init__(self):
                self.delays = [2, 2, 2, 2, 2, 2, 3, 5, 5, 5, 5, 5, 5, 5, 5, 5,
                               10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 30]

            def wait(self):
                delay = self.delays[0]
                if len(self.delays) > 1:
                    self.delays.pop(0)
                assert (2 <= delay <= 30)
                time.sleep(delay)

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                pass

        class StatusLogger:
            def __init__(self, initial_state):
                self.last_state = initial_state
                print(initial_state, end='', flush=True)

            def log_state(self, state):
                if state == self.last_state:
                    print('.', end='', flush=True)
                else:
                    if self.last_state != '':
                        print('')
                    # else: if state was empty, no need to end the line
                    print(state, end='', flush=True)
                    self.last_state = state

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                pass

        client = self._get_or_make_client()
        with StatusLogger('') as status_logger, \
                ProgressiveDelay() as delayer:
            while True:
                job_details = self.client_get_job_details(client, job_id, with_filter='solve_state,status')
                do = job_details['entity']['decision_optimization']
                status = self._get_job_status_from_details(job_details)
                self._logger.info(f'Job status: {status}')
                if print_activity:
                    status_logger.log_state(f'Job is {status}.')
                if status in ['completed', 'failed', 'canceled']:
                    break
                if print_activity:
                    # There may be a bit of log to look at
                    try:
                        activity = do['solve_state']['latest_engine_activity']
                        if activity:
                            # Because of the StatusLogger, we're not at the beginning of a line
                            print('')
                            # We are joining the lines in the activity with a CR,
                            # only to remove them if they were already included...
                            act = '\n'.join(activity)
                            act = self.remove_empty_lines(act)
                            print(act)
                    except KeyError:
                        # This must mean that no activity is available yet
                        pass
                delayer.wait()
        if print_activity:
            # The status_logger printed something, but didn't end the line yet
            print('')
        return status, job_details

    @staticmethod
    def get_file_as_data(path):
        """Return the base-64 encoded content of a file."""
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
            self._logger.warning('Error while fetching type of a job!')
            return '?????'

    def _get_version_from_details(self, job):
        try:
            deployment_id = job['entity']['deployment']['id']
            deployment = self._get_deployment_from_id(deployment_id)
            model_id = deployment['entity']['asset']['id']
            model = self._get_model_definition_from_id(model_id)
            deployment_type = model['entity']['wml_model']['type']
            match = re.fullmatch(r"do-....*_([0-9.]*)", deployment_type)
            engine_version = '?????'
            if match:
                engine_version = match.group(1)
            return engine_version
        except KeyError:
            # Something changed. But let's not fail just for that
            self._logger.warning('Error while fetching version of a job!')
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
            self._logger.warning('Error while fetching size of a job!')
            return '?'

    def get_jobs(self):
        """Return the list of tuples (status, id, ...) for all jobs in the deployment."""
        client = self._get_or_make_client()
        self._logger.debug('Getting job details...')
        job_details = client.deployments.get_job_details()
        self._logger.debug('Done.')
        self._logger.debug('Getting information about deployments and models...')
        result = []
        for job in job_details['resources']:
            status = self._get_job_status_from_details(job)
            job_id = self._get_job_id_from_details(job)
            created = self._get_creation_time_from_details(job, self.tz)
            names = self._get_input_names_from_details(job)
            deployment_type = self._get_type_from_details(job)
            engine_version = self._get_version_from_details(job)
            size = self._get_size_from_details(job)
            JobTuple = namedtuple('Job', ['status', 'id', 'created', 'names', 'type', 'version', 'size'])
            j = JobTuple(status=status, id=job_id, created=created, names=names,
                         type=deployment_type, version=engine_version, size=size)
            result.append(j)
        result.sort(key=attrgetter('created'))
        self._logger.debug('Done.')
        return result

    def parse_paths(self, paths):
        """Expand wildcards that may appear in the input assets list."""
        self._logger.debug(f'Parsing input list: {paths}')
        globbed = []
        for path in paths.split():
            # Let's first get rid of the 'force' flag that glob
            # would not understand
            path, _, force = _get_file_spec(path)
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

    def create_inputs(self, paths, cdd_inputdata, solve_payload):
        # First deal with wildcards
        globbed = self.parse_paths(paths)
        # And let's now create the inputs from these files
        names = []
        for path in globbed:
            path, basename, force = _get_file_spec(path)
            if basename in names:
                raise SimilarNamesInJob(basename)
            names.append(basename)
            if self.inputs == 'inline':
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
                        "href": "/v2/assets/" + data_asset_id + "?space_id=" + self.space_id
                    }
                }
            solve_payload[cdd_inputdata].append(input_data)

    def create_job(self, paths, deployment_id):
        """Create a deployment job (aka a run) and return its id."""
        client = self._get_or_make_client()
        cdd = client.deployments.DecisionOptimizationMetaNames
        assert(self.outputs == 'inline' or self.outputs == 'assets')
        cdd_outputdata = cdd.OUTPUT_DATA
        if self.outputs == 'assets':
            cdd_outputdata = cdd.OUTPUT_DATA_REFERENCES
        # Assume we use inline data (i.e. content in the job request)
        cdd_inputdata = cdd.INPUT_DATA
        if self.inputs == 'assets':
            # But if we don't want inline data, we have to submit
            # input references instead
            cdd_inputdata = cdd.INPUT_DATA_REFERENCES
        solve_payload = {
            cdd.SOLVE_PARAMETERS: {
                'oaas.logAttachmentName': 'log.txt',
                'oaas.logTailEnabled': 'true',
                'oaas.resultsFormat': 'JSON'
            },
            cdd_inputdata: [],
            cdd_outputdata: [
                {'id': '.*'}
            ]
        }
        if self.outputs == 'assets':
            out = solve_payload[cdd_outputdata][0]
            out['type'] = 'data_asset'
            # PyCharm assumes that, because we added a string in the dict on
            # the previous line, we should only add strings in the same dict.
            # But this is not how WML does...
            # noinspection PyTypeChecker
            out['connection'] = {}
            # noinspection PyTypeChecker
            out['location'] = {'name': '${job_id}/${attachment_name}'}
        if self.timelimit:
            params = solve_payload[cdd.SOLVE_PARAMETERS]
            params['oaas.timeLimit'] = 1000 * self.timelimit
        self.create_inputs(paths, cdd_inputdata, solve_payload)
        self._logger.debug('Creating the job...')
        if self.inputs == 'inline':
            self._logger.debug('Data is inline. Let\'s not print the payload...')
        else:
            self._logger.debug(repr(solve_payload))
        dt = datetime.now()
        job_details = client.deployments.create_job(deployment_id, solve_payload)
        submit_time = (datetime.now() - dt).total_seconds()
        self._logger.debug(f'Done in {submit_time}. Getting its id...')
        job_id = client.deployments.get_job_uid(job_details)
        return job_id

    def _get_deployment_id(self):
        # Which deployment we want depends on a number of configuration values
        # in the library.  In order for the cache to work correctly, and not always
        # return the same deployment id, the cached function must be given these
        # values as parameters.
        return self._get_deployment_id_with_params(self.DEPLOYMENT_NAME,
                                                   self.model_type,
                                                   self.do_version,
                                                   self.tshirt_size)

    @lru_cache
    def _get_deployment_id_with_params(self, deployment_name_prefix, model_type, do_version, tshirt_size):
        # The point of this forwarding function is to allow testing.
        # Specifically, counting the number of calls to the _cached
        # function, and deciding what it returns.
        # Mocking this function would remove the lru_cache...
        return self._get_deployment_id_with_params_cached(deployment_name_prefix, model_type, do_version, tshirt_size)

    def _get_deployment_id_with_params_cached(self, deployment_name_prefix, model_type, do_version, tshirt_size):
        """Create the deployment if doesn't exist already, return its id."""
        self._logger.debug('Getting deployments...')
        client = self._get_or_make_client()
        deployment_details = client.deployments.get_details()
        self._logger.debug('Done.')
        resources = deployment_details['resources']
        deployment_name = f'{deployment_name_prefix}-{model_type}-{do_version}-{tshirt_size}'
        self._logger.debug(f'Got the list. Looking for deployment named \'{deployment_name}\'')
        deployment_id = None
        for r in resources:
            if r['entity']['name'] == deployment_name:
                deployment_id = r['metadata']['id']
                self._logger.debug('Found it.')
                break
        if deployment_id is not None:
            return deployment_id

        self._logger.debug('This deployment doesn\'t exist yet. Creating it...')
        deployment_id = self._create_deployment(deployment_name)
        return deployment_id

    def _create_deployment(self, deployment_name):
        # We need a model to create a deployment
        model_id = self._get_model_id()
        # Create the deployment
        self._logger.debug('Creating the deployment itself...')
        client = self._get_or_make_client()
        cdc = client.deployments.ConfigurationMetaNames
        meta_props = {
            cdc.NAME: deployment_name,
            cdc.DESCRIPTION: "Deployment for the Solve on WML Python script",
            cdc.BATCH: {},
            cdc.HARDWARE_SPEC: {'name': self.tshirt_size, 'num_nodes': 2}
        }
        deployment = client.deployments.create(artifact_uid=model_id, meta_props=meta_props)
        self._logger.debug('Deployment created.')
        deployment_id = client.deployments.get_id(deployment)
        return deployment_id

    def _get_model_id(self):
        """Create an empty model if one doesn't exist, return its id."""
        self._logger.debug('Getting models...')
        client = self._get_or_make_client()
        details = client.repository.get_details()
        self._logger.debug('Done.')
        resources = details['models']['resources']
        model_name = f'{self.MODEL_NAME}-{self.model_type}-{self.do_version}'
        self._logger.debug(f'Got the list. Looking for model named \'{model_name}\'...')
        model_id = None
        for r in resources:
            if r['metadata']['name'] == model_name:
                model_id = r['metadata']['id']
                self._logger.debug('Found it.')
                self._logger.debug(f'Model id: {model_id}')
                break
        if model_id is None:
            self._logger.debug('This model doesn\'t exist yet. Creating it...')
            model_id = self._create_model(model_name)
        return model_id

    def get_available_versions(self):
        """Return the list of available DO versions on the platform."""
        client = self._get_or_make_client()
        target_version = "1.0.92"
        if not version_is_greater(client.version, target_version):
            return [f'Error: need WML client version {target_version} or better to retrieve available versions']
        available_versions = []
        for s in client.software_specifications.get_details()['resources']:
            name = s['metadata']['name']
            match = re.fullmatch(r"do_([0-9.]*)", name)
            if match:
                available_versions.append(match.group(1))
        return available_versions

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
        self._logger.debug('Model created.')
        model_id = client.repository.get_model_id(model_details)
        self._logger.debug(f'Model id: {model_id}')
        return model_id

    def _find_or_create_space(self):
        """Find the Space to use from its name, create it if it doesn't exist."""
        assert self._client
        client = self._client
        self._logger.debug('Fetching existing spaces...')
        space_details = client.spaces.get_details()
        resources = space_details['resources']
        self._logger.debug(f'Got the list. Looking for space named \'{self.space_name}\'...')
        space_id = None
        for r in resources:
            if r['entity']['name'] == self.space_name:
                space_id = r['metadata']['id']
                self._logger.debug('Found it.')
                break
        if space_id is None:
            self._logger.debug('This space doesn\'t exist yet. Creating it...')
            # Prepare necessary information

            wml_credentials = self._wml_credentials
            cos_crn = _CredentialsProvider.COS_CRN
            ml_crn = _CredentialsProvider.ML_CRN
            if cos_crn not in wml_credentials or ml_crn not in wml_credentials:
                raise NoCredentialsToCreateSpace(f'WML credentials do not contain the information necessary '
                                                 f'to create a deployment space. \nMissing \'{cos_crn}\' '
                                                 f'and/or \'{ml_crn}\'.')
            assert type(wml_credentials[cos_crn]) is str
            assert type(wml_credentials[ml_crn]) is str

            csc = client.spaces.ConfigurationMetaNames
            metadata = {
                csc.NAME: self.space_name,
                csc.DESCRIPTION: self.space_name + ' description',
                csc.STORAGE: {
                    "type": "bmcos_object_storage",
                    "resource_crn": self._wml_credentials[cos_crn]
                },
                csc.COMPUTE: {
                    "name": "existing_instance_id",
                    "crn": self._wml_credentials[ml_crn]
                }
            }
            # Create the space
            # We want the space to be ready as soon as the code returns from
            # spaces.store(), so we use background_mode=False. In addition, this
            # gives us error checking which doesn't happen in default mode.
            space_details = client.spaces.store(meta_props=metadata, background_mode=False)
            state = space_details['entity']['status'].get('state')
            self._logger.debug(f'Space created, with state={state}.')
            space_id = client.spaces.get_uid(space_details)
        self._logger.info(f'Space id: {space_id}')
        return space_id

    def _get_asset_details(self):
        """Return the list of all the data assets in the space."""
        client = self._get_or_make_client()
        # This is the first version where data_assets.get_details() works
        assert(version_is_greater(client.version, "1.0.95.1"))
        results = client.data_assets.get_details()['resources']
        return results

    def _find_asset_id_by_name(self, name):
        """Looks for a data asset with the given name, returns its id, or None"""
        assets = self._get_asset_details()
        for asset in assets:
            metadata = asset['metadata']
            if metadata['name'] == name:
                return metadata['asset_id']
        return None

    def create_asset(self, path, basename):
        """Create a data asset with the given name.

        A Watson Studio data asset is an entity that mimicks a file."""
        client = self._get_or_make_client()
        asset_details = client.data_assets.create(basename, path)
        return asset_details['metadata']['guid']

    def delete_asset(self, uid):
        """Delete an existing asset. Return True if ok, False if not."""
        client = self._get_or_make_client()
        status = client.data_assets.delete(uid)
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
            self._logger.debug('Creating new asset with local content.')
            asset_to_delete = data_asset_id
        else:
            self._logger.debug('No, creating the data asset.')
        data_asset_id = self.create_asset(path, basename)
        self._logger.debug('Done.')
        if asset_to_delete:
            self._logger.debug('Deleting the old data asset.')
            if self.delete_asset(asset_to_delete):
                self._logger.debug('Done.')
            else:
                self._logger.warning('Could not delete pre-existing asset.')
        return data_asset_id
