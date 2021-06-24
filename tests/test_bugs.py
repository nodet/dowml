import datetime
import pprint
from collections import namedtuple
from logging import Logger
from unittest import TestCase, main, mock
from unittest.mock import Mock, ANY, call

from ibm_watson_machine_learning import APIClient
from ibm_watson_machine_learning.Set import Set
from ibm_watson_machine_learning.deployments import Deployments
from ibm_watson_machine_learning.spaces import Spaces

from dowml import DOWMLInteractive, \
    CommandNeedsJobID, CommandNeedsNonNullInteger, CommandNeedsBool
from dowmllib import DOWMLLib

TEST_CREDENTIALS_FILE_NAME = 'tests/test_credentials.txt'
EXPECTED = 'expected'


class Test_GH40(TestCase):
    # Creation time is incorrect in the 'jobs' output
    # https://github.ibm.com/xavier-nodet/dowml/issues/40

    def setUp(self) -> None:
        lib = DOWMLLib(TEST_CREDENTIALS_FILE_NAME)
        lib.tz = datetime.timezone(datetime.timedelta(hours=1))
        lib._logger = Mock(spec=Logger)
        lib._client = Mock(spec=APIClient)
        lib._client.set = Mock(spec=Set)
        lib._client.deployments = Mock(spec=Deployments)
        lib._client.spaces = Mock(spec=Spaces)
        lib._client.spaces.get_details.return_value = {'resources': []}
        self.lib = lib

    def test_jobs_should_display_local_time(self):
        self.lib._client.deployments.get_job_details.return_value = {'resources': [
            # One job
            {
                'entity': {'decision_optimization': {'status': {'state': 'state'}}},
                'metadata': {
                    'id': 'id',
                    'created_at': '2021-05-02T15:58:02Z'
                }
            }
        ]}
        self.assertEqual('2021-05-02 16:58:02', self.lib.get_jobs()[0].created)


if __name__ == '__main__':
    main()