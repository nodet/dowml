import datetime
from logging import Logger
from unittest import TestCase, main
from unittest.mock import Mock

import requests
from ibm_watson_machine_learning import APIClient
from ibm_watson_machine_learning.Set import Set
from ibm_watson_machine_learning.deployments import Deployments
from ibm_watson_machine_learning.spaces import Spaces

from dowml.dowmllib import DOWMLLib
from dowml.interactive import DOWMLInteractive, main_loop

TEST_CREDENTIALS_FILE_NAME = 'test_credentials.txt'
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


class TestTimeoutsShouldNotStopDowml(TestCase):
    """https://github.com/nodet/dowml/issues/27"""

    def setUp(self) -> None:
        self.cli = DOWMLInteractive(TEST_CREDENTIALS_FILE_NAME)
        self.cli.lib = Mock(spec=DOWMLLib)

    def test_main_loop_doesnt_throw_on_timeout(self):
        instance = Mock(spec=DOWMLInteractive)
        # main_loop restarts if there is an error. Therefore, if catching an
        # exception, we restart.  And thus we need a second side_effect that
        # will not be an error and will stop the loop.
        #
        # Note that raising a ConnectTimeout didn't correctly test that we would
        # catch a requests.exception.TimeOut, because it also inherits from
        # ConnectionError and that one was already caught.
        instance.cmdloop = Mock(side_effect=[requests.exceptions.ReadTimeout(), None])
        try:
            main_loop(instance, [], False)
        except requests.exceptions.ConnectTimeout:
            self.fail("Expected main_loop to catch requests.exceptions.ConnectTimeout")


if __name__ == '__main__':
    main()
