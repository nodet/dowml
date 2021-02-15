from logging import Logger
from unittest.mock import Mock

from ibm_watson_machine_learning import APIClient
from ibm_watson_machine_learning.deployments import Deployments

from dowmlclient import DOWMLClient
from unittest import TestCase, main


class TestSolve(TestCase):

    def setUp(self) -> None:
        client = DOWMLClient('test_credentials.txt')
        client._logger = Mock(spec=Logger)
        client._client = Mock(spec=APIClient)
        client._client.deployments = Mock(spec=Deployments)
        client._client.deployments.DecisionOptimizationMetaNames = Mock()
        client.get_file_as_data = lambda path: 'base-64-content'
        client._space_id = 'space-id'
        client._get_deployment_id = Mock(spec=DOWMLClient._get_deployment_id)
        self.client = client

    def test_solve_multiple_files(self):
        self.client.solve('afiro.mps', False)
        self.client._client.deployments.create_job.assert_called_once()


if __name__ == '__main__':
    main()
