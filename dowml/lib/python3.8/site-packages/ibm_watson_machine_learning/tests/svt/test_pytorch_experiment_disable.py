import unittest
import time
import io
import sys
import logging
from ibm_watson_machine_learning.experiments import Experiments
from preparation_and_cleaning import *


class TestWMLClientWithKeras(unittest.TestCase):
    deployment_uid = None
    definition_1_uid = None
    definition_2_uid = None
    definition_1_url = None
    definition_2_url = None
    experiment_uid = None
    experiment_run_uid = None
    trained_model_uid = None
    train_run_id = None
    scoring_url = None
    logger = logging.getLogger(__name__)

    @classmethod
    def setUpClass(self):
        TestWMLClientWithKeras.logger.info("Service Instance: setting up credentials")
        self.wml_credentials = get_wml_credentials()
        self.client = get_client()
        self.cos_resource = get_cos_resource()
        self.bucket_names = prepare_cos(self.cos_resource, data_code=BOSTON)

    @classmethod
    def tearDownClass(self):
        clean_cos(self.cos_resource, self.bucket_names)

    def test_01_service_instance_details(self):
        TestWMLClientWithKeras.logger.info("Check client ...")
        self.assertTrue(self.client.__class__.__name__ == 'APIClient')

        TestWMLClientWithKeras.logger.info("Getting instance details ...")
        details = self.client.service_instance.get_details()
        TestWMLClientWithKeras.logger.debug(details)

        self.assertTrue("published_models" in str(details))
        self.assertEqual(type(details), dict)

    def test_02_save_definition1(self):
        TestWMLClientWithKeras.logger.info("Save model definition ...")

        self.client.repository.DefinitionMetaNames.show()

        metadata = {
            self.client.repository.DefinitionMetaNames.NAME: "PyTorch model definition",
            self.client.repository.DefinitionMetaNames.AUTHOR_NAME: "John Smith",
            self.client.repository.DefinitionMetaNames.AUTHOR_EMAIL: "js@js.com",
            self.client.repository.DefinitionMetaNames.FRAMEWORK_NAME: "pytorch",
            self.client.repository.DefinitionMetaNames.FRAMEWORK_VERSION: "0.3",
            self.client.repository.DefinitionMetaNames.RUNTIME_NAME: "python",
            self.client.repository.DefinitionMetaNames.RUNTIME_VERSION: "3.5",
            self.client.repository.DefinitionMetaNames.EXECUTION_COMMAND: "python3 torch_check.py"
        }

        model_content_path = './artifacts/torch_train.zip'
        definition_details = self.client.repository.store_definition(training_definition=model_content_path, meta_props=metadata)
        TestWMLClientWithKeras.definition_1_url = self.client.repository.get_definition_url(definition_details)
        TestWMLClientWithKeras.definition_1_uid = self.client.repository.get_definition_uid(definition_details)
        TestWMLClientWithKeras.logger.info("Saved model definition url: " + str(TestWMLClientWithKeras.definition_1_url))

    def test_03_save_definition2(self):
        TestWMLClientWithKeras.logger.info("Save model definition ...")

        self.client.repository.DefinitionMetaNames.show()

        metadata = {
            self.client.repository.DefinitionMetaNames.NAME: "PyTorch model definition",
            self.client.repository.DefinitionMetaNames.AUTHOR_NAME: "John Smith",
            self.client.repository.DefinitionMetaNames.AUTHOR_EMAIL: "js@js.com",
            self.client.repository.DefinitionMetaNames.FRAMEWORK_NAME: "pytorch",
            self.client.repository.DefinitionMetaNames.FRAMEWORK_VERSION: "0.3",
            self.client.repository.DefinitionMetaNames.RUNTIME_NAME: "python",
            self.client.repository.DefinitionMetaNames.RUNTIME_VERSION: "3.5",
            self.client.repository.DefinitionMetaNames.EXECUTION_COMMAND: "python3 torch_check.py"
        }

        model_content_path = './artifacts/torch_train.zip'
        definition_details = self.client.repository.store_definition(training_definition=model_content_path, meta_props=metadata)
        TestWMLClientWithKeras.definition_2_url = self.client.repository.get_definition_url(definition_details)
        TestWMLClientWithKeras.definition_2_uid = self.client.repository.get_definition_uid(definition_details)
        TestWMLClientWithKeras.logger.info("Saved model definition url: " + str(TestWMLClientWithKeras.definition_2_url))

    def test_04_get_definition_details(self):
        TestWMLClientWithKeras.logger.info("Getting definition details ...")
        details = self.client.repository.get_definition_details()
        TestWMLClientWithKeras.logger.info(details)
        self.assertTrue('PyTorch model definition' in str(details))
        TestWMLClientWithKeras.logger.info("List definitions")
        self.client.repository.list_definitions()

    def test_05_save_experiment(self):
        metadata = {
                    self.client.repository.ExperimentMetaNames.NAME: "xxx",
                    self.client.repository.ExperimentMetaNames.AUTHOR_EMAIL: "js@js.com",
                    self.client.repository.ExperimentMetaNames.EVALUATION_METHOD: "binary",
                    self.client.repository.ExperimentMetaNames.EVALUATION_METRICS: [],
                    self.client.repository.ExperimentMetaNames.TRAINING_DATA_REFERENCE: {
                        "type": "s3",
                        "connection": {
                            "access_key_id": "xxx",
                            "secret_access_key": "xxx",
                            "endpoint_url": "https://xxx.ibm.com"
                        },
                        "source": {
                            "bucket": "xxx",
                            "key": "xxx"
                        }
                    },
                    self.client.repository.ExperimentMetaNames.TRAINING_RESULTS_REFERENCE: {
                        "type": "s3",
                        "connection": {
                            "access_key_id": "xxx",
                            "secret_access_key": "xxx",
                            "endpoint_url": "https://xxx.ibm.com"
                        },
                        "target": {
                            "bucket": "xxx",
                            "key": "xxx"
                        }
                    },
                    self.client.repository.ExperimentMetaNames.TRAINING_REFERENCES: [
                        {
                            "name": "xxx",
                            "training_definition_url": "https://xxx.ibm.com",
                            "command": "test",
                            "compute_configuration": {
                                "name": "k80"
                            }
                        }
                    ]
                }

        print(get_cos_training_data_reference(self.bucket_names))
        print(get_cos_training_results_reference(self.bucket_names))
        experiment_details = self.client.repository.store_experiment(meta_props=metadata)

        TestWMLClientWithKeras.experiment_uid = self.client.repository.get_experiment_uid(experiment_details)

        experiment_specific_details = self.client.repository.get_experiment_details(TestWMLClientWithKeras.experiment_uid)
        self.assertTrue(TestWMLClientWithKeras.experiment_uid in str(experiment_specific_details))

    def test_06_update_experiment(self):
        metadata = {
            self.client.repository.ExperimentMetaNames.NAME: "my_experiment",
            self.client.repository.ExperimentMetaNames.DESCRIPTION: "mnist best model",
            self.client.repository.ExperimentMetaNames.AUTHOR_NAME: "John Smith",
            self.client.repository.ExperimentMetaNames.EVALUATION_METHOD: "multiclass",
            self.client.repository.ExperimentMetaNames.EVALUATION_METRICS: ["accuracy"],
            self.client.repository.ExperimentMetaNames.TRAINING_DATA_REFERENCE: get_cos_training_data_reference(
                self.bucket_names),
            self.client.repository.ExperimentMetaNames.TRAINING_RESULTS_REFERENCE: get_cos_training_results_reference(
                self.bucket_names),
            self.client.repository.ExperimentMetaNames.TRAINING_REFERENCES: [
                {
                    "name": "mnist_nn",
                    "training_definition_url": TestWMLClientWithKeras.definition_1_url,
                    "compute_configuration": {"name": "k80"}
                },
                {
                    "name": "mnist_cnn",
                    "training_definition_url": TestWMLClientWithKeras.definition_2_url,
                    "compute_configuration": {"name": "k80"}
                },
            ]
        }

        experiment_details = self.client.repository.update_experiment(TestWMLClientWithKeras.experiment_uid, metadata)
        self.assertTrue('my_experiment' in str(experiment_details))
        print(experiment_details)
        self.assertTrue('xxx' not in str(experiment_details))

    def test_07_get_experiment_details(self):
        details = self.client.repository.get_experiment_details()
        self.assertTrue(TestWMLClientWithKeras.experiment_uid in str(details))

        details2 = self.client.repository.get_experiment_details(TestWMLClientWithKeras.experiment_uid)
        self.assertTrue(TestWMLClientWithKeras.experiment_uid in str(details2))

    def test_08_run_experiment(self):
        created_experiment_run_details = self.client.experiments.run(TestWMLClientWithKeras.experiment_uid, asynchronous=True)
        self.assertIsNotNone(created_experiment_run_details)
        TestWMLClientWithKeras.experiment_run_uid = Experiments.get_run_uid(created_experiment_run_details)

    def test_09_monitor(self):
        captured_output = io.StringIO()  # Create StringIO object
        sys.stdout = captured_output  # and redirect stdout.
        self.client.experiments.monitor_logs(TestWMLClientWithKeras.experiment_run_uid)  # Call function.
        sys.stdout = sys.__stdout__  # Reset redirect.
        self.assertTrue('training-' in captured_output.getvalue())

    def test_10_get_status(self):
        while True:
            time.sleep(3)
            status = self.client.experiments.get_status(TestWMLClientWithKeras.experiment_run_uid)
            if status['state'] == 'completed' or status['state'] == 'error' or status['state'] == 'canceled':
                break
        self.assertIsNotNone(status)
        self.assertTrue(status['state'] == 'completed')

    def test_11_get_all_experiments_runs_details(self):
        details = self.client.experiments.get_details()
        self.assertIsNotNone(details)

    def test_12_get_experiment_details(self):
        details = self.client.experiments.get_details(TestWMLClientWithKeras.experiment_uid)
        self.assertIsNotNone(details)

    def test_13_get_experiment_run_details(self):
        details = self.client.experiments.get_run_details(TestWMLClientWithKeras.experiment_run_uid)
        self.assertIsNotNone(details)

        self.assertIsNotNone(self.client.experiments.get_training_runs(details))
        self.assertIsNotNone(self.client.experiments.get_training_uids(details))

    def test_14_list_experiments(self):
        captured_output = io.StringIO()  # Create StringIO object
        sys.stdout = captured_output  # and redirect stdout.
        self.client.experiments.list_runs()  # Call function.
        sys.stdout = sys.__stdout__  # Reset redirect.
        self.assertTrue(TestWMLClientWithKeras.experiment_uid in captured_output.getvalue())
        self.client.experiments.list_runs()  # Just to see values.
        captured_output = io.StringIO()  # Create StringIO object
        sys.stdout = captured_output  # and redirect stdout.
        self.client.repository.list_experiments()  # Call function.
        sys.stdout = sys.__stdout__  # Reset redirect.
        self.assertTrue(TestWMLClientWithKeras.experiment_uid in captured_output.getvalue())
        self.client.repository.list_experiments()  # Just to see values.

    def test_15_list_experiment_runs_for_experiment(self):
        captured_output = io.StringIO()  # Create StringIO object
        sys.stdout = captured_output  # and redirect stdout.
        self.client.experiments.list_runs(TestWMLClientWithKeras.experiment_uid)  # Call function.
        sys.stdout = sys.__stdout__  # Reset redirect.
        self.assertTrue(TestWMLClientWithKeras.experiment_uid in captured_output.getvalue())
        self.client.experiments.list_runs(TestWMLClientWithKeras.experiment_uid)  # Just to see values.

    # def test_16_create_deployment(self):
    #     TestWMLClientWithCaffe.logger.info("Create deployment")
    #     TestWMLClientWithCaffe.train_run_uid = self.client.experiments.get_details(TestWMLClientWithCaffe.experiment_uid)['resources'][0]['entity']['training_statuses'][0]['training_guid']
    #     TestWMLClientWithCaffe.trained_model_uid = self.client.repository.store_model(TestWMLClientWithCaffe.train_run_uid, "test pytorch")
    #     deployment_details = self.client.deployments.create(artifact_uid=TestWMLClientWithCaffe.trained_model_uid, name="Test deployment", asynchronous=False)
    #     TestWMLClientWithCaffe.logger.debug("Deployment details: " + str(deployment_details))
    #     TestWMLClientWithCaffe.deployment_uid = self.client.deployments.get_uid(deployment_details)
    #     TestWMLClientWithCaffe.scoring_url = self.client.deployments.get_scoring_url(deployment_details)
    #     self.assertTrue('online' in str(TestWMLClientWithCaffe.scoring_url))
    #
    # def test_17_scoring(self):
    #     TestWMLClientWithCaffe.logger.info("Score model")
    #     scoring_data = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #                              0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 18, 18, 18,
    #                              126, 136, 175, 26, 166, 255, 247, 127, 0, 0, 0, 0, 0,
    #                              0, 0, 0, 0, 0, 0, 0, 30, 36, 94, 154, 170, 253,
    #                              253, 253, 253, 253, 225, 172, 253, 242, 195, 64, 0, 0, 0,
    #                              0, 0, 0, 0, 0, 0, 0, 0, 49, 238, 253, 253, 253,
    #                              253, 253, 253, 253, 253, 251, 93, 82, 82, 56, 39, 0, 0,
    #                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 18, 219, 253,
    #                              253, 253, 253, 253, 198, 182, 247, 241, 0, 0, 0, 0, 0,
    #                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #                              80, 156, 107, 253, 253, 205, 11, 0, 43, 154, 0, 0, 0,
    #                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #                              0, 0, 0, 14, 1, 154, 253, 90, 0, 0, 0, 0, 0,
    #                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #                              0, 0, 0, 0, 0, 0, 0, 139, 253, 190, 2, 0, 0,
    #                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #                              0, 0, 0, 0, 0, 0, 0, 0, 0, 11, 190, 253, 70,
    #                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 35,
    #                              241, 225, 160, 108, 1, 0, 0, 0, 0, 0, 0, 0, 0,
    #                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #                              0, 0, 81, 240, 253, 253, 119, 25, 0, 0, 0, 0, 0,
    #                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #                              0, 0, 0, 0, 0, 45, 186, 253, 253, 150, 27, 0, 0,
    #                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #                              0, 0, 0, 0, 0, 0, 0, 0, 16, 93, 252, 253, 187,
    #                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 249,
    #                              253, 249, 64, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 46, 130,
    #                              183, 253, 253, 207, 2, 0, 0, 0, 0, 0, 0, 0, 0,
    #                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 39, 148,
    #                              229, 253, 253, 253, 250, 182, 0, 0, 0, 0, 0, 0, 0,
    #                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 24, 114,
    #                              221, 253, 253, 253, 253, 201, 78, 0, 0, 0, 0, 0, 0,
    #                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 23, 66,
    #                              213, 253, 253, 253, 253, 198, 81, 2, 0, 0, 0, 0, 0,
    #                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 18, 171,
    #                              219, 253, 253, 253, 253, 195, 80, 9, 0, 0, 0, 0, 0,
    #                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 55, 172,
    #                              226, 253, 253, 253, 253, 244, 133, 11, 0, 0, 0, 0, 0,
    #                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #                              136, 253, 253, 253, 212, 135, 132, 16, 0, 0, 0, 0, 0,
    #                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #                              0, 0, 0, 0]
    #
    #     scoring_payload = {'values': [scoring_data, scoring_data]}
    #     scores = self.client.deployments.score(TestWMLClientWithCaffe.scoring_url, scoring_payload)
    #     self.assertIsNotNone(scores)
    #
    # def test_18_delete_deployment(self):
    #     TestWMLClientWithCaffe.logger.info("Delete deployment")
    #     self.client.deployments.delete(TestWMLClientWithCaffe.deployment_uid)
    #
    # def test_19_delete_model(self):
    #     TestWMLClientWithCaffe.logger.info("Delete model")
    #     self.client.repository.delete(TestWMLClientWithCaffe.trained_model_uid)

    def test_20_delete_experiment_run(self):
        self.client.experiments.delete(TestWMLClientWithKeras.experiment_run_uid)

    def test_21_list(self):
        captured_output = io.StringIO()  # Create StringIO object
        sys.stdout = captured_output  # and redirect stdout.
        self.client.repository.list_definitions()  # Call function.
        sys.stdout = sys.__stdout__  # Reset redirect.
        self.assertTrue(TestWMLClientWithKeras.definition_1_uid in captured_output.getvalue())
        self.assertTrue(TestWMLClientWithKeras.definition_2_uid in captured_output.getvalue())
        self.client.repository.list_definitions() # just to see

        captured_output = io.StringIO()  # Create StringIO object
        sys.stdout = captured_output  # and redirect stdout.
        self.client.repository.list_experiments()  # Call function.
        sys.stdout = sys.__stdout__  # Reset redirect.
        self.assertTrue(TestWMLClientWithKeras.experiment_uid in captured_output.getvalue())
        self.client.repository.list_experiments()  # just to see

        self.client.repository.list()

    def test_22_delete_experiment(self):
        self.client.repository.delete(TestWMLClientWithKeras.experiment_uid)

    def test_23_delete_definitions(self):
        self.client.repository.delete(TestWMLClientWithKeras.definition_1_uid)
        self.client.repository.delete(TestWMLClientWithKeras.definition_2_uid)

    def test_24_list(self):
        captured_output = io.StringIO()  # Create StringIO object
        sys.stdout = captured_output  # and redirect stdout.
        self.client.repository.list()  # Call function.
        sys.stdout = sys.__stdout__  # Reset redirect.
        self.assertTrue(TestWMLClientWithKeras.definition_1_uid not in captured_output.getvalue())
        self.assertTrue(TestWMLClientWithKeras.definition_2_uid not in captured_output.getvalue())
        self.assertTrue(TestWMLClientWithKeras.experiment_uid not in captured_output.getvalue())
        self.client.repository.list()  # Just to see values.


if __name__ == '__main__':
    unittest.main()
