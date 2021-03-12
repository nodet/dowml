import unittest
import time
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
    experiment_run_details = None
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
            self.client.repository.DefinitionMetaNames.NAME: "my_training_definition",
            self.client.repository.DefinitionMetaNames.DESCRIPTION: "my_description",
            self.client.repository.DefinitionMetaNames.AUTHOR_NAME: "John Smith",
            self.client.repository.DefinitionMetaNames.AUTHOR_EMAIL: "js@js.com",
            self.client.repository.DefinitionMetaNames.FRAMEWORK_NAME: "tensorflow",
            self.client.repository.DefinitionMetaNames.FRAMEWORK_VERSION: "1.5",
            self.client.repository.DefinitionMetaNames.RUNTIME_NAME: "python",
            self.client.repository.DefinitionMetaNames.RUNTIME_VERSION: "3.5",
            self.client.repository.DefinitionMetaNames.EXECUTION_COMMAND: "python3 keras_boston.py"
        }

        model_content_path = './artifacts/keras_boston.zip'
        definition_details = self.client.repository.store_definition(training_definition=model_content_path, meta_props=metadata)
        TestWMLClientWithKeras.definition_1_url = self.client.repository.get_definition_url(definition_details)
        TestWMLClientWithKeras.definition_1_uid = self.client.repository.get_definition_uid(definition_details)
        TestWMLClientWithKeras.logger.info("Saved model definition url: " + str(TestWMLClientWithKeras.definition_1_url))

    def test_03_get_definition_details(self):
        TestWMLClientWithKeras.logger.info("Getting definition details ...")
        details = self.client.repository.get_definition_details()
        TestWMLClientWithKeras.logger.info(details)
        self.assertTrue('my_training_definition' in str(details))
        TestWMLClientWithKeras.logger.info("List definitions")
        self.client.repository.list_definitions()

    def test_04_save_experiment(self):
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
                }
            ]
        }

        print(get_cos_training_data_reference(self.bucket_names))
        print(get_cos_training_results_reference(self.bucket_names))
        experiment_details = self.client.repository.store_experiment(meta_props=metadata)

        TestWMLClientWithKeras.experiment_uid = self.client.repository.get_experiment_uid(experiment_details)

        experiment_specific_details = self.client.repository.get_experiment_details(TestWMLClientWithKeras.experiment_uid)
        self.assertTrue(TestWMLClientWithKeras.experiment_uid in str(experiment_specific_details))

    def test_05_get_experiment_details(self):
        details = self.client.repository.get_experiment_details()
        self.assertTrue(TestWMLClientWithKeras.experiment_uid in str(details))

        details2 = self.client.repository.get_experiment_details(TestWMLClientWithKeras.experiment_uid)
        self.assertTrue(TestWMLClientWithKeras.experiment_uid in str(details2))

    def test_06_run_experiment(self):
        created_experiment_run_details = self.client.experiments.run(TestWMLClientWithKeras.experiment_uid, asynchronous=False)
        self.assertIsNotNone(created_experiment_run_details)
        TestWMLClientWithKeras.experiment_run_uid = Experiments.get_run_uid(created_experiment_run_details)
        print("Experiment run id:\n{}".format(TestWMLClientWithKeras.experiment_run_uid))

    def test_07_get_status(self):
        start_time = time.time()
        diff_time = start_time - start_time
        while True and diff_time < 60 * 10:
            time.sleep(3)
            status = self.client.experiments.get_status(TestWMLClientWithKeras.experiment_run_uid)
            if status['state'] == 'completed' or status['state'] == 'error' or status['state'] == 'canceled':
                break
            diff_time = time.time() - start_time
        self.assertIsNotNone(status)
        self.assertTrue(status['state'] == 'completed')

    def test_08_get_experiment_details(self):
        details = self.client.experiments.get_details(TestWMLClientWithKeras.experiment_uid)
        print("Experiment details:\n{}".format(details))
        self.assertIsNotNone(details)

    def test_09_get_experiment_run_details(self):
        TestWMLClientWithKeras.experiment_run_details = self.client.experiments.get_run_details(TestWMLClientWithKeras.experiment_run_uid)
        print("Experiment run details:\n{}".format(TestWMLClientWithKeras.experiment_run_details))
        self.assertIsNotNone(TestWMLClientWithKeras.experiment_run_details)
        self.assertIsNotNone(self.client.experiments.get_training_runs(TestWMLClientWithKeras.experiment_run_details))

    def test_10_store_experiment_as_model(self):
        training_run_uids = self.client.experiments.get_training_uids(TestWMLClientWithKeras.experiment_run_details)
        training_run_uid = training_run_uids[0]

        model_props = {
            self.client.repository.ModelMetaNames.NAME: "Keras Experiment Model",
        }
        model_details = self.client.repository.store_model(training_run_uid, meta_props=model_props)
        TestWMLClientWithKeras.trained_model_uid = self.client.repository.get_model_uid(model_details)

    def test_11_create_learning_system(self):
        self.client.learning_system.ConfigurationMetaNames.show()

        meta_prop = {
            self.client.learning_system.ConfigurationMetaNames.FEEDBACK_DATA_REFERENCE: get_cos_training_data_reference(self.bucket_names),
            self.client.learning_system.ConfigurationMetaNames.TRAINING_RESULTS_REFERENCE: get_cos_training_results_reference(self.bucket_names),
            self.client.learning_system.ConfigurationMetaNames.MIN_FEEDBACK_DATA_SIZE: 10,
            self.client.learning_system.ConfigurationMetaNames.AUTO_RETRAIN: "always",
            self.client.learning_system.ConfigurationMetaNames.AUTO_REDEPLOY: "always",
            self.client.learning_system.ConfigurationMetaNames.COMPUTE_CONFIGURATION: {
              "name": "k80x2",
              "nodes": 1
            }
        }

        details = self.client.learning_system.setup(TestWMLClientWithKeras.trained_model_uid, meta_prop)
        self.assertIsNotNone(details)

    def test_12_create_deployment(self):
        deployment_details = self.client.deployments.create(artifact_uid=TestWMLClientWithKeras.trained_model_uid, name="Test deployment", asynchronous=False)
        TestWMLClientWithKeras.logger.debug("Deployment details: " + str(deployment_details))
        TestWMLClientWithKeras.deployment_uid = self.client.deployments.get_uid(deployment_details)
        TestWMLClientWithKeras.scoring_url = self.client.deployments.get_scoring_url(deployment_details)
        self.assertTrue('online' in str(TestWMLClientWithKeras.scoring_url))

    def test_13_list(self):
        self.client.learning_system.list()  # Call function.

    def test_14_run_learning_system(self):
        run_details = self.client.learning_system.run(TestWMLClientWithKeras.trained_model_uid, asynchronous=False)
        TestWMLClientWithKeras.run_uid = self.client.learning_system.get_run_uid(run_details)
        url = self.client.learning_system.get_run_href(run_details)
        self.assertIsNotNone(url)

    def test_15_get_runs(self):
        runs = self.client.learning_system.get_runs(TestWMLClientWithKeras.trained_model_uid)
        self.assertIsNotNone(runs)
        self.assertTrue(TestWMLClientWithKeras.run_uid in str(runs))

    def test_16_get_deployment_status(self):
        print(self.client.deployments.get_status(TestWMLClientWithKeras.deployment_uid))

    def test_17_get_metrics(self):
        metrics = self.client.learning_system.get_metrics(TestWMLClientWithKeras.trained_model_uid)
        print(metrics)
        self.assertTrue('values' in str(metrics))

    def test_18_list_metrics(self):
        self.client.learning_system.list_metrics(TestWMLClientWithKeras.trained_model_uid)

    def test_19_list_runs(self):
        self.client.learning_system.list_runs(TestWMLClientWithKeras.trained_model_uid)

    def test_20_list_all_runs(self):
        self.client.learning_system.list_runs()

    def test_21_get_run_details(self):
        details = self.client.learning_system.get_run_details(TestWMLClientWithKeras.run_uid)
        self.assertTrue('COMPLETED' in str(details))

    def test_22_delete_deployment(self):
        TestWMLClientWithKeras.logger.info("Delete deployment")
        self.client.deployments.delete(TestWMLClientWithKeras.deployment_uid)

    def test_23_delete_model(self):
        TestWMLClientWithKeras.logger.info("Delete model")
        self.client.repository.delete(TestWMLClientWithKeras.trained_model_uid)

    def test_24_delete_experiment_run(self):
        self.client.experiments.delete(TestWMLClientWithKeras.experiment_run_uid)

    def test_25_delete_experiment(self):
        self.client.repository.delete(TestWMLClientWithKeras.experiment_uid)

    def test_26_delete_definition(self):
        self.client.repository.delete(TestWMLClientWithKeras.definition_1_uid)


if __name__ == '__main__':
    unittest.main()
