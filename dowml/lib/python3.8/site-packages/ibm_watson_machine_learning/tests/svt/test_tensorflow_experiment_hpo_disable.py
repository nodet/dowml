import unittest
import time
import sys
import io
import logging
from ibm_watson_machine_learning.experiments import Experiments
from preparation_and_cleaning import *


class TestWMLClientWithExperiment(unittest.TestCase):
    deployment_uid = None
    model_uid = None
    scoring_url = None
    definition_uid = None
    definition_url = None
    trained_model_uid = None
    experiment_uid = None
    experiment_run_uid = None
    logger = logging.getLogger(__name__)

    @classmethod
    def setUpClass(self):
        TestWMLClientWithExperiment.logger.info("Service Instance: setting up credentials")
        self.wml_credentials = get_wml_credentials()
        # reload(site)
        self.client = get_client()
        self.cos_resource = get_cos_resource()
        self.bucket_names = prepare_cos(self.cos_resource)
        clean_env(self.client, self.cos_resource)

    @classmethod
    def tearDownClass(self):
        clean_cos(self.cos_resource, self.bucket_names)

    def test_01_service_instance_details(self):
        TestWMLClientWithExperiment.logger.info("Check client ...")
        self.assertTrue(self.client.__class__.__name__ == 'APIClient')

        TestWMLClientWithExperiment.logger.info("Getting instance details ...")
        details = self.client.service_instance.get_details()
        TestWMLClientWithExperiment.logger.debug(details)

        self.assertTrue("published_models" in str(details))
        self.assertEqual(type(details), dict)

    def test_02_save_definition1(self):
        TestWMLClientWithExperiment.logger.info("Save model definition ...")

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
            self.client.repository.DefinitionMetaNames.EXECUTION_COMMAND: "python3 cnn.py --dataDir ${DATA_DIR} config.json"
            }

        model_content_path = './artifacts/tf_hpo/hpo_mnist_tf.zip'

        definition_details = self.client.repository.store_definition(training_definition=model_content_path, meta_props=metadata)
        TestWMLClientWithExperiment.definition_uid = self.client.repository.get_definition_uid(definition_details)
        TestWMLClientWithExperiment.definition_url = self.client.repository.get_definition_url(definition_details)
        TestWMLClientWithExperiment.logger.info("Saved model definition uid: " + str(TestWMLClientWithExperiment.definition_uid))

    def test_03_get_definition_details(self):
        TestWMLClientWithExperiment.logger.info("Getting definition details ...")
        details_1 = self.client.repository.get_definition_details(TestWMLClientWithExperiment.definition_uid)
        TestWMLClientWithExperiment.logger.info(details_1)
        self.assertTrue('my_training_definition' in str(details_1))

    def test_04_save_experiment(self):
        metadata = {
                    self.client.repository.ExperimentMetaNames.NAME: "xxx",
                    self.client.repository.ExperimentMetaNames.AUTHOR_EMAIL: "js@js.com",
                    self.client.repository.ExperimentMetaNames.EVALUATION_METHOD: "binary",
                    self.client.repository.ExperimentMetaNames.EVALUATION_METRICS: [],
                    self.client.repository.ExperimentMetaNames.TRAINING_DATA_REFERENCE: get_cos_training_data_reference(self.bucket_names),
                    self.client.repository.ExperimentMetaNames.TRAINING_RESULTS_REFERENCE: get_cos_training_results_reference(self.bucket_names),
                    self.client.repository.ExperimentMetaNames.TRAINING_REFERENCES: [
                        {
                            "name": "mnist_nn",
                            "training_definition_url": TestWMLClientWithExperiment.definition_url,
                            "compute_configuration": {"name": "k80"}
                        }
                    ]
                }

        TestWMLClientWithExperiment.logger.info(get_cos_training_data_reference(self.bucket_names))
        TestWMLClientWithExperiment.logger.info(get_cos_training_results_reference(self.bucket_names))
        experiment_details = self.client.repository.store_experiment(meta_props=metadata)

        TestWMLClientWithExperiment.experiment_uid = self.client.repository.get_experiment_uid(experiment_details)

        experiment_specific_details = self.client.repository.get_experiment_details(TestWMLClientWithExperiment.experiment_uid)
        self.assertTrue(TestWMLClientWithExperiment.experiment_uid in str(experiment_specific_details))

    def test_05_update_experiment(self):
        metadata = {
            self.client.repository.ExperimentMetaNames.NAME: "my HPO experiment",
            self.client.repository.ExperimentMetaNames.DESCRIPTION: "mnist best model",
            self.client.repository.ExperimentMetaNames.AUTHOR_NAME: "John Smith",
            self.client.repository.ExperimentMetaNames.EVALUATION_METHOD: "multiclass",
            self.client.repository.ExperimentMetaNames.EVALUATION_METRICS: ["accuracy"],
            self.client.repository.ExperimentMetaNames.TRAINING_DATA_REFERENCE: get_cos_training_data_reference(self.bucket_names),
            self.client.repository.ExperimentMetaNames.TRAINING_RESULTS_REFERENCE: get_cos_training_results_reference(self.bucket_names),
            self.client.repository.ExperimentMetaNames.TRAINING_REFERENCES: [
                {
                    "name": "mnist_nn",
                    "training_definition_url": TestWMLClientWithExperiment.definition_url,
                    "compute_configuration": {"name": "p100"},
                    "hyper_parameters_optimization": {
                        "method": {
                            "name": "rbfopt",
                            "parameters": [
                                self.client.experiments.HPOMethodParam("filename", "val_dict_list.json"),
                                self.client.experiments.HPOMethodParam("time_interval", "steps"),
                                self.client.experiments.HPOMethodParam("objective", "accuracy"),
                                self.client.experiments.HPOMethodParam("maximize_or_minimize", "maximize"),
                                self.client.experiments.HPOMethodParam("num_optimizer_steps", 2)
                            ]
                        },
                        "hyper_parameters": [
                            self.client.experiments.HPOParameter('learning_rate', min=0.001, max=0.002, step=0.001)
                        ]
                    }
                }
            ]
        }

        experiment_details = self.client.repository.update_experiment(TestWMLClientWithExperiment.experiment_uid, metadata)
        self.assertTrue('my HPO experiment' in str(experiment_details))
        TestWMLClientWithExperiment.logger.info(experiment_details)
        self.assertTrue('xxx' not in str(experiment_details))

    def test_06_get_experiment_details(self):
        TestWMLClientWithExperiment.logger.info("Getting experiment details")
        details = self.client.repository.get_experiment_details()
        self.assertTrue(TestWMLClientWithExperiment.experiment_uid in str(details))

        details2 = self.client.repository.get_experiment_details(TestWMLClientWithExperiment.experiment_uid)
        self.assertTrue(TestWMLClientWithExperiment.experiment_uid in str(details2))
        TestWMLClientWithExperiment.logger.info("Experiment details: {}".format(details2))

    def test_07_run_experiment(self):
        TestWMLClientWithExperiment.logger.info("Running experiment")
        created_experiment_run_details = self.client.experiments.run(TestWMLClientWithExperiment.experiment_uid)
        self.assertIsNotNone(created_experiment_run_details)
        TestWMLClientWithExperiment.experiment_run_uid = Experiments.get_run_uid(created_experiment_run_details)

    def test_08_get_status(self):
        TestWMLClientWithExperiment.logger.info("Get experiment run status")
        start_time = time.time()
        diff_time = start_time - start_time
        while True and diff_time < 60 * 10:
            time.sleep(10)
            status = self.client.experiments.get_status(TestWMLClientWithExperiment.experiment_run_uid)

            TestWMLClientWithExperiment.logger.info('Train STATUS: ' + str(self.client.experiments.get_run_details(TestWMLClientWithExperiment.experiment_run_uid)))
            if status['state'] == 'completed' or status['state'] == 'error' or status['state'] == 'canceled':
                break
            diff_time = time.time() - start_time

        self.assertIsNotNone(status)
        self.assertTrue(status['state'] == 'completed')

    # def test_09_1_monitor_logs(self):
    #     captured_output = io.StringIO()  # Create StringIO object
    #     sys.stdout = captured_output  # and redirect stdout.
    #     self.client.experiments.monitor_logs(TestWMLClientWithExperiment.experiment_run_uid) # Call function.
    #     sys.stdout = sys.__stdout__  # Reset redirect.
    #     print('MONITOR LOGS')
    #     self.client.experiments.monitor_logs(TestWMLClientWithExperiment.experiment_run_uid) # to see
    #     self.assertTrue('training-' in captured_output.getvalue())

    """
    def test_09_2_monitor_training_logs(self):
        experiment_run_details = self.client.experiments.get_run_details(TestWMLClientWithExperiment.experiment_run_uid)
        training_uids = self.client.experiments.get_training_uids(experiment_run_details)
        captured_output = io.StringIO()  # Create StringIO object
        sys.stdout = captured_output  # and redirect stdout.
        self.client.training.monitor_logs(training_uids[1]) # Call function.
        sys.stdout = sys.__stdout__  # Reset redirect.
        self.assertTrue('training-' in captured_output.getvalue())
        print('MONITOR LOG')
        self.client.training.monitor_logs(training_uids[1])  # to see
    """
    def test_10_get_all_experiments_runs_details(self):
        details = self.client.experiments.get_details()
        self.assertIsNotNone(details)

    def test_11_get_experiment_run_details(self):
        details = self.client.experiments.get_details(TestWMLClientWithExperiment.experiment_uid)
        self.assertIsNotNone(details)

    def test_12_get_experiment_run_details(self):
        details = self.client.experiments.get_run_details(TestWMLClientWithExperiment.experiment_run_uid)
        self.assertIsNotNone(details)

        self.assertIsNotNone(self.client.experiments.get_training_runs(details))
        self.assertIsNotNone(self.client.experiments.get_training_uids(details))

    def test_13_list_experiments(self):
        stdout_ = sys.stdout
        captured_output = io.StringIO()  # Create StringIO object
        sys.stdout = captured_output  # and redirect stdout.
        self.client.experiments.list_runs()  # Call function.
        sys.stdout = stdout_  # Reset redirect.
        self.assertTrue(TestWMLClientWithExperiment.experiment_uid in captured_output.getvalue())
        self.client.experiments.list_runs()  # Just to see values.

        stdout_ = sys.stdout
        captured_output = io.StringIO()  # Create StringIO object
        sys.stdout = captured_output  # and redirect stdout.
        self.client.repository.list_experiments()  # Call function.
        sys.stdout = stdout_ # Reset redirect.
        self.assertTrue(TestWMLClientWithExperiment.experiment_uid in captured_output.getvalue())
        self.client.repository.list_experiments()  # Just to see values.

    def test_14_list_experiment_runs_for_experiment(self):
        stdout_ = sys.stdout
        captured_output = io.StringIO()  # Create StringIO object
        sys.stdout = captured_output  # and redirect stdout.
        self.client.experiments.list_runs(TestWMLClientWithExperiment.experiment_uid)  # Call function.
        sys.stdout = stdout_  # Reset redirect.
        self.assertTrue(TestWMLClientWithExperiment.experiment_uid in captured_output.getvalue())
        self.client.experiments.list_runs(TestWMLClientWithExperiment.experiment_uid)  # Just to see values.

    def test_15_check_number_of_training_runs(self):
        TestWMLClientWithExperiment.logger.info("List training runs for this experiment run: ")
        self.client.experiments.list_training_runs(TestWMLClientWithExperiment.experiment_run_uid)

        TestWMLClientWithExperiment.logger.info("Getting run details:")
        details = self.client.experiments.get_run_details(TestWMLClientWithExperiment.experiment_run_uid)
        TestWMLClientWithExperiment.logger.info("Details: ".format(details))
        training_run_uids = self.client.experiments.get_training_uids(experiment_run_details=details)

        TestWMLClientWithExperiment.logger.info("Training run uids: " + str(training_run_uids))

        TestWMLClientWithExperiment.logger.info("NUMBER OF RUNS: " + str(len(self.client.experiments.get_details(TestWMLClientWithExperiment.experiment_uid)['resources'][0]['entity']["training_statuses"])))
        self.assertTrue(len(training_run_uids) == 2 + 1)

    def test_16_delete_experiment_run(self):
        self.client.experiments.delete(TestWMLClientWithExperiment.experiment_run_uid)

    def test_17_list(self):
        stdout_ = sys.stdout
        captured_output = io.StringIO()  # Create StringIO object
        sys.stdout = captured_output  # and redirect stdout.
        self.client.repository.list_definitions()  # Call function.
        sys.stdout = stdout_  # Reset redirect.
        self.assertTrue(TestWMLClientWithExperiment.definition_uid in captured_output.getvalue())
        self.client.repository.list_definitions()  # Just to see values.

        stdout_ = sys.stdout
        captured_output = io.StringIO()  # Create StringIO object
        sys.stdout = captured_output  # and redirect stdout.
        self.client.repository.list_experiments()  # Call function.
        sys.stdout = stdout_  # Reset redirect.
        self.assertTrue(TestWMLClientWithExperiment.experiment_uid in captured_output.getvalue())
        self.client.repository.list_experiments()  # Just to see values.

        self.client.repository.list()

    def test_18_delete_experiment(self):
        self.client.repository.delete(TestWMLClientWithExperiment.experiment_uid)

    def test_19_delete_definition(self):
        self.client.repository.delete(TestWMLClientWithExperiment.definition_uid)

    def test_20_list(self):
        stdout_ = sys.stdout
        captured_output = io.StringIO()  # Create StringIO object
        sys.stdout = captured_output  # and redirect stdout.
        self.client.repository.list()  # Call function.
        sys.stdout = stdout_  # Reset redirect.
        self.assertTrue(TestWMLClientWithExperiment.definition_uid not in captured_output.getvalue())
        self.assertTrue(TestWMLClientWithExperiment.experiment_uid not in captured_output.getvalue())
        self.client.repository.list()  # Just to see values.

if __name__ == '__main__':
    unittest.main()
