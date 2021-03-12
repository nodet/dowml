import unittest
import sys
import io
from multiprocessing import Process, Queue
import logging
from ibm_watson_machine_learning.experiments import Experiments
from preparation_and_cleaning import *


class TestWMLClientWithExperiment(unittest.TestCase):
    deployment_uid = None
    model_uid = None
    model2_uid = None
    scoring_url = None
    definition_1_uid = None
    definition_1_url = None
    trained_model_uid = None
    experiment_uid = None
    experiment_run_uid = None
    trained_model_uids = None
    logger = logging.getLogger(__name__)

    @classmethod
    def setUpClass(self):
        TestWMLClientWithExperiment.logger.info("Service Instance: setting up credentials")
        self.wml_credentials = get_wml_credentials()
        # reload(site)
        self.client = get_client()
        self.cos_resource = get_cos_resource()
        self.bucket_names = prepare_cos(self.cos_resource)

    @classmethod
    def tearDownClass(self):
        clean_cos(self.cos_resource, self.bucket_names)
        pass

    def test_01_service_instance_details(self):
        TestWMLClientWithExperiment.logger.info("Check client ...")
        self.assertTrue(self.client.__class__.__name__ == 'APIClient')

        TestWMLClientWithExperiment.logger.info("Getting instance details ...")
        details = self.client.service_instance.get_details()
        TestWMLClientWithExperiment.logger.debug(details)

        self.assertTrue("published_models" in str(details))
        self.assertEqual(type(details), dict)

    def test_02_save_definition1(self):
        import os

        TestWMLClientWithExperiment.logger.info("Save model definition ...")
        metadata = {
            self.client.repository.DefinitionMetaNames.NAME: "DDL with metrics",
            self.client.repository.DefinitionMetaNames.DESCRIPTION: "my_description",
            self.client.repository.DefinitionMetaNames.AUTHOR_NAME: "John Smith",
            self.client.repository.DefinitionMetaNames.AUTHOR_EMAIL: "js@js.com",
            self.client.repository.DefinitionMetaNames.FRAMEWORK_NAME: "tensorflow-ddl",
            self.client.repository.DefinitionMetaNames.FRAMEWORK_VERSION: "1.5",
            self.client.repository.DefinitionMetaNames.RUNTIME_NAME: "python",
            self.client.repository.DefinitionMetaNames.RUNTIME_VERSION: "3.5",
            self.client.repository.DefinitionMetaNames.EXECUTION_COMMAND: "python mnist-env.py"
            }

        model_content_path = os.path.join('artifacts', 'tf_distributed_ddl', 'tf_ddl.zip')
        definition_details = self.client.repository.store_definition(training_definition=model_content_path, meta_props=metadata)
        TestWMLClientWithExperiment.definition_1_uid = self.client.repository.get_definition_uid(definition_details)
        TestWMLClientWithExperiment.definition_1_url = self.client.repository.get_definition_url(definition_details)
        TestWMLClientWithExperiment.logger.info("Saved model definition uid: " + str(TestWMLClientWithExperiment.definition_1_uid))

    def test_03_get_definition_details(self):
        TestWMLClientWithExperiment.logger.info("Getting definition details ...")
        details_1 = self.client.repository.get_definition_details(TestWMLClientWithExperiment.definition_1_uid)
        TestWMLClientWithExperiment.logger.info(details_1)
        self.assertTrue('DDL with metrics' in str(details_1))

    def test_04_save_experiment(self):
        metadata = {
                    self.client.repository.ExperimentMetaNames.NAME: "tf_distributed_DDL_experiment",
                    self.client.repository.ExperimentMetaNames.DESCRIPTION: "mnist best model",
                    self.client.repository.ExperimentMetaNames.AUTHOR_NAME: "John Smith",
                    self.client.repository.ExperimentMetaNames.AUTHOR_EMAIL: "js@js.com",
                    self.client.repository.ExperimentMetaNames.TRAINING_DATA_REFERENCE: get_cos_training_data_reference(self.bucket_names),
                    self.client.repository.ExperimentMetaNames.TRAINING_RESULTS_REFERENCE: get_cos_training_results_reference(self.bucket_names),
                    self.client.repository.ExperimentMetaNames.TRAINING_REFERENCES: [
                        {
                            "name": "mnist_DDL_distributed",
                            "training_definition_url": TestWMLClientWithExperiment.definition_1_url,
                            "compute_configuration": {"name": "p100x2", "nodes": 1},
                        }
                    ]
                }

        experiment_details = self.client.repository.store_experiment(meta_props=metadata)
        self.assertTrue('tf_distributed_DDL_experiment' in str(experiment_details))

        TestWMLClientWithExperiment.experiment_uid = self.client.repository.get_experiment_uid(experiment_details)

        experiment_specific_details = self.client.repository.get_experiment_details(TestWMLClientWithExperiment.experiment_uid)
        self.assertTrue(TestWMLClientWithExperiment.experiment_uid in str(experiment_specific_details))

    def test_06_get_experiment_details(self):
        details = self.client.repository.get_experiment_details()
        self.assertTrue(TestWMLClientWithExperiment.experiment_uid in str(details))

        details2 = self.client.repository.get_experiment_details(TestWMLClientWithExperiment.experiment_uid)
        self.assertTrue(TestWMLClientWithExperiment.experiment_uid in str(details2))

    def test_07_run_experiment(self):
        created_experiment_run_details = self.client.experiments.run(TestWMLClientWithExperiment.experiment_uid, asynchronous=False)
        self.assertIsNotNone(created_experiment_run_details)
        TestWMLClientWithExperiment.experiment_run_uid = Experiments.get_run_uid(created_experiment_run_details)

    def test_08_get_status(self):
        status = self.client.experiments.get_status(TestWMLClientWithExperiment.experiment_run_uid)
        self.assertIsNotNone(status)
        self.assertTrue(status['state'] == 'completed')

    def test_09_monitor_logs(self):
        training_uids = self.client.experiments.get_training_uids(self.client.experiments.get_run_details(TestWMLClientWithExperiment.experiment_run_uid))

        queue = Queue()
        process = Process(target=run_monitor_training, args=(self.client, training_uids[0], queue))
        process.start()
        process.join(timeout=90)
        process.terminate()
        self.assertFalse(queue.empty())
        self.assertTrue("GPU" in queue.get())

    def test_10_get_all_experiments_runs_details(self):
        details = self.client.experiments.get_details()
        self.assertIsNotNone(details)

    def test_11_get_experiment_details(self):
        details = self.client.experiments.get_details(TestWMLClientWithExperiment.experiment_uid)
        self.assertIsNotNone(details)

    def test_12_get_experiment_run_details(self):
        details = self.client.experiments.get_run_details(TestWMLClientWithExperiment.experiment_run_uid)
        self.assertIsNotNone(details)

    def test_13_list_experiments(self):
        stdout_ = sys.stdout
        captured_output = io.StringIO()  # Create StringIO object
        sys.stdout = captured_output  # and redirect stdout.
        self.client.experiments.list_runs()  # Call function.
        sys.stdout = stdout_  # Reset redirect.
        self.assertTrue(TestWMLClientWithExperiment.experiment_uid in captured_output.getvalue())
        self.client.experiments.list_runs()  # Just to see values.

    def test_14_list_experiment_runs_for_experiment(self):
        stdout_ = sys.stdout
        captured_output = io.StringIO()  # Create StringIO object
        sys.stdout = captured_output  # and redirect stdout.
        self.client.experiments.list_runs(TestWMLClientWithExperiment.experiment_uid)  # Call function.
        sys.stdout = stdout_  # Reset redirect.
        self.assertTrue(TestWMLClientWithExperiment.experiment_uid in captured_output.getvalue())
        self.client.experiments.list_runs(TestWMLClientWithExperiment.experiment_uid)  # Just to see values.

    def test_15_list_training_runs_for_experiment_run(self):
        stdout_ = sys.stdout
        captured_output = io.StringIO()  # Create StringIO object
        sys.stdout = captured_output  # and redirect stdout.
        self.client.experiments.list_training_runs(TestWMLClientWithExperiment.experiment_run_uid)  # Call function.
        sys.stdout = stdout_  # Reset redirect.
        self.assertTrue('GUID' in captured_output.getvalue())
        self.assertTrue('completed' in captured_output.getvalue())
        self.client.experiments.list_training_runs(TestWMLClientWithExperiment.experiment_run_uid)  # to see values

    def test_16_get_training_runs_uid(self):
        details = self.client.experiments.get_run_details(TestWMLClientWithExperiment.experiment_run_uid)
        TestWMLClientWithExperiment.trained_model_uids = self.client.experiments.get_training_uids(details)
        self.assertTrue(len(TestWMLClientWithExperiment.trained_model_uids) == 1)

    """
    def test_17_experiment_metrics(self):
        print("Get experiment metrics ..")
        metrics = self.client.experiments.get_metrics(TestWMLClientWithExperiment.experiment_run_uid)
        final_metrics = self.client.experiments.get_latest_metrics(TestWMLClientWithExperiment.experiment_run_uid)

        print('metrics: ' + str(metrics))
        print('final metrics: ' + str(final_metrics))

        self.assertTrue('accuracy' in str(metrics))
        self.assertTrue('accuracy' in str(final_metrics))
        self.assertTrue(len(final_metrics) == 1)

    def test_18_store_trained_models(self):
        model_1 = self.client.repository.store_model(TestWMLClientWithExperiment.trained_model_uids[0], {'name': 'distributed mnist'})
        TestWMLClientWithExperiment.model_uid = self.client.repository.get_model_uid(model_1)

        print(TestWMLClientWithExperiment.model_uid)
        self.assertIsNotNone(TestWMLClientWithExperiment.model_uid)
    """
    def test_19_delete_experiment_run(self):
        self.client.experiments.delete(TestWMLClientWithExperiment.experiment_run_uid)

    def test_20_list_repository_1(self):
        captured_output = io.StringIO()  # Create StringIO object
        sys.stdout = captured_output  # and redirect stdout.
        self.client.repository.list()  # Call function.
        sys.stdout = sys.__stdout__  # Reset redirect.
        self.assertTrue(TestWMLClientWithExperiment.definition_1_uid in captured_output.getvalue())
        self.assertTrue(TestWMLClientWithExperiment.experiment_uid in captured_output.getvalue())
        self.client.repository.list()  # Just to see values.

    def test_21_delete_experiment(self):
        self.client.repository.delete(TestWMLClientWithExperiment.experiment_uid)

    def test_22_delete_definitions(self):
        self.client.repository.delete(TestWMLClientWithExperiment.definition_1_uid)

    def test_23_list_repository_2(self):
        stdout_ = sys.stdout
        captured_output = io.StringIO()  # Create StringIO object
        sys.stdout = captured_output  # and redirect stdout.
        self.client.repository.list()  # Call function.
        sys.stdout = stdout_  # Reset redirect.
        self.assertTrue(TestWMLClientWithExperiment.definition_1_uid not in captured_output.getvalue())
        self.assertTrue(TestWMLClientWithExperiment.experiment_uid not in captured_output.getvalue())
        self.client.repository.list()  # Just to see values.

    """
    def test_24_get_models_details(self):
        print(TestWMLClientWithExperiment.model_uid)
        self.assertTrue(TestWMLClientWithExperiment.model_uid in str(self.client.repository.get_model_details(TestWMLClientWithExperiment.model_uid)))

    def test_25_delete_models(self):
        self.client.repository.delete(TestWMLClientWithExperiment.model_uid)
        self.client.repository.delete(TestWMLClientWithExperiment.model2_uid)
    """


if __name__ == '__main__':
    unittest.main()
