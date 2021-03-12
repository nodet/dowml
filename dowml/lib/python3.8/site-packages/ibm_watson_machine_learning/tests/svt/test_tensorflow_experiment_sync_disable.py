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
    definition_2_uid = None
    definition_2_url = None
    trained_model_uid = None
    experiment_uid = None
    experiment_run_uid = None
    trained_model_uids = None
    logger = logging.getLogger(__name__)

    @classmethod
    def setUpClass(self):
        TestWMLClientWithExperiment.logger.info("Service Instance: setting up credentials")
        self.wml_credentials = get_wml_credentials()
        self.client = get_client()
        self.cos_resource = get_cos_resource()
        self.bucket_names = prepare_cos(self.cos_resource)

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
            self.client.repository.DefinitionMetaNames.EXECUTION_COMMAND: "python3 tensorflow_mnist_softmax.py --trainingIters 20"
            }

        model_content_path = './artifacts/tf-softmax-model.zip'
        definition_details = self.client.repository.store_definition(training_definition=model_content_path, meta_props=metadata)
        TestWMLClientWithExperiment.definition_1_uid = self.client.repository.get_definition_uid(definition_details)
        TestWMLClientWithExperiment.definition_1_url = self.client.repository.get_definition_url(definition_details)
        TestWMLClientWithExperiment.logger.info("Saved model definition uid: " + str(TestWMLClientWithExperiment.definition_1_uid))

    def test_03_save_definition2(self):
        TestWMLClientWithExperiment.logger.info("Save model definition ...")
        metadata = {
            self.client.repository.DefinitionMetaNames.NAME: "definition with metrics",
            self.client.repository.DefinitionMetaNames.DESCRIPTION: "my_description",
            self.client.repository.DefinitionMetaNames.AUTHOR_NAME: "John Smith",
            #self.client.repository.DefinitionMetaNames.AUTHOR_EMAIL: "js@js.com",
            self.client.repository.DefinitionMetaNames.FRAMEWORK_NAME: "tensorflow",
            self.client.repository.DefinitionMetaNames.FRAMEWORK_VERSION: "1.3",
            self.client.repository.DefinitionMetaNames.RUNTIME_NAME: "python",
            self.client.repository.DefinitionMetaNames.RUNTIME_VERSION: "3.5",
            self.client.repository.DefinitionMetaNames.EXECUTION_COMMAND: "python3 convolutional_network.py --trainImagesFile ${DATA_DIR}/train-images-idx3-ubyte.gz --trainLabelsFile ${DATA_DIR}/train-labels-idx1-ubyte.gz --testImagesFile ${DATA_DIR}/t10k-images-idx3-ubyte.gz --testLabelsFile ${DATA_DIR}/t10k-labels-idx1-ubyte.gz --learningRate 0.001 --trainingIters 20000"
            }

        model_content_path = './artifacts/mnist_cnn_metrics.zip'
        definition_details = self.client.repository.store_definition(training_definition=model_content_path, meta_props=metadata)
        TestWMLClientWithExperiment.definition_2_uid = self.client.repository.get_definition_uid(definition_details)
        TestWMLClientWithExperiment.definition_2_url = self.client.repository.get_definition_url(definition_details)
        TestWMLClientWithExperiment.logger.info("Saved model definition uid: " + str(TestWMLClientWithExperiment.definition_2_uid))

    def test_04_get_definition_details(self):
        TestWMLClientWithExperiment.logger.info("Getting definition details ...")
        details_1 = self.client.repository.get_definition_details(TestWMLClientWithExperiment.definition_1_uid)
        TestWMLClientWithExperiment.logger.info(details_1)
        self.assertTrue('my_training_definition' in str(details_1))

        details_2 = self.client.repository.get_definition_details(TestWMLClientWithExperiment.definition_2_uid)
        TestWMLClientWithExperiment.logger.info(details_2)
        self.assertTrue('definition with metrics' in str(details_2))

    def test_05_save_experiment(self):
        TestWMLClientWithExperiment.logger.info("Saving experiment ...")
        metadata = {
                    self.client.repository.ExperimentMetaNames.NAME: "my_experiment",
                    self.client.repository.ExperimentMetaNames.DESCRIPTION: "mnist best model",
                    self.client.repository.ExperimentMetaNames.TAGS: [{"value": "project_guid", "description": "DSX project guid"}],
                    self.client.repository.ExperimentMetaNames.AUTHOR_NAME: "John Smith",
                    #self.client.repository.ExperimentMetaNames.AUTHOR_EMAIL: "js@js.com",
                    self.client.repository.ExperimentMetaNames.EVALUATION_METHOD: "multiclass",
                    self.client.repository.ExperimentMetaNames.EVALUATION_METRICS: ["accuracy"],
                    self.client.repository.ExperimentMetaNames.TRAINING_DATA_REFERENCE: get_cos_training_data_reference(self.bucket_names),
                    self.client.repository.ExperimentMetaNames.TRAINING_RESULTS_REFERENCE: get_cos_training_results_reference(self.bucket_names),
                    self.client.repository.ExperimentMetaNames.TRAINING_REFERENCES: [
                        {
                            "name": "mnist_nn",
                            "training_definition_url": TestWMLClientWithExperiment.definition_1_url,
                            "command": "python3 tensorflow_mnist_softmax.py --trainingIters 20",
                            "compute_configuration": {"name": "k80"},
                        },
                        {
                            "training_definition_url": TestWMLClientWithExperiment.definition_2_url,
                            "compute_configuration": {"name": "k80"},
                        }
                    ]
                }

        experiment_details = self.client.repository.store_experiment(meta_props=metadata)
        self.assertTrue('my_experiment' in str(experiment_details))

        TestWMLClientWithExperiment.experiment_uid = self.client.repository.get_experiment_uid(experiment_details)

        experiment_specific_details = self.client.repository.get_experiment_details(TestWMLClientWithExperiment.experiment_uid)
        self.assertTrue(TestWMLClientWithExperiment.experiment_uid in str(experiment_specific_details))

    def test_06_get_experiment_details(self):
        TestWMLClientWithExperiment.logger.info("Get experiment details ...")
        details = self.client.repository.get_experiment_details()
        self.assertTrue(TestWMLClientWithExperiment.experiment_uid in str(details))

        details2 = self.client.repository.get_experiment_details(TestWMLClientWithExperiment.experiment_uid)
        self.assertTrue(TestWMLClientWithExperiment.experiment_uid in str(details2))

        self.assertTrue("project_guid" in str(details2))

    def test_07_run_experiment(self):
        TestWMLClientWithExperiment.logger.info("Running experiment ...")
        created_experiment_run_details = self.client.experiments.run(TestWMLClientWithExperiment.experiment_uid, asynchronous=False)
        self.assertIsNotNone(created_experiment_run_details)
        TestWMLClientWithExperiment.experiment_run_uid = Experiments.get_run_uid(created_experiment_run_details)

    def test_08_get_status(self):
        TestWMLClientWithExperiment.logger.info("Get experiment status ...")
        status = self.client.experiments.get_status(TestWMLClientWithExperiment.experiment_run_uid)
        self.assertIsNotNone(status)
        self.assertTrue(status['state'] == 'completed')

    def test_09_monitor_metrics(self):
        queue = Queue()
        process = Process(target=run_monitor, args=(self.client, TestWMLClientWithExperiment.experiment_run_uid, queue))
        process.start()
        process.join(timeout=600)
        process.terminate()
        self.assertFalse(queue.empty())
        self.assertTrue('accuracy' in queue.get())

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
        self.assertTrue('accuracy' in captured_output.getvalue())
        self.client.experiments.list_training_runs(TestWMLClientWithExperiment.experiment_run_uid)  # to see values

    def test_16_get_training_runs_uid(self):
        details = self.client.experiments.get_run_details(TestWMLClientWithExperiment.experiment_run_uid)
        TestWMLClientWithExperiment.trained_model_uids = self.client.experiments.get_training_uids(details)
        self.assertTrue(len(TestWMLClientWithExperiment.trained_model_uids) == 2)

    def test_17_experiment_metrics(self):
        TestWMLClientWithExperiment.logger.info("Get experiment metrics ..")
        metrics = self.client.experiments.get_metrics(TestWMLClientWithExperiment.experiment_run_uid)
        final_metrics = self.client.experiments.get_latest_metrics(TestWMLClientWithExperiment.experiment_run_uid)

        TestWMLClientWithExperiment.logger.info('metrics: ' + str(metrics))
        TestWMLClientWithExperiment.logger.info('final metrics: ' + str(final_metrics))

        self.assertTrue('accuracy' in str(metrics))
        self.assertTrue('accuracy' in str(final_metrics))
        self.assertTrue(len(final_metrics) == 1)

    def test_18_store_trained_models(self):
        model_1 = self.client.repository.store_model(TestWMLClientWithExperiment.trained_model_uids[0], {'name': 'nn mnist'})
        model_2 = self.client.repository.store_model(TestWMLClientWithExperiment.trained_model_uids[1], {'name': 'cnn mnist'})
        TestWMLClientWithExperiment.model_uid = self.client.repository.get_model_uid(model_1)
        TestWMLClientWithExperiment.model2_uid = self.client.repository.get_model_uid(model_2)
        TestWMLClientWithExperiment.logger.info(TestWMLClientWithExperiment.model_uid)
        TestWMLClientWithExperiment.logger.info(TestWMLClientWithExperiment.model2_uid)
        self.assertIsNotNone(TestWMLClientWithExperiment.model_uid)
        self.assertIsNotNone(TestWMLClientWithExperiment.model2_uid)

    def test_19_delete_experiment_run(self):
        self.client.experiments.delete(TestWMLClientWithExperiment.experiment_run_uid)

    def test_20_list_repository_1(self):
        stdout_ = sys.stdout
        captured_output = io.StringIO()  # Create StringIO object
        sys.stdout = captured_output  # and redirect stdout.
        self.client.repository.list()  # Call function.
        sys.stdout = stdout_  # Reset redirect.
        self.assertTrue(TestWMLClientWithExperiment.definition_1_uid in captured_output.getvalue())
        self.assertTrue(TestWMLClientWithExperiment.definition_2_uid in captured_output.getvalue())
        self.assertTrue(TestWMLClientWithExperiment.experiment_uid in captured_output.getvalue())
        self.client.repository.list()  # Just to see values.

    def test_21_delete_experiment(self):
        self.client.repository.delete(TestWMLClientWithExperiment.experiment_uid)

    def test_22_delete_definitions(self):
        self.client.repository.delete(TestWMLClientWithExperiment.definition_1_uid)
        self.client.repository.delete(TestWMLClientWithExperiment.definition_2_uid)

    def test_23_list_repository_2(self):
        stdout_ = sys.stdout
        captured_output = io.StringIO()  # Create StringIO object
        sys.stdout = captured_output  # and redirect stdout.
        self.client.repository.list()  # Call function.
        sys.stdout = stdout_  # Reset redirect.
        self.assertTrue(TestWMLClientWithExperiment.definition_1_uid not in captured_output.getvalue())
        self.assertTrue(TestWMLClientWithExperiment.definition_2_uid not in captured_output.getvalue())
        self.assertTrue(TestWMLClientWithExperiment.experiment_uid not in captured_output.getvalue())
        self.client.repository.list()  # Just to see values.

    def test_24_get_models_details(self):
        print(TestWMLClientWithExperiment.model_uid)
        self.assertTrue(TestWMLClientWithExperiment.model_uid in str(self.client.repository.get_model_details(TestWMLClientWithExperiment.model_uid)))

    def test_25_delete_models(self):
        self.client.repository.delete(TestWMLClientWithExperiment.model_uid)
        self.client.repository.delete(TestWMLClientWithExperiment.model2_uid)

if __name__ == '__main__':
    unittest.main()
