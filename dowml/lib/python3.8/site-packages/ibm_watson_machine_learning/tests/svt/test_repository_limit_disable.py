import unittest
import logging
from preparation_and_cleaning import *
from models_preparation import *

ADDITIONAL_LIST_LINES = 4

class TestWMLClientWithScikitLearn(unittest.TestCase):
    definition_url = None
    experiments_uids = []
    definitions_uids = []
    models_uids = []
    logger = logging.getLogger(__name__)
    models_no = None
    definitions_no = None
    experiments_no = None

    @classmethod
    def setUpClass(self):
        TestWMLClientWithScikitLearn.logger.info("Service Instance: setting up credentials")

        self.wml_credentials = get_wml_credentials()
        self.client = get_client()

        self.client = get_client()
        self.cos_resource = get_cos_resource()
        self.bucket_names = prepare_cos(self.cos_resource)

    def test_00_check_client_version(self):
        TestWMLClientWithScikitLearn.logger.info("Check client version...")

        self.logger.info("Getting version ...")
        version = self.client.version
        TestWMLClientWithScikitLearn.logger.debug(version)
        self.assertTrue(len(version) > 0)

    def test_01_service_instance_details(self):
        TestWMLClientWithScikitLearn.logger.info("Check client ...")
        self.assertTrue(self.client.__class__.__name__ == 'APIClient')

        self.logger.info("Getting instance details ...")
        details = self.client.service_instance.get_details()
        TestWMLClientWithScikitLearn.logger.debug(details)

        self.assertTrue("published_models" in str(details))
        self.assertEqual(type(details), dict)

    def test_02_prepare(self):
        TestWMLClientWithScikitLearn.models_no = len(self.client.repository.get_model_details()['resources'])
        TestWMLClientWithScikitLearn.definitions_no = len(self.client.repository.get_definition_details()['resources'])
        TestWMLClientWithScikitLearn.experiments_no = len(self.client.repository.get_experiment_details()['resources'])

    def test_03_publish_models(self):
        TestWMLClientWithScikitLearn.logger.info("Creating scikit-learn models ...")

        model_data = create_scikit_learn_model_data()
        predicted = model_data['prediction']

        TestWMLClientWithScikitLearn.logger.debug(predicted)
        self.assertIsNotNone(predicted)

        self.logger.info("Publishing scikit-learn model ...")

        self.client.repository.ModelMetaNames.show()

        model_props = {self.client.repository.ModelMetaNames.AUTHOR_NAME: "IBM",
                       self.client.repository.ModelMetaNames.NAME: "LOCALLY created Digits prediction model"
                       }

        for i in range(0, 51):
            published_model_details = self.client.repository.store_model(model=model_data['model'], meta_props=model_props, training_data=model_data['training_data'], training_target=model_data['training_target'])
            TestWMLClientWithScikitLearn.models_uids.append(self.client.repository.get_model_uid(published_model_details))

    def test_04_get_models_details(self):
        details = self.client.repository.get_model_details()
        self.assertTrue(len(details['resources']) == TestWMLClientWithScikitLearn.models_no + 51)

        details = self.client.repository.get_model_details(limit=5)
        self.assertTrue(len(details['resources']) == 5)

    def test_05_list_models(self):
        stdout_ = sys.stdout
        captured_output = io.StringIO()  # Create StringIO object
        sys.stdout = captured_output  # and redirect stdout.
        self.client.repository.list_models(limit=5)  # Call function.
        sys.stdout = stdout_  # Reset redirect.
        self.assertTrue('Note: Only first 50 records were displayed.' not in captured_output.getvalue())
        self.assertTrue(len(captured_output.getvalue().split('\n')) == 5 + ADDITIONAL_LIST_LINES)

        stdout_ = sys.stdout
        captured_output = io.StringIO()  # Create StringIO object
        sys.stdout = captured_output  # and redirect stdout.
        self.client.repository.list_models()  # Call function.
        sys.stdout = stdout_  # Reset redirect.
        self.assertTrue('Note: Only first 50 records were displayed.' in captured_output.getvalue())
        self.assertTrue(len(captured_output.getvalue().split('\n')) == 51 + ADDITIONAL_LIST_LINES)

        stdout_ = sys.stdout
        captured_output = io.StringIO()  # Create StringIO object
        sys.stdout = captured_output  # and redirect stdout.
        self.client.repository.list_models(limit=51)  # Call function.
        sys.stdout = stdout_  # Reset redirect.
        self.assertTrue('Note: Only first 50 records were displayed.' not in captured_output.getvalue())
        self.assertTrue(len(captured_output.getvalue().split('\n')) == 51 + ADDITIONAL_LIST_LINES)

    def test_06_publish_definitions(self):
        metadata = {
            self.client.repository.DefinitionMetaNames.NAME: "my_training_definition",
            self.client.repository.DefinitionMetaNames.DESCRIPTION: "my_description",
            self.client.repository.DefinitionMetaNames.AUTHOR_NAME: "John Smith",
            self.client.repository.DefinitionMetaNames.FRAMEWORK_NAME: "tensorflow",
            self.client.repository.DefinitionMetaNames.FRAMEWORK_VERSION: "1.5",
            self.client.repository.DefinitionMetaNames.RUNTIME_NAME: "python",
            self.client.repository.DefinitionMetaNames.RUNTIME_VERSION: "3.5",
            self.client.repository.DefinitionMetaNames.EXECUTION_COMMAND: "python3 tensorflow_mnist_softmax.py --trainingIters 20"
        }

        model_content_path = './artifacts/tf-softmax-model.zip'

        definition_details = None

        for i in range(0, 51):
            definition_details = self.client.repository.store_definition(training_definition=model_content_path, meta_props=metadata)
            TestWMLClientWithScikitLearn.definitions_uids.append(self.client.repository.get_definition_uid(definition_details))

        TestWMLClientWithScikitLearn.definition_url = self.client.repository.get_definition_url(definition_details)

    def test_07_get_definitions_details(self):
        details = self.client.repository.get_definition_details()
        self.assertTrue(len(details['resources']) == TestWMLClientWithScikitLearn.definitions_no + 51)

        details = self.client.repository.get_definition_details(limit=5)
        self.assertTrue(len(details['resources']) == 5)

    def test_08_list_definitions(self):
        stdout_ = sys.stdout
        captured_output = io.StringIO()  # Create StringIO object
        sys.stdout = captured_output  # and redirect stdout.
        self.client.repository.list_definitions(limit=5)  # Call function.
        sys.stdout = stdout_  # Reset redirect.
        self.assertTrue('Note: Only first 50 records were displayed.' not in captured_output.getvalue())
        self.assertTrue(len(captured_output.getvalue().split('\n')) == 5 + ADDITIONAL_LIST_LINES)

        stdout_ = sys.stdout
        captured_output = io.StringIO()  # Create StringIO object
        sys.stdout = captured_output  # and redirect stdout.
        self.client.repository.list_definitions()  # Call function.
        sys.stdout = stdout_  # Reset redirect.
        self.assertTrue('Note: Only first 50 records were displayed.' in captured_output.getvalue())
        self.assertTrue(len(captured_output.getvalue().split('\n')) == 51 + ADDITIONAL_LIST_LINES)

        stdout_ = sys.stdout
        captured_output = io.StringIO()  # Create StringIO object
        sys.stdout = captured_output  # and redirect stdout.
        self.client.repository.list_definitions(limit=51)  # Call function.
        sys.stdout = stdout_  # Reset redirect.
        self.assertTrue('Note: Only first 50 records were displayed.' not in captured_output.getvalue())
        self.assertTrue(len(captured_output.getvalue().split('\n')) == 51 + ADDITIONAL_LIST_LINES)

    def test_09_publish_experiments(self):
        metadata = {
            self.client.repository.ExperimentMetaNames.NAME: "xxx",
            self.client.repository.ExperimentMetaNames.TRAINING_DATA_REFERENCE: get_cos_training_data_reference(self.bucket_names),
            self.client.repository.ExperimentMetaNames.TRAINING_RESULTS_REFERENCE: get_cos_training_results_reference(self.bucket_names),
            self.client.repository.ExperimentMetaNames.TRAINING_REFERENCES: [
                {
                    "name": "mnist_nn",
                    "training_definition_url": TestWMLClientWithScikitLearn.definition_url,
                    "compute_configuration": {
                        "name": "p100"
                    }
                }
            ]
        }

        for i in range(0, 51):
            experiment_details = self.client.repository.store_experiment(meta_props=metadata)
            TestWMLClientWithScikitLearn.models_uids.append(self.client.experiments.get_definition_uid(experiment_details))

    def test_10_get_experiments_details(self):
        details = self.client.repository.get_experiment_details()
        self.assertTrue(len(details['resources']) == TestWMLClientWithScikitLearn.experiments_no + 51)

        details = self.client.repository.get_experiment_details(limit=5)
        self.assertTrue(len(details['resources']) == 5)

    def test_11_list_experiments(self):
        stdout_ = sys.stdout
        captured_output = io.StringIO()  # Create StringIO object
        sys.stdout = captured_output  # and redirect stdout.
        self.client.repository.list_experiments(limit=5)  # Call function.
        sys.stdout = stdout_  # Reset redirect.
        self.assertTrue('Note: Only first 50 records were displayed.' not in captured_output.getvalue())
        self.assertTrue(len(captured_output.getvalue().split('\n')) == 5 + ADDITIONAL_LIST_LINES)

        stdout_ = sys.stdout
        captured_output = io.StringIO()  # Create StringIO object
        sys.stdout = captured_output  # and redirect stdout.
        self.client.repository.list_experiments()  # Call function.
        sys.stdout = stdout_  # Reset redirect.
        self.assertTrue('Note: Only first 50 records were displayed.' in captured_output.getvalue())
        self.assertTrue(len(captured_output.getvalue().split('\n')) == 51 + ADDITIONAL_LIST_LINES)

        stdout_ = sys.stdout
        captured_output = io.StringIO()  # Create StringIO object
        sys.stdout = captured_output  # and redirect stdout.
        self.client.repository.list_experiments(limit=51)  # Call function.
        sys.stdout = stdout_  # Reset redirect.
        self.assertTrue('Note: Only first 50 records were displayed.' not in captured_output.getvalue())
        self.assertTrue(len(captured_output.getvalue().split('\n')) == 51 + ADDITIONAL_LIST_LINES)

    def test_12_list(self):
        stdout_ = sys.stdout
        captured_output = io.StringIO()  # Create StringIO object
        sys.stdout = captured_output  # and redirect stdout.
        self.client.repository.list()  # Call function.
        sys.stdout = stdout_  # Reset redirect.
        self.assertTrue('Note: Only first 50 records were displayed.' in captured_output.getvalue())
        self.assertTrue(len(captured_output.getvalue().split('\n')) == 51 + ADDITIONAL_LIST_LINES)

    def test_13_delete_models(self):
        for uid in TestWMLClientWithScikitLearn.models_uids:
            self.client.repository.delete(uid)

    def test_14_delete_definitions(self):
        for uid in TestWMLClientWithScikitLearn.definitions_uids:
            self.client.repository.delete(uid)

    def test_15_delete_experiments(self):
        for uid in TestWMLClientWithScikitLearn.experiments_uids:
            self.client.repository.delete(uid)


if __name__ == '__main__':
    unittest.main()
