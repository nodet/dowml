import unittest
import logging
from ibm_watson_machine_learning.wml_client_error import ApiRequestFailure, WMLClientError
from preparation_and_cleaning import *


class TestWMLClientWithTensorflow(unittest.TestCase):
    deployment_uid = None
    model_uid = None
    definition_uid = None
    trained_model_uid = None
    scoring_url = None
    runtime_uid = None
    logger = logging.getLogger(__name__)

    @classmethod
    def setUpClass(self):
        TestWMLClientWithTensorflow.logger.info("Service Instance: setting up credentials")
        self.wml_credentials = get_wml_credentials()
        self.client = get_client()
        self.cos_resource = get_cos_resource()
        self.bucket_names = prepare_cos(self.cos_resource)

    @classmethod
    def tearDownClass(self):
        clean_cos(self.cos_resource, self.bucket_names)
        #pass

    def test_01_service_instance_details(self):
        TestWMLClientWithTensorflow.logger.info("Check client ...")
        self.assertTrue(self.client.__class__.__name__ == 'APIClient')

        TestWMLClientWithTensorflow.logger.info("Getting instance details ...")
        details = self.client.service_instance.get_details()
        TestWMLClientWithTensorflow.logger.debug(details)

        self.assertTrue("published_models" in str(details))
        self.assertEqual(type(details), dict)

    def test_02_get_trained_models(self):
        TestWMLClientWithTensorflow.logger.info("Get trained models details...")
        models = self.client.training.get_details()
        TestWMLClientWithTensorflow.logger.debug("Models: " + str(models))
        self.assertIsNotNone(models)

    def test_03_save_definition(self):
        TestWMLClientWithTensorflow.logger.info("Save model definition ...")

        self.client.repository.DefinitionMetaNames.show()

        metadata = {
            self.client.repository.DefinitionMetaNames.NAME: "distributed with metrics",
            self.client.repository.DefinitionMetaNames.DESCRIPTION: "my_description",
            self.client.repository.DefinitionMetaNames.AUTHOR_NAME: "John Smith",
            self.client.repository.DefinitionMetaNames.AUTHOR_EMAIL: "js@js.com",
            self.client.repository.DefinitionMetaNames.FRAMEWORK_NAME: "tensorflow",
            self.client.repository.DefinitionMetaNames.FRAMEWORK_VERSION: "1.5",
            self.client.repository.DefinitionMetaNames.RUNTIME_NAME: "python",
            self.client.repository.DefinitionMetaNames.RUNTIME_VERSION: "3.5",
            self.client.repository.DefinitionMetaNames.EXECUTION_COMMAND: "PS_HOSTS_COUNT=2 ./launcher.py python3 ${MODEL_DIR}/mnist_dist.py --data_dir ${DATA_DIR}"
        }

        model_content_path = './artifacts/tf_distributed/tf_distributed.zip'
        definition_details = self.client.repository.store_definition(training_definition=model_content_path, meta_props=metadata)
        TestWMLClientWithTensorflow.definition_uid = self.client.repository.get_definition_uid(definition_details)
        TestWMLClientWithTensorflow.logger.info("Saved model definition uid: " + str(TestWMLClientWithTensorflow.definition_uid))

    def test_04_get_definition_details(self):
        TestWMLClientWithTensorflow.logger.info("Getting definition details ...")
        details = self.client.repository.get_definition_details()
        TestWMLClientWithTensorflow.logger.info(details)
        self.assertTrue('distributed with metrics' in str(details))
        TestWMLClientWithTensorflow.logger.info("List definitions")
        self.client.repository.list_definitions()

    def test_05_train_using_interactive_mode_s3(self):
        TestWMLClientWithTensorflow.logger.info("Train TensorFlow model ...")

        self.client.training.ConfigurationMetaNames.show()

        training_configuration_dict = {
            self.client.training.ConfigurationMetaNames.NAME: "Hand-written Digit Recognition",
            self.client.training.ConfigurationMetaNames.AUTHOR_NAME: "John Smith",
            self.client.training.ConfigurationMetaNames.AUTHOR_EMAIL: "JohnSmith@js.com",
            self.client.training.ConfigurationMetaNames.DESCRIPTION: "Hand-written Digit Recognition training",
            self.client.training.ConfigurationMetaNames.TRAINING_DATA_REFERENCE: get_cos_training_data_reference(self.bucket_names),
            self.client.training.ConfigurationMetaNames.TRAINING_RESULTS_REFERENCE: get_cos_training_results_reference(self.bucket_names),
            self.client.training.ConfigurationMetaNames.COMPUTE_CONFIGURATION: {"name": "p100x2", "nodes": 3}
        }

        training_details = self.client.training.run(TestWMLClientWithTensorflow.definition_uid, training_configuration_dict, asynchronous=False)
        TestWMLClientWithTensorflow.trained_model_uid = self.client.training.get_run_uid(training_details)
        TestWMLClientWithTensorflow.logger.info(
            "Trained model guid: " + TestWMLClientWithTensorflow.trained_model_uid)
        self.assertTrue('training' in TestWMLClientWithTensorflow.trained_model_uid)

    def test_06_get_trained_status(self):
        status = self.client.training.get_status(TestWMLClientWithTensorflow.trained_model_uid)
        TestWMLClientWithTensorflow.logger.info("Training status: " + str(status))
        self.assertTrue('state' in status)

    def test_07_get_trained_details(self):
        details = self.client.training.get_details(TestWMLClientWithTensorflow.trained_model_uid)
        state = details['entity']['status']['state']

        TestWMLClientWithTensorflow.logger.info("Training state:" + state)
        TestWMLClientWithTensorflow.logger.debug("Training details: " + str(details))

        self.assertTrue('state' in str(details))

        TestWMLClientWithTensorflow.logger.info('Getting details DONE.')
        self.assertTrue('completed' in state)

    def test_08_prepare_runtime(self):
        meta = {
            self.client.runtimes.ConfigurationMetaNames.NAME: "runtime_spec_python_3.5",
            self.client.runtimes.ConfigurationMetaNames.PLATFORM: {
                "name": "python",
                "version": "3.5"
            }
        }
        runtime_details = self.client.runtimes.store(meta)
        TestWMLClientWithTensorflow.runtime_uid = self.client.runtimes.get_uid(runtime_details)

    def test_09_save_trained_model_in_repository(self):
        TestWMLClientWithTensorflow.logger.info("Saving trained model in repo ...")
        meta_props = {
            self.client.repository.ModelMetaNames.NAME: "My supercool TF sample model",
            self.client.repository.ModelMetaNames.FRAMEWORK_NAME: "tensorflow",
            self.client.repository.ModelMetaNames.FRAMEWORK_VERSION: "1.5-py3",
            self.client.repository.ModelMetaNames.RUNTIME_UID: TestWMLClientWithTensorflow.runtime_uid
        }
        saved_model_details = self.client.repository.store_model(TestWMLClientWithTensorflow.trained_model_uid, meta_props)
        TestWMLClientWithTensorflow.model_uid = self.client.repository.get_model_uid(saved_model_details)
        TestWMLClientWithTensorflow.logger.info("Saved model details: {}".format(saved_model_details))
        self.assertIsNotNone(saved_model_details)

    def test_10_check_runtime_uid(self):
        model_details = self.client.repository.get_model_details(TestWMLClientWithTensorflow.model_uid)
        runtime_uid = self.client.runtimes.get_uid(model_details)

        self.assertTrue(runtime_uid == TestWMLClientWithTensorflow.runtime_uid)

    def test_11_check_meta_names(self):
        TestWMLClientWithTensorflow.logger.info("Check meta names")
        author_name = self.client.repository.DefinitionMetaNames.AUTHOR_NAME
        self.assertTrue(author_name == 'author_name')

        author_name2 = self.client.training.ConfigurationMetaNames.AUTHOR_NAME
        self.assertTrue(author_name2 == 'author_name')

    def test_12_get_trained_models_table(self):
        TestWMLClientWithTensorflow.logger.info("List trained models")
        self.client.training.list()

    def test_13_delete_train_run(self):
        TestWMLClientWithTensorflow.logger.info("Delete train run")
        deleted = self.client.training.delete(TestWMLClientWithTensorflow.trained_model_uid)
        TestWMLClientWithTensorflow.logger.info("Deleted run: {}".format(str(deleted)))

        trained_models = self.client.training.get_details()
        self.assertTrue(str(TestWMLClientWithTensorflow.trained_model_uid) not in str(trained_models))

    def test_14_check_if_train_run_deleted(self):
        TestWMLClientWithTensorflow.logger.info("Checking if model has been deleted ...")
        self.assertRaises(WMLClientError, self.client.training.get_status, TestWMLClientWithTensorflow.trained_model_uid)
        self.assertRaises(WMLClientError, self.client.training.get_details, TestWMLClientWithTensorflow.trained_model_uid)
        self.assertRaises(ApiRequestFailure, self.client.repository.store_model, TestWMLClientWithTensorflow.trained_model_uid, {"name": "test"})

    # TODO uncomment when ready
    # def test_15_create_deployment(self):
    #     TestWMLClientWithTensorflow.logger.info("Create deployment")
    #     deployment_details = self.client.deployments.create(artifact_uid=TestWMLClientWithTensorflow.model_uid, name="Test deployment", asynchronous=False)
    #     TestWMLClientWithTensorflow.logger.debug("Deployment details: " + str(deployment_details))
    #     TestWMLClientWithTensorflow.deployment_uid = self.client.deployments.get_uid(deployment_details)
    #     TestWMLClientWithTensorflow.scoring_url = self.client.deployments.get_scoring_url(deployment_details)
    #     self.assertTrue('online' in str(TestWMLClientWithTensorflow.scoring_url))
    #
    # def test_16_scoring(self):
    #     TestWMLClientWithTensorflow.logger.info("Score model")
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
    #     scores = self.client.deployments.score(TestWMLClientWithTensorflow.scoring_url, scoring_payload)
    #     self.assertIsNotNone(scores)
    #
    # def test_17_delete_deployment(self):
    #     TestWMLClientWithTensorflow.logger.info("Delete deployment")
    #     self.client.deployments.delete(TestWMLClientWithTensorflow.deployment_uid)

    def test_18_delete_model(self):
        TestWMLClientWithTensorflow.logger.info("Delete model")
        self.client.repository.delete(TestWMLClientWithTensorflow.model_uid)

    def test_19_delete_definition(self):
        TestWMLClientWithTensorflow.logger.info("Delete definition")
        self.client.repository.delete(TestWMLClientWithTensorflow.definition_uid)

    def test_20_delete_runtime(self):
        self.client.repository.delete(TestWMLClientWithTensorflow.runtime_uid)


if __name__ == '__main__':
    unittest.main()
