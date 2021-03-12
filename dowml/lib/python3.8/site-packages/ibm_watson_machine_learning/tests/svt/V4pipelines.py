import unittest

import logging
from preparation_and_cleaning import *
from models_preparation import *


class TestAIFunction(unittest.TestCase):
    runtime_uid = None
    lib_uid = None
    deployment_uid = None
    pipeline_uid = None
    scoring_url = None
    logger = logging.getLogger(__name__)

    @classmethod
    def setUpClass(self):
        TestAIFunction.logger.info("Service Instance: setting up credentials")
        self.lib_filepath = os.path.join(os.getcwd(), 'artifacts', 'ai_function.gz')
        self.wml_credentials = get_wml_credentials()
        self.client = get_client()

        self.function_name = 'sample v4 pipeline'
        self.deployment_name = "Test deployment"

    def test_01_service_instance_details(self):
        TestAIFunction.logger.info("Check client ...")
        self.assertTrue(self.client.__class__.__name__ == 'APIClient')

        TestAIFunction.logger.info("Getting instance details ...")
        details = self.client.service_instance.get_details()
        TestAIFunction.logger.debug(details)

        self.assertTrue("published_models" in str(details))
        self.assertEqual(type(details), dict)

    def test_02_create_pipeline(self):

        self.client.repository.PipelineMetaNames.show()
        lib_meta = {
            self.client.runtimes.LibraryMetaNames.NAME: "sample v4 pipeline2",
            self.client.runtimes.LibraryMetaNames.PLATFORM:  {"name": "python","versions": ["3.6"]},
            self.client.runtimes.LibraryMetaNames.VERSION: "1",
            self.client.runtimes.LibraryMetaNames.FILEPATH: TestAIFunction.lib_filepath
        }
        lib_details = self.client.runtimes.store_library(lib_meta)
        TestAIFunction.lib_uid = self.client.runtimes.get_library_uid(lib_details)
        metadata = {
            self.client.pipelines.ConfigurationMetaNames.NAME: "sample v4 pipeline",
            self.client.pipelines.ConfigurationMetaNames.DESCRIPTION: "sample description",
            self.client.pipelines.ConfigurationMetaNames.DOCUMENT: {
                "doc_type": "pipeline",
                "version": "2.0",
                "primary_pipeline": "dlaas_only",
                "pipelines": [
                    {
                        "id": "dlaas_only",
                        "runtime_ref": "hybrid",
                        "nodes": [
                            {
                                "id": "training",
                                "type": "model_node",
                                "op": "dl_train",
                                "runtime_ref": "DL",
                                "inputs": [
                                ],
                                "outputs": [],
                                "parameters": {
                                    "name": "tf-mnist",
                                    "description": "Simple MNIST model implemented in TF",
                                    "command": "python3 convolutional_network.py --trainImagesFile ${DATA_DIR}/train-images-idx3-ubyte.gz --trainLabelsFile ${DATA_DIR}/train-labels-idx1-ubyte.gz --testImagesFile ${DATA_DIR}/t10k-images-idx3-ubyte.gz --testLabelsFile ${DATA_DIR}/t10k-labels-idx1-ubyte.gz --learningRate 0.001 --trainingIters 6000",
                                    "model_definition_url": "/v4/libraries"+TestAIFunction.lib_uid,
                                    "compute_configuration_name": "k80",
                                    "compute_configuration_nodes": 1,
                                    "target_bucket": "wml-dev-results"
                                }
                            }
                        ]
                    }
                ],
                "schemas": [
                    {
                        "id": "schema1",
                        "fields": [
                            {
                                "name": "text",
                                "type": "string"
                            }
                        ]
                    }
                ],
                "runtimes": [
                    {
                        "id": "DL",
                        "name": "tensorflow",
                        "version": "1.5-py3"
                    }
                ]
            }

        }
        pipeline_details = self.client.pipelines.store(metadata)
        TestAIFunction.pipeline_uid = self.client.pipelines.get_uid(pipeline_details)



    def test_03_update_pipeline(self):
        pipeline_props = {
            self.client.repository.FunctionMetaNames.NAME: 'sample_v4_pipeline_new',
        }

        details = self.client.pipelines.update(TestAIFunction.pipeline_uid, pipeline_props)
        self.assertFalse('sample v4 pipeline' in json.dumps(details))

    def test_05_get_details(self):

        details = self.client.pipelines.get_details(self.pipeline_uid)
        self.assertTrue('sample_v4_pipeline_new' in str(details))

    def test_06_list(self):
        self.client.pipelines.list()

    def test_12_delete_pipeline(self):

        TestAIFunction.logger.info("Delete function")
        self.client.pipelines.delete(TestAIFunction.pipeline_uid)
        self.client.runtimes.delete_library(TestAIFunction.lib_uid)


if __name__ == '__main__':
    unittest.main()
