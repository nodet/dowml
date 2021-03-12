import unittest
import os
import json
import logging
from preparation_and_cleaning import *


class TestWMLClientWithSpark(unittest.TestCase):
    deployment_uid = None
    model_uid = None
    scoring_url = None
    definition_url = None
    logger = logging.getLogger(__name__)

    @classmethod
    def setUpClass(self):
        TestWMLClientWithSpark.logger.info("Service Instance: setting up credentials")

        self.wml_credentials = get_wml_credentials()
        self.client = get_client()
        self.model_path = os.path.join(os.getcwd(), 'artifacts', 'heart-drug-sample', 'drug-selection-model.tgz')
        self.pipeline_path = os.path.join(os.getcwd(), 'artifacts', 'heart-drug-sample', 'drug-selection-pipeline.tgz')
        self.meta_path = os.path.join(os.getcwd(), 'artifacts', 'heart-drug-sample', 'drug-selection-meta.json')

        with open(TestWMLClientWithSpark.meta_path) as json_data:
            metadata = json.load(json_data)

        self.model_meta = metadata['model_meta']
        self.pipeline_meta = metadata['pipeline_meta']

    def test_1_service_instance_details(self):
        TestWMLClientWithSpark.logger.info("Check client ...")
        self.assertTrue(type(self.client).__name__ == 'APIClient')

        TestWMLClientWithSpark.logger.info("Getting instance details ...")
        details = self.client.service_instance.get_details()
        TestWMLClientWithSpark.logger.debug(details)

        self.assertTrue("published_models" in str(details))
        self.assertEqual(type(details), dict)


    def test_3_publish_model(self):
        TestWMLClientWithSpark.logger.info("Publishing spark model ...")
        self.client.repository.ModelMetaNames.show()

        model_props = {
            self.client.repository.ModelMetaNames.NAME: "SparkModel-from-tar",
            self.client.repository.ModelMetaNames.TYPE: "mllib_2.3",
            self.client.repository.ModelMetaNames.RUNTIME_UID: "spark-mllib_2.3",
            self.client.repository.ModelMetaNames.TRAINING_DATA_REFERENCES:
                [
                    {
                        'type': 's3',
                        'connection': {
                            'endpoint_url': 'not_applicable',
                            'access_key_id': 'not_applicable',
                            'secret_access_key': 'not_applicable'
                        },
                        'location': {
                            'bucket': 'not_applicable'
                        },
                        'schema': {
                            'id': '1',
                            'type': 'struct',
                            'fields': [{
                                'name': 'AGE',
                                'type': 'float',
                                'nullable': True,
                                'metadata': {
                                    'modeling_role': 'target'
                                }
                            }, {
                                'name': 'SEX',
                                'type': 'string',
                                'nullable': True,
                                'metadata': {}
                            }, {
                                'name': 'CHOLESTEROL',
                                'type': 'string',
                                'nullable': True,
                                'metadata': {}
                            }, {
                                'name': 'BP',
                                'type': 'string',
                                'nullable': True,
                                'metadata': {}
                            }, {
                                'name': 'NA',
                                'type': 'float',
                                'nullable': True,
                                'metadata': {}
                            }, {
                                'name': 'K',
                                'type': 'float',
                                'nullable': True,
                                'metadata': {}
                            }]
                        }
                    }
                    ]}


        print('XXX' + str(model_props))
        published_model = self.client.repository.store_model(model=self.model_path, meta_props=model_props)
        print("Model details: " + str(published_model))

        TestWMLClientWithSpark.model_uid = self.client.repository.get_model_uid(published_model)
        TestWMLClientWithSpark.logger.info("Published model ID:" + str(TestWMLClientWithSpark.model_uid))
        self.assertIsNotNone(TestWMLClientWithSpark.model_uid)

    def test_4_get_details(self):
        TestWMLClientWithSpark.logger.info("Get model details")
        details = self.client.repository.get_details(self.model_uid)
        print(details)
        TestWMLClientWithSpark.logger.debug("Model details: " + str(details))
        self.assertTrue("SparkModel" in str(details))

    def test_5_create_deployment(self):
        TestWMLClientWithSpark.logger.info("Create deployment")
        deployment = self.client.deployments.create(self.model_uid, meta_props={self.client.deployments.ConfigurationMetaNames.NAME: "best-drug model deployment",self.client.deployments.ConfigurationMetaNames.ONLINE: {}})
        TestWMLClientWithSpark.logger.info("model_uid: " + self.model_uid)
        TestWMLClientWithSpark.logger.debug("Online deployment: " + str(deployment))
        TestWMLClientWithSpark.scoring_url = self.client.deployments.get_scoring_href(deployment)
        TestWMLClientWithSpark.deployment_uid = self.client.deployments.get_uid(deployment)
        self.assertTrue("online" in str(deployment))

    def test_6_get_deployment_details(self):
        TestWMLClientWithSpark.logger.info("Get deployment details")
        deployment_details = self.client.deployments.get_details(TestWMLClientWithSpark.deployment_uid)
        print(deployment_details)
        self.assertTrue('best-drug model deployment' in str(deployment_details))

    def test_6_score(self):
        TestWMLClientWithSpark.logger.info("Score the model")
        scoring_data = {
            self.client.deployments.ScoringMetaNames.INPUT_DATA: [
                {
                    "fields": ["AGE", "SEX", "BP", "CHOLESTEROL", "NA", "K"],
                    "values": [[20.0, "F", "HIGH", "HIGH", 0.71, 0.07], [55.0, "M", "LOW", "HIGH", 0.71, 0.07]]
                }
            ]
        }

        predictions = self.client.deployments.score(TestWMLClientWithSpark.deployment_uid, scoring_data)
        print(predictions)
        self.assertTrue("predictedLabel" in str(predictions))

    def test_7_delete_deployment(self):
        TestWMLClientWithSpark.logger.info("Delete deployment")
        self.client.deployments.delete(TestWMLClientWithSpark.deployment_uid)

    def test_8_delete_model(self):
        TestWMLClientWithSpark.logger.info("Delete model")
        self.client.repository.delete(TestWMLClientWithSpark.model_uid)


if __name__ == '__main__':
    unittest.main()
