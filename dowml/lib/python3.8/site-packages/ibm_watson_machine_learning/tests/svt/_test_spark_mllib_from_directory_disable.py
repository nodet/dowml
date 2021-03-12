import unittest
import os
import sys
from os.path import join as path_join

SPARK_HOME_PATH = os.environ['SPARK_HOME']
PYSPARK_PATH = str(SPARK_HOME_PATH) + "/python/"
sys.path.insert(1, path_join(PYSPARK_PATH))

from pyspark.sql import SparkSession
import logging
from preparation_and_cleaning import *


class TestWMLClientWithSpark(unittest.TestCase):
    deployment_uid = None
    model_uid = None
    scoring_url = None
    logger = logging.getLogger(__name__)

    @classmethod
    def setUpClass(self):
        TestWMLClientWithSpark.logger.info("Service Instance: setting up credentials")
        # reload(site)

        self.wml_credentials = get_wml_credentials()
        self.client = get_client()
        self.model_path = os.path.join(os.getcwd(), 'artifacts', 'spark_mllib_model')

        self.filename = "artifacts/GoSales_Tx_NaiveBayes.csv"
        self.model_name = "SparkMLlibFromObjectLocal Model"
        self.deployment_name = "Test deployment"

        self.spark = SparkSession.builder.getOrCreate()
        global df
        df = self.spark.read.load(
            os.path.join(os.environ['SPARK_HOME'], 'data', 'mllib', 'sample_binary_classification_data.txt'),
            format='libsvm')

    def test_1_service_instance_details(self):
        TestWMLClientWithSpark.logger.info("Check client ...")
        self.assertTrue(type(self.client).__name__ == 'APIClient')

        TestWMLClientWithSpark.logger.info("Getting instance details ...")
        details = self.client.service_instance.get_details()
        TestWMLClientWithSpark.logger.debug(details)

        self.assertTrue("published_models" in str(details))
        self.assertEqual(type(details), dict)

    def test_2_publish_model(self):
        df_data = self.spark.read \
            .format('org.apache.spark.sql.execution.datasources.csv.CSVFileFormat') \
            .option('header', 'true') \
            .option('inferSchema', 'true') \
            .load(self.filename)

        splitted_data = df_data.randomSplit([0.8, 0.18, 0.02], 24)
        train_data = splitted_data[0]

        TestWMLClientWithSpark.logger.info("Publishing spark model ...")

        self.client.repository.ModelMetaNames.show()

        model_props = {self.client.repository.ModelMetaNames.AUTHOR_NAME: "IBM",
                       self.client.repository.ModelMetaNames.AUTHOR_EMAIL: "ibm@ibm.com",
                       self.client.repository.ModelMetaNames.NAME: self.model_name,
                       self.client.repository.ModelMetaNames.FRAMEWORK_NAME: "mllib"
                       }
        published_model = self.client.repository.store_model(model=self.model_path, meta_props=model_props, training_data=train_data)
        TestWMLClientWithSpark.model_uid = self.client.repository.get_model_uid(published_model)
        TestWMLClientWithSpark.logger.info("Published model ID:" + str(TestWMLClientWithSpark.model_uid))
        self.assertIsNotNone(TestWMLClientWithSpark.model_uid)

    def test_3_publish_model_details(self):
        details = self.client.repository.get_details(self.model_uid)
        TestWMLClientWithSpark.logger.debug("Model details: " + str(details))
        self.assertTrue(self.model_name in str(details))

    def test_4_create_deployment(self):
        deployment = self.client.deployments.create(artifact_uid=self.model_uid, name=self.deployment_name, asynchronous=False)
        TestWMLClientWithSpark.logger.info("model_uid: " + self.model_uid)
        TestWMLClientWithSpark.logger.debug("Online deployment: " + str(deployment))
        TestWMLClientWithSpark.scoring_url = self.client.deployments.get_scoring_url(deployment)
        TestWMLClientWithSpark.deployment_uid = self.client.deployments.get_uid(deployment)
        self.assertTrue("online" in str(deployment))

    def test_5_get_deployment_details(self):
        deployment_details = self.client.deployments.get_details()
        self.assertTrue(self.deployment_name in str(deployment_details))

    def test_6_score(self):
        scoring_data = {"fields": ["GENDER","AGE","MARITAL_STATUS","PROFESSION"],"values": [["M",23,"Single","Student"],["M",55,"Single","Executive"]]}
        predictions = self.client.deployments.score(TestWMLClientWithSpark.scoring_url, scoring_data)
        self.assertTrue("prediction" in str(predictions))

    def test_7_delete_deployment(self):
        self.client.deployments.delete(TestWMLClientWithSpark.deployment_uid)

    def test_8_delete_model(self):
        self.client.repository.delete(TestWMLClientWithSpark.model_uid)

if __name__ == '__main__':
    unittest.main()
