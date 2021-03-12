import unittest
import io
import sys
import logging
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, IndexToString, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline
from preparation_and_cleaning import *


class TestWMLClientWithSpark(unittest.TestCase):
    model_uid = None
    run_uid = None
    deployment_uid = None
    logger = logging.getLogger(__name__)
    predict_data = None

    @classmethod
    def setUpClass(self):
        TestWMLClientWithSpark.logger.info("Service Instance: setting up credentials")

        self.wml_credentials = get_wml_credentials()
        self.client = get_client()

        self.filename = "artifacts/GoSales_Tx_NaiveBayes.csv"
        self.data_file = "artifacts/sample_binary_classification_data.txt"
        self.model_name = "SparkMLlibFromObjectLocal Model"

        clean_env(self.client, None)

        self.spark = SparkSession.builder.getOrCreate()
        global df

        df = self.spark.read.load(self.data_file, format='libsvm')

    def test_00_check_client_version(self):
        TestWMLClientWithSpark.logger.info("Check client version...")

        self.logger.info("Getting version ...")
        version = self.client.version
        TestWMLClientWithSpark.logger.debug(version)
        self.assertTrue(len(version) > 0)

    def test_01_service_instance_details(self):
        TestWMLClientWithSpark.logger.info("Check client ...")
        self.assertTrue(self.client.__class__.__name__ == 'APIClient')

        self.logger.info("Getting instance details ...")
        details = self.client.service_instance.get_details()
        TestWMLClientWithSpark.logger.debug(details)

        self.assertTrue("published_models" in str(details))
        self.assertEqual(type(details), dict)

    def test_02_publish_model(self):
        TestWMLClientWithSpark.logger.info("Creating spark model ...")

        df_data = self.spark.read \
            .format('org.apache.spark.sql.execution.datasources.csv.CSVFileFormat') \
            .option('header', 'true') \
            .option('inferSchema', 'true') \
            .load(self.filename)

        splitted_data = df_data.randomSplit([0.8, 0.18, 0.02], 24)
        train_data = splitted_data[0]
        test_data = splitted_data[1]
        TestWMLClientWithSpark.predict_data = splitted_data[2]

        stringIndexer_label = StringIndexer(inputCol="PRODUCT_LINE", outputCol="label").fit(df_data)
        stringIndexer_prof = StringIndexer(inputCol="PROFESSION", outputCol="PROFESSION_IX")
        stringIndexer_gend = StringIndexer(inputCol="GENDER", outputCol="GENDER_IX")
        stringIndexer_mar = StringIndexer(inputCol="MARITAL_STATUS", outputCol="MARITAL_STATUS_IX")

        vectorAssembler_features = VectorAssembler(inputCols=["GENDER_IX", "AGE", "MARITAL_STATUS_IX", "PROFESSION_IX"],
                                                   outputCol="features")
        rf = RandomForestClassifier(labelCol="label", featuresCol="features")
        labelConverter = IndexToString(inputCol="prediction", outputCol="predictedLabel",
                                       labels=stringIndexer_label.labels)
        pipeline_rf = Pipeline(stages=[stringIndexer_label, stringIndexer_prof, stringIndexer_gend, stringIndexer_mar,
                                       vectorAssembler_features, rf, labelConverter])
        model_rf = pipeline_rf.fit(train_data)

        TestWMLClientWithSpark.logger.info("Publishing spark model ...")

        self.client.repository.ModelMetaNames.show()

        model_props = {self.client.repository.ModelMetaNames.AUTHOR_NAME: "IBM",
                       self.client.repository.ModelMetaNames.AUTHOR_EMAIL: "ibm@ibm.com",
                       self.client.repository.ModelMetaNames.NAME: self.model_name,
                       self.client.repository.ModelMetaNames.TRAINING_DATA_REFERENCE: {
                           "name": "DRUG training data",
                           "connection": get_feedback_data_reference(),
                           "source": {
                               "tablename": "DRUG_TRAIN_DATA_UPDATED",
                               "type": "dashdb"
                           }
                       },
                       self.client.repository.ModelMetaNames.EVALUATION_METHOD: "multiclass",
                       self.client.repository.ModelMetaNames.EVALUATION_METRICS: [
                           {
                               "name": "accuracy",
                               "value": 0.64,
                               "threshold": 0.8
                           }
                       ]
                       }

        TestWMLClientWithSpark.logger.info(pipeline_rf)
        TestWMLClientWithSpark.logger.info(type(pipeline_rf))

        published_model = self.client.repository.store_model(model=model_rf, meta_props=model_props,
                                                             training_data=train_data, pipeline=pipeline_rf)
        TestWMLClientWithSpark.model_uid = self.client.repository.get_model_uid(published_model)
        TestWMLClientWithSpark.logger.info("Published model ID:" + str(TestWMLClientWithSpark.model_uid))
        self.assertIsNotNone(TestWMLClientWithSpark.model_uid)

    def test_03_deploy_model(self):
        deployment_details = self.client.deployments.create(TestWMLClientWithSpark.model_uid, "my deployment")
        TestWMLClientWithSpark.deployment_uid = self.client.deployments.get_uid(deployment_details)

    def test_04_create_learning_system(self):
        self.client.learning_system.ConfigurationMetaNames.show()

        meta_prop = {
            self.client.learning_system.ConfigurationMetaNames.FEEDBACK_DATA_REFERENCE: {
                "name": "DRUG feedback",
                "connection": get_feedback_data_reference(),
                "source": {
                    "tablename": "DRUG_FEEDBACK_DATA",
                    "type": "dashdb"
                }
            },
            self.client.learning_system.ConfigurationMetaNames.MIN_FEEDBACK_DATA_SIZE: 100,
            self.client.learning_system.ConfigurationMetaNames.SPARK_REFERENCE: get_spark_reference(),
            self.client.learning_system.ConfigurationMetaNames.AUTO_RETRAIN: "never",
            self.client.learning_system.ConfigurationMetaNames.AUTO_REDEPLOY: "never"
        }
        details = self.client.learning_system.setup(TestWMLClientWithSpark.model_uid, meta_prop)
        self.assertIsNotNone(details)

    def test_05_update_learning_system(self):
        meta_prop = {
            self.client.learning_system.ConfigurationMetaNames.FEEDBACK_DATA_REFERENCE: {
                "name": "DRUG feedback",
                "connection": get_feedback_data_reference(),
                "source": {
                    "tablename": "DRUG_FEEDBACK_DATA",
                    "type": "dashdb"
                }
            },
            self.client.learning_system.ConfigurationMetaNames.MIN_FEEDBACK_DATA_SIZE: 10,
            self.client.learning_system.ConfigurationMetaNames.SPARK_REFERENCE: get_spark_reference(),
            self.client.learning_system.ConfigurationMetaNames.AUTO_RETRAIN: "always",
            self.client.learning_system.ConfigurationMetaNames.AUTO_REDEPLOY: "always"
        }
        details = self.client.learning_system.update(TestWMLClientWithSpark.model_uid, meta_prop)
        self.assertIsNotNone(details)

    def test_06_list(self):
        captured_output = io.StringIO()  # Create StringIO object
        sys.stdout = captured_output  # and redirect stdout.
        self.client.learning_system.list()  # Call function.
        sys.stdout = sys.__stdout__  # Reset redirect.
        self.assertTrue(TestWMLClientWithSpark.model_uid in captured_output.getvalue())

        self.client.learning_system.list()

    def test_07_run_learning_system(self):
        run_details = self.client.learning_system.run(TestWMLClientWithSpark.model_uid, asynchronous=False)
        TestWMLClientWithSpark.run_uid = self.client.learning_system.get_run_uid(run_details)
        url = self.client.learning_system.get_run_href(run_details)
        self.assertIsNotNone(url)

    def test_08_get_runs(self):
        runs = self.client.learning_system.get_runs(TestWMLClientWithSpark.model_uid)
        self.assertIsNotNone(runs)
        self.assertTrue(TestWMLClientWithSpark.run_uid in str(runs))

    def test_09_get_deployment_status(self):
        print(self.client.deployments.get_status(TestWMLClientWithSpark.deployment_uid))

    def test_10_get_metrics(self):
        metrics = self.client.learning_system.get_metrics(TestWMLClientWithSpark.model_uid)
        self.assertTrue('setup' in str(metrics))

    def test_11_list_metrics(self):
        captured_output = io.StringIO()  # Create StringIO object
        sys.stdout = captured_output  # and redirect stdout.
        self.client.learning_system.list_metrics(TestWMLClientWithSpark.model_uid)  # Call function.
        sys.stdout = sys.__stdout__  # Reset redirect.
        self.assertTrue('setup' in captured_output.getvalue())

        self.client.learning_system.list_metrics(TestWMLClientWithSpark.model_uid)

    def test_12_list_runs(self):
        captured_output = io.StringIO()  # Create StringIO object
        sys.stdout = captured_output  # and redirect stdout.
        self.client.learning_system.list_runs(TestWMLClientWithSpark.model_uid)  # Call function.
        sys.stdout = sys.__stdout__  # Reset redirect.
        self.assertTrue(TestWMLClientWithSpark.run_uid in captured_output.getvalue())

        self.client.learning_system.list_runs(TestWMLClientWithSpark.model_uid)

    def test_13_list_all_runs(self):
        captured_output = io.StringIO()  # Create StringIO object
        sys.stdout = captured_output  # and redirect stdout.
        self.client.learning_system.list_runs()  # Call function.
        sys.stdout = sys.__stdout__  # Reset redirect.
        self.assertTrue(TestWMLClientWithSpark.run_uid in captured_output.getvalue())

        self.client.learning_system.list_runs()

    def test_14_get_run_details(self):
        details = self.client.learning_system.get_run_details(TestWMLClientWithSpark.run_uid)
        self.assertTrue('COMPLETED' in str(details))

    def test_15_send_feedback(self):
        result = self.client.learning_system.send_feedback(
            TestWMLClientWithSpark.model_uid,
            [list(x) for x in TestWMLClientWithSpark.predict_data.collect()]
        )
        self.assertTrue('rows_inserted' in str(result))

    def test_16_delete_deployment(self):
        self.client.deployments.delete(TestWMLClientWithSpark.deployment_uid)

    def test_17_delete_model(self):
        TestWMLClientWithSpark.logger.info("Delete model")
        self.client.repository.delete(TestWMLClientWithSpark.model_uid)


if __name__ == '__main__':
    unittest.main()
