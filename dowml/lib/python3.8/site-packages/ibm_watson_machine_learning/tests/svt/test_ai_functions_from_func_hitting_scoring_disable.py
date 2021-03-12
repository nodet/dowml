import unittest

import logging
from preparation_and_cleaning import *
from models_preparation import *

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StringIndexer, IndexToString, HashingTF, IDF, Tokenizer
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml import Pipeline

class TestAIFunction(unittest.TestCase):
    runtime_uid = None
    deployment_uid = None
    function_uid = None
    scoring_url = None
    area_model_uid = None
    recommendation_model_uid = None
    area_deployment_uid = None
    recommendation_deployment_uid = None
    logger = logging.getLogger(__name__)

    @classmethod
    def setUpClass(self):
        TestAIFunction.logger.info("Service Instance: setting up credentials")
        self.data_path = os.path.join(os.getcwd(), 'datasets', 'cars4u', 'car_rental_training_data_v3.csv')

        self.wml_credentials = get_wml_credentials()
        self.client = get_client()

        self.function_name = 'simplest AI function'
        self.deployment_name = "Test deployment"

    def test_01_service_instance_details(self):
        TestAIFunction.logger.info("Check client ...")
        self.assertTrue(self.client.__class__.__name__ == 'APIClient')

        TestAIFunction.logger.info("Getting instance details ...")
        details = self.client.service_instance.get_details()
        TestAIFunction.logger.debug(details)

        self.assertTrue("published_models" in str(details))
        self.assertEqual(type(details), dict)

    def test_02_create_ai_function(self):

        self.client.repository.FunctionMetaNames.show()

        function_props = {
            self.client.repository.FunctionMetaNames.NAME: 'simplest AI function',
            self.client.repository.FunctionMetaNames.DESCRIPTION: 'desc',
            self.client.repository.FunctionMetaNames.TAGS: [{"value": "ProjectA", "description": "Functions created for ProjectA"}],
        }

        from pyspark.sql import SparkSession

        spark = SparkSession \
            .builder \
            .appName("test") \
            .getOrCreate()

        df_data = spark.read.csv(self.data_path, header=True, sep=";", inferSchema = True)

        # action
        train_data, test_data = df_data.select("ID", "Customer_Service", "Business_Area").randomSplit([0.8, 0.2], 24)

        tokenizer = Tokenizer(inputCol="Customer_Service", outputCol="words")
        hashing_tf = HashingTF(inputCol=tokenizer.getOutputCol(), outputCol='hash')
        idf = IDF(inputCol=hashing_tf.getOutputCol(), outputCol="features", minDocFreq=5)

        string_indexer_label = StringIndexer(inputCol="Business_Area", outputCol="label").fit(train_data)
        dt_area = DecisionTreeClassifier(labelCol="label", featuresCol=idf.getOutputCol())
        label_converter = IndexToString(inputCol="prediction", outputCol="predictedLabel", labels=string_indexer_label.labels)
        pipeline_area = Pipeline(stages=[tokenizer, hashing_tf, idf, string_indexer_label, dt_area, label_converter])
        model_area = pipeline_area.fit(train_data)

        # recommendation
        train_data, test_data = df_data.randomSplit([0.8, 0.2], 24)

        string_indexer_gender = StringIndexer(inputCol="Gender", outputCol="gender_ix")
        string_indexer_customer_status = StringIndexer(inputCol="Customer_Status", outputCol="customer_status_ix")
        string_indexer_status = StringIndexer(inputCol="Status", outputCol="status_ix")
        string_indexer_owner = StringIndexer(inputCol="Car_Owner", outputCol="owner_ix")
        string_business_area = StringIndexer(inputCol="Business_Area", outputCol="area_ix")

        assembler = VectorAssembler(inputCols=["gender_ix", "customer_status_ix", "status_ix", "owner_ix", "area_ix", "Children", "Age", "Satisfaction"], outputCol="features")
        string_indexer_action = StringIndexer(inputCol="Action", outputCol="label").fit(df_data)
        label_action_converter = IndexToString(inputCol="prediction", outputCol="predictedLabel", labels=string_indexer_action.labels)
        dt_action = DecisionTreeClassifier()
        pipeline_recommendation = Pipeline(stages=[string_indexer_gender, string_indexer_customer_status, string_indexer_status, string_indexer_action, string_indexer_owner, string_business_area, assembler, dt_action, label_action_converter])

        model_recommendation = pipeline_recommendation.fit(train_data)

        # create model area
        area_model_details = self.client.repository.store_model(model_area, {
            self.client.repository.ModelMetaNames.NAME: "area_model",
            self.client.repository.ModelMetaNames.FRAMEWORK_NAME: "spss-modeler",
            self.client.repository.ModelMetaNames.FRAMEWORK_VERSION: "18.0"
        }, pipeline=pipeline_area, training_data=train_data)
        TestAIFunction.area_model_uid = self.client.repository.get_model_uid(area_model_details)

        # create deployment area
        area_deployment_details = self.client.deployments.create(TestAIFunction.area_model_uid, 'area_deployment')
        TestAIFunction.area_deployment_uid = self.client.deployments.get_uid(area_deployment_details)
        area_scoring_url = self.client.deployments.get_scoring_url(area_deployment_details)

        # create model action
        recommendation_model_details = self.client.repository.store_model(model_recommendation, {
            self.client.repository.ModelMetaNames.NAME: "recommendation_model",
            self.client.repository.ModelMetaNames.FRAMEWORK_NAME: "spss-modeler",
            self.client.repository.ModelMetaNames.FRAMEWORK_VERSION: "18.0"
        }, pipeline=pipeline_recommendation, training_data=train_data)
        TestAIFunction.recommendation_model_uid = self.client.repository.get_model_uid(recommendation_model_details)

        # create deployment action
        recommendation_deployment_details = self.client.deployments.create(TestAIFunction.recommendation_model_uid, 'recommendation_deployment')
        TestAIFunction.recommendation_deployment_uid = self.client.deployments.get_uid(recommendation_deployment_details)
        recommendation_scoring_url = self.client.deployments.get_scoring_url(recommendation_deployment_details)

        def score_wrapper(wml_credentials=self.client.wml_credentials, modelver_area='ver1', scoring_url_area=area_scoring_url, modelver_action='ver2', scoring_url_action=recommendation_scoring_url):
            def score(payload):
                """AI function with model version.

                Example:
                  {"fields": ["ID", "Gender", "Status", "Children", "Age", "Customer_Status", "Car_Owner", "Customer_Service"],
                   "values": [[2624, 'Male', 'S', 0, 49.27, 'Active', 'No', "Good experience with all the rental co.'s I contacted. I Just called with rental dates and received pricing and selected rental co."]]}
                """
                from ibm_watson_machine_learning import APIClient

                client = APIClient(wml_credentials)

                scores_area = client.deployments.score(scoring_url_area, payload)
                scores_action = client.deployments.score(scoring_url_action, payload)

                values = [rec + scores_area['values'][i][slice(15, 17)] + [modelver_area] + scores_action['values'][i][
                    slice(20, 22)] + [modelver_action] for i, rec in enumerate(payload['values'])]

                fields = payload['fields'] + ['Probability_Area', 'Prediction_Area', 'Model_Version_Area'] + [
                    'Probability_Action', 'Prediction_Action', 'Model_Version_Action']

                return {'fields': fields, 'values': values}

            return score

        ai_function_details = self.client.repository.store_function(score_wrapper, function_props)

        TestAIFunction.function_uid = self.client.repository.get_function_uid(ai_function_details)
        function_url = self.client.repository.get_function_url(ai_function_details)
        TestAIFunction.logger.info("AI function ID:" + str(TestAIFunction.function_uid))
        TestAIFunction.logger.info("AI function URL:" + str(function_url))
        self.assertIsNotNone(TestAIFunction.function_uid)
        self.assertIsNotNone(function_url)
        TestAIFunction.runtime_uid = self.client.runtimes.get_uid(ai_function_details)

    def test_03_download_ai_function_content(self):
        try:
            os.remove('test_ai_function.tar.gz')
        except:
            pass

        self.client.repository.download(TestAIFunction.function_uid, filename='test_ai_function.tar.gz')

        try:
            os.remove('test_ai_function.tar.gz')
        except:
            pass

    def test_04_get_details(self):
        details = self.client.repository.get_function_details()
        self.assertTrue(self.function_name in str(details))

        details = self.client.repository.get_function_details(self.function_uid)
        self.assertTrue(self.function_name in str(details))

        details = self.client.repository.get_details()
        self.assertTrue("functions" in details)

        details = self.client.repository.get_details(self.function_uid)
        self.assertTrue(self.function_name in str(details))

    def test_05_list(self):
        self.client.repository.list()

        self.client.repository.list_functions()

    def test_06_create_deployment(self):
        TestAIFunction.logger.info("Create deployment")
        deployment = self.client.deployments.create(artifact_uid=self.function_uid, name=self.deployment_name, asynchronous=False)
        TestAIFunction.logger.debug("Online deployment: " + str(deployment))
        TestAIFunction.scoring_url = self.client.deployments.get_scoring_url(deployment)
        TestAIFunction.logger.debug("Scoring url: {}".format(TestAIFunction.scoring_url))
        TestAIFunction.deployment_uid = self.client.deployments.get_uid(deployment)
        TestAIFunction.logger.debug("Deployment uid: {}".format(TestAIFunction.deployment_uid))
        self.assertTrue("online" in str(deployment))

    def test_07_get_deployment_details(self):
        TestAIFunction.logger.info("Get deployment details")
        deployment_details = self.client.deployments.get_details()
        TestAIFunction.logger.debug("Deployment details: {}".format(deployment_details))
        self.assertTrue(self.deployment_name in str(deployment_details))

    def test_08_score(self):
        # sample_payload = {
        #     "fields": ["gender_ix", "customer_status_ix", "status_ix", "owner_ix", "area_ix", "Children", "Age", "Customer_Service", "Satisfaction"],
        #     "values": [
        #         ['Female', 'Inactive', 'M', 'Yes', 'Product: Availability/Variety/Size', 2, 48.85, 'I thought the representative handled the initial situation badly.  The company was out of cars, with none coming in that day.  Then the representative tried to find us a car at another franchise.  There they were successful.', 0],
        #         ['Female', 'Inactive', 'M', 'No', 'Product: Availability/Variety/Size', 0, 55.00, 'I have had a few recent rentals that have taken a very very long time, with no offer of apology.  In the most recent case, the agent subsequently offered me a car type on an upgrade coupon and then told me it was no longer available because it had just be', 0]
        #     ]
        # }
        sample_payload = {
            "fields": ["ID", "Gender", "Status", "Children", "Age", "Customer_Status", "Car_Owner", "Business_Area", "Customer_Service", "Satisfaction"],
            "values": [[2624, 'Male', 'S', 0, 49.27, 'Active', 'No', "Service: Knowledge", "Good experience with all the rental co.'s I contacted. I Just called with rental dates and received pricing and selected rental co.", 0]]}
        predictions = self.client.deployments.score(TestAIFunction.scoring_url, sample_payload)
        print("Predictions: {}".format(predictions))
        self.assertTrue("values" in str(predictions))

    def test_09_delete_deployment(self):
        TestAIFunction.logger.info("Delete deployment")
        self.client.deployments.delete(TestAIFunction.deployment_uid)

    def test_10_delete_function(self):
        TestAIFunction.logger.info("Delete function")
        self.client.repository.delete(TestAIFunction.function_uid)

    def test_11_delete_runtime(self):
        self.client.runtimes.delete(TestAIFunction.runtime_uid)

    def test_12_clean(self):
        self.client.repository.delete(TestAIFunction.area_model_uid)
        self.client.repository.delete(TestAIFunction.recommendation_model_uid)


if __name__ == '__main__':
    unittest.main()
