import unittest
import os
from sklearn.datasets import load_svmlight_file
import logging
from ibm_watson_machine_learning.tests.utils import get_wml_credentials, get_cos_credentials, get_space_id
from ibm_watson_machine_learning.tests.utils.cleanup import space_cleanup
from ibm_watson_machine_learning import APIClient


class TestWMLClientWithDO(unittest.TestCase):
    deployment_uid = None
    model_uid = None
    scoring_url = None
    space_name = 'tests_sdk_space'
    space_id = None
    model_path = os.path.join('.', 'svt', 'artifacts', 'do-model.tar.gz')
    logger = logging.getLogger(__name__)

    @classmethod
    def setUpClass(cls) -> None:
        """
        Load WML credentials from config.ini file based on ENV variable.
        """

        cls.wml_credentials = get_wml_credentials()

        cls.wml_client = APIClient(wml_credentials=cls.wml_credentials)

        if not cls.wml_client.ICP:
            cls.cos_credentials = get_cos_credentials()
            cls.cos_endpoint = cls.cos_credentials.get('endpoint_url')
            cls.cos_resource_instance_id = cls.cos_credentials.get('resource_instance_id')

        cls.wml_client = APIClient(wml_credentials=cls.wml_credentials)
        cls.project_id = cls.wml_credentials.get('project_id')

    def test_00a_space_cleanup(self):
        space_cleanup(self.wml_client,
                      get_space_id(self.wml_client, self.space_name,
                                   cos_resource_instance_id=self.cos_resource_instance_id),
                      days_old=7)
        TestWMLClientWithDO.space_id = get_space_id(self.wml_client, self.space_name,
                                                    cos_resource_instance_id=self.cos_resource_instance_id)

        # if self.wml_client.ICP:
        #     self.wml_client.set.default_project(self.project_id)
        # else:
        self.wml_client.set.default_space(self.space_id)

    def test_02_publish_model(self):
        TestWMLClientWithDO.logger.info("Publishing decision optimization model ...")

        self.wml_client.repository.ModelMetaNames.show()

        self.wml_client.software_specifications.list()
        sw_spec_id = self.wml_client.software_specifications.get_id_by_name("do_12.9")

        output_data_schema = [{'id': 'stest',
                               'type': 'list',
                               'fields': [{'name': 'age', 'type': 'float'},
                                          {'name': 'sex', 'type': 'float'},
                                          {'name': 'cp', 'type': 'float'},
                                          {'name': 'restbp', 'type': 'float'},
                                          {'name': 'chol', 'type': 'float'},
                                          {'name': 'fbs', 'type': 'float'},
                                          {'name': 'restecg', 'type': 'float'},
                                          {'name': 'thalach', 'type': 'float'},
                                          {'name': 'exang', 'type': 'float'},
                                          {'name': 'oldpeak', 'type': 'float'},
                                          {'name': 'slope', 'type': 'float'},
                                          {'name': 'ca', 'type': 'float'},
                                          {'name': 'thal', 'type': 'float'}]
                               }, {'id': 'teste2',
                                   'type': 'test',
                                   'fields': [{'name': 'age', 'type': 'float'},
                                              {'name': 'sex', 'type': 'float'},
                                              {'name': 'cp', 'type': 'float'},
                                              {'name': 'restbp', 'type': 'float'},
                                              {'name': 'chol', 'type': 'float'},
                                              {'name': 'fbs', 'type': 'float'},
                                              {'name': 'restecg', 'type': 'float'},
                                              {'name': 'thalach', 'type': 'float'},
                                              {'name': 'exang', 'type': 'float'},
                                              {'name': 'oldpeak', 'type': 'float'},
                                              {'name': 'slope', 'type': 'float'},
                                              {'name': 'ca', 'type': 'float'},
                                              {'name': 'thal', 'type': 'float'}]}]

        model_props = {self.wml_client.repository.ModelMetaNames.NAME: "LOCALLY created DO model",
            self.wml_client.repository.ModelMetaNames.TYPE: "do-docplex_12.9",
            self.wml_client.repository.ModelMetaNames.SOFTWARE_SPEC_UID: sw_spec_id,
            self.wml_client.repository.ModelMetaNames.OUTPUT_DATA_SCHEMA: output_data_schema
                       }
        published_model = self.wml_client.repository.store_model(model=self.model_path, meta_props=model_props)
        TestWMLClientWithDO.model_uid = self.wml_client.repository.get_model_uid(published_model)
        TestWMLClientWithDO.logger.info("Published model ID:" + str(TestWMLClientWithDO.model_uid))
        self.assertIsNotNone(TestWMLClientWithDO.model_uid)

    def test_03_publish_model_details(self):
        TestWMLClientWithDO.logger.info("Get published model details ...")
        details = self.wml_client.repository.get_details(self.model_uid)

        TestWMLClientWithDO.logger.debug("Model details: " + str(details))

    def test_04_create_deployment(self):
        TestWMLClientWithDO.logger.info("Create deployment ...")
        global deployment
        deployment = self.wml_client.deployments.create(
            self.model_uid,
            meta_props={
                self.wml_client.deployments.ConfigurationMetaNames.NAME: "Test deployment",
                self.wml_client.deployments.ConfigurationMetaNames.BATCH:{},
                self.wml_client.deployments.ConfigurationMetaNames.HARDWARE_SPEC: {"name": "S", "num_nodes": 1}
            }
        )

        TestWMLClientWithDO.logger.info("model_uid: " + self.model_uid)
        TestWMLClientWithDO.logger.info("Batch deployment: " + str(deployment))
        TestWMLClientWithDO.deployment_uid = self.wml_client.deployments.get_uid(deployment)
        self.assertTrue("batch" in str(deployment))


    # def test_06_update_model_version(self):
    #     deployment_details = self.wml_client.deployments.get_details(TestWMLClientWithXGBoost.deployment_uid)
    #
    #     self.wml_client.deployments.update(TestWMLClientWithXGBoost.deployment_uid)
    #     new_deployment_details = self.wml_client.deployments.get_details(TestWMLClientWithXGBoost.deployment_uid)
    #
    #     self.assertNotEquals(deployment_details['entity']['deployed_version']['guid'], new_deployment_details['entity']['deployed_version']['guid'])

    def test_07_get_deployment_details(self):
        TestWMLClientWithDO.logger.info("Get deployment details ...")
        deployment_details = self.wml_client.deployments.get_details()
        self.assertTrue("Test deployment" in str(deployment_details))

        TestWMLClientWithDO.logger.debug("Batch deployment: " + str(deployment_details))
        TestWMLClientWithDO.deployment_uid = self.wml_client.deployments.get_uid(deployment)
        self.assertIsNotNone(TestWMLClientWithDO.deployment_uid)

    def test_08_score(self):
        TestWMLClientWithDO.logger.info("Batch model scoring ...")

        import pandas as pd
        diet_food = pd.DataFrame([["Roasted Chicken", 0.84, 0, 10],
                                  ["Spaghetti W/ Sauce", 0.78, 0, 10],
                                  ["Tomato,Red,Ripe,Raw", 0.27, 0, 10],
                                  ["Apple,Raw,W/Skin", 0.24, 0, 10],
                                  ["Grapes", 0.32, 0, 10],
                                  ["Chocolate Chip Cookies", 0.03, 0, 10],
                                  ["Lowfat Milk", 0.23, 0, 10],
                                  ["Raisin Brn", 0.34, 0, 10],
                                  ["Hotdog", 0.31, 0, 10]], columns=["name", "unit_cost", "qmin", "qmax"])

        diet_food_nutrients = pd.DataFrame([
            ["Spaghetti W/ Sauce", 358.2, 80.2, 2.3, 3055.2, 11.6, 58.3, 8.2],
            ["Roasted Chicken", 277.4, 21.9, 1.8, 77.4, 0, 0, 42.2],
            ["Tomato,Red,Ripe,Raw", 25.8, 6.2, 0.6, 766.3, 1.4, 5.7, 1],
            ["Apple,Raw,W/Skin", 81.4, 9.7, 0.2, 73.1, 3.7, 21, 0.3],
            ["Grapes", 15.1, 3.4, 0.1, 24, 0.2, 4.1, 0.2],
            ["Chocolate Chip Cookies", 78.1, 6.2, 0.4, 101.8, 0, 9.3, 0.9],
            ["Lowfat Milk", 121.2, 296.7, 0.1, 500.2, 0, 11.7, 8.1],
            ["Raisin Brn", 115.1, 12.9, 16.8, 1250.2, 4, 27.9, 4],
            ["Hotdog", 242.1, 23.5, 2.3, 0, 0, 18, 10.4]
        ], columns=["Food", "Calories", "Calcium", "Iron", "Vit_A", "Dietary_Fiber", "Carbohydrates", "Protein"])

        diet_nutrients = pd.DataFrame([
            ["Calories", 2000, 2500],
            ["Calcium", 800, 1600],
            ["Iron", 10, 30],
            ["Vit_A", 5000, 50000],
            ["Dietary_Fiber", 25, 100],
            ["Carbohydrates", 0, 300],
            ["Protein", 50, 100]
        ], columns=["name", "qmin", "qmax"])

        job_payload_ref = {
            self.wml_client.deployments.DecisionOptimizationMetaNames.INPUT_DATA: [
                {
                    "id": "diet_food.csv",
                    "values": diet_food
                },
                {
                    "id": "diet_food_nutrients.csv",
                    "values": diet_food_nutrients
                },
                {
                    "id": "diet_nutrients.csv",
                    "values": diet_nutrients
                }
            ],
            self.wml_client.deployments.DecisionOptimizationMetaNames.OUTPUT_DATA: [
                {
                    "id": ".*.csv"
                }
            ]
        }

        TestWMLClientWithDO.logger.debug("Scoring data: {}".format(
            job_payload_ref[self.wml_client.deployments.DecisionOptimizationMetaNames.INPUT_DATA]))
        job = self.wml_client.deployments.create_job(self.deployment_uid, meta_props=job_payload_ref)

        import time

        job_id = self.wml_client.deployments.get_job_uid(job)

        elapsed_time = 0
        while self.wml_client.deployments.get_job_status(job_id).get('state') != 'completed' and elapsed_time < 300:
            elapsed_time += 10
            time.sleep(10)
        if self.wml_client.deployments.get_job_status(job_id).get('state') == 'completed':
            job_details_do = self.wml_client.deployments.get_job_details(job_id)
            kpi = job_details_do['entity']['decision_optimization']['solve_state']['details']['KPI.Total Calories']
            print(f"KPI: {kpi}")
        else:
            print("Job hasn't completed successfully in 5 minutes.")

        TestWMLClientWithDO.logger.debug("Prediction: " + str(job_details_do))
        self.assertTrue("output_data" in str(job_details_do))

    def test_09_delete_deployment(self):
        TestWMLClientWithDO.logger.info("Delete model deployment ...")
        self.wml_client.deployments.delete(TestWMLClientWithDO.deployment_uid)

    def test_10_delete_model(self):
        TestWMLClientWithDO.logger.info("Delete model ...")
        self.wml_client.repository.delete(TestWMLClientWithDO.model_uid)

if __name__ == '__main__':
    unittest.main()
