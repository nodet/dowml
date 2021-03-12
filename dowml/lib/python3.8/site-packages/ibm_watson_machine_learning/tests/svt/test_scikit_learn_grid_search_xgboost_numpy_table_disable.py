import unittest
import logging
from ibm_watson_machine_learning.wml_client_error import WMLClientError
from preparation_and_cleaning import *
from models_preparation import *
import pandas as pd
from xgboost import XGBClassifier
import numpy as np
from sklearn.grid_search import GridSearchCV
import pickle


class TestWMLClientWithScikitLearn(unittest.TestCase):
    logger = logging.getLogger(__name__)

    @classmethod
    def setUpClass(self):
        TestWMLClientWithScikitLearn.logger.info("Service Instance: setting up credentials")

        self.wml_credentials = get_wml_credentials()
        self.client = get_client()

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

    def test_02_publish_model(self):
        TestWMLClientWithScikitLearn.logger.info("Creating scikit-learn model ...")

        df_iris = pd.read_csv(os.path.join('artifacts', 'iris.csv'))
        y = np.ravel(df_iris[['Species']])
        X = df_iris[['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width']]

        GradParameter_candidates = {'n_estimators': [10],
                                    'learning_rate': [0.1],
                                    'max_depth': [2, 3],
                                    'subsample': [1.0]}

        clfGrad = GridSearchCV(estimator=XGBClassifier(),
                               param_grid=GradParameter_candidates,
                               scoring='accuracy',
                               n_jobs=1,
                               cv=5,
                               refit=True)

        print(X)

        clfGrad.fit(X.values, y)
        clfGrad.predict(X.head(5).values)
        clfGrad.predict_proba(X.head(5).values)

        dir_path = os.path.join('artifacts', 'scikit_learn_grid_search_xgboost_model')

        try:
            os.mkdir(dir_path)
        except:
            pass

        model_path = os.path.join(dir_path, 'scikit_learn_grid_search_xgboost_model.pkl')

        # with open(model_path, 'wb') as f:
        #     print(dir(clfGrad.best_estimator_))
        #     print(clfGrad.best_estimator_.get_params)
        #     print(clfGrad.best_estimator_.get_xgb_params)
        #     pickle.dump(clfGrad.best_estimator_, f)

        with open(model_path, 'wb') as f:
            print(dir(clfGrad))
            pickle.dump(clfGrad, f)

        self.logger.info("Publishing scikit-learn grid search model ...")

        self.client.repository.ModelMetaNames.show()

        model_props = {self.client.repository.ModelMetaNames.AUTHOR_NAME: "IBM",
                       self.client.repository.ModelMetaNames.AUTHOR_EMAIL: "ibm@ibm.com",
                       self.client.repository.ModelMetaNames.NAME: "Locally created XGB",
                       self.client.repository.ModelMetaNames.FRAMEWORK_NAME: 'scikit-learn',
                       #self.client.repository.ModelMetaNames.FRAMEWORK_VERSION: '0.6'
                       # self.client.repository.ModelMetaNames.TRAINING_DATA_REFERENCE: {},
                       # self.client.repository.ModelMetaNames.EVALUATION_METHOD: "multiclass",
                       # self.client.repository.ModelMetaNames.EVALUATION_METRICS: [
                       #     {
                       #         "name": "accuracy",
                       #         "value": 0.64,
                       #         "threshold": 0.8
                       #     }
                       # ]
                       }
        published_model_details = self.client.repository.store_model(model=dir_path, meta_props=model_props)
        TestWMLClientWithScikitLearn.model_uid = self.client.repository.get_model_uid(published_model_details)
        TestWMLClientWithScikitLearn.model_url = self.client.repository.get_model_url(published_model_details)
        self.logger.info("Published model ID:" + str(TestWMLClientWithScikitLearn.model_uid))
        self.logger.info("Published model URL:" + str(TestWMLClientWithScikitLearn.model_url))
        self.assertIsNotNone(TestWMLClientWithScikitLearn.model_uid)
        self.assertIsNotNone(TestWMLClientWithScikitLearn.model_url)

        import shutil
        shutil.rmtree(dir_path)

    def test_03_download_model(self):
        TestWMLClientWithScikitLearn.logger.info("Download model")
        try:
            os.remove('download_test_url')
        except OSError:
            pass

        try:
            file = open('download_test_uid', 'r')
        except IOError:
            file = open('download_test_uid', 'w')
            file.close()

        self.client.repository.download(TestWMLClientWithScikitLearn.model_uid, filename='download_test_url')
        self.assertRaises(WMLClientError, self.client.repository.download, TestWMLClientWithScikitLearn.model_uid, filename='download_test_uid')

    def test_04_get_details(self):
        TestWMLClientWithScikitLearn.logger.info("Get model details")
        details = self.client.repository.get_details(self.model_uid)
        TestWMLClientWithScikitLearn.logger.debug("Model details: " + str(details))
        self.assertTrue("Locally created XGB" in str(details))

        details_all = self.client.repository.get_details()
        TestWMLClientWithScikitLearn.logger.debug("All artifacts details: " + str(details_all))
        self.assertTrue("Locally created XGB" in str(details_all))

        details_models = self.client.repository.get_model_details()
        TestWMLClientWithScikitLearn.logger.debug("All models details: " + str(details_models))
        self.assertTrue("Locally created XGB" in str(details_models))

    def test_05_create_deployment(self):
        TestWMLClientWithScikitLearn.logger.info("Create deployments")
        deployment = self.client.deployments.create(artifact_uid=self.model_uid, name="Test deployment", asynchronous=False)
        TestWMLClientWithScikitLearn.logger.info("model_uid: " + self.model_uid)
        TestWMLClientWithScikitLearn.logger.info("Online deployment: " + str(deployment))
        self.assertTrue(deployment is not None)
        TestWMLClientWithScikitLearn.scoring_url = self.client.deployments.get_scoring_url(deployment)
        TestWMLClientWithScikitLearn.deployment_uid = self.client.deployments.get_uid(deployment)
        self.assertTrue("online" in str(deployment))
        self.client.deployments.get_status(TestWMLClientWithScikitLearn.deployment_uid)

    def test_06_get_deployment_details(self):
        TestWMLClientWithScikitLearn.logger.info("Get deployment details")
        deployment_details = self.client.deployments.get_details()
        self.assertTrue("Test deployment" in str(deployment_details))

    def test_07_get_deployment_details_using_uid(self):
        TestWMLClientWithScikitLearn.logger.info("Get deployment details using uid")
        deployment_details = self.client.deployments.get_details(TestWMLClientWithScikitLearn.deployment_uid)
        self.assertIsNotNone(deployment_details)

    def test_08_score(self):
        TestWMLClientWithScikitLearn.logger.info("Score model")
        scoring_data = {'fields': ['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width'], 'values': [[5.1, 3.5, 1.4, 0.2]]}
        predictions = self.client.deployments.score(TestWMLClientWithScikitLearn.scoring_url, scoring_data)
        self.assertTrue("prediction" in str(predictions))

    def test_09_delete_deployment(self):
        TestWMLClientWithScikitLearn.logger.info("Delete deployment")
        self.client.deployments.delete(TestWMLClientWithScikitLearn.deployment_uid)

    def test_10_delete_model(self):
        TestWMLClientWithScikitLearn.logger.info("Delete model")
        self.client.repository.delete(TestWMLClientWithScikitLearn.model_uid)


if __name__ == '__main__':
    unittest.main()
