import unittest
import logging
from preparation_and_cleaning import *
from ibm_watson_machine_learning.utils import delete_directory, extract_mlmodel_from_archive
import coremltools
from sys import platform


class TestWMLClientWithKeras2CoreML(unittest.TestCase):
    deployment_uid = None
    model_uid = None
    converted_model_path = None
    download_url = None
    logger = logging.getLogger(__name__)

    @classmethod
    def setUpClass(self):
        TestWMLClientWithKeras2CoreML.logger.info("Service Instance: setting up credentials")
        self.wml_credentials = get_wml_credentials()
        self.client = get_client()
        self.model_path = os.path.join(os.getcwd(), 'artifacts', 'core_ml', 'keras', 'mnistCNN.h5.tgz')

    def test_01_service_instance_details(self):
        TestWMLClientWithKeras2CoreML.logger.info("Check client ...")
        self.assertTrue(self.client.__class__.__name__ == 'APIClient')

        self.logger.info("Getting instance details ...")
        details = self.client.service_instance.get_details()
        TestWMLClientWithKeras2CoreML.logger.debug(details)

        self.assertTrue("published_models" in str(details))
        self.assertEqual(type(details), dict)

    def test_02_publish_model(self):

        self.logger.info("Publishing Keras model ...")

        self.client.repository.ModelMetaNames.show()

        model_props = {
                        self.client.repository.ModelMetaNames.NAME: "Core ML - keras mnist model",
                        self.client.repository.ModelMetaNames.FRAMEWORK_NAME: "tensorflow",
                        self.client.repository.ModelMetaNames.FRAMEWORK_VERSION: "1.5",
                        self.client.repository.ModelMetaNames.RUNTIME_NAME: "python",
                        self.client.repository.ModelMetaNames.RUNTIME_VERSION: "3.5"}

        published_model_details = self.client.repository.store_model(model=self.model_path, meta_props=model_props)
        TestWMLClientWithKeras2CoreML.model_uid = self.client.repository.get_model_uid(published_model_details)
        TestWMLClientWithKeras2CoreML.model_url = self.client.repository.get_model_url(published_model_details)
        self.logger.info("Published model ID:" + str(TestWMLClientWithKeras2CoreML.model_uid))
        self.logger.info("Published model URL:" + str(TestWMLClientWithKeras2CoreML.model_url))
        self.assertIsNotNone(TestWMLClientWithKeras2CoreML.model_uid)
        self.assertIsNotNone(TestWMLClientWithKeras2CoreML.model_url)

    def test_04_publish_model_details(self):
        details_models = self.client.repository.get_model_details()
        TestWMLClientWithKeras2CoreML.logger.debug("All models details: " + str(details_models))
        self.assertTrue("Core ML - keras mnist model" in str(details_models))

    def test_05_convert_2_coreml(self):
        self.logger.info("Converting keras model to core ml ...")
        print('Model_UID: ' + TestWMLClientWithKeras2CoreML.model_uid)
        deployment_details = self.client.deployments.create(TestWMLClientWithKeras2CoreML.model_uid, deployment_type='virtual')

        self.assertTrue('DEPLOY_SUCCESS' in str(deployment_details))
        TestWMLClientWithKeras2CoreML.deployment_uid = self.client.deployments.get_uid(deployment_details)
        TestWMLClientWithKeras2CoreML.download_url = self.client.deployments.get_download_url(deployment_details)

        print('Deployment UID: ' + TestWMLClientWithKeras2CoreML.deployment_uid)
        print('Deployment download url: ' + TestWMLClientWithKeras2CoreML.download_url)

        self.assertTrue(TestWMLClientWithKeras2CoreML.download_url is not None)

    def test_06_load_CoreML_model(self):
        filepath = self.client.deployments.download(TestWMLClientWithKeras2CoreML.deployment_uid)

        print('Downloaded model path: ' + filepath)

        TestWMLClientWithKeras2CoreML.converted_model_path = extract_mlmodel_from_archive(filepath, TestWMLClientWithKeras2CoreML.model_uid)

        loaded_model = coremltools.models.MLModel(TestWMLClientWithKeras2CoreML.converted_model_path)
        loaded_model.short_description = 'this is a test model'
        self.assertTrue('this is a test model' in str(loaded_model.short_description))

        if not platform in ['linux']:
            scoring_data = {'input1': [[[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.,0.,0.,0.32941177,0.87058824,0.2627451,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.,0.,0.,0.7254902,0.99607843,0.44705883,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.,0.,0.,0.62352943,0.99607843,0.28235295,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.,0.,0.,0.5921569,0.99607843,0.44705883,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.,0.,0.,0.23529412,0.99607843,0.6392157,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.23921569,0.4745098,0.4745098,0.],[0.,0.,0.,0.,0.,0.,0.,0.14117648,0.94509804,0.8901961,0.06666667,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.12156863,0.52156866,0.9490196,0.99607843,0.99607843,0.],[0.,0.,0.,0.,0.,0.,0.,0.,0.7764706,0.99607843,0.25882354,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.01176471,0.14901961,0.8784314,0.99607843,0.99607843,0.99607843,0.8117647,0.],[0.,0.,0.,0.,0.,0.,0.,0.,0.7764706,0.88235295,0.05490196,0.,0.,0.,0.,0.,0.,0.,0.,0.07450981,0.79607844,0.99607843,0.99607843,0.99607843,0.99607843,0.85882354,0.07058824,0.],[0.,0.,0.,0.,0.,0.,0.,0.,0.7764706,0.99607843,0.2627451,0.,0.,0.,0.,0.,0.,0.,0.29411766,0.8666667,0.99607843,0.99607843,0.4509804,0.20392157,0.20392157,0.15686275,0.,0.],[0.,0.,0.,0.,0.,0.,0.,0.,0.7764706,0.99607843,0.2627451,0.,0.,0.,0.,0.,0.03529412,0.49411765,0.9843137,0.99607843,0.85882354,0.3019608,0.00392157,0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.,0.,0.,0.,0.7764706,0.99607843,0.2627451,0.,0.,0.,0.23137255,0.52156866,0.8039216,0.99607843,0.9411765,0.6509804,0.13725491,0.,0.,0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.,0.,0.,0.,0.7764706,0.98039216,0.23137255,0.,0.08627451,0.5058824,0.9764706,0.99607843,0.972549,0.7137255,0.22352941,0.,0.,0.,0.,0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.,0.,0.,0.,0.7764706,0.8980392,0.08235294,0.3254902,0.9137255,0.99607843,0.99607843,0.73333335,0.22745098,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.,0.,0.,0.,0.7764706,0.99607843,0.9254902,0.99215686,1.,0.93333334,0.24313726,0.01960784,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.,0.,0.,0.,0.6666667,0.99607843,0.99607843,0.81960785,0.3254902,0.17254902,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.,0.,0.,0.,0.20392157,0.54901963,0.41568628,0.07058824,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]]]}

            predictions = loaded_model.predict(scoring_data)
            self.assertTrue('output1' in str(predictions))
            delete_directory(TestWMLClientWithKeras2CoreML.model_uid)

    def test_07_update_coreml_with_metadata(self):
        self.logger.info("Converting keras model to core ml ...")

        meta_data = {
            "input_names": "image",
            "output_names": "output",
            "class_labels": ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
            "image_scale": 1 / 255.
        }

        deployment_details = self.client.deployments.update(deployment_uid=TestWMLClientWithKeras2CoreML.deployment_uid, meta_props=meta_data)
        print(str(deployment_details))

        self.assertTrue('DEPLOY_SUCCESS' in str(deployment_details))
        self.assertTrue(TestWMLClientWithKeras2CoreML.deployment_uid == self.client.deployments.get_uid(deployment_details))

        TestWMLClientWithKeras2CoreML.deployment_uid = self.client.deployments.get_uid(deployment_details)
        self.assertTrue(TestWMLClientWithKeras2CoreML.download_url == self.client.deployments.get_download_url(deployment_details))
        print(TestWMLClientWithKeras2CoreML.deployment_uid)

    def test_08_load_CoreML_model(self):
        filepath = self.client.deployments.download(TestWMLClientWithKeras2CoreML.deployment_uid)

        print('Downloaded model path: ' + filepath)

        TestWMLClientWithKeras2CoreML.converted_model_path = extract_mlmodel_from_archive(filepath, TestWMLClientWithKeras2CoreML.model_uid)

        loaded_model = coremltools.models.MLModel(TestWMLClientWithKeras2CoreML.converted_model_path)
        loaded_model.short_description = 'this is a test model'
        self.assertTrue('this is a test model' in str(loaded_model.short_description))

        if not platform in ['linux']:
            scoring_data = {'image': [[[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.,0.,0.,0.32941177,0.87058824,0.2627451,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.,0.,0.,0.7254902,0.99607843,0.44705883,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.,0.,0.,0.62352943,0.99607843,0.28235295,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.,0.,0.,0.5921569,0.99607843,0.44705883,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.,0.,0.,0.23529412,0.99607843,0.6392157,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.23921569,0.4745098,0.4745098,0.],[0.,0.,0.,0.,0.,0.,0.,0.14117648,0.94509804,0.8901961,0.06666667,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.12156863,0.52156866,0.9490196,0.99607843,0.99607843,0.],[0.,0.,0.,0.,0.,0.,0.,0.,0.7764706,0.99607843,0.25882354,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.01176471,0.14901961,0.8784314,0.99607843,0.99607843,0.99607843,0.8117647,0.],[0.,0.,0.,0.,0.,0.,0.,0.,0.7764706,0.88235295,0.05490196,0.,0.,0.,0.,0.,0.,0.,0.,0.07450981,0.79607844,0.99607843,0.99607843,0.99607843,0.99607843,0.85882354,0.07058824,0.],[0.,0.,0.,0.,0.,0.,0.,0.,0.7764706,0.99607843,0.2627451,0.,0.,0.,0.,0.,0.,0.,0.29411766,0.8666667,0.99607843,0.99607843,0.4509804,0.20392157,0.20392157,0.15686275,0.,0.],[0.,0.,0.,0.,0.,0.,0.,0.,0.7764706,0.99607843,0.2627451,0.,0.,0.,0.,0.,0.03529412,0.49411765,0.9843137,0.99607843,0.85882354,0.3019608,0.00392157,0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.,0.,0.,0.,0.7764706,0.99607843,0.2627451,0.,0.,0.,0.23137255,0.52156866,0.8039216,0.99607843,0.9411765,0.6509804,0.13725491,0.,0.,0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.,0.,0.,0.,0.7764706,0.98039216,0.23137255,0.,0.08627451,0.5058824,0.9764706,0.99607843,0.972549,0.7137255,0.22352941,0.,0.,0.,0.,0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.,0.,0.,0.,0.7764706,0.8980392,0.08235294,0.3254902,0.9137255,0.99607843,0.99607843,0.73333335,0.22745098,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.,0.,0.,0.,0.7764706,0.99607843,0.9254902,0.99215686,1.,0.93333334,0.24313726,0.01960784,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.,0.,0.,0.,0.6666667,0.99607843,0.99607843,0.81960785,0.3254902,0.17254902,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.,0.,0.,0.,0.20392157,0.54901963,0.41568628,0.07058824,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]]]}

            predictions = loaded_model.predict(scoring_data)
            self.assertTrue('output' in str(predictions))

    def test_10_delete_model(self):
        delete_directory(TestWMLClientWithKeras2CoreML.model_uid)
        self.client.repository.delete(TestWMLClientWithKeras2CoreML.model_uid)


if __name__ == '__main__':
    unittest.main()
