import unittest
import logging
from preparation_and_cleaning import *
from ibm_watson_machine_learning.utils import delete_directory, extract_mlmodel_from_archive
import coremltools
from sys import platform
from os import remove as remove_file
from os.path import isfile
import tarfile


class TestWMLClientWithCaffe200MBCoreML(unittest.TestCase):
    deployment_uid = None
    model_uid = None
    converted_model_path = None
    logger = logging.getLogger(__name__)

    @classmethod
    def setUpClass(self):
        TestWMLClientWithCaffe200MBCoreML.logger.info("Service Instance: setting up credentials")
        self.wml_credentials = get_wml_credentials()
        self.client = get_client()
        self.cos_resource = get_cos_resource()
        self.bucket_names = prepare_cos(self.cos_resource, data_code=BAIR_BVLC)
        self.model_path = os.path.join(os.getcwd(), 'datasets', 'BAIR_BVLC')

    def test_01_service_instance_details(self):
        TestWMLClientWithCaffe200MBCoreML.logger.info("Check client ...")
        self.assertTrue(self.client.__class__.__name__ == 'APIClient')

        self.logger.info("Getting instance details ...")
        details = self.client.service_instance.get_details()
        TestWMLClientWithCaffe200MBCoreML.logger.debug(details)

        self.assertTrue("published_models" in str(details))
        self.assertEqual(type(details), dict)

    def test_02_publish_model(self):
        self.logger.info("Publishing Caffe model ...")

        self.client.repository.ModelMetaNames.show()

        archive_name = 'bvlc_reference_caffenet.tar.gz'

        with tarfile.open(archive_name, 'w:gz') as tar:
            for file_path in os.listdir(self.model_path):
                tar.add(os.path.join(self.model_path, file_path), file_path)

        model_props = {
            self.client.repository.ModelMetaNames.NAME: "Core ML - caffe bvlc model",
            self.client.repository.ModelMetaNames.FRAMEWORK_NAME: "caffe",
            self.client.repository.ModelMetaNames.FRAMEWORK_VERSION: "1.0",
            self.client.repository.ModelMetaNames.RUNTIME_NAME: "python",
            self.client.repository.ModelMetaNames.RUNTIME_VERSION: "3.5"}

        published_model_details = self.client.repository.store_model(model=archive_name, meta_props=model_props)

        os.remove(archive_name)

        TestWMLClientWithCaffe200MBCoreML.model_uid = self.client.repository.get_model_uid(published_model_details)
        TestWMLClientWithCaffe200MBCoreML.model_url = self.client.repository.get_model_url(published_model_details)
        self.logger.info("Published model ID:" + str(TestWMLClientWithCaffe200MBCoreML.model_uid))
        self.assertIsNotNone(TestWMLClientWithCaffe200MBCoreML.model_uid)

    def test_04_publish_model_details(self):
        details_models = self.client.repository.get_model_details()
        TestWMLClientWithCaffe200MBCoreML.logger.debug("All models details: " + str(details_models))
        self.assertTrue("Core ML - caffe bvlc model" in str(details_models))

    def test_05_convert_2_coreml(self):
        self.logger.info("Converting caffe model to core ml ...")

        deployment_details = self.client.deployments.create(TestWMLClientWithCaffe200MBCoreML.model_uid,
                                                            deployment_type='virtual')

        self.assertTrue('DEPLOY_SUCCESS' in str(deployment_details))
        TestWMLClientWithCaffe200MBCoreML.deployment_uid = self.client.deployments.get_uid(deployment_details)
        print('Deployment UID: ' + TestWMLClientWithCaffe200MBCoreML.deployment_uid)

    def test_06_load_CoreML_model(self):
        filepath = self.client.deployments.download(TestWMLClientWithCaffe200MBCoreML.deployment_uid)
        print('Downloaded model path: ' + filepath)
        TestWMLClientWithCaffe200MBCoreML.converted_model_path = extract_mlmodel_from_archive(filepath,
                                                                                           TestWMLClientWithCaffe200MBCoreML.model_uid)
        loaded_model = coremltools.models.MLModel(TestWMLClientWithCaffe200MBCoreML.converted_model_path)
        loaded_model.short_description = 'this is a test model'
        self.assertTrue('this is a test model' in str(loaded_model.short_description))

        if not platform in ['linux']:
            # TODO - sample input for scoring (imagenet data)
            input_param = 3*[227*[227*[0.0]]]
            scoring_data = {'data': input_param}

            predictions = loaded_model.predict(scoring_data)
            self.assertTrue('prob' in str(predictions))
            delete_directory(TestWMLClientWithCaffe200MBCoreML.model_uid)

    def test_10_delete_model(self):
        delete_directory(TestWMLClientWithCaffe200MBCoreML.model_uid)
        self.client.repository.delete(TestWMLClientWithCaffe200MBCoreML.model_uid)

        if isfile(TestWMLClientWithCaffe200MBCoreML.converted_model_path):
            try:
                remove_file(TestWMLClientWithCaffe200MBCoreML.converted_model_path)
            except OSError:
                self.logger.info("Converted model file could not be removed: " + str(TestWMLClientWithCaffe200MBCoreML.converted_model_path))
            else:
                self.logger.info("Converted model file is removed: " + str(TestWMLClientWithCaffe200MBCoreML.converted_model_path))


if __name__ == '__main__':
    unittest.main()
