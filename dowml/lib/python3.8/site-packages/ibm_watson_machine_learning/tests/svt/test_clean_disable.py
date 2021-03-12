import unittest
import logging
from preparation_and_cleaning import *
from datetime import datetime


class TestWMLClientWithExperiment(unittest.TestCase):
    logger = logging.getLogger(__name__)

    @classmethod
    def setUpClass(self):
        self.wml_credentials = get_wml_credentials()
        self.client = get_client()
        self.cos_resource = get_cos_resource()

    def test_01_clean_experiments(self):
        clean_experiments(self.client, datetime.now(timezone('UTC')))

    def test_02_clean_training_runs(self):
        clean_training_runs(self.client, datetime.now(timezone('UTC')))

    def test_03_clean_definitions(self):
        clean_definitions(self.client, datetime.now(timezone('UTC')))

    def test_04_clean_models(self):
        clean_models(self.client, datetime.now(timezone('UTC')))

    def test_05_clean_deployments(self):
        clean_deployments(self.client, datetime.now(timezone('UTC')))

    def test_06_clean_functions(self):
        clean_ai_functions(self.client, datetime.now(timezone('UTC')))

    def test_07_clean_runtimes(self):
        clean_runtimes(self.client, datetime.now(timezone('UTC')))

    def test_08_clean_libraries(self):
        clean_custom_libraries(self.client, datetime.now(timezone('UTC')))

    def test_09_clean_buckets(self):
        if self.cos_resource is not None:
            for bucket in self.cos_resource.buckets.all():
                if 'wml-test-' in bucket.name and bucket.creation_date < datetime.now(timezone('UTC')):
                    print('Deleting \'{}\' bucket.'.format(bucket.name))
                    try:
                        for upload in bucket.multipart_uploads.all():
                            upload.abort()
                        for o in bucket.objects.all():
                            o.delete()
                        bucket.delete()
                    except Exception as e:
                        print("Exception during bucket deletion occured: " + str(e))