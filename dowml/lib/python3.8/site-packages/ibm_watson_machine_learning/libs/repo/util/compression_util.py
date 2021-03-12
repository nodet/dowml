# (C) Copyright IBM Corp. 2020.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tarfile, gzip, shutil


class CompressionUtil(object):
    @staticmethod
    def create_tar(path, arcname, archive_path):
        tar = tarfile.open(archive_path, "w")
        tar.add(path, arcname)
        tar.close()

    @staticmethod
    def extract_tar(archive_path, path):
        tar = tarfile.open(archive_path)
        tar.extractall(path)
        tar.close()

    @staticmethod
    def compress_file_gzip(filepath, gzip_filepath):
        with open(filepath, 'rb') as f_in, gzip.open(gzip_filepath, 'wb+') as f_out:
            shutil.copyfileobj(f_in, f_out)

    @staticmethod
    def decompress_file_gzip(gzip_filepath, filepath):
        with gzip.open(gzip_filepath, 'rb') as f:
            content = f.read()
            output_f = open(filepath, 'wb+')
            output_f.write(content)
            output_f.close()
