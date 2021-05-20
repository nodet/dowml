# (C) Copyright IBM Corp. 2021.
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

__all__ = [
    'FlightService'
]

import os
import json
import sys
import threading
from typing import List, Optional, Iterable, TYPE_CHECKING
from .errors import DataStreamError, WrongLocationProperty, FlightServiceInfoNotAvailable, WrongFileLocation, FileUploadFailed
from .utils import prepare_interaction_props_for_cos

import pandas as pd

if TYPE_CHECKING:
    from ibm_watson_machine_learning import APIClient

# the upper bound limitation for training data size
DATA_SIZE_LIMIT = 1073741824  # 1GB in Bytes


class FlightService:
    """FlightService object unify the work for training data reading from different types of data sources,
        including databases. It uses a Flight Service and `pyarrow` library to connect and transfer the data.

    Parameters
    ----------
    wml_client: APIClient, required
        WML Client

    n_batches: int, optional
        Defines for how many parts / batches data source should be split.
        Default is 1.

    data_location: dict, optional
        Data location information passed by user.

    """

    def __init__(self,
                 wml_client: 'APIClient',
                 params: dict,
                 n_batches: Optional[int] = 1,
                 data_location: dict = None,) -> None:
        self.wml_client = wml_client
        self.params = params
        self.n_batches = n_batches
        self.data_source_type = None

        self.data_location = data_location

        self.lock_read = threading.Lock()
        self.stop_reading = False

        from pyarrow import flight

        flight_hostname = os.environ.get('FLIGHT_SERVICE_LOCATION', 'wdp-connect-flight')
        flight_port = os.environ.get('FLIGHT_SERVICE_PORT', '443')

        if flight_hostname == 'wdp-connect-flight':
            try:
                from project_lib import Project

            except ModuleNotFoundError:
                raise FlightServiceInfoNotAvailable()

        self.flight_client = flight.FlightClient(
            location=f"grpc+tls://{flight_hostname}:{flight_port}",
            disable_server_verification=True,
            override_hostname=flight_hostname
        )

    def authenticate(self) -> 'flight.ClientAuthHandler':
        """Creates an authenticator object for Flight Service."""
        from pyarrow import flight

        class TokenClientAuthHandler(flight.ClientAuthHandler):
            """Authenticator implementation from pyarrow flight."""

            def __init__(self, token):
                super().__init__()
                self.token = bytes('Bearer ' + token, 'utf-8')

            def authenticate(self, outgoing, incoming):
                outgoing.write(self.token)
                self.token = incoming.read()

            def get_token(self):
                return self.token

        if self.wml_client is None:
            return TokenClientAuthHandler(token=os.environ.get('USER_ACCESS_TOKEN'))
        else:
            return TokenClientAuthHandler(token=self.wml_client.wml_token)

    def get_endpoints(self) -> Iterable[List['flight.FlightEndpoint']]:
        """Listing all available Flight Service endpoints (one endpoint corresponds to one batch)"""
        from pyarrow import flight

        self.flight_client.authenticate(self.authenticate())
        try:

            for source_command in self._select_source_command():
                info = self.flight_client.get_flight_info(
                    flight.FlightDescriptor.for_command(source_command)
                )
                yield info.endpoints

        except flight.FlightInternalError as e:
            if 'CDICO2034E' in str(e):
                raise WrongLocationProperty(e)

            elif 'CDICO2015E' in str(e):
                raise WrongFileLocation(e)

            else:
                raise e

    def _get_data(self, endpoint: 'flight.FlightEndpoint') -> 'pd.DataFrame':
        """
        Read data from Flight Service (only one batch).

        Properties
        ----------
        endpoint: flight.FlightEndpoint, required

        Returns
        -------
        pd.DataFrame with batch data
        """
        from pyarrow import flight
        import pyarrow as pa

        try:
            reader = self.flight_client.do_get(endpoint.ticket)

        except flight.FlightUnavailableError as e:
            raise DataStreamError(reason=str(e))

        chunks = []

        # Flight Service could split one batch into several chunks to have better performance
        while True:
            try:
                chunk, metadata = reader.read_chunk()
                chunks.append(chunk)
            except StopIteration:
                break

        data = pa.Table.from_batches(chunks)
        return data.to_pandas()

    def _read_batch(self, batch_number: int, sequence_number: int,
                    endpoint: 'flight.FlightEndpoint', dfs: List['pd.DataFrame']) -> None:
        """This method should be used as a separate thread for downloading specific batch of data in parallel.

        Parameters
        ----------
        batch_number: int, required
            Specific number of the batch to download.

        sequence_number: int required
            Sequence number, it tells the number of file that we reads for file storage.

        endpoint: flight.FlightEndpoint, required
            Flight Service endpoint to connect.

        dfs: List['pd.DataFrame'], required
            List where we will be storing the downloaded batch. Shared across threads.
        """
        if not self.stop_reading:
            df = self._get_data(endpoint)
            row_size = sys.getsizeof(df.iloc[:2]) - sys.getsizeof(df.iloc[:1])

            if row_size != 0:   # append batches only when we have data
                with self.lock_read:
                    if not self.stop_reading:
                        # note: what to do if we have batch too large (over 1 GB limit)
                        if sys.getsizeof(df) > DATA_SIZE_LIMIT:  # this is GB in Bytes
                            row_limit = round(DATA_SIZE_LIMIT / row_size)
                            dfs.insert(batch_number, df.iloc[:row_limit])
                            self.stop_reading = True
                            # print(f"STOP READING SET by batch: {batch_number}")
                        # --- end note
                        else:
                            total_size = sum([row_size * len(data) for data in dfs])

                            # note: what to do when we have total size nearly under the limit
                            if total_size <= DATA_SIZE_LIMIT:
                                upper_row_limit = (DATA_SIZE_LIMIT - total_size) // row_size
                                df = df.iloc[:upper_row_limit]
                                dfs.insert(batch_number, df)
                            # --- end note

                            else:
                                # print(f"total size stop: {total_size}")
                                self.stop_reading = True

            # print(f"Downloaded batch number: {batch_number}, sequence: {sequence_number}")
            # print(f"Batch shape: {df.shape}")
            # print(f"Estimated batch size: {sys.getsizeof(df)} Bytes")
            # print(f'Thread {batch_number}, sequence: {sequence_number} completed reading the batch.')

    def read(self) -> 'pd.DataFrame':
        """Fetch the data from Flight Service. Fetching is done in batches.
            There is an upper top limit of data size to be fetched configured to 1 GB.

        Returns
        -------
        Pandas DataFrame with fetched data.
        """
        dfs = []
        sequences = []

        # Note: endpoints are created by Flight Service based on number of partitions configured
        # one endpoint serves one batch of the data
        for n, endpoints in enumerate(self.get_endpoints()):
            threads = []
            sequences.append(threads)

            for i, endpoint in enumerate(endpoints):
                reading_thread = threading.Thread(target=self._read_batch, args=(i, n, endpoint, dfs))
                threads.append(reading_thread)
                # print(f"Starting batch reading thread: {i}, sequence: {n}...")
                reading_thread.start()

        for n, sequence in enumerate(sequences):
            for i, thread in enumerate(sequence):
                # print(f"Joining batch reading thread {i}, sequence: {n}...")
                thread.join()

        dfs = pd.concat(dfs)

        # Note: be sure that we do not cross upper data size limit
        estimated_data_size = sys.getsizeof(dfs)
        estimated_row_size = estimated_data_size // len(dfs)
        size_over_limit = estimated_data_size - DATA_SIZE_LIMIT

        if size_over_limit > 0:
            upper_limit = len(dfs) - (size_over_limit // estimated_row_size) * 2
            dfs = dfs.iloc[:upper_limit]
        # --- end note

        print(f"TOTAL DOWNLOADED DATA SIZE: {sys.getsizeof(dfs)} Bytes")

        return dfs

    def get_asset_id(self) -> dict:
        _dict = {
            "data_asset_id": None,
            "connection_asset_id": None,
        }

        if self.data_location['type'] == 'data_asset':
            _dict['data_asset_id'] = self.data_location['href'].split('/v2/assets/')[-1].split('?')[0]

        else:
            _dict['connection_asset_id'] = self.data_location['connection']['id']

        return _dict

    def _select_source_command(self) -> List[str]:
        """Based on a data source type, select appropriate commands for flight service configuration."""
        # when dataset is small we received empty dfs, it is ok (15 is optimum for larger than 1 GB)
        command = {
            "num_partitions": 15,
        }

        # note: WS scenario
        if self.wml_client is None:
            command['project_id'] = os.environ.get('PROJECT_ID')

        else:
            if self.wml_client.default_space_id is not None:
                command['space_id'] = self.wml_client.default_space_id

            elif self.wml_client.default_project_id is not None:
                command['project_id'] = self.wml_client.default_project_id

        ids = self.get_asset_id()

        if ids['data_asset_id']:
            command['asset_id'] = ids['data_asset_id']

        elif ids['connection_asset_id']:
            command['asset_id'] = ids['connection_asset_id']
            command['interaction_properties'] = {}

            if 'bucket' in self.data_location['location']:
                command['interaction_properties'] = prepare_interaction_props_for_cos(
                    self.params, self.data_location['location']['path'])
                command['interaction_properties']['infer_schema'] = "true"
                command['interaction_properties']['file_name'] = self.data_location['location']['path']
                command['interaction_properties']['bucket'] = self.data_location['location']['bucket']

            elif 'schema_name' in self.data_location['location']:
                command['interaction_properties']['schema_name'] = self.data_location['location']['schema_name']
                command['interaction_properties']['table_name'] = self.data_location['location']['table_name']

            else:
                command['interaction_properties'] = self.data_location['location']

        return [json.dumps(command)]

    def write_data(self, data: 'pd.DataFrame'):
        from pyarrow import flight
        import pyarrow as pa

        schema = pa.Schema.from_pandas(data)
        ids = self.get_asset_id()
        command = {}

        # note: WS scenario
        if self.wml_client is None:
            command['project_id'] = os.environ.get('PROJECT_ID')

        else:
            if self.wml_client.default_space_id is not None:
                command['space_id'] = self.wml_client.default_space_id

            elif self.wml_client.default_project_id is not None:
                command['project_id'] = self.wml_client.default_project_id

        if ids['data_asset_id']:
            command['asset_id'] = ids['data_asset_id']

        elif ids['connection_asset_id']:
            command['asset_id'] = ids['connection_asset_id']
            command['interaction_properties'] = {}

            if 'bucket' in self.data_location['location']:
                command['interaction_properties']['file_name'] = self.data_location['location']['path']
                command['interaction_properties']['bucket'] = self.data_location['location']['bucket']

            elif 'schema_name' in self.data_location['location']:
                command['interaction_properties']['schema_name'] = self.data_location['location']['schema_name']
                command['interaction_properties']['table_name'] = self.data_location['location']['table_name']

            else:
                command['interaction_properties'] = self.data_location['location']

        self.flight_client.authenticate(self.authenticate())
        writer, reader = self.flight_client.do_put(flight.FlightDescriptor.for_command(json.dumps(command)), schema)

        with writer:
            writer.write_table(pa.Table.from_pandas(data))

        return writer, reader
