import logging
import pprint
import sys
import time

from ibm_watson_machine_learning import APIClient

from dowmllib import _CredentialsProvider

logger = None


def find_space_in_list(client, space_name):
    global logger
    spaces = client.spaces.get_details()
    logger.debug(f'Retrieved spaces.')
    resources = spaces['resources']
    logger.debug(f'There are {len(resources)} spaces.')
    for s in resources:
        if s['entity']['name'] == space_name:
            logger.debug(f'Found one with name \'{space_name}\'')
            space_id = client.spaces.get_id(s)
            logger.debug(f'Its id is {space_id}')
            return space_id
    return None


def main():
    global logger
    logging.basicConfig(force=True, format='%(asctime)s %(message)s', level=logging.DEBUG)
    logger = logging.getLogger('delete_space')

    # Credentials are provided by the environment
    wml_credentials = _CredentialsProvider().credentials

    if _CredentialsProvider.SPACE_NAME not in wml_credentials:
        logger.error(f'Refusing to destroy the default space')
        sys.exit(1)

    space_name = wml_credentials[_CredentialsProvider.SPACE_NAME]
    client = APIClient(wml_credentials)
    logger.debug(f'Created client.')
    space_id = find_space_in_list(client, space_name)
    if not space_id:
        logger.error(f'No space found named \'{space_name}\'')
        sys.exit(1)

    logger.debug(f'Deleting space...')
    client.spaces.delete(space_id)
    logger.debug(f'Done.')

    counter = 0
    while space_id:
        time.sleep(2)
        space_id = find_space_in_list(client, space_name)
        counter = counter + 1
        if counter > 50:
            logger.error(f'Waited too long... Exiting.')
            sys.exit(1)


if __name__ == '__main__':
    main()
