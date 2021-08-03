import pprint
import sys

from ibm_watson_machine_learning import APIClient

from dowmllib import _CredentialsProvider


def main():
    # Credentials are provided by the environment
    wml_credentials = _CredentialsProvider().credentials

    if _CredentialsProvider.SPACE_NAME not in wml_credentials:
        print(f'Refusing to destroy the default space')
        sys.exit(1)

    space_name = wml_credentials[_CredentialsProvider.SPACE_NAME]
    client = APIClient(wml_credentials)
    print(f'Created client.')
    spaces = client.spaces.get_details()
    print(f'Retrieved spaces.')
    resources = spaces['resources']
    print(f'There are {len(resources)} spaces.')
    space_id = None
    for s in resources:
        if s['entity']['name'] == space_name:
            print(f'Found one with name \'{space_name}\'')
            space_id = client.spaces.get_id(s)
            print(f'Its id is {space_id}')
    if not space_id:
        print(f'No space found named \'{space_name}\'')
        sys.exit(1)

    print(f'Deleting space...')
    client.spaces.delete(space_id)
    print(f'Done.')



if __name__ == '__main__':
    main()
