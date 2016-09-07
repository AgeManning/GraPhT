
import sys
# Add Directory to run Tests
sys.path.append('./tests')
sys.path.append('./classes')
import tests


# Main Function - For the time being, we will just run tests
def main():

    # Load Parameters from config file
    configData = load_config('./config.cfg')

    # Run a test
    tests.run_test1()

    # Run another test
    tests.run_test(configData)

def load_config(configFile):
    """
    Loads potential and other data from configFile
    :param configFile: Path to config file
    :return: A data structure containing required configuration
    """

    # TODO: Implement this
    return False

# Call the main function
if __name__ == '__main__':
    print(__name__)
    main()
