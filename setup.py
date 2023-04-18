from setuptools import find_packages, setup


def get_requirements(file_path):
    '''
        This function will get all the requirements and return the list of requirements
    '''
    HYPHEN_DOT_E = "-e ."
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [x.replace("\n", "") for x in requirements ]

        if HYPHEN_DOT_E in requirements:
            requirements.remove(HYPHEN_DOT_E)
    return requirements
    
setup(
    name= 'ml project',
    version= '0.0.1',
    author= 'Amruth',
    author_email = 'amruthglr@gmail.com',
    packages= find_packages(),
    install_requires = get_requirements('requirements.txt')
)