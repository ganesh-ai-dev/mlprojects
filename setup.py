from setuptools import  find_packages,setup
from typing import List
gsg='-e .'
def get_requirements(file_path:str)->list[str]:
    requirements=[]
    with open(file_path)as file_obj:
        requirements=file_obj.readlines()
        [req.replace("\n","")for req in requirements]
        if gsg in requirements:
            requirements.remove(gsg)



setup(
    name='mlproject',
    version='0.0.2',
    author='Ganesh',author_email='tarigondaganesh1234@gmail.com',
    packages=find_packages(),
    install_requirements=get_requirements('requirements.txt')
)