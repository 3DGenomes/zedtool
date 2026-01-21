#control version
from setuptools import setup
import os

def parse_requirements(filename):
    with open(filename) as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

def get_version():
    version_file = os.path.join('src', 'zedtool', 'VERSION')
    with open(version_file) as f:
        return f.read().strip()

setup(
    name='zedtool',
    version=get_version(),
    author='John Markham',
    author_email='john.markham@gmail.com',
	packages=['zedtool'],
    url='https://github.com/johnfmarkham/zedtool',
    description='Z Estimate Diagnostics Tool',
    package_dir = {'zedtool':'src/zedtool'},
    package_data = {'zedtool': ['test/*']},
    install_requires=parse_requirements("requirements.txt"),
    python_requires=">=3.12",
    entry_points={
        'console_scripts': [
            'zedtool=zedtool.cli:main',
        ],
    },
    include_package_data=True
)
