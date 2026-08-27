from setuptools import setup, find_packages

def parse_requirements(filename):
    with open(filename, 'r') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name='landslide-detection-ne-india',
    version='1.0.0',
    description='Landslide Detection System for North Eastern India',
    packages=find_packages(),
    install_requires=parse_requirements('requirements.txt'),
    python_requires='>=3.10'
)
