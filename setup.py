"""
Setup script for Causal Learning Based Video Anomaly Detection.
"""

from setuptools import setup, find_packages
import os

# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Read requirements
with open(os.path.join(this_directory, 'requirements.txt'), encoding='utf-8') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name='causal-video-anomaly-detection',
    version='1.0.0',
    author='BJIT Research Team',
    author_email='research@bjitgroup.com',
    description='Self-Discovering Temporal Anomaly Patterns in Video Anomaly Detection via Causal Representation Learning',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/pvvkishore/Causal-Learning-Based-Video-Anomaly-Detection_Paper_Code_Raw',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Image Recognition',
    ],
    python_requires='>=3.8',
    install_requires=requirements,
    extras_require={
        'dev': [
            'pytest>=7.0.0',
            'pytest-cov>=4.0.0',
            'black>=23.0.0',
            'flake8>=6.0.0',
            'mypy>=1.0.0',
        ],
    },
    entry_points={
        'console_scripts': [
            'causal-vad-train=src.train:main',
            'causal-vad-eval=src.evaluate:main',
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
