from setuptools import setup, find_packages

setup(
    name="horguesc",
    version="0.1.0",
    packages=find_packages(),
    description="horguesc Python package",
    author="S. Shibasaki",
    author_email="sdeb6vm@gmail.com",
    url="https://github.com/s-shibasaki/horguesc",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    entry_points={
        'console_scripts': [
            'horguesc=horguesc.cli:main',
        ],
    },
)