from setuptools import setup, find_packages

DESCRIPTION = '(Continuous) Ab initio Generalized Langevin equations '
setup(
    name='AIGLE',
    version='0.1.0',
    author="Pinchen Xie",
    author_email="<pinchenx@math.princeton.edu>",
    packages=find_packages(),
    description=DESCRIPTION,
    python_requires=">=3.7",
    install_requires=[
        'numpy',
        'torch'
    ],
    # entry_points={
    #     'console_scripts': [
    #         'aigle = AIGLE.module1:main',
    #     ],
    # },
)