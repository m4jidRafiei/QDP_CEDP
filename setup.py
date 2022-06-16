import setuptools

import pm_cedp_qdp

with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
    name=pm_cedp_qdp.__name__,
    version=pm_cedp_qdp.__version__,
    author=pm_cedp_qdp.__author__,
    author_email=pm_cedp_qdp.__author_email__,
    description="Quantifying Temporal Privacy Leakage in Continuous Event Data Publishing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/m4jidRafiei/QDP_CEDP",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'numpy>=1.19.5',
        'pandas>=1.1.5',
        'pm4py>=2.2.15',
        'setuptools>=59.6.0',
        'matplotlib>=2.2.2'
    ],
    project_urls={
        'Source': 'https://github.com/m4jidRafiei/QDP_CEDP'
    }
)