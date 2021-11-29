"""
Setup script for url_classifier package
"""

from distutils.core import setup

setup(
    name='url_classifier',
    version='0.1',
    install_requires=['numpy', 'scikit-learn', 'pandas', 'joblib', 'xgboost', 'pytest', 'mlflow'],
    # install_requires=[],
    # package_dir={'': 'url_classifier'},
)