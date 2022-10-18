from setuptools import setup

setup(
    name='mftma',
    version='0.1',
    description='mean-field theoretic manifold analysis',
    author='Intel AI Lab',
    license='Apache 2.0',
    packages=['mftma'],
    zip_safe=False,
    install_requires=[
        'autograd>=1.3',
        'numpy>=1.17.0',
        'pymanopt>=1.0.0',
        'scipy>=1.2.0',
        'cvxopt>=1.2.3',
        'scikit-learn>=0.21.3',
        'torch>=1.1',
    ]
)
