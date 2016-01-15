from setuptools import setup


def readme():
    with open('README.rst') as f:
        return f.read()

setup(name='pdsimage',
      version='1.1.0',
      description='Package for creating images from NASA PDS files',
      keywords='PDS, LOLA, WAC, crater, dome, FFC, floor-fractured craters',
      url='https://github.com/cthorey/pdsimage',
      download_url='https://github.com/cthorey/pdsimage/tarball/1.0',
      author='Thorey Clement',
      author_email='clement.thorey@gmail.com',
      license='MIT',
      packages=['pdsimage'],
      install_requires=[
          'numpy>=1.6',
          'six>=1.3.0',
          'setuptools>=0.7.2',
          'matplotlib>=1.3.0',
          'scipy>=0.10',
          'pandas',
          'pvl',
          'requests',
          'sphinxcontrib-napoleon'],
      include_package_data=True,
      zip_safe=False)
