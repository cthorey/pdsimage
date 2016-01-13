from setuptools import setup


def readme():
    with open('README.rst') as f:
        return f.read()

setup(name='pdsimage',
      version='1.1',
      description='Package for creating images from NASA PDS files',
      keywords='PDS, LOLA, WAC, crater, dome, FFC, floor-fractured craters',
      url='https://github.com/cthorey/pdsimage',
      download_url='https://github.com/cthorey/pdsimage/tarball/1.0',
      author='Thorey Clement',
      author_email='clement.thorey@gmail.com',
      license='MIT',
      packages=['pdsimage'],
      install_requires=[
          'Cython',
          'numpy',
          'pandas',
          'pvl',
          'requests',
          'scipy',
          'matplotlib'],
      dependency_links=[
          'https://downloads.sourceforge.net/project/matplotlib/matplotlib-toolkits/basemap-1.0.7/basemap-1.0.7.tar.gz'],
      include_package_data=True,
      zip_safe=False)
