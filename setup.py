from setuptools import setup


def readme():
    with open('README.md') as f:
        return f.read()

setup(name='pdsimage',
      version='1.0',
      description='Package for creating images from NASA PDS files',
      classifiers=[
          'Development Status :: 1 - Alpha',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 2.7',
          'Topic :: PDS Image :: Planetary Science',
      ],
      keywords='PDS, LOLA, WAC, crater, dome, FFC, floor-fractured craters',
      url='https://github.com/cthorey/pdsimage',
      author='Thorey Clement',
      author_email='clement.thorey@gmail.com',
      license='MIT',
      packages=['pdsimage'],
      install_requires=[
          'numpy',
          'pandas',
          'basemap',
          'pvl',
          'requests',
          'scipy',
          'matplotlib',
          'cartopy'],
      include_package_data=True,
      zip_safe=False)
