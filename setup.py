from distutils.core import setup

setup(
  name='eegtools',
  url='https://github.com/breuderink/eegtools',
  author='Boris Reuderink',
  author_email='b.reuderink@gmail.com',
  license='New BSD',
  version='0.2.1',
  # Needed for PIP, see http://stackoverflow.com/questions/8295644/:
  install_requires=open('requirements.txt').readlines(),
  packages=['eegtools', 'eegtools.io', 'eegtools.data'],
  )
