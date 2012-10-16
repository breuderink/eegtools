from distutils.core import setup

setup(
  name='EEGtools',
  url='https://github.com/breuderink/eegtools',
  author='Boris Reuderink',
  author_email='b.reuderink@gmail.com',
  license='BSD',
  version='0.2.1',
  requires=[ # FIXME does this work? docs seem inconsistent....
    'numpy(>=1.5.1)',
    'scipy'
    ],
  packages=['eegtools', 'eegtools.io', 'eegtools.data'],
  )
