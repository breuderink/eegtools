#Introduction

EEGtools is a set of Python libraries for EEG analysis. Most of the code was
developed as a part of the [PhD
work](https://github.com/downloads/breuderink/phdthesis/reuderink2011rbc.zip)
of [Boris Reuderink](http://borisreuderink.nl) in the form of the library
[Psychic](https://github.com/breuderink/psychic). EEGtools is the successor of
Psychic, and does not attempt to provide a framework for analysis, but rather a
small set of well-tested functions for scientific EEG analysis.


# Status
EEGtools is currently in alpha phase. That said, code in the master branch is
thoroughly tested, and probably existed in
[Psychic](https://github.com/breuderink/psychic). For versioning we use
[semantic versioning](semver.org).


### Planned features:
- Examples!
- Importing of [BDF](http://www.biosemi.com/faq/file_format.htm) and
  [GDF](http://arxiv.org/abs/cs.DB/0608052) file formats.
- Publication-ready visualisation of topographic activation (scalp plots)
- PARAFAC tensor decomposition for summarization of tensors (higher order
  arrays).

### 0.2
- Added various spatial filters (common spatial patterns), channel selection,
  whitening and the common average reference.
- Added functions for feature extraction (windowing, spectral estimation,
  filtering, narrow-band covariance tensors).

### 0.1
- Included automatically downloading importers for public brain-computer interfacing (BCI) data sets,
  such as [BCI Competition
  3.4a](http://www.bbci.de/competition/iii/#data_set_iva), Reuderink's
  Affective Pacman and [Schalk's
  Physiobank](http://www.physionet.org/pn4/eegmmidb/) datasets.
- Added importer for EDF+ files including annotations.


# Installation

The preferred method of installation is using
[PIP](http://www.pip-installer.org). The latest development version can be
installed with:

    $ pip install git+https://github.com/breuderink/eegtools.git --user

The latest stable release can be installed with:

    $ pip install eegtools --user
