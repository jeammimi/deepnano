### DeepNano: alternative basecaller for MinION reads - R9(.4) version

Requirements
================

We use Python 2.7.

Here are versions of Python packages, that we used:

- Cython==0.23.4
- numpy==1.10.2
- h5py==2.5.0
- python-dateutil==2.5.0

Basic usage:
================

`python basecall.py --chemistry <r9/r9.4> <list of fast5 files>`

Advanced arguments:
=================

- `-h` - prints help message
- `--output FILENAME` - output filename
- `--directory DIRECTORY` Directory where read files are stored

