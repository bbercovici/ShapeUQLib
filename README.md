# ShapeUQLib

Implementation of an analytical shape uncertainty and inertia statistics formulation tailored for small body analyses. This library is not stand alone and is meant to be called by [SBGAT](https://github.com/bbercovici/SBGAT). Implementation and derivation details can be found  

## Requires
1. Armadillo
2. CMake

## Installation: 

### Mac users

ShapeUQLib can be retrieved from Homebrew:

    brew tap bbercovici/self
    brew update
    brew install shapeuqlib

### Unix users (Mac and Linux)

    git clone https://github.com/bbercovici/ShapeUQLib.git
    cd ShapeUQLib/build
    cmake ..
    make
    make install

## Getting updates

    git pull
    cd build
    cmake ..
    make
    make install

## License

[This software is distributed under the MIT License](https://choosealicense.com/licenses/mit/)




