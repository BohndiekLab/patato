# Contributing to PATATO

We'd love to get more people involved with PATATO to add additional features and make it better. Feel free to take a look at the list of issues on our GitHub repository or fork the repository and make your own updates.

* Please add tests for any new features.
* Please add documentation to any new features (see the existing codebase for examples). 
* Please follow the existing code style.
* Consider adding to the documentation (e.g. an example script to illustrate any new functionality). 

If you'd like to help out or for more information,
please contact Thomas Else at [thomas.else@cruk.cam.ac.uk](mailto:thomas.else@cruk.cam.ac.uk).

## Getting started

To get started, fork this repository on GitHub ([here](https://github.com/BohndiekLab/patato)). You should then clone your fork of the repository to your local machine, install it as editable (-e flag in pip), where you can make changes as desired.

```bash
git clone https://github.com/YOUR_USER_NAME/patato.git
cd patato
pip install -e .
```

## Running tests

Please add tests corresponding to any code you would like to contribute. The test suites are currently located in the /tests directory. Please add them here. The following sequence of commands should then run successfully (from the patato root directory).

```bash
cd tests
pythom -m unittest
```

## Rebuilding the documentation

If you contribute to the documentation, you can test that documentation will build properly by running the following commands (from the patato root directory).

```bash
cd docs
make html
```

This will build the documentation in html format in the /docs/build directory.
