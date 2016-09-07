# GraPhT

Gravity Waves from Phase Transitions - Automative tool for calculating gravitational wave spectra originating from first order phase transitions in the early universe.

## Getting Started

Once cloned, change to the directory and set up a virtual environment. This can be done with `virtualenv`, however we will be using `pyvenv`. Run
``` bash
$ pyvenv venv
$ source ./venv/bin/activate
$ pip install -r requirements.txt
```

This will install all necessary python modules for this project.

## Dependencies

The current dependencies for this project are; `tk`.

To install these packages on arch, run the following

``` bash
$ pacman -S tk
```

> For other operating systems, look up the packages and install through your OS's package manager.

## Development...

> This project is currently under active development. Details will ensue as the project develops.

## Developers

This section is a reference for developers

### Adding packages

If you need to add a package, ensure you are working in the virtual environment. Then install using pip:
``` bash
$ pip install <package-name>
```

Then update the requirements file. First change directory to GraPhT/ Then
``` bash
$ pip freeze > requirements.txt
```

### Clean commits

Any files generated in /output or /venv are automatically removed from your git commits. Please ensure your commits only add files that are required by the project. To check which files are being commited, remember you can check with ` git status`.
