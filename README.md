Tube Segmentation Framework
===================================

The tube segmentation framework is a software for fast segmentation and centerline extraction of tubular structures (e.g. blood vessels and airways) from different modalities (e.g. CT, MR and US) and organs using GPUs and OpenCL.

See the file LICENSE for license information.

Dependencies
----------------------------------

* OpenCL. You need an OpenCL implementation installed on your system to use this software (AMD, NVIDIA, Intel or Apple)
* Boost iostreams. E.g. on Ubuntu linux use the following package: libboost-iostreams-dev
* The two submodules: SIPL and OpenCLUtilities
* GTK 2 for visualization (not required, used by the SIPL module). On Ubuntu linux use the following package: libgtk2.0-dev

Compiling
----------------------------------

To compile the software first make sure that all software dependencies are installed and set up correctly.
Next, use cmake.

For instance on linux, do the following:
```bash
cmake .
make -j8
```

Usage
----------------------------------

Parameters
----------------------------------

