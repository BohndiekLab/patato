---
title: 'POTATO: A Python Photoacoustic Analysis Toolkit'
tags:
  - Python
authors:
  - name: Thomas R. Else
    orcid: 0000-0002-2652-4190
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
  - name: Janek Gröhl
    affiliation: "1, 2"
  - name: Lina Hacker
    affiliation: "1, 2"
  - name: Sarah Bohndiek
    corresponding: true # (This is how to denote the corresponding author)
    affiliation: "1, 2"
affiliations:
  - name:  CRUK Cambridge Institute, University of Cambridge, UK
    index: 1
  - name: Department of Physics, University of Cambridge, UK
    index: 2
date: 07 October 2022
bibliography: paper.bib
---
 
# Summary
POTATO is a Python-based toolkit for the analysis of photoacoustic (PA) tomography data. We provide a convenient, transparent and OpenSource library to the community to enable more reproducible image analysis. As an emerging technique, PA imaging often suffers from fairly opaque processing algorithms. Here, we provide convenient Python interfaces for optimised implementations of reconstruction, spectral unmixing and post-processing of typical forms of PA data. This includes dynamic contrast enhanced (DCE) datasets and oxygen-enhanced (OE) datasets.

# Statement of Need

POTATO is designed to fill a gap in the PA imaging community. Currently, PA image analysis code is generally developed separately by different research groups, often unpublished, and not having a consistent interface. This makes it difficult to a. verify whether the code is implemented accurately and b. harder to make use of advanced algorithms which can improve image quality. Alternatively, research groups may use proprietary software for image processing, which can make image processing less reproducible, harder to apply to large datasets and less transparent. 

## Related Work
* Matlab toolkit
* Berkan's code
* IPASC

# Usage

## Python Interface

## Command-Line Interface

### 1. Convert iThera Data to HDF5 Format

Converts all scans within a folder to hdf5 format. Outputs hdf5 
format files in the second folder given.

```shell script
msot-import-ithera /path/to/itherastudyfolder /path/to/processeddatafolder
```

### 2. Set speed of sounds for each scan

Will loop through each scan within the specified folder, 
allowing you to tune the speed of sound. (this does a quick
default backprojection, unless you specify a different preset
e.g. for clinical data). Specify a folder and a starting speed
of sound.

Not you can optionally include a line plot of the photoacoustic signal
through the centre of the images by adding the option ```-L True``` after
the command.

```shell script
msot-set-speed-of-sound /path/to/processeddatafolder 1465
```

### 3. Reconstruct scan(s)

Once you have set the speed of sound for a desired scan, you
can then run the reconstruction algorithm. By default this 
uses a suitable backprojection, but you 
can change this by passing a different json file on the command
line as optional argument ```-p /path/to/presets/___.json```.

The argument here can now be either a file or a folder.

```shell script 
msot-reconstruct /path/to/processeddatafolder
```

### 4. Draw regions of interest

The possible roi names are specified in the python file
(sorry), it's easy enough to delete this, I just didn't want
to accidentally include ROIs with typos in their names. 
You can also specify the position (e.g. ```-n tumour -p left```).

```shell script
msot-draw-roi /path/to/processeddatafolder -n ROINAME
```
### 5. Spectral Unmixing

By  default this downscales the MSOT images
by a factor of 2 and then unmixes into Hb/HbO2 and calculates sO2. 
Like recons, you can specify a preset, which gives you more freedom
e.g. melanin/ICG etc. You can also specify which wavelengths to 
include in the unmixing. This will automatically run for all scans
unless you include the ```-f``` option, which will filter
out scans which have the text following this in their name 
(for example, I quite often look at gas challenge data. When
I take the scan I always include GC_SS in the name, so I filter this).

```shell script
msot-unmix /path/to/processeddatafolder
```

### 6. Further analysis

You can do automated OE/DCE analysis with 
```shell script
msot-analyse-dce <folder name> -p <label e.g. DCE>
``` 

and
 
```shell script
msot-analyse-gc <folder name> -p <label e.g. GC_SS>```. This will put delta so2 or ICG measurements
````

in your hdf5 files. You should replace the label with however you've named your data,
e.g. I often name the data like so: name-mouse-GC_SS for gas challenge single slice, for that I would run:
```shell script
msot-analyse-gc /path/to/folder -p GC_SS
```


# Availability, documentation and development
POTATO can be installed on Windows, MacOS and Linux via ```pip install potato```, or the most up-to-date code can be obtained on GitHub **here - add link**. Certain components rely on CUDA so require an nVidia graphics card and may require a substantial amount of graphics RAM depending on the particular use case. Full documentation is available here **add link here**. Contributions can be made to the GitHub repository here: **add link here**. Feedback, bug reports or suggestions for further development can be made via GitHub issues, or by email. 

# Acknowledgements

We acknowledge helpful feedback from the users of this tool within our lab, particularly Mariam-Eleni Oraiopoulou and Ellie Bunce. 

# References