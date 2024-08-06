Content:
This software is used to systematically assess how climate change affected the probability and intensity of extreme events, and to assess the contributions of emitters. This work relies on Extreme Event Attribution and climate emulators, and has been developed by Yann Quilcaille. A manuscript based on this software has been submitted, specifically attributing heatwaves to the emissions of the carbon majors. The DOI will be provided if accepted.

Citation:
When using this code, please acknowledge this software, and cite the incoming DOI for the manuscript.

How to install:
This software has been developed in Python 3.9.17, and has been validated on Linux (openSUSE 15.5). To install this software, please follow these instructions:
1. Install python on your machine (necessary to run): https://www.python.org/downloads/
2. Install conda on your machine (highly recommended to create environments): https://conda.io/projects/conda/en/latest/user-guide/install/index.html
3. Install mamba on your machine (recommended to handle environments): https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html
4. Install the depencies of this code using mamba: https://fig.io/manual/mamba/install
5. Install this code: the easy option is to download the code using the green button "<> Code" in the top-right corner. If you want to contribute to the development of the software, please use the steps outlined in the next section.
6. Unzip in your repository, and have fun.
The steps to set up Python (1-4) may take up to few hours, depending on your internet and knowledge of python. The steps to set up the code (5-6) would take few minutes.

How to contribute to the developments:
Contributions to the development of this software are welcomed. They will be handled through pull-requests on GitHub. To contribute, please follow these instructions:
1. Set up GitHub (online): create a GitHub account, and fork this code on your GitHub.
2. Set up Git (locally): install Git; clone your fork on your machine; create your branch for your developments
3. Edit your branch with your developments.
4. Push your developments using Git from your local repository to your remote repository, and create a pull-request from your GitHub.

How to use:
The first script to open is "attribution_majors.py". Here, you will have the overall structure: defining options for the use, loading data, analyzing, creating outputs.
 - Defining options for the run: we recommend keeping the preset ones, but you are welcome to try others.
 - Loading data: the different data sources will be loaded using the provided paths (code from "fcts_support_io.py"). It is necessary that you download the required data to run this script. Please check the next section, so that this software may access the data on your machine.
 - Analyzing: we highly recommend not to edit this section. Here, the extreme events will be defined with figures if required (code from "fcts_support_event.py"). Then, conditional distributions will be trained with figures if required (code from "fcts_support_training.py"). Then, the work is extended to the emitters (code from "fcts_support_synthesis.py").
 - Creating outputs: all results are completely synthesized in a so-called panorama (code from "fcts_support_synthesis.py"). Then figures & tables are created (code from "fcts_support_plot_v4.py")
Please mind that some scripts have been used to facilitate the work ("treatment*.py").
Please mind that two shell scripts are also provided ("run*.sh"), for those that have access to a server.
Any feedback is welcome, to make this code as user-friendly as possible given its technical aspects.

Data:
This software uses different sources of data, that cannot be provided here due to their size. We encourage the users to access this data using the following modes:
 - EM-DAT: available at https://public.emdat.be/
 - GADM: available at https://gadm.org/download_world.html
 - ERA5: available at https://doi.org/10.24381/cds.adbb2d47
 - BEST: available at http://berkeleyearth.org/data
 - CMIP6: available at https://aims2.llnl.gov/search/cmip6
 - Carbon Majors database: available at https://carbonmajors.org/Downloads
 - OSCAR computations: provided on request
Due to the limits set by this amount of data, a demo cannot be provided for this software.

System requirements:
This software has been run on an ETHZ server (64 cores, 1.5 TB RAM, 200 Gbit/s LAN, 64-bit Linux system). In theory, it could run on a laptop, but it has not been tested. The bottleneck would be CMIP6 data, that would represent a lot of storage & RAM. If you decide to work on a laptop and not a server, we would highly recommend going step by step to validate that your machine would be sufficient.

Dependencies:
 - cartopy (0.22.0)
 - cdo (1.6.0)
 - cftime (1.6.2)
 - geopandas (0.13.2)
 - igraph (0.10.4)
 - numpy (1.24.4)
 - openpyxl (3.1.2)
 - pandas (1.5.3)
 - regionmask (0.10.0)
 - scipy (1.11.2)
 - seaborn (0.12.2)
 - shapely (2.0.1)
 - statsmodels (0.14.0)
 - xarray (2023.8.0)
For information, the following dependencies will already be installed with Python: copy, csv, difflib, itertools, math, matplotlib, mpl_toolkits, os, sys, time, unicodedata, warnings.

Reproduction instructions for the manuscript "Systematic attribution of heatwaves to the emissions of carbon majors":
1. Follow instructions to install the software
2. Follow links to download the data
3. Edit paths where you store data in the script "attribution_majors.py" 
4. Run "attribution_majors.py" without changing the other options.
In case of issue, please contact Yann Quilcaille (yann.quilcaille@env.ethz.ch).
