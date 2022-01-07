# Script to download all 200 observational ensemble memebers of the HadCRUT5 dataset.
# Script downloads and assimilates the data, producing a GMST, a single location for all ensmeble memebers and and overall mean and std

import requests
from glob import glob
import os
from zipfile import ZipFile

obs_dir = '/gws/nopw/j04/lancs_atmos/users/amosm1/bayesian_ensembling/obs/'

# Download files - files are saved on the GWS
url_template = "https://www.metoffice.gov.uk/hadobs/hadcrut5/data/current/analysis/HadCRUT.5.0.1.0.analysis.anomalies.{}_to_{}_netcdf.zip"
for i in range(20):
    idx0 = 1 + i * 10
    idx1 = (i + 1) * 10
    url = url_template.format(idx0, idx1)
    save_file_name = url.split('/')[-1]
    if not os.path.exists(save_file_name):
        print(f'Downloading: {save_file_name}')
        response = requests.get(url)
        open(os.path.join(obs_dir, save_file_name), "wb").write(response.content)
    else:
        print(f'Already downloaded - skipping: {save_file_name}')

# Unzip files
zip_files = glob(obs_dir + '*.zip')
for file in zip_files:
    with ZipFile(file, 'r') as zipfile:
        zipfile.extractall(obs_dir)
        # Clean up and remove zip files
        os.remove(file)

# Produce mean file
