import pandas as pd
from tqdm import tqdm

"""
This is script to convert the raw exoplanets data, downloaded from the NASA
exoplanet archive, to a CSV file for each system. A subset of columns is used
"""

COLUMNS_TO_SELECT = [
    'pl_hostname', 'pl_letter', 'pl_orbper','pl_orbsmax','pl_orbeccen', 
    'pl_orbincl',  'pl_radj', 'st_mass', 'pl_angsep', 'pl_orbtper', 
    'pl_orblper', 'pl_bmassj', 'pl_bmassprov', 'pl_tranmid'
]

df = pd.read_csv('data/planets_raw.csv', comment='#')

# Save new files
for name, system in tqdm(df.groupby('pl_hostname')):
    file_name = name.lower().replace(' ', '-')
    system[COLUMNS_TO_SELECT].to_csv('data/systems/{}.csv'.format(file_name))

