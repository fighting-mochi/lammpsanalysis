import sys
import numpy as np
import pandas as pd
from pathlib import Path

def process_file(filename):
    ti_data = []
    with open(filename, 'r') as file:
        ti_found = False
        for line in file:
            if 'Ti' in line:
                ti_found = True
            elif ti_found:
                elements = line.split()
                if len(elements) >= 4:
                    ti_data.append(elements[-4:])
                ti_found = False
    return ti_data


data = pd.DataFrame({}, columns=['Px', 'Py', 'Pz', 'Ptot'])

for ts in range(0,600000,5000):
    filename = Path(f'./converted_timestep_{ts}_edm_all.cfg')
    if filename.is_file():
        results = process_file(filename)

        Px = np.mean([float(sublist[0]) for sublist in results])
        Py = np.mean([float(sublist[1]) for sublist in results])
        Pz = np.mean([float(sublist[2]) for sublist in results])
        Ptot = np.mean([float(sublist[3]) for sublist in results])

        data.loc[ts] = [Px, Py, Pz, Ptot]

# print(data)

data.to_csv('Ti_dipolemoment.csv', index=True, header=True, index_label='Step')



# with open(filename+'_out', 'w') as outfile:
#     # outfile.write("\n".join(results))
#     outfile.writelines(results)


