import sys
import pathlib
import numpy as np
import pandas as pd
import heapq as hq


'''
cd <folder> # this <folder> contains .lmp files (e.g. converted_timestep_<timestep>.lmp can be generated by extract_lmp_fromtrajectory.py)
python polarization.py

search "USER" for parameters and setting, that you might want to change based on your use case
'''

unit_eA2_to_muCcm2 = 1602.176634  # e/Ang^2 -> muC/cm^2

def calculate_polarizationperunitcell(unitcell: pd.DataFrame) -> list[float]:
    '''    This is how unitcell look like; atomtype 1, 2, 3 refers to Ba, O, Ti

    unit: charge (e), distance (Ang)

          atomtype        x        y        z  charge  weight
    1880       1.0  28.0640  32.1431  15.9005     2.0   0.125
    1825       1.0  24.0622  28.0952  16.0106     2.0   0.125
    1830       1.0  28.0093  28.1048  15.9362     2.0   0.125
    1875       1.0  24.0724  32.1496  16.0610     2.0   0.125
    1380       1.0  28.0440  32.1338  11.9469     2.0   0.125
    1375       1.0  23.8237  32.0746  11.9631     2.0   0.125
    1330       1.0  28.0709  28.0860  11.9405     2.0   0.125
    1325       1.0  24.0328  28.1337  12.0208     2.0   0.125
    1381       2.0  26.0104  30.1939  11.9385    -2.0   0.500
    1832       2.0  26.0977  28.2621  13.8121    -2.0   0.500
    1883       2.0  28.1233  30.3029  13.9472    -2.0   0.500
    1882       2.0  26.0058  32.2514  13.8832    -2.0   0.500
    1881       2.0  26.0271  30.2215  15.9246    -2.0   0.500
    1878       2.0  23.9907  30.1572  14.0518    -2.0   0.500
    1884       3.0  25.9456  30.0481  13.9087     4.0   1.000
    '''

    # estimate the unit cell volume by the 6 averaged planes of the Ba corners 
    latt_x = np.mean(hq.nlargest(4, unitcell[unitcell['atomtype']==1.0]['x'])) - np.mean(hq.nsmallest(4, unitcell[unitcell['atomtype']==1.0]['x']))
    latt_y = np.mean(hq.nlargest(4, unitcell[unitcell['atomtype']==1.0]['y'])) - np.mean(hq.nsmallest(4, unitcell[unitcell['atomtype']==1.0]['y']))
    latt_z = np.mean(hq.nlargest(4, unitcell[unitcell['atomtype']==1.0]['z'])) - np.mean(hq.nsmallest(4, unitcell[unitcell['atomtype']==1.0]['z']))
    unitcell_vol = latt_x * latt_y * latt_z
    # print(unitcell)
    # print(f'{latt_x  = }', f'{latt_y  = }', f'{latt_z = }', f'{unitcell_vol = }')

    displacement = unitcell.copy()
    # these two reference points (com or surrounded Ti) end up at the same results.
    '''
    com = pd.Series({'atomtype': 'COM',
        'x': float(np.mean(unitcell['x'])),
        'y': float(np.mean(unitcell['y'])),
        'z': float(np.mean(unitcell['z']))})
    displacement[['x','y','z']] = unitcell[['x','y','z']] - com[['x','y','z']]                # take center of mass as reference point for the displacement
    '''
    displacement[['x','y','z']] = unitcell[['x','y','z']].sub(unitcell[['x','y','z']].iloc[-1]) # take the surrounded Ti as reference point for the displacement, following the definition from Speliasky&Cohen, doi: 10.1088/0953-8984/23/43/435902

    displacement['dp_x'] = displacement['x'] * displacement['charge'] * displacement['weight']
    displacement['dp_y'] = displacement['y'] * displacement['charge'] * displacement['weight']
    displacement['dp_z'] = displacement['z'] * displacement['charge'] * displacement['weight']
    # print(displacement)

    Px = np.sum(displacement['dp_x']) / unitcell_vol * unit_eA2_to_muCcm2
    Py = np.sum(displacement['dp_y']) / unitcell_vol * unit_eA2_to_muCcm2
    Pz = np.sum(displacement['dp_z']) / unitcell_vol * unit_eA2_to_muCcm2
    # print(f'{Px = }', f'{Py = }', f'{Pz = }')

    com_x, com_y, com_z = [ float(np.mean(unitcell['x'])), float(np.mean(unitcell['y'])), float(np.mean(unitcell['z'])) ] # a list of the position (x, y, z) of the center of mass (com) of the unitcell
    # print([com_x, com_y, com_z, Px, Py, Pz])
    return [com_x, com_y, com_z, Px, Py, Pz]

def read_lmp(filename: str) -> list[ list[float], pd.DataFrame ]:

    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()
            if "atoms" in line:
                num_atoms = int(line.split()[0])
            elif "xlo xhi" in line:
                xlo, xhi = line.split()[0:2]
            elif "ylo yhi" in line:
                ylo, yhi = line.split()[0:2]
            elif "zlo zhi" in line:
                zlo, zhi = line.split()[0:2]
            # look up python match
    box = [float(xlo), float(xhi), float(ylo), float(yhi), float(zlo), float(zhi)]

    data = pd.read_csv(filename, names = ['id', 'atomtype', 'x', 'y', 'z', 'ppx', 'ppy', 'ppz'], skiprows=18, sep=r'\s+')

    conditions = [(data['atomtype'] == 1),    (data['atomtype'] == 2),    (data['atomtype'] == 3)]
    charges = [2, -2, 4]  # USER: nominal charges are used for BaTiO3 (Ba: +2, O: -2, Ti: +4)
    data['charge'] = np.select(conditions, charges, default=0)

    conditions = [(data['atomtype'] == 1),    (data['atomtype'] == 2),    (data['atomtype'] == 3)]
    weights = [1/8, 1/2, 1] # USER: weights of atoms in a unit cell
    data['weight'] = np.select(conditions, weights, default=0)

    if len(data) == num_atoms:
        data = data.drop(['id'], axis=1)
        data = data.sort_values(by=['atomtype'], ascending=False)
    else:
        print('atoms lost...')
        return -1

    return [box, data]


def loopthroughTi(dic: dict) -> pd.DataFrame:
    xlo, xhi, ylo, yhi, zlo, zhi = dic[0]
    data = dic[1].drop(['ppx', 'ppy', 'ppz'], axis = 1)

    x_start = (xlo + xhi) / 4 # /4 or 3/4 is just to avoid being too close to the system boundary (thus, can be changed), as periodic image is not yet implemented in the current code.
    y_start = (ylo + yhi) / 4
    z_start = (zlo + zhi) / 4
    x_end   = (xlo + xhi) / 4 * 3
    y_end   = (ylo + yhi) / 4 * 3
    z_end   = (zlo + zhi) / 4 * 3

    mask_Ti = (data['atomtype'] == 3)
    mask_x  = (data['x'] > x_start) & (data['x'] < x_end)
    mask_y  = (data['y'] > y_start) & (data['y'] < y_end)
    mask_z  = (data['z'] > z_start) & (data['z'] < z_end)
    Tis = data[ mask_Ti & mask_x & mask_y & mask_z ].sort_values(by=['x', 'y', 'z'], ascending=[True, True, True])    # find the Ti that is not too close to the edge. the number of such Ti is different at each time step, i.e. print(len(Tis)) is different at each time step
    
    j = 0 # the index of unit cell; final_j + 1 == len(Tis)
    sys_data = pd.DataFrame({}, columns=['x', 'y', 'z', 'px', 'py', 'pz']) 
    for i, _ in Tis.iterrows():
        unitcell = find_surroundingTi(data, i)
        sys_data.loc[j] = calculate_polarizationperunitcell(unitcell)
        j += 1

    # print(sys_data)
    return sys_data
    


def find_surroundingTi(data: pd.DataFrame, index: int) -> pd.DataFrame:
    # this function finds the 8 surrounding Ba and 6 surrounding O of each Ti.
    surrounded_Ti = pd.DataFrame(data.loc[index]).T
    pos_x, pos_y, pos_z = [ float(surrounded_Ti.iloc[0][axis]) for axis in ['x', 'y', 'z'] ]   # pos_x, pos_y, pos_z = [ 21.8977 , 21.9272 , 21.8901] # one Ti position
    mask_Ba = (data['atomtype'] == 1)
    mask_O  = (data['atomtype'] == 2)
    mask_x  = (data['x'] > pos_x - 3) & (data['x'] < pos_x + 3)
    mask_y  = (data['y'] > pos_y - 3) & (data['y'] < pos_y + 3)
    mask_z  = (data['z'] > pos_z - 3) & (data['z'] < pos_z + 3)
    surrounding_Ba = data[mask_Ba & mask_x & mask_y & mask_z]
    surrounding_O  = data[mask_O  & mask_x & mask_y & mask_z]
    if len(surrounding_Ba) != 8:
        print('wrong')
    if len(surrounding_O) != 6:
        print('wrong')
    return pd.concat([surrounding_Ba, surrounding_O, surrounded_Ti])


# def write_ovito(ovito_timeframe: int, sourcefile: str, num_unitcell: int, box: list) -> :
#     with open(ovito_file)



if __name__ == "__main__":
    ovt_filename = sys.argv[1] + '.ovt'
    pathlib.Path(ovt_filename).unlink(missing_ok=True)


    # USER: specify the timesteps where polarization needs to be estimated
    start    = 990000     # start
    end      = 995000    # end
    interval = 5000       # depends on the output frequency of lammps

    for i in range(start, end+1, interval):   
        filename = f'./converted_timestep_{i}.lmp'
        info = read_lmp(filename) # info = [box, data]
        xlo, xhi, ylo, yhi, zlo, zhi = info[0]
        sys_data = loopthroughTi(info)

        with open(ovt_filename, 'a+') as ovt:
            ovt.write('ITEM: TIMESTEP\n'
                f'{i-start} {filename}\n'
                'ITEM: NUMBER OF ATOMS\n'
                f'{len(sys_data)}\n'
                'ITEM: BOX BOUNDS pp pp pp\n'
                f'{xlo} {xhi}\n'
                f'{ylo} {yhi}\n'
                f'{zlo} {zhi}\n'
                'ITEM: ATOMS id x y z px py pz\n')
            sys_data.to_csv(ovt, sep='\t', header=False)

