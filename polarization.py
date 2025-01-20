import numpy as np
import pandas as pd

def calculate_polarizationperunitcell(unitcell: pd.DataFrame) -> list:
    '''
    atoms = pd.DataFrame({},  index = ['x','y','z'])
    atoms['Ba1'] = [   0.0245698,   -0.287814,  -0.104884 ]
    atoms['Ba2'] = [     3.99987,   -0.285513,  -0.164884 ]
    atoms['Ba3'] = [     0.11437,   3.62799  ,  -0.0912843]
    atoms['Ba4'] = [     4.11987,   3.71019  ,  0.026616  ]
    atoms['Ba5'] = [ 0.000770142,   -0.438614,  4.01512   ]
    atoms['Ba6'] = [     4.02727,   -0.375714,  3.97562   ]
    atoms['Ba7'] = [   0.0400701,   3.65649  ,  3.88302   ]
    atoms['Ba8'] = [     4.04987,   3.83999  ,  3.95772   ]
    atoms['O1']  = [     2.01917,  1.64999   , -0.115484  ]
    atoms['O2']  = [     2.08747,  -0.267314 ,     1.95712]
    atoms['O3']  = [   0.0425698,     1.70019,     1.90012]
    atoms['O4']  = [     1.93387,     1.67559,     3.86762]
    atoms['O5']  = [     2.05907,     3.83779,     1.87682]
    atoms['O6']  = [     3.99507,     1.89169,     1.85322]
    atoms['Ti']  = [     2.03547,     1.61119,     2.02362]
    '''
    '''
              atomtype        x        y        z  ppx  ppy  ppz
    1735       1.0  31.9334  19.9624  16.0193  0.0  0.0  0.0
    1730       1.0  27.9978  19.9820  15.9751  0.0  0.0  0.0
    1685       1.0  32.0322  15.9680  15.8836  0.0  0.0  0.0
    1680       1.0  28.1184  16.0189  15.8560  0.0  0.0  0.0
    1230       1.0  27.9740  20.1385  11.9737  0.0  0.0  0.0
    1185       1.0  32.0597  15.9045  11.9897  0.0  0.0  0.0
    1180       1.0  28.0108  16.0065  11.9308  0.0  0.0  0.0
    1235       1.0  31.9973  19.9329  11.9196  0.0  0.0  0.0
    1236       2.0  30.0022  18.0472  12.0102  0.0  0.0  0.0
    1733       2.0  27.9080  18.0047  14.1104  0.0  0.0  0.0
    1687       2.0  30.0399  16.0620  13.9414  0.0  0.0  0.0
    1738       2.0  31.9812  18.1157  14.0358  0.0  0.0  0.0
    1737       2.0  29.9913  20.0380  14.0454  0.0  0.0  0.0
    1736       2.0  29.9697  18.0981  16.0517  0.0  0.0  0.0
    1739       3.0  30.0482  18.0595  13.9687  0.0  0.0  0.0
    '''


    displacement = unitcell.copy()

    '''
    com = pd.Series({'atomtype': 'COM',
        'x': float(np.mean(unitcell['x'])),
        'y': float(np.mean(unitcell['y'])),
        'z': float(np.mean(unitcell['z']))})
    displacement[['x','y','z']] = unitcell[['x','y','z']] - com[['x','y','z']]                # take center of mass as reference point for the displacement
    '''
    displacement[['x','y','z']] = unitcell[['x','y','z']].sub(unitcell[['x','y','z']].iloc[-1]) # take the surrounded Ti as reference point for the displacement, following the definition from Speliasky&Cohen, doi: 10.1088/0953-8984/23/43/435902
    # these two reference points (com or surrounded Ti) end up at the same results, it seems to be always like this.
    
    displacement['dp_x'] = displacement['x'] * displacement['charge'] * displacement['weight']
    displacement['dp_y'] = displacement['y'] * displacement['charge'] * displacement['weight']
    displacement['dp_z'] = displacement['z'] * displacement['charge'] * displacement['weight']
    
    latt_x = float(np.max(displacement['x'])) - float(np.min(displacement['x']))
    latt_y = float(np.max(displacement['y'])) - float(np.min(displacement['y']))
    latt_z = float(np.max(displacement['z'])) - float(np.min(displacement['z']))
    
    unitcell_vol = latt_x * latt_y * latt_z
    Px = np.sum(displacement['dp_x']) / unitcell_vol * 1600
    Py = np.sum(displacement['dp_y']) / unitcell_vol * 1600
    Pz = np.sum(displacement['dp_z']) / unitcell_vol * 1600
    return [Px, Py, Pz]

def read_lmp(filename):

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
    charges = [2, -2, 4]
    data['charge'] = np.select(conditions, charges, default=0)

    conditions = [(data['atomtype'] == 1),    (data['atomtype'] == 2),    (data['atomtype'] == 3)]
    weights = [1/8, 1/2, 1]
    data['weight'] = np.select(conditions, weights, default=0)

    if len(data) == num_atoms:
        data = data.drop(['id'], axis=1)
        data = data.sort_values(by=['atomtype'], ascending=False)
    else:
        print('atoms lost...')
        return -1

    return [box, data]


def loopthroughTi(dic:dict) -> list:
    xlo, xhi, ylo, yhi, zlo, zhi = dic[0]
    data = dic[1].drop(['ppx', 'ppy', 'ppz'], axis = 1)
    x_start = (xlo + xhi) / 4
    y_start = (ylo + yhi) / 4
    z_start = (zlo + zhi) / 4
    x_end   = (xlo + xhi) / 4 * 3
    y_end   = (ylo + yhi) / 4 * 3
    z_end   = (zlo + zhi) / 4 * 3
    mask_Ti = (data['atomtype'] == 3)
    mask_x  = (data['x'] > x_start) & (data['x'] < x_end)
    mask_y  = (data['y'] > y_start) & (data['y'] < y_end)
    mask_z  = (data['z'] > z_start) & (data['z'] < z_end)
    Tis = data[ mask_Ti & mask_x & mask_y & mask_z ].sort_values(by=['x', 'y', 'z'], ascending=[True, True, True])
    Px = []
    Py = []
    Pz = []
    for i, _ in Tis.iterrows():
        unitcell = find_surroundingTi(data, i)
        px, py, pz = calculate_polarizationperunitcell(unitcell)
        Px.append(px)
        Py.append(py)
        Pz.append(pz)
    return [ sum(Px)/len(Px), sum(Py)/len(Py), sum(Pz)/len(Pz)]

    


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



    

if __name__ == "__main__":
    # filename = sys.argv[1]
    polarization = pd.DataFrame({}, columns = ['Px', 'Py', 'Pz'])

    for i in range(80000, 100000, 500):
        filename = f'./converted_timestep_{i}.lmp'
        info = read_lmp(filename)
        polarization.loc[i] = loopthroughTi(info)

    polarization.to_csv('fullunitcellpolarization.csv')
