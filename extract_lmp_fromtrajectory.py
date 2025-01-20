import sys
import numpy as np
from datetime import datetime

def read_trajectory(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    timesteps = []
    num_atoms = 0
    box_bounds = []
    atoms = []

    i = 0
    while i < len(lines):
        if 'ITEM: TIMESTEP' in lines[i]:
            timesteps.append(int(lines[i+1]))
            i += 2
        elif 'ITEM: NUMBER OF ATOMS' in lines[i]:
            num_atoms = int(lines[i+1])
            i += 2
        elif 'ITEM: BOX BOUNDS' in lines[i]:
            box_bounds.append([list(map(float, lines[i+1].split())),
                               list(map(float, lines[i+2].split())),
                               list(map(float, lines[i+3].split()))])
            i += 4
        elif 'ITEM: ATOMS' in lines[i]:
            atoms_data = []
            for j in range(num_atoms):
                atoms_data.append(list(map(float, lines[i+j+1].split())))
            atoms.append(atoms_data)
            i += num_atoms + 1
        else:
            i += 1

    return timesteps, num_atoms, box_bounds, atoms

def write_lmp(filename, timestep, num_atoms, box_bounds, atoms, masses):
    with open(filename, 'w') as f:
        f.write(f"LAMMPS data file via conversion script, version {datetime.now().strftime('%d %b %Y')}, timestep {timestep}\n\n")
        f.write(f"{num_atoms} atoms\n")
        f.write(f"{len(masses)} atom types\n\n")

        # Write box bounds
        f.write(f"{box_bounds[0][0]:.14f} {box_bounds[0][1]:.14f} xlo xhi\n")
        f.write(f"{box_bounds[1][0]:.14f} {box_bounds[1][1]:.14f} ylo yhi\n")
        f.write(f"{box_bounds[2][0]:.14f} {box_bounds[2][1]:.14f} zlo zhi\n")
        f.write(f"{box_bounds[0][2]:.14f} {box_bounds[1][2]:.14f} {box_bounds[2][2]:.14f} xy xz yz\n\n")

        # Write masses
        f.write("Masses\n\n")
        for i, mass in enumerate(masses):
            f.write(f"{i+1} {mass}\n")
        f.write("\n")

        # Write atoms
        f.write("Atoms # atomic\n\n")
        for atom in atoms:
            f.write(f"{int(atom[0])} {int(atom[1])} {atom[3]:.14f} {atom[4]:.14f} {atom[5]:.14f} 0 0 0\n")

def convert_trajectory_to_lmp(traj_filename, lmp_filename_template, output_folder):
    timesteps, num_atoms, box_bounds, atoms = read_trajectory(traj_filename)
    masses = [137.327, 15.999, 47.867]  # Hardcoded masses from your example

    for i, timestep in enumerate(timesteps):
        if (timestep % 500 == 0): # and (timestep > (74999 - 1)):  #== 100000: #
            lmp_filename = output_folder+'/'+lmp_filename_template.format(timestep)
            write_lmp(lmp_filename, timestep, num_atoms, box_bounds[i], atoms[i], masses)

# Usage
traj_filename = sys.argv[1] #"./heating.lammpstraj"]
output_folder = sys.argv[2] # "ef0.001"
lmp_filename_template = "converted_timestep_{}.lmp"
convert_trajectory_to_lmp(traj_filename, lmp_filename_template, output_folder)
