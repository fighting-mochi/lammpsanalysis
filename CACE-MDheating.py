import sys
import os
import pickle
import numpy as np
import torch
import torch.nn as nn

import cace
from cace.calculators import CACECalculator
from cace.models.atomistic import NeuralNetworkPotential

from ase import units
from ase.md.langevin import Langevin
from ase.md.npt import NPT
from ase.md.nptberendsen import Inhomogeneous_NPTBerendsen, NPTBerendsen, NVTBerendsen
from ase.md import MDLogger
from ase.io import read, write

from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from types import SimpleNamespace

############# set up the atoms object #############
cuda_device = "cuda"

ref_atom = read('/capstor/scratch/cscs/lhsu/MD_ferroBTO/initialstructure/BaTiO3_R_uppertriangular.traj')

n = 2
ref_atom = ref_atom.repeat((n, n, n))

timestep = 2
temp = 50
steps_equi = 2 #1000//timestep * 5
#steps_ramp = 5 #1000//timestep * 200

args_dict = {"timestep": timestep,
             "temperature": temp,
             "pressure": (1.01325)* 1e-4 + 2.8,
             "taut": 20 * timestep,
             "taup": 200 * timestep,
             "bulk_modulus": 10,
             "trajector_equi":f'equil_{temp}K.traj',
             "logfile_equi":f'equil_{temp}K.log',
             "trajector_heat": 'heating_ramp.traj',
             "logfile_heat": 'heating_ramp.log',
             "loginterval": 1,}

###################################################

cace_nnp_noStress = torch.load('/capstor/scratch/cscs/lhsu/ferroBTO/CACE_NNP_phase_4.pth', weights_only = False, map_location=torch.device('cuda'))
if os.path.exists('/capstor/scratch/cscs/lhsu/ferroBTO/avge0.pkl'):
    with open('/capstor/scratch/cscs/lhsu/ferroBTO/avge0.pkl', 'rb') as f:
        avge0 = pickle.load(f)
else:
    print('there is no avge0.pkl')

forces = cace.modules.Forces(energy_key='CACE_energy',
                             forces_key='CACE_forces',
                             stress_key='CACE_stress',)

output_modules =  cace_nnp_noStress.output_modules[0:4] + [forces]

cace_nnp = NeuralNetworkPotential(
    representation=cace_nnp_noStress.representation,
    output_modules= output_modules,
    keep_graph=True
)

for param_noS, param_withS in zip(cace_nnp_noStress.parameters(), cace_nnp.parameters()):
    if param_noS.shape == param_withS.shape:
        param_withS.data = param_noS.data.clone()
    else:
        print(f"Skipping parameter with shape {param_noS.shape} as it does not match {param_withS.shape}")

cace_nnp = cace_nnp.to(torch.device(cuda_device))

############################################

calculator = CACECalculator(model_path=cace_nnp,
                            device= cuda_device,
                            energy_key='CACE_energy',
                            forces_key='CACE_forces',
                            stress_key='CACE_stress',
                            compute_stress=True,
                            atomic_energies= avge0,
                            )
ref_atom.set_calculator(calculator)

args = SimpleNamespace(**args_dict)

ptime = args.taup * units.fs
bulk_modulus_au = args.bulk_modulus / 160.2176  # GPa to eV/A^3
compressibility_au = 1 / bulk_modulus_au

# Equilibration at 50 K
dyn_ber = NPTBerendsen(
    atoms=ref_atom,
    timestep=args.timestep * units.fs,
    temperature_K=temp,
    pressure_au=args.pressure * units.GPa,
    taut=args.taut * units.fs,
    taup=args.taup * units.fs,
    compressibility_au=compressibility_au,
    trajectory=f'equil_{temp}K.traj',
    logfile=f'equil_{temp}K.log',
    loginterval=args.loginterval,
    append_trajectory=False,
)
dyn_ber.run(steps_equi)  # steps_equi: number of steps for initial equilibration

# Get the final equilibrated structure
#atoms_equil = ref_atom.copy()

####### Sample MD #######
# Prepare trajectory and log files for the heating ramp

#########################
# Custom logger section #
#########################

def custom_md_logger(atoms, step, logfile=args.logfile_heat):
    etotal = atoms.get_total_energy()
    pe = atoms.get_potential_energy()
    ke = atoms.get_kinetic_energy()
    temp = atoms.get_temperature()
    stress = atoms.get_stress()  # returns stress in eV/Å^3
    s1, s2, s3, s4, s5, s6 = stress
    press = -np.mean(np.array(stress[:3])) * 160.21766208  # GPa
    # Try to get custom properties, else set to nan
    # ecoul = getattr(atoms, 'ecoul', np.nan)
    # print(ecoul)
    # elong = getattr(atoms, 'elong', np.nan)
    # print(elong)
    fmax = np.max(np.linalg.norm(atoms.get_forces(), axis=1))
    vol = atoms.get_volume()
    a, b, c = atoms.cell.lengths()
    alpha, beta, gamma = atoms.cell.angles()
    # dists = atoms.get_all_distances(mic=True)
    n_atoms = len(atoms)

    with open(args.logfile_heat, 'a') as f:
        f.write(f"{step:6d} {n_atoms:4d} {etotal[0]:12.6f} {pe[0]:12.6f} {ke:12.6f} {press:10.3f} {temp:10.2f} "
                f"{s1:10.4f} {s2:10.4f} {s3:10.4f} {s4:10.4f} {s5:10.4f} {s6:10.4f} {a:10.4f} {b:10.4f} {c:10.4f} "
                f"{alpha:8.3f} {beta:8.3f} {gamma:8.3f} {fmax:10.4f} {vol:12.4f}\n")

# Write header once
with open(args.logfile_heat, 'w') as f:
    f.write("# step n_atoms etotal pe ke press temp s1 s2 s3 s4 s5 s6 boxa boxb boxc boxalpha boxbeta boxgamma fmax vol\n")

#########################
# Heating ramp with custom logging
#########################

atoms_ramp = read(args.trajector_equi, index=-1)

calculator = CACECalculator(model_path=cace_nnp,
                            device= cuda_device,
                            energy_key='CACE_energy',
                            forces_key='CACE_forces',
                            stress_key='CACE_stress',
                            compute_stress=True,
                            atomic_energies= avge0,
                            )
atoms_ramp.set_calculator(calculator)


T_start = temp
T_end = 70 #0
T_step = 5
temps = range(T_start, T_end + T_step, T_step)
steps_per_temp = 1 #000  # e.g., 1000 steps per temperature

# Start from the equilibrated structure
# atoms = atoms_equil.copy()


step_counter = 0
for T in temps:
    dyn = NPT(
        atoms=atoms_ramp,
        timestep=args.timestep * units.fs,
        temperature_K=T,
        externalstress=args.pressure * units.GPa,
        ttime=args.taut * units.fs,
        pfactor=args.bulk_modulus * units.GPa * (args.taup * units.fs)**2,
        trajectory=args.trajector_heat,
        logfile=None,  # We'll use our own logger
        loginterval=args.loginterval,
        append_trajectory=True,
        mask=(1, 1, 1),
    )
    print(f"Heating at {T} K")

    # Attach custom logger
    def logger_func():
        global step_counter
        custom_md_logger(atoms_ramp, step_counter, logfile=args.logfile_heat)
        step_counter += args.loginterval

    dyn.attach(logger_func, interval=args.loginterval)
    dyn.run(steps_per_temp)
#     # atoms_ramp is now updated and ready for the next T
#    atoms_ramp = read(args.trajector_heat, index=-1)

#########################

