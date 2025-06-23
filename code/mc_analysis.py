import mdtraj as md
import glob
import numpy as np
import pandas as pd
import os
import rdkit
from rdkit import Chem
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks
from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity
import parmed as pmd
from parmed import gromacs

# Predefined constants and SMARTS patterns
ONE_4PI_EPS0 = 138.93545764438198
SCAFFOLD = Chem.MolFromSmarts('COC([C@H]1N(C([C@H](c3cc(O)c(O)c(O)c3)C2C~CCCC2)=O)CCCC1)=O')
ALIPHATIC_SCAFFOLD = Chem.MolFromSmarts('COC([C@H]1N(C([C@H](C3CC(O)C(O)C(O)C3)C2C~CCCC2)O)CCCC1)O') # If you're loading a topology with no bond orders (e.g. from PDB)
linker_SMARTS = lambda length: f"OccO{'*' * length}OC=O" # Get the linker atoms for a given length
aliphatic_linker_SMARTS = lambda length: f"OCCO{'*' * length}OC(O)CN" # Same as above but with no bond orders

def get_scaffold_atoms(mol: rdkit.Chem.rdchem.Mol, scaff) -> list:
    """
    Get the atoms of the scaffold in the molecule.
    """
    scaff_atms = mol.GetSubstructMatches(scaff)
    if len(scaff_atms) == 0:
        return []
    elif len(scaff_atms) > 1:
        print(scaff_atms)
        return scaff_atms[0]
    else:
        return scaff_atms[0]
    
def get_linker_atoms(mol: rdkit.Chem.rdchem.Mol, aliphatic_top=False) -> list:
    """
    Get the atoms of the linker in the molecule. This tries to find the largest possible match (starting from 15) for the linker SMARTS.
    """
    # Get the largest possible match for the SMARTS
    if not aliphatic_top:
        SMARTS = linker_SMARTS
    else:
        SMARTS = aliphatic_linker_SMARTS
    for i in range(15, 0, -1):
        linker = Chem.MolFromSmarts(SMARTS(i))
        linker_atms = mol.GetSubstructMatches(linker)
        if len(linker_atms) > 0:
            break
    is_para = False
    if len(linker_atms) == 0:
        return None, None
    # The linker SMARTS defined above has two possible matches in the para systems, that are both symmetric and lie on the same plane.
    # We can therefore only take the first match.
    elif len(linker_atms) > 1:
        if len(linker_atms) == 2 and linker_atms[0][2:] == linker_atms[1][2:]:
            is_para = True
        else:
            is_para = None
            print("MULTIPLE SUBSTRUCTURE MATCHES FOUND!", linker_atms)
    linker_atms = linker_atms[0]
    if aliphatic_top:
        linker_atms = linker_atms[:-2] # Remove the last two atoms which are only there for the SMARTS matching
    return linker_atms, is_para

# Define the dihedral atoms. These are defined relative to the scaffold atoms.
default_dihedral_atoms = {
    # Torsion dihedrals
    'A': [0, 1, 2, 3],
    'B': [1, 2, 3, 4],
    'C': [3, 4, 5, 6],
    'D': [4, 5, 6, 7],
    # E, F and G have two possible matches, so here we give them both and we figure out which is which later
    'E_0': [5,6,16,17],
    'E_1': [5,6,16,21],
    'F_0': [5,6,7,8],
    'F_1': [5,6,7,15],
    'G_0': [6,16,17,18],
    'G_1': [6,16,21,20],
}

# This angle is part of the scaffold but is not used in the main analysis as it is always constant
# Only used for supplementary figure
extra_dihedral_atoms = {
    'Bbis': [2,3,4,5],
}

def get_scaffold_dihedrals(path, scaff=SCAFFOLD):
    """
    Get the dihedral angles of the scaffold from the trajectory.
    """
    dcd_file = path + '/ensemble-0.dcd'
    top_file = glob.glob(path + '/reference.pdb')[0]
    sdf_file = glob.glob(path + '/*.sdf')[0]
    traj = md.load(dcd_file, top=top_file)
    mol = Chem.SDMolSupplier(sdf_file)[0]
    scaff_atoms = get_scaffold_atoms(mol, scaff)
    lig_dihedral_atoms = [[scaff_atoms[i] for i in dihedral] for dihedral in default_dihedral_atoms.values()]
    angles = md.compute_dihedrals(traj, lig_dihedral_atoms)
    angles = np.rad2deg(angles)
    angles = np.where(angles < 0, angles + 360, angles)
    # We are using the default dihedral atoms
    # Angle E has two possible matches that are separated by 120 degrees, so always take the +120 degrees match as reference
    angle_E_diff = angles[:, 4] - angles[:, 5]
    angle_E_diff = (angle_E_diff + 180) % 360 - 180 # this is either (around) +120 or -120
    assert np.all(abs(angle_E_diff) < 150) and np.all(abs(angle_E_diff) > 90)
    angle_E = angles[:, 4] if np.mean(angle_E_diff) > 0 else angles[:, 5]
    # Angle F is invariant under C2 rotation, so we just take the angle that is most often in the range [0, 180] as reference
    n_below_180 = np.sum(angles[:, 6] < 180)
    angle_F = angles[:, 6] if n_below_180 > len(angles) / 2 else angles[:, 7]
    # Angle G has two possible matches that we assume are symmetric, so we take the angle that is most often in the range [0, 180] as reference
    n_below_180 = np.sum(angles[:, 8] < 180)
    angle_G = angles[:, 8] if n_below_180 > len(angles) / 2 else angles[:, 9]
    angles = np.concatenate((angles[:, :4], angle_E[:, np.newaxis], angle_F[:, np.newaxis], angle_G[:, np.newaxis]), axis=1)
    return angles

def get_extra_dihedrals(path: str, scaff=SCAFFOLD) -> np.ndarray:
    """
    Get the one extra dihedral angle for supplementary figure.
    """
    dcd_file = path + '/ensemble-0.dcd'
    top_file = glob.glob(path + '/reference.pdb')[0]
    sdf_file = glob.glob(path + '/*.sdf')[0]
    traj = md.load(dcd_file, top=top_file)
    mol = Chem.SDMolSupplier(sdf_file)[0]
    scaff_atoms = get_scaffold_atoms(mol, scaff)
    lig_dihedral_atoms = [[scaff_atoms[i] for i in dihedral] for dihedral in extra_dihedral_atoms.values()]
    angles = md.compute_dihedrals(traj, lig_dihedral_atoms)
    angles = np.rad2deg(angles)
    angles = np.where(angles < 0, angles + 360, angles)
    return angles

def get_linker_dihedrals(path: str, aliphatic_top=False) -> np.ndarray:
    """
    Get the dihedral angles of the linker from the trajectory.
    """
    dcd_file = path + '/ensemble-0.dcd'
    top_file = glob.glob(path + '/reference.pdb')[0]
    sdf_file = glob.glob(path + '/*.sdf')[0]
    traj = md.load(dcd_file, top=top_file)
    mol = Chem.SDMolSupplier(sdf_file)[0]
    linker_atms, is_para = get_linker_atoms(mol, aliphatic_top)
    linker_dihedrals = [linker_atms[i:i+4] for i in range(0, len(linker_atms)-3)] # A linear chain of N atoms defines N-3 dihedrals
    linker_dihedrals = linker_dihedrals[1:] # The first dihedral (OccO, planar) is uninteresting so we skip it
    angles = md.compute_dihedrals(traj, linker_dihedrals)
    angles = np.rad2deg(angles)
    angles = np.where(angles < 0, angles + 360, angles)
    if is_para:  
        # The first angle of the para linkers is invariant under C2 rotation, but for consistency we shift it so that the most occuring angle is in the range [0, 180]
        # As that can be a hard degree of freedom for some systems
        angles = angles.T
        most_common_angle = np.argmax(np.histogram(angles[0], bins=2, range=(0, 360))[0])
        if most_common_angle == 1:
            angles[0] = [angle + 180 for angle in angles[0]]
            angles = np.where(angles >= 360, angles - 360, angles)
        angles = angles.T
    return angles

def get_hbond_energy(distances, parameters):
    """
    Calculate the hydrogen bond energy as a sum of the electrostatic and van der Waals interactions of the dipole-dipole interaction.
    Distances is an array of shape (4,4,N) containing all distances between the 4 atoms in N frames.
    Parameters is an array of shape (4,3) containing [charge, sigma, epsilon] for each atom type [acceptor_pos, acceptor_neg, hydrogen, donor].
    """
    coulomb = np.zeros((4,4, distances.shape[2]))
    lj = np.zeros((4,4, distances.shape[2]))
    # iterate over all atom pairs and calculate the energies
    for i in range(4):
        for j in range(i+1, 4):
            if (i == 0 and j == 1) or (i == 2 and j == 3):
                # don't calculate interactions between bonded atoms
                continue
            charge_i, sigma_i, epsilon_i = parameters[i]
            charge_j, sigma_j, epsilon_j = parameters[j]
            r = distances[i,j]
            coulomb[i,j] = ONE_4PI_EPS0 * charge_i * charge_j / r
            sigma = (sigma_i + sigma_j) / 2
            epsilon = np.sqrt(epsilon_i * epsilon_j)
            lj[i,j] = 4 * epsilon * ((sigma / r)**12 - (sigma / r)**6)
    return coulomb, lj

def get_parameters(processed_top_file, search_atoms):
    """
    Get the parameters from a gromacs topology file.
    search_atoms is a list of (resname, resid, atomname) tuples to extract parameters for.
    """
    basepath = os.path.dirname(processed_top_file)
    gromacs.GROMACS_TOPDIR = basepath
    s = pmd.load_file(processed_top_file)
    parameters = []
    for resname, resid, atomname in search_atoms:
        assert s.residues[resid].name == resname
        res_atoms = s.residues[resid].atoms
        at = [a for a in res_atoms if a.name == atomname][0]
        charge = at.charge
        sigma = at.sigma / 10 # angstrom to nm
        epsilon = at.epsilon * 4.184 # kcal/mol to kJ/mol
        parameters.append([charge, sigma, epsilon])
    return np.array(parameters)

def best_periodic_shift(angles):
    """
    Shifts only the angles that are outside the cutoff point based on the lowest density in the angle distribution (for plotting).
    """
    normalized_angles = np.mod(angles, 360)
    
    kde = gaussian_kde(normalized_angles, bw_method='scott')
    angle_grid = np.linspace(0, 360, 100)
    density = kde(angle_grid)
    
    cutoff_angle = angle_grid[np.argmin(density)]
    
    adjusted_angles = np.copy(angles)
    
    # Angles greater than cutoff_angle should be shifted by -360°
    adjusted_angles[adjusted_angles > cutoff_angle] -= 360
    
    return adjusted_angles, cutoff_angle

### Convergence analysis functions

def kl_kde(samples_p, samples_q, bandwidth=0.2):
    """
    Kernel density estimation of Kullback-Leibler divergence D(p || q) = E_p[log(p(x)) - log(q(x))].
    """
    kde_p = KernelDensity(bandwidth=bandwidth).fit(samples_p)
    kde_q = KernelDensity(bandwidth=bandwidth).fit(samples_q)

    log_p = kde_p.score_samples(samples_p)
    log_q = kde_q.score_samples(samples_p)

    return np.mean(log_p - log_q)

def convergence_curve(dihedral_angles, variance_to_explain=0.9, min_samples=50, num_points=10):
    """
    angles: (n_frames, n_dihedrals) in radians
    variance_to_explain: fraction of variance to explain with PCA
    min_samples: minimum samples to start the convergence curve
    num_points: number of points in the convergence curve
    Returns times t and D[t].
    """
    # sin and cos transform to project angles onto a circle
    X = np.hstack([np.sin(dihedral_angles), np.cos(dihedral_angles)])
    # fit PCA on entire dataset first
    pca = PCA()
    pca.fit(X)
    # find the number of dimensions that explain the desired variance
    explained_var = np.cumsum(pca.explained_variance_ratio_)
    n_dimensions = np.argmax(explained_var >= variance_to_explain) + 1
    print(f"Number of PCs to explain {variance_to_explain*100}% variance: {n_dimensions}")
    Xp = pca.fit_transform(X)[:, :n_dimensions]
    # build the convergence curve
    D = []
    T = Xp.shape[0]
    times = np.geomspace(min_samples, T, num=num_points, dtype=int)
    for t in times:
        D.append(kl_kde(Xp[:t+1], Xp))
    return times, np.array(D)

### Trajectory extraction functions

def extract_hbond_data(simulation_dir, out_file):
    """
    Extract hydrogen bond data from the simulation directories and save it to a CSV file.

    If the trajectories are not present (as in the github repository), try to load from precomputed data.
    """
    if not os.path.exists(simulation_dir):
        print("Reading precomputed file")
        df = pd.read_csv(out_file)
        return df
    
    paths = glob.glob(simulation_dir+'/*')
    paths = [p for p in paths if os.path.isdir(p)]
    systems = [os.path.basename(f) for f in paths]

    # Use 15a as a reference for the parameters, as they don't change (much) between the different systems thanks to dash charges
    residues = ["Mol0", "Mol0", "ILE", "ILE"]
    indices = [0, 0, 75, 75]
    # Hardcoded selection for 15a atom names
    atom_names = ["C13x", "O6x", "H", "N"]
    params = get_parameters('md_input_files/complex/15a/model.top', [(res, ind, name) for res, ind, name in zip(residues, indices, atom_names)])

    # Get hydrogen bond desciptors (distance and angle) and calculate the energy
    hbond_energy = []
    theta = []
    d = []
    for p in paths:
        t = md.load(f"{p}/ensemble-0.dcd", top=f"{p}/reference.pdb")
        mol = Chem.MolFromMolFile(f'{p}/ligand.sdf', removeHs=False)
        acc_pos = t.top._atoms[get_scaffold_atoms(mol, SCAFFOLD)[2]].index
        acc_neg = t.top._atoms[get_scaffold_atoms(mol, SCAFFOLD)[27]].index
        hyd = t.top.select('resname ILE and resid 75 and name H')[0]
        don = t.top.select('resname ILE and resid 75 and name N')[0]
        distances_to_calculate = [[acc_pos, hyd], [acc_pos, don], [acc_neg, hyd], [acc_neg, don]]
        distances_flat = md.compute_distances(t, distances_to_calculate)
        distances = np.zeros((4,4,distances_flat.shape[0]))
        distances[0,2] = distances_flat[:,0]
        distances[0,3] = distances_flat[:,1]
        distances[1,2] = distances_flat[:,2]
        distances[1,3] = distances_flat[:,3]
        # we only need the upper triangle of the matrix
        coulomb, lj = get_hbond_energy(distances, params)
        hbond_energy.append(np.sum(coulomb+lj, axis=(0,1)))
        d.append(distances[1,2])
        angle_to_calculate = [[don, hyd, acc_neg]]
        t = md.compute_angles(t, angle_to_calculate).T[0]
        t = np.rad2deg(t)
        theta.append(t)
    df = pd.DataFrame({
        'system' : np.repeat(np.array(systems), len(d[0])),
        'frame' : np.tile(np.arange(len(d[0])), len(d)),
        'distance' : np.concatenate(d),
        'angle' : np.concatenate(theta),
        'energy' : np.concatenate(hbond_energy)
    })
    df.to_csv(out_file)
    return df

def extract_scaffold_dihedrals(simulation_dir, out_file):
    """
    Extract scaffold dihedrals and save them to a CSV file.

    If the trajectories are not present (as in the github repository), try to load from precomputed data.
    """
    if not os.path.exists(simulation_dir):
        print("Reading precomputed file")
        df = pd.read_csv(out_file)
        return df

    paths = glob.glob(simulation_dir+'/*')
    paths = [p for p in paths if os.path.isdir(p)]
    angles = np.array([get_scaffold_dihedrals(p) for p in paths], dtype=np.float16)
    systems = [os.path.basename(f) for f in paths]
    df = pd.DataFrame({
        'system': np.repeat(systems, angles.shape[1]),
        'frame': np.tile(np.arange(angles.shape[1]), len(systems)),
    })
    for i in range(angles.shape[2]):
        df[f'angle_{i}'] = np.concatenate([a[:,i] for a in angles])
    df.to_csv(out_file)
    return df

def extract_other_dihedrals(simulation_dir, out_file):
    """
    Extract all non-scaffold dihedrals and save them to a CSV file.
    Only extracts the dihedrals with linker length 5 (on which the main analysis is based).

    If the trajectories are not present (as in the github repository), try to load from precomputed data.
    """
    if not os.path.exists(simulation_dir):
        print("Reading precomputed file")
        df = pd.read_csv(out_file)
        return df

    paths = glob.glob(simulation_dir+'/*')
    paths = [p for p in paths if os.path.isdir(p)]
    angles = []
    systems = []
    for path in paths:
        mol = path+"/ligand.sdf"
        linker_length = len(get_linker_atoms(Chem.MolFromMolFile(mol))[0])-7
        if linker_length != 5 or "CBR-182" in path: # We only care about linker length 5 and para-linked
            continue
        angles.append(np.concatenate([get_linker_dihedrals(path), get_extra_dihedrals(path)], axis=1))
        systems.append(os.path.basename(path))
    angles = np.array(angles, dtype=np.float16)
    df = pd.DataFrame({
        'system': np.repeat(systems, angles.shape[1]),
        'frame': np.tile(np.arange(angles.shape[1]), len(systems)),
    })
    for i in range(angles.shape[2]):
        df[f'angle_{i}'] = np.concatenate([a[:,i] for a in angles])
    df.to_csv(out_file)
    return df

def extract_other_dihedrals_acyclic(simulation_dir, out_file):
    """
    Extract non-scaffold dihedrals of acyclic systems (there is only one) and save them to a CSV file.

    If the trajectories are not present (as in the github repository), try to load from precomputed data.
    """
    if not os.path.exists(simulation_dir):
        print("Reading precomputed file")
        df = pd.read_csv(out_file)
        return df

    paths = glob.glob(simulation_dir+'/*')
    paths = [p for p in paths if os.path.isdir(p)]
    angles = []
    systems = []
    for path in paths:
        mol = path+"/ligand.sdf"
        if get_linker_atoms(Chem.MolFromMolFile(mol))[0] is not None: # We only care about the acyclic systems
            continue
        angles.append(get_extra_dihedrals(path))
        systems.append(os.path.basename(path))
    angles = np.array(angles, dtype=np.float16)
    df = pd.DataFrame({
        'system': np.repeat(systems, angles.shape[1]),
        'frame': np.tile(np.arange(angles.shape[1]), len(systems)),
        'angle_0': np.squeeze(np.concatenate(angles)),
    })
    df.to_csv(out_file)
    return df