import pickle as pkl
import mdtraj as mdj
import numpy as np
import time
import matplotlib.pyplot as plt
from geomm.grouping import group_pair
from geomm.superimpose import superimpose
from geomm.centering import center_around
from geomm.rmsd import calc_rmsd

colors = ['seagreen', 'tomato', 'dodgerblue', 'chocolate']
base_path= '/dickson/s1/bosesami/cbrXA/'
write_path = f'{base_path}/analysis/features/'

sys_dict = {
           'Protonated': [f'{base_path}/cbrXA_HIS_POPC', 5, 'cbrXA_HIS_POPC', [1110,1111,900,900,900], 'charmm-gui-2451287232/openmm'],
           'Deprotonated': [f'{base_path}/cbrXA_HIS_Lys196H_POPC', 5, 'cbrXA_HIS_Lys196H_POPC', [1110,1110,900,900,900], 'charmm-gui-2451288367/openmm']
           }

for keys in sys_dict.keys():
    
    features = {}
    counter = 0

    folder = sys_dict[keys][0]
    n_runs = sys_dict[keys][1]
    string = sys_dict[keys][2]
    charmm_folder = sys_dict[keys][4]

    print(f'Running for system: {keys}')
    for run in range(1,n_runs+1):
        last_frame = sys_dict[keys][3][run-1]

        if run <= 2:
            pdb = mdj.load_pdb(f'{folder}/run{run}/step5_input.pdb')
            trj_all_frames = mdj.load_dcd(f'{folder}/run{run}/{string}_run{run}_nowater_{last_frame}frames.dcd', top = f'{folder}/run{run}/{string}_run{run}_nowater_frame0.pdb')
        if run >=3:
            pdb = mdj.load_pdb(f'{folder}/{charmm_folder}/step5_input.pdb')
            trj_all_frames = mdj.load_dcd(f'{folder}/{charmm_folder}/{string}_run{run}_nowater_{last_frame}frames.dcd', top=f'{folder}/{charmm_folder}/{string}_run{run}_nowater_frame0.pdb')
        
        print(f'Trajectory loaded: run{run}...')

        # Getting the important indices:
        #bs_idxs = pdb.top.select('segname PROB and name CA and (residue 48 49 50 51 52 53 54 55 56 247 248 249 250 251 252 253 254 255 256 257 258 132 135 136 196 192 283 74 75 333 337 406 405 402)')
        
        #bs_idxs = pdb.top.select('segname PROB and name CA and (residue 48 49 50 51 52 53 54 55 56 252 253 254 255 256 257)')
        lig_idxs = pdb.top.select('segname PROC and name CA')
        
        idxs = pdb.top.select('segname PROB and backbone and (residue 49 50 51 52 53 252 253 254 255 196 192)') 
        
        pos = trj_all_frames.xyz
        unitcell_len = trj_all_frames.unitcell_lengths
        n_frames = trj_all_frames.n_frames
        
        features[run]  = np.zeros((n_frames, lig_idxs.shape[0]*idxs.shape[0]))


        print(f'Calculating features for run{run}...')
        for frame in range(n_frames):
            grouped_walker_pos = group_pair(pos[frame], unitcell_len[frame], idxs,lig_idxs)

            counter = 0

            # For Distance Features
            for lig_atom in lig_idxs: 
                for bs_atom in idxs:
                    features[run][frame,counter] = np.sqrt(np.sum(np.square(grouped_walker_pos[lig_atom] - grouped_walker_pos[bs_atom])))
                    counter += 1

        print(f'Run{run}: Done...')
    pkl.dump(features, open(f'{write_path}/Hist_dist_feat_{idxs.shape[0]*lig_idxs.shape[0]}dim_{keys}_all{n_runs}runs.pkl','wb'))
