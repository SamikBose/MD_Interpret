import pickle as pkl
import mdtraj as mdj
import numpy as np
import time
import matplotlib.pyplot as plt
from geomm.grouping import group_pair
from geomm.superimpose import superimpose
from geomm.centering import center_around
from geomm.rmsd import calc_rmsd

def concat_atom_idxs(residue_array, pdb, label, segname):

    out = []
    for i, r in enumerate(residue_array):
        if label == "sidechain_heavy":
            idxs = pdb.topology.select(f"segname {segname} and residue {r} and sidechain and not element H")
            out.extend(idxs)
            
        elif label == "CA":
            idxs = pdb.topology.select(f"segname {segname} and residue {r} and name CA")
            out.extend(idxs)
                    

        elif label == "backCB":
            names = ['N', 'CA', 'C', 'O', 'CB']  # Gly will lack CB
            for nm in names:
                idx = pdb.topology.select(f"segname {segname} and residue {r} and name {nm}")
                if idx.size:
                    out.append(idx[0])

        elif label == "heavy":
            idxs = pdb.topology.select(f"segname {segname} and residue {r} and not element H")
            out.extend(idxs)

        elif label == 'heavywramp':
            if i < len(residue_array) - 1:  # all but last residue → PROB
                idxs = pdb.topology.select(f"segname {segname} and residue {r} and not element H")
            else:  # last residue → PROA
                idxs = pdb.topology.select(f"segname {segname} and residue {r} and not element H")
            out.extend(idxs)

    return np.asarray(out, dtype=int)



if __name__ == '__main__':

    base_path= '/dickson/s1/bosesami/clr_ramp_work/'
    MD_data_path = f'{base_path}/standard_MD/'
    write_path = f'/dickson/s1/bosesami/REVO_tica_attempts/clr_swing_in_out/distance_based/distance_features/'

    sys_dict = {
               'R1': [f'{MD_data_path}/R1/', 5, 2034, [2000,2000,2000,2000,2000], 'charmm-gui-5614072231/openmm', [1063,1064,1065,1066,1067]],
               'R3': [f'{MD_data_path}/R3/', 5, 88, [2000,2000,2000,2000,2000], 'charmm-gui-5614871995/openmm', [37,38,39,40,41]]
               }

    for keys in sys_dict.keys():
        
        leu_idx = sys_dict[keys][2]
        leu_region = np.arange(leu_idx-1, leu_idx+2)

        ramp_idx = np.array(sys_dict[keys][5])

        features = {}
        folder = sys_dict[keys][0]
        n_runs = sys_dict[keys][1]
        charmm_folder = sys_dict[keys][4]

        # Without water all 5 runs have the same topology
        # Getting the important indices
        pdb = mdj.load_pdb(f'{folder}/{charmm_folder}/step3_input.pdb')
        
        leu_region_idxs = concat_atom_idxs(residue_array=leu_region, pdb=pdb, label="backCB",segname='PROB')
        ramp_region_idxs = concat_atom_idxs(residue_array=ramp_idx, pdb=pdb, label="CA", segname='PROA')

        print(leu_region_idxs.shape, ramp_region_idxs.shape)

        print(f'Running for system: {keys}')
        for run in range(1,n_runs+1):

            trj_all_frames = mdj.load_dcd(f'{folder}/{charmm_folder}/run{run}.dcd', top = f'{folder}/{charmm_folder}/step3_input.pdb')

            print(f'Trajectory loaded: run{run}...')
            pos = trj_all_frames.xyz
            unitcell_len = trj_all_frames.unitcell_lengths
            n_frames = trj_all_frames.n_frames

            features[run]  = np.zeros((n_frames, leu_region_idxs.shape[0]*ramp_region_idxs.shape[0]))

            print(f'Calculating features for run{run}...')
            for frame in range(n_frames):
                grouped_walker_pos = group_pair(pos[frame], unitcell_len[frame], leu_region_idxs, ramp_region_idxs)
                # For Distance Features
                for counter, clr_atom in enumerate(leu_region_idxs): 
                    for ramp_atom in ramp_region_idxs:
                        features[run][frame,counter] = np.sqrt(np.sum(np.square(grouped_walker_pos[clr_atom] - grouped_walker_pos[ramp_atom])))

            print(f'Run{run}: Done...')
        pkl.dump(features, open(f'{write_path}/{keys}_clrBackboneCB_rampCA_distances_{leu_region_idxs.shape[0]*ramp_region_idxs.shape[0]}dim.pkl','wb'))
