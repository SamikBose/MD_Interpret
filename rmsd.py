import pickle as pkl
import mdtraj as mdj
import numpy as np
import time
import matplotlib.pyplot as plt
from geomm.grouping import group_pair
from geomm.superimpose import superimpose
from geomm.centering import center_around
from geomm.rmsd import calc_rmsd

colors = ['seagreen', 'magenta']
base_path='/dickson/s1/bosesami/cbrXA/'

sys_dict = {
           'prot': [f'{base_path}/cbrXA_HIS_POPC', 2, 'cbrXA_HIS_POPC'],
           'deprot': [f'{base_path}/cbrXA_HIS_Lys196H_POPC', 2, 'cbrXA_HIS_Lys196H_POPC']
           }

for keys in sys_dict.keys():

    folder = sys_dict[keys][0]
    n_runs = sys_dict[keys][1]
    string = sys_dict[keys][2]

    print(f'Running for system: {keys}')
    for run in range(1,n_runs+1):
        print(f'Running for run: {run}...')
        pdb = mdj.load_pdb(f'{folder}/run{run}/{string}_run{run}_nowater_frame0.pdb')
        lig_idxs = pdb.top.select('resname HIS and segname PROC and not element H')
        protein_idxs = pdb.top.select('segname PROB PROA and not element H')
        bs_idxs = mdj.compute_neighbors(pdb, 1.0, lig_idxs, haystack_indices=protein_idxs, periodic=True)[0]

        trj_all_frames = mdj.load_dcd(f'{folder}/run{run}/{string}_run{run}_nowater_allframes.dcd', top = f'{folder}/run{run}/{string}_run{run}_nowater_frame0.pdb')
        pos = trj_all_frames.xyz
        unitcell_len = trj_all_frames.unitcell_lengths
        
        ref_pos = pos[0]
        ref_centroid = np.average(ref_pos[bs_idxs], axis =0)
        ref_centered_pos = ref_pos - ref_centroid
        n_frames = trj_all_frames.n_frames
        
        rmsd_arr = np.zeros(n_frames)
        for frame in range(n_frames):
            grouped_pos_nowater = group_pair(pos[frame], unitcell_len[frame], bs_idxs, lig_idxs)

            centroid = np.average(grouped_pos_nowater[bs_idxs],axis=0)
            grouped_pos_nowater_cent = grouped_pos_nowater - centroid

            superimposed_pos, rotmat , qcp = superimpose(ref_centered_pos,grouped_pos_nowater_cent, idxs=bs_idxs)

            rmsd = calc_rmsd(ref_centered_pos, superimposed_pos, idxs=lig_idxs)*10
            rmsd_arr[frame] = rmsd
            #print(rmsd_arr[500], rmsd_arr[5000], rmsd_arr[10000])
        plt.plot(rmsd_arr, label = f'Run{run}', color=colors[run-1])
    

    x_ticks_loc = [i for i in range(0,n_frames,2000)]
    xticks = [0.1*i for i in range(0,n_frames,2000)] 

    plt.legend()
    plt.xticks(ticks=x_ticks_loc, labels=xticks)
    plt.xlabel('Time (ns)')
    plt.ylabel('Histidine RMSD (Angs)')
    plt.savefig(f'{keys}_HIS_rmsd_{n_runs}runs_bs_idx_as_cent.pdf', dpi =300)
    plt.close()

