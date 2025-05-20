import pickle as pkl
import mdtraj as mdj
import numpy as np
import time
import matplotlib.pyplot as plt
from geomm.grouping import group_pair
from geomm.superimpose import superimpose
from geomm.centering import center_around
from geomm.rmsd import calc_rmsd


auto_time_list = [10,50,100,200,300,400,500,750,1000,1250,1500,1750,2000,2250,2500]
x_ticks_loc = [i for i in auto_time_list]
xticks = [int(0.1*i) for i in auto_time_list]


colors = ['seagreen', 'tomato', 'dodgerblue', 'peru']
base_path='/dickson/s1/bosesami/cbrXA/'

sys_dict = {
           'prot': [f'{base_path}/cbrXA_HIS_POPC', 2, 'cbrXA_HIS_POPC'],
           'deprot': [f'{base_path}/cbrXA_HIS_Lys196H_POPC', 2, 'cbrXA_HIS_Lys196H_POPC']
           }

counter = 0
for keys in sys_dict.keys():

    folder = sys_dict[keys][0]
    n_runs = sys_dict[keys][1]
    string = sys_dict[keys][2]

    all_run_acf =  np.zeros((n_runs, len(auto_time_list)))
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
        superimp_arr = np.zeros((n_frames, lig_idxs.shape[0], 3))
        for frame in range(n_frames):
            grouped_pos_nowater = group_pair(pos[frame], unitcell_len[frame], bs_idxs, lig_idxs)

            centroid = np.average(grouped_pos_nowater[bs_idxs],axis=0)
            grouped_pos_nowater_cent = grouped_pos_nowater - centroid

            superimposed_pos, rotmat , qcp = superimpose(ref_centered_pos,grouped_pos_nowater_cent, idxs=bs_idxs)
            superimp_arr[frame] = superimposed_pos[lig_idxs]

        mean_superimp = np.mean(superimp_arr, axis=0)
        
        acf_list = []
        corr_zero = np.zeros((lig_idxs.shape[0], 3))
        for frame in range(n_frames):
            frame2 = frame
            yi = superimp_arr[frame] - mean_superimp
            corr_zero += yi*yi
        corr_zero = corr_zero/n_frames

        for auto_time in auto_time_list:
            t1 = time.time()
            print(f'Running for auto correlation lag-time: {auto_time} steps')
            corr = np.zeros((lig_idxs.shape[0], 3))
            for frame in range(n_frames-auto_time):
                frame2 = frame + auto_time
                yi = superimp_arr[frame] - mean_superimp
                yj = superimp_arr[frame2] - mean_superimp
                corr += yi*yj

            corr = corr/(n_frames-auto_time)
            # take a mean
            #acf = np.mean((corr/corr_zero), axis)
            acf_list.append(corr/corr_zero)
            t2  = time.time()
            print(t2 - t1)
        acf = np.mean(acf_list, axis=(1,2))
        all_run_acf[run-1] = acf
    
    avg_acf = np.mean(all_run_acf, axis=0)
    std_err = np.std(all_run_acf, axis=0)/np.sqrt(n_runs)
    print(avg_acf)

    plt.plot(auto_time_list ,avg_acf, color = colors[counter],marker='o',label=f'{keys}')
    plt.fill_between(x=auto_time_list, y1=np.subtract(avg_acf, std_err), y2=np.add(avg_acf, std_err), color=colors[counter], alpha=0.2)
    counter += 1


plt.legend()
plt.xlabel('Lag times (ns)')
plt.xticks(ticks=x_ticks_loc, labels=xticks, rotation=60, size='small')
plt.ylabel('Auto correlation function')
plt.title(f'Auto correlation of HIS metabolite')
plt.savefig(f'2sys_ACF_{run}runs_maxlag_{auto_time_list[-1]}.pdf', dpi=300)
plt.close()

    #x_ticks_loc = [i for i in range(0,n_frames,2000)]
    #xticks = [0.1*i for i in range(0,n_frames,2000)] 
    #
    #plt.legend()
    #plt.xticks(ticks=x_ticks_loc, labels=xticks)
    #plt.xlabel('Time (ns)')
    #plt.ylabel('Histidine RMSD (Angs)')
    #plt.savefig(f'{keys}_HIS_rmsd_{n_runs}runs_bs_idx_as_cent.pdf', dpi =300)
    #plt.close()

