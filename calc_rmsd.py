import pickle as pkl
import mdtraj as mdj
import numpy as np
import time
import sys
import matplotlib.pyplot as plt
from feature_extraction import ref_centered_pose, grouped_centered_frames, aligned_frames, rmsd_all_frames

if __name__ == '__main__':


    # some important paths for input and output
    base_path= '/dickson/s1/bosesami/CTR/'
    rmsd_path = f'{base_path}/analyze_sf_MD/rmsd_files/'
    
    out_str = 'TM6_backbone_rmsd_wrt_inac'

    # the main dictionary containing all the bookkeepping information
    sys_dict = {'active': ['/dickson/s1/bosesami/CTR/MD_sims/active/charmm-gui-6790505779/openmm' , 5, [2000,2000,2000,2000,2000]],
              'inactive': ['/dickson/s1/bosesami/CTR/MD_sims/inactive/charmm-gui-6791034540/openmm',5,[2000,2000,2000,2000,2000]] }


    #load a reference for both systems, all frames
    pdb_system = sys.argv[1]
    ref_pdb_path = f'{sys_dict[pdb_system][0]}/step5_input_nowater.pdb'
    ref_pdb = mdj.load_pdb(ref_pdb_path)
    pdb_pos = ref_pdb.xyz[0]
    alignment_atoms = ref_pdb.top.select('(segname PROB and backbone) or (segname PROA and backbone and not resid 295 to 327)')
        
    reference_centered_pose = ref_centered_pose(pos = pdb_pos, alignment_idxs = alignment_atoms, frame_idx = 0)


    # main loop (over each of the systems, here protonated and deprotonated)
    for keys in sys_dict.keys():
        
        # the main output 
        rmsd_arr = np.zeros((5,2000))

        # some variable definition based on the way data is stored!
        folder = sys_dict[keys][0]
        n_runs = sys_dict[keys][1]
        
        # defining the standard topology
        pdb_path = f'{folder}/step5_input_nowater.pdb'
        pdb = mdj.load_pdb(pdb_path)
        pdb_pos = pdb.xyz[0]

        # some key indices being defined here
        # PROA: CTR, PROB: RAMP,
        alignment_atoms =  pdb.top.select('(segname PROB and backbone) or (segname PROA and backbone and not resid 295 to 327)')
        important_atoms = pdb.top.select('segname PROA and resid 295 to 327 and backbone')
        group_pair_idx1 = alignment_atoms 
        group_pair_idx2 = important_atoms

        # Getting the reference for superimposing later (Without water all 5 runs have the same topology)
        #reference_centered_pose = ref_centered_pose(pos = pdb_pos, alignment_idxs = alignment_atoms, frame_idx = 0)

        print(f'Running for system: {keys}')

        for run in range(1,n_runs+1):
            
            # specially required for this work, since I stored the md sim data with the final frame number in the filename
            last_frame = sys_dict[keys][2][run-1]
            trj_all_frames =  mdj.load_dcd(f'{folder}/{keys}_run{run}_{last_frame}frames_nowater.dcd',  top = f'{folder}/step5_input_nowater.pdb')

            print(f'Trajectory loaded: run{run}...')
            
            # use mdtraj to get all the positions and unitcell lengths and number of frames
            pos = trj_all_frames.xyz
            unitcell_len = trj_all_frames.unitcell_lengths
            n_frames = trj_all_frames.n_frames

            # get the rmsd of the entire trajectory w.r.t a particular frame.
            # the function below first does group pairing, then centering, then superimposing and then rmsd calculation...
            rmsd_list = rmsd_all_frames(coords = pos, 
                                            ref_coords = reference_centered_pose, 
                                            unitcell_length = unitcell_len, 
                                            alignment_idxs = alignment_atoms, 
                                            pair_idx1 = group_pair_idx1, 
                                            pair_idx2 = group_pair_idx2,
                                            important_idxs= important_atoms)
            
            rmsd_arr[run-1] = np.array(rmsd_list)

            print(f'Run{run}: Done...')
        pkl.dump(rmsd_arr, open(f'{rmsd_path}/{keys}_{out_str}.pkl','wb'))
