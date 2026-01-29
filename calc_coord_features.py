import pickle as pkl
import mdtraj as mdj
import numpy as np
import time
import matplotlib.pyplot as plt
from geomm.grouping import group_pair
from geomm.superimpose import superimpose
from geomm.centering import center_around
from geomm.rmsd import calc_rmsd
from feature_extraction import ref_centered_pose, grouped_centered_frames, aligned_frames

if __name__ == '__main__':


    # some important paths for input and output
    base_path= '/dickson/s1/bosesami/CTR/'
    write_path = f'{base_path}/analyze_sf_MD/features/'
    
    out_str = 'TM6_coord_feats_all_runs'

    # the main dictionary containing all the bookkeepping information
    sys_dict = {'active': ['/dickson/s1/bosesami/CTR/MD_sims/active/charmm-gui-6790505779/openmm' , 5, [2000,2000,2000,2000,2000]],
              'inactive': ['/dickson/s1/bosesami/CTR/MD_sims/inactive/charmm-gui-6791034540/openmm',5,[2000,2000,2000,2000,2000]] }


    # main loop (over each of the systems, here protonated and deprotonated)
    for keys in sys_dict.keys():
        
        # the main output dictionary
        features = {}

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
        reference_centered_pose = ref_centered_pose(pos = pdb_pos, alignment_idxs = alignment_atoms, frame_idx = 0)

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

            # initialize the feature array
            # since this is a coordinate feature so we use 3 (for x,y and z coords) as the final dimension value
            features[run]  = np.zeros((n_frames, important_atoms.shape[0], 3))
            
            # aling the frames with respect to the backbone atoms.
            # the function below first does group pairing, then centering and then superimposing...
            all_pos_aligned = aligned_frames(coords = pos, 
                                            ref_coords = reference_centered_pose, 
                                            unitcell_length = unitcell_len, 
                                            alignment_idxs = alignment_atoms, 
                                            pair_idx1 = group_pair_idx1, 
                                            pair_idx2 = group_pair_idx2)
            
            # store the features
            for frame in range(n_frames):
                features[run][frame] =  all_pos_aligned[frame][important_atoms]

            print(f'Run{run}: Done...')
        pkl.dump(features, open(f'{write_path}/{keys}_{out_str}_{important_atoms.shape[0]}dim.pkl','wb'))
