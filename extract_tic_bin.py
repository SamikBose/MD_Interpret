import sys
import pickle as pkl
import numpy as np
import mdtraj as mdj


def frames_in_2d_bin(tica_arr, bin_center, bin_width):
    """
    Return indices of frames whose 2D tICA coordinates fall within a rectangular bin.

    A frame i is selected if:
      |tica_arr[i, 0] - bin_center[0]| <= bin_width[0]
      |tica_arr[i, 1] - bin_center[1]| <= bin_width[1]

    Parameters
    ----------
    tica_arr : np.ndarray, shape (n_frames, 2)
        2D tICA coordinates per frame.
    bin_center : array-like of length 2
        Center of the bin [c0, c1].
    bin_width : array-like of length 2
        Half-widths in each dimension [w0, w1].

    Returns
    -------
    np.ndarray
        1D array of integer frame indices satisfying both conditions.
    """
    tica_arr = np.asarray(tica_arr)
    c = np.asarray(bin_center)
    w = np.asarray(bin_width)

    if tica_arr.ndim != 2 or tica_arr.shape[1] != 2:
        raise ValueError(f"tica_arr must have shape (n_frames, 2); got {tica_arr.shape}")

    mask = (np.abs(tica_arr[:, 0] - c[0]) <= w[0]) & (np.abs(tica_arr[:, 1] - c[1]) <= w[1])
    return np.where(mask)[0]

# Example:
# idx = frames_in_2d_bin(tica_arr, bin_center=[0.5, -0.6], bin_width=[0.1, 0.2])


if __name__ == '__main__':

    base_path= '/dickson/s1/bosesami/cbrXA/'
    analysis_path = f'{base_path}/analysis'
    tica_path = f'{analysis_path}/tica'
    #out_str = 'his_coord_feats_all_runs'
    # the main dictionary containing all the bookkeepping information
    sys_dict = {
               'Protonated': [f'{base_path}/cbrXA_HIS_POPC', 5, 'cbrXA_HIS_POPC', [1110,1111,900,900,900], 'charmm-gui-2451287232/openmm'],
               'Deprotonated': [f'{base_path}/cbrXA_HIS_Lys196H_POPC', 5, 'cbrXA_HIS_Lys196H_POPC', [1110,1110,900,900,900], 'charmm-gui-2451288367/openmm']
               }

    traj_out_path = f'{analysis_path}/FE_tica'

    #inp_string = 'coTICA_his_coord_based'
    inp_string = 'coTICA_Break_heavyCoordBased'
    #out_string = 'his_coord_coTICA'
    out_string = 'break_coord_coTICA'

    
    keys = sys.argv[1]
    bin_center = [float(sys.argv[2]), float(sys.argv[3])]
    bin_width = [float(sys.argv[4]), float(sys.argv[5])]
    
    tica_dict = pkl.load(open(f'{tica_path}/{inp_string}_2nTIC_100lag.pkl', 'rb'))

    folder = sys_dict[keys][0]
    n_runs = sys_dict[keys][1]
    string = sys_dict[keys][2]
    charmm_folder = sys_dict[keys][4]
    
    print(f'The system is: {keys}')
    
    extract_traj = []
    
    for run in tica_dict[keys].keys():  
        print(f'Running for {keys}, Run{run}...')
        last_frame = sys_dict[keys][3][run-1] 
        tica_arr = np.array(tica_dict[keys][run])
        idxs = frames_in_2d_bin(tica_arr, bin_center=bin_center, bin_width=bin_width)
        if len(idxs) > 0:
            print(f'Snapshots in the bin of {bin_center} with width {bin_width}: {len(idxs)}')
            if run <= 2:
                trj = mdj.load_dcd(f'{folder}/run{run}/{string}_run{run}_nowater_{last_frame}frames.dcd', top = f'{folder}/run{run}/{string}_run{run}_nowater_frame0.pdb')
            if run >=3:
                trj = mdj.load_dcd(f'{folder}/{charmm_folder}/{string}_run{run}_nowater_{last_frame}frames.dcd', top=f'{folder}/{charmm_folder}/{string}_run{run}_nowater_frame0.pdb')
            
            sliced = trj.slice(idxs)
            imp_atoms = sliced.topology.select('segname PROA PROB PROC')
            sliced_protein = sliced.atom_slice(imp_atoms)
            extract_traj.append(sliced_protein)


    if extract_traj:  # check it's not empty
        combined = mdj.join(extract_traj)
        combined.save_dcd(f'{traj_out_path}/{keys}_{out_string}_bincenters_{bin_center[0]}_{bin_center[1]}.dcd')
        combined[0].save_pdb(f'{traj_out_path}/{keys}_frame0_{out_string}_bincenters_{bin_center[0]}_{bin_center[1]}.pdb')
