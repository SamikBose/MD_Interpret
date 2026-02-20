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

    base_path = '/dickson/s1/bosesami/REVO_tica_attempts/clr_swing_in_out/distance_based'
    MD_data_path = f'/dickson/s1/bosesami/clr_ramp_work/standard_MD/'
    tica_path = f'{base_path}/tica'
    
    # the main dictionary containing all the bookkeepping information
    out_string = 'clrBackboneCB_rampCA_distance_cotica'
    inp_string = 'coTICA_clrBackboneCB_rampCA_distancefeat_allsystems'

    sys_dict = {
               'R1': [f'{MD_data_path}/R1/', 5, 2034, [2000,2000,2000,2000,2000], 'charmm-gui-5614072231/openmm', [1063,1064,1065,1066,1067]],
               'R3': [f'{MD_data_path}/R3/', 5, 88, [2000,2000,2000,2000,2000], 'charmm-gui-5614871995/openmm', [37,38,39,40,41]]
               }
    
    traj_out_path = f'{base_path}/FE_tica_snaps'

   
    ### Important bin centers: Get these from the corresponding free energy plots.
    keys = sys.argv[1]
    bin_center = [float(sys.argv[2]), float(sys.argv[3])]
    bin_width = [float(sys.argv[4]), float(sys.argv[5])]
    
    tica_dict = pkl.load(open(f'{tica_path}/{inp_string}_2nTIC_1lag.pkl', 'rb'))

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
            trj = mdj.load_dcd(f'{folder}/{charmm_folder}/run{run}.dcd', top = f'{folder}/{charmm_folder}/step3_input.pdb')

            sliced = trj.slice(idxs)
            imp_atoms = sliced.topology.select('segname PROA PROB')
            sliced_protein = sliced.atom_slice(imp_atoms)
            extract_traj.append(sliced_protein)


    if extract_traj:  # check it's not empty
        combined = mdj.join(extract_traj)
        combined.save_dcd(f'{traj_out_path}/{keys}_{out_string}_bincenters_{bin_center[0]}_{bin_center[1]}.dcd')
        combined[0].save_pdb(f'{traj_out_path}/{keys}_frame0_{out_string}_bincenters_{bin_center[0]}_{bin_center[1]}.pdb')
