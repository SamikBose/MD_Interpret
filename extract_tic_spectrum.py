import sys
import pickle as pkl
import numpy as np
import mdtraj as mdj


if __name__ == '__main__':

    base_path= '/dickson/s1/bosesami/CTR/'
    tica_path = f'{base_path}/analyze_sf_MD/tica/'
    traj_out_path = f'{base_path}/analyze_sf_MD/FE_tica'

    inp_string = 'coTICA_TM6_coord_feats_all_systems'
    out_string = 'TM6'

    sys_dict = {'active': ['/dickson/s1/bosesami/CTR/MD_sims/active/charmm-gui-6790505779/openmm' , 5, [2000,2000,2000,2000,2000]],
                  'inactive': ['/dickson/s1/bosesami/CTR/MD_sims/inactive/charmm-gui-6791034540/openmm',5,[2000,2000,2000,2000,2000]] }


    low_cut = float(sys.argv[1])
    high_cut = float(sys.argv[2])
    ntic = int(sys.argv[3])

    tica_dict = pkl.load(open(f'{tica_path}/{inp_string}_2nTIC_100lag.pkl', 'rb'))

    for keys in tica_dict.keys():

        folder = sys_dict[keys][0]
        n_runs = sys_dict[keys][1]

        print(f'The system is: {keys}')
        
        highval_traj = []
        lowval_traj = []
        
        for run in tica_dict[keys].keys():  
            print(f'Running for Run{run} and tic {ntic}...')
            last_frame = sys_dict[keys][2][run-1] 
            tica_arr = np.array(tica_dict[keys][run])[:,ntic]
            
            high_idxs = np.where(tica_arr > high_cut)[0]
            if len(high_idxs) > 0:
                print(f'Snapshots in high_val spectrum: {len(high_idxs)}')
                trj =  mdj.load_dcd(f'{folder}/{keys}_run{run}_{last_frame}frames_nowater.dcd',  top = f'{folder}/step5_input_nowater.pdb')
                print(f'Trajectory loaded: run{run}...') 
                
                sliced = trj.slice(high_idxs)
                imp_atoms = sliced.topology.select('not water')
                sliced_protein = sliced.atom_slice(imp_atoms)
                highval_traj.append(sliced_protein)
                    
            low_idxs = np.where(tica_arr < low_cut)[0]
            if len(low_idxs) > 0:
                print(f'Snapshots in low_val spectrum: {len(low_idxs)}')
                trj =  mdj.load_dcd(f'{folder}/{keys}_run{run}_{last_frame}frames_nowater.dcd',  top = f'{folder}/step5_input_nowater.pdb')
                print(f'Trajectory loaded: run{run}...')

                sliced_2 = trj.slice(low_idxs)
                imp_atoms = sliced_2.topology.select('not water')
                sliced_protein_2 = sliced_2.atom_slice(imp_atoms)
                lowval_traj.append(sliced_protein_2)

        if highval_traj:  # check it's not empty
            combined = mdj.join(highval_traj)
            combined.save_dcd(f'{traj_out_path}/tic{ntic}_{out_string}_highcut{high_cut}_{keys}.dcd')
            combined[0].save_pdb(f'{traj_out_path}/ref_frame_{out_string}_high_tic{ntic}_{keys}.pdb')

        if lowval_traj:  # check it's not empty
            combined = mdj.join(lowval_traj)
            combined.save_dcd(f'{traj_out_path}/tic{ntic}_{out_string}_lowcut{low_cut}_{keys}.dcd')
            combined[0].save_pdb(f'{traj_out_path}/ref_frame_{out_string}_low_tic{ntic}_{keys}.pdb')

