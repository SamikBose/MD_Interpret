from deeptime.decomposition import TICA
from deeptime.util.data import TimeLaggedDataset
from deeptime.clustering import KMeans
import mdtraj as mdj
import numpy as np
import pickle as pkl
import sys
import time
import sys


def get_init_final_frame_index(feat_dict):
    system_index = {}
    count = 0

    # init the dict as list
    for keys in feat_dict.keys():
        system_index[keys] = []

    # load the list with initial and final frame index
    for keys in feat_dict.keys():
        system_index[keys].append(count)
        for runs in feat_dict[keys].keys():
            count += len(feat_dict[keys][runs])
        system_index[keys].append(count)
    return(system_index)




tau_list = [1,2,5,10,20,50,100]
n_tica_dim_list = [2,5]

base_path='/dickson/s1/bosesami/cbrXA/'
analysis_path = f'{base_path}/analysis'
tica_path = f'{analysis_path}/tica'

systems = {
           'Protonated': [f'{base_path}/cbrXA_HIS_POPC', 5, 'cbrXA_HIS_POPC'],
           'Deprotonated': [f'{base_path}/cbrXA_HIS_Lys196H_POPC', 5, 'cbrXA_HIS_Lys196H_POPC']
           }

if int(sys.argv[1]) == 1 or int(sys.argv[1]) == 3:
    n_reshape = int(sys.argv[1])
    
else:
    print('Feature type argument must be provided:')
    print('For distance features use argument 1')
    print('For coordinate features use argument 3')


for tau in tau_list:
    for n_tica_dim in n_tica_dim_list:

        print(f'Running for lag, n_tic: {tau}, {n_tica_dim}')
        ### Building the FULL time lagged dataset
        t0_data = []
        t1_data = []

        for keys in systems.keys():
            n_runs = systems[keys][1]
            filename = f'{analysis_path}/features/Hist_dist_feat_44dim_{keys}_all5runs.pkl' 
            feat_dict = pkl.load(open(filename, 'rb'))
            for runs in feat_dict.keys():
                run_feat_arr = feat_dict[runs]
                data_length = run_feat_arr.shape[0] - tau
                #print(f'System{keys}, run{runs}, data length: {data_length}')
                n_features = run_feat_arr.shape[1]
                for i in range(data_length):
                    t0_data.append(run_feat_arr[i])
                    t1_data.append(run_feat_arr[i+tau])

        reshaped_t0 = np.reshape(t0_data, (len(t0_data),n_features*n_reshape))
        reshaped_t1 = np.reshape(t1_data, (len(t1_data),n_features*n_reshape))
        big_tld = TimeLaggedDataset(reshaped_t0, reshaped_t1)

        ### Building the tica estimator object
        t1= time.time()
        print("Start: Training TICA..")
        estimator = TICA(dim=n_tica_dim, lagtime=tau).fit(big_tld).fetch_model()
        pkl.dump(estimator, open(f'{tica_path}/ticaModel_His_DistanceBased_{n_tica_dim}nTIC_{tau}lag.pkl','wb'))

        
        ## Store the tica model
        if n_reshape == 1:
            pkl.dump(estimator, open(f'{tica_path}/ticaModel_His_DistanceBased_{n_tica_dim}nTIC_{tau}lag.pkl','wb'))
        if n_reshape == 3:
            pkl.dump(estimator, open(f'{tica_path}/ticaModel_Break_CoordBased_{n_tica_dim}nTIC_{tau}lag.pkl','wb'))


        ### Finally get the components: system and runs as dictionary keys.
        tica_feats = {}
        for keys in systems.keys():
            tica_feats[keys] = {}
            filename = f'{analysis_path}/features/Hist_dist_feat_44dim_{keys}_all5runs.pkl'
            feat_dict = pkl.load(open(filename, 'rb'))
            for runs in feat_dict.keys():
                print(f'System:{keys}, run{runs}...')
                run_feat_arr = feat_dict[runs].reshape(feat_dict[runs].shape[0], n_features*n_reshape)
                tica_feats[keys][runs] = [estimator.transform(item) for item in run_feat_arr]
        t2 = time.time()

        print(f"TICA featurization with n_TIC {n_tica_dim}, time taken: {t2 -t1} seconds...")

        if n_reshape == 1:
            pkl.dump(tica_feats, open(f'{tica_path}/coTICA_His_DistanceBased_{n_tica_dim}nTIC_{tau}lag.pkl','wb'))
        if n_reshape == 3:
            pkl.dump(tica_feats, open(f'{tica_path}/coTICA_Break_CoordBased_{n_tica_dim}nTIC_{tau}lag.pkl','wb'))

