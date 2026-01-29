import matplotlib.pyplot as plt
import pickle as pkl 
import numpy as np


color_list = ['tomato', 'dodgerblue', 'seagreen']

sys_dict = {'active': ['/dickson/s1/bosesami/CTR/MD_sims/active/charmm-gui-6790505779/openmm' , 5, [2000,2000,2000,2000,2000]],
              'inactive': ['/dickson/s1/bosesami/CTR/MD_sims/inactive/charmm-gui-6791034540/openmm',5,[2000,2000,2000,2000,2000]] }


for i, system in enumerate(sys_dict.keys()):
    
    rmsd_arr = pkl.load(open(f'rmsd_files/{system}_TM6_backbone_rmsd_wrt_inac.pkl','rb'))
    
    avg = np.average(rmsd_arr, axis=0)
    std = np.std(rmsd_arr, axis=0)/np.sqrt(5)
    
    x_list = [i for i in range(2000)]
    plt.plot( x_list, avg, color = color_list[i], marker ='.',markersize=2, label=f'{system}')
    plt.fill_between(x=x_list, y1=np.subtract(avg, std), y2=np.add(avg, std), color=color_list[i], alpha=0.3)        
    
plt.legend()
plt.xlabel('Snapshots')
plt.ylabel('RMSD (\AA)')
plt.savefig(f'rmsd_files/both_systems_rmsd_wrt-inactive.png', dpi=600)
plt.close()
pkl.load(open(f'rmsd_files/{system}_TM6_backbone_rmsd_wrt_inac.pkl','rb'))

