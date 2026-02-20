import numpy as np
import pickle as pkl
import sys
import os.path as osp
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import pyplot as plt

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


def closest_value(input_list, input_value):


    arr = np.asarray(input_list)
    i = (np.abs(arr - input_value)).argmin()
    print(input_value, i, np.abs(arr - input_value))
    return arr[i]



def min_max_axes(tic_all, Ntic1, Ntic2):

    
    x_max = []
    x_min = []
    y_max = []
    y_min = []

    for system in tic_all.keys():
        tic1 = []
        tic2 = []

        for run in tic_all[system].keys():
            n_frames = len(tic_all[system][run])
            for frame in range(n_frames):
                tic1.append(tic_all[system][run][frame][Ntic1])
                tic2.append(tic_all[system][run][frame][Ntic2])

        x_max.append(np.max(tic1))
        y_max.append(np.max(tic2))

        x_min.append(np.min(tic1))
        y_min.append(np.min(tic2))


        print(f'From {system}:')
        print(f'TIC{Ntic1} min and max: {np.min(tic1)}, {np.max(tic1)}')
        print(f'TIC{Ntic2} min and max: {np.min(tic2)}, {np.max(tic2)}')

    xmax_val = np.max(x_max)
    xmin_val = np.min(x_min)
    ymax_val = np.max(y_max)
    ymin_val = np.min(y_min)

    print(f'Min and max of axes with all systems: ({xmin_val}, {xmax_val}), ({ymin_val},{ymax_val}) ')

    return [[xmin_val, xmax_val],[ymin_val, ymax_val]]

def get_fe_max_val(info, num_bin, num_system, scale_up_fe):

    min_fe_list = []
    num_system = num_system
    num_bin = num_bin
    
    fe = np.zeros((num_system, num_bin, num_bin))
    
    for counter, sys in enumerate(info.keys()):
        hist = info[sys]
        #print(np.max(hist), np.min(hist))
        mask = hist > 0
        mask2 = hist == 0
        #print(counter, sys)
        fe[counter] = -np.log10(hist, where=mask)
        fe[counter][mask2] = scale_up_fe
        #print(fe[counter][mask].min())
        #print(fe[counter][mask2].max())
        min_fe_list.append(fe[counter][mask].min())

    min_fe = np.min(min_fe_list)
    fe -= min_fe
    max_val = fe.max()
    for i in range(num_system):
        assert fe[i].max() == max_val, "Each system should have the same FE max"

    return fe, max_val


def get_clean_ticks(bins1, bins2, xtick_stride, ytick_stride):

    x_ticks = [i for i in bin1]
    y_ticks = [i for i in bin2]


    xtick_max = np.ceil(np.max(x_ticks))
    xtick_min = np.floor(np.min(x_ticks))
    ytick_max = np.ceil(np.max(y_ticks))
    ytick_min = np.floor(np.min(y_ticks))
    
    
    
    x_ticks_mod = np.arange(start=xtick_min, stop=xtick_max, step=xtick_stride)
    y_ticks_mod = np.arange(start=ytick_min, stop=ytick_max, step=ytick_stride)

    x_ticks_mod = x_ticks_mod[1:]
    y_ticks_mod = y_ticks_mod[1:]

    ticklabels_list = [x_ticks_mod.astype(int), y_ticks_mod.astype(int)]

    pos_xticks = []
    pos_yticks = []
    for i in x_ticks_mod:
        val = closest_value(x_ticks, i)
        pos_xticks.append(x_ticks.index(val))
    
    for i in y_ticks_mod:
        val = closest_value(y_ticks, i)
        pos_yticks.append(y_ticks.index(val))
    
    tickpositions_list = [pos_xticks, pos_yticks]

    return  ticklabels_list , tickpositions_list


if __name__ == '__main__':

    base_path= '/dickson/s1/bosesami/REVO_tica_attempts/clr_swing_in_out/distance_based'
    tica_path = f'{base_path}/tica/'
    fig_out_path = f'{base_path}/tica/' 

    inp_string = 'coTICA_clrBackboneCB_rampCA_distancefeat_allsystems'
    out_string = 'clr_ramp_distancebased_ticaFE'

    num_bin = int(sys.argv[1])
    tau = int(sys.argv[2])
    ntica = int(sys.argv[3])
    
    Ntic1 = 0 
    Ntic2 = 1
    num_system = 2
    scale_up_fe = 0.1 # test this number

    # load the data
    tic_all = pkl.load(open(f'{tica_path}/{inp_string}_{ntica}nTIC_{tau}lag.pkl','rb'))

    # gives you a range for computing the histograms after considering all the systems.
    axes_ranges = min_max_axes(tic_all, Ntic1, Ntic2)
    
    hist_dict = {}

    for system in tic_all.keys():
        tic1 = []
        tic2 = []

        for run in tic_all[system].keys():
            n_frames = len(tic_all[system][run])
        
            for frame in range(n_frames):
                tic1.append(tic_all[system][run][frame][Ntic1])
                tic2.append(tic_all[system][run][frame][Ntic2])


        hist, bins1, bins2 = np.histogram2d(tic1, tic2, bins = num_bin, density=False, range=axes_ranges)

        hist_dict[system] = hist

    # get the free energy as an array of n_system X num_bin X num_bin
    # and the max_val for the heatmap
    fe_arr , max_val = get_fe_max_val(hist_dict, num_bin, num_system, scale_up_fe)

    # since the ranges and number of bins are same in all systems 
    # we can use the bins from the last system iteration
    bin1_c = 0.5*(bins1[1:] + bins1[:-1])
    bin2_c = 0.5*(bins2[1:] + bins2[:-1])
    
    # rounding off to two decimal places
    bin1 = np.round(bin1_c, decimals=2)
    bin2 = np.round(bin2_c, decimals=2)

    # get the ticklabels and positions for a clean plotting
    ticklabels_list , tickpositions_list = get_clean_ticks(bins1, bins2, xtick_stride=1, ytick_stride=2)


    for i,system in enumerate(tic_all.keys()):
        
        
        fe = fe_arr[i]
        
        fig, ax = plt.subplots(figsize=(10,10))
        
        # this is something I need to have a look. Why do we make it transpose? There is a reason but I forgot!
        heatmap = ax.pcolor(fe.T, cmap=plt.cm.magma, vmin=0, vmax =max_val)

        # correct way to set tick labels and positions 
        ax.set_xticks(tickpositions_list[0] , labels= ticklabels_list[0], minor=False,fontsize=30)
        ax.set_yticks(tickpositions_list[1] , labels= ticklabels_list[1], minor=False, fontsize=30)
        ax.set_xlabel('TIC-0',fontsize=30)
        ax.set_ylabel('TIC-1',fontsize=30)
        
        # need to divide the space to plot the colorbar pad
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.15)
        plt.colorbar(heatmap, cax=cax)

        # saving etc...
        #plt.savefig(f'{fig_out_path}/free_en_bins{num_bin}_TIC{Ntic1}vsTIC{Ntic2}_{system}.pdf')
        plt.savefig(f'{fig_out_path}/{system}_{out_string}_{num_bin}bins_{tau}_{ntica}tics_magma.png', dpi=300)
        plt.close()
