import mdtraj as mdj
import numpy as np
import time

def combine_dcds(systems, keep_water=False, n_interval=1):

    """This function combines dcds from different substeps of a particular 
    openmm run. Please ensure the substep dcds form a continuous run.
    Each substep*.dcd file contains number of frames printed into the 
    file as per the production input file. Check the submission script 
    and the input file to understand the how many substeps you have used.
    Also, this function by default removes all the waters (and ions) from 
    the system while building the trajectory. If you want to keep the waters
    and ions use 'keep_water = True'.


    Parameters
    ----------
    systems: Dictionary with keys of different type of systems/conditions that have been simulated.
    
        The system dictionary has to follow some certain rules.
        The keys of the dictionary must denote all the different type of systems
        (or conditions) that have been simulated for comparison and that you want to combine.
        The first value should be the string containing the path where one would get all topology/dcds.
        The second value should be the number of independent runs (replicates).
        The third value should be a list that contains how many substeps are there
        in each individual run of the systems.

        An example systems dictionary looks like this:

        systems = {'active': ['/dickson/s1/bosesami/CTR/MD_sims/active/charmm-gui-6790505779/openmm' , 5, [20,20,20,20,20]],
              'inactive': ['/dickson/s1/bosesami/CTR/MD_sims/inactive/charmm-gui-6791034540/openmm',5,[20,20,20,20,20]] }

        Here the two systems are active and inactive. The paths are the first entry in the value list. 
        The second entry is the number of runs. 
        The 3rd entry in the value list is a list that contains how many substeps are there in each 
        individual run of the systems. I have 20 here for all of them.

        VERY IMPORTANT: The dcd and topology file nomenclature must match the following string

        substep{i}_run{run}.dcd 
        step5_input.pdb

        where i: substep idx ranging from 1 to n_substep (in the example above n_substep=20)
            run: independent run idx ranging from 1 to (in the example above n_runs=5)

    Returns
    -------
     trj.save_pdb(): Builds a frame of pdb for each run.
     trj.save_dcd(): Builds the entire continuous trajetory for each run.

    TODO: Make the input dcd and topology files as string inputs so that people can use whatever naming convension they want.

    """
    
    #Check these variables in the individual folders
    systems = {'active': ['/dickson/s1/bosesami/CTR/MD_sims/active/charmm-gui-6790505779/openmm' , 5, [20,20,20,20,20]], 
              'inactive': ['/dickson/s1/bosesami/CTR/MD_sims/inactive/charmm-gui-6791034540/openmm',5,[20,20,20,20,20]] }

    ####################################
    for sys in systems.keys():
        assert len(systems[sys]) >= 3, "Must contain three items (i) the path to dcd/top, (ii) n_runs and (iii) list of substeps.. Check the docstring please."

    for sys in systems.keys():
        path = systems[sys][0]
        n_runs = systems[sys][1]

        
        for run in range(1, n_runs+1):
            t1 = time.time()
            n_steps = systems[sys][2][run-1]
            for i in range(1,n_steps+1):
                print(f'{sys}: Running for run and substep: {run},{i}')
        
                if i == 1:
                    trj1 = mdj.load(f'{path}/substep{i}_run{run}.dcd' ,top=f'{path}/step5_input.pdb')
                    
                    if n_interval > 1:
                        trj1 = trj1[::n_interval]
                    if keep_water == False:                   
                        trj1 = trj1.remove_solvent()
                
                else:    
                    trj2 = mdj.load(f'{path}/substep{i}_run{run}.dcd' ,top=f'{path}/step5_input.pdb')
                    
                    if n_interval > 1:
                        trj2 = trj2[::n_interval]
                    if keep_water == False:
                        trj2 = trj2.remove_solvent()
                    
                    trj1 = trj1.join([trj2])
            print(time.time()-t1)
        
            print(f'Final number of frames: run{run}', trj1.n_frames)

            if keep_water == False:
                string = 'nowater'
            if keep_water == True:
                string = 'withwater'

            trj1[0].save_pdb(f'{path}/{sys}_run{run}_frame0_{string}.pdb')
            trj1.save_dcd(f'{path}/{sys}_run{run}_{trj1.n_frames}frames_{string}.dcd')
