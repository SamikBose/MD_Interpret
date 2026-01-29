import mdtraj as mdj
import numpy as np
import time
from geomm.grouping import group_pair
from geomm.superimpose import superimpose
from geomm.rmsd import calc_rmsd


def ref_centered_pose(pos, alignment_idxs = None, frame_idx = None):
    """For a snapshot, this centers the snapshot around a
    set of provided atom indices. 
    This function is often handy to create a refernce, centered 
    position (coordinates), which is essential for alignment of 
    the MD frames using geomm.superimpose function. 
    
    Parameters
    ----------

    pos : arraylike
        The position (coordinates) array of the snapshot that you 
        will be using as reference afterwards.

    alignment_idxs : arraylike 
        Collection of the indices which will be used to center the 
        snapshot.

    Returns
    -------

    ref_centered_pos : arraylike
        Centered reference frame coordinates to be used in superimpose 
        or other geomm based functionalities.

    """    
    
    ref_pos = pos

    if alignment_idxs is not None:
        ref_centroid = np.average(ref_pos[alignment_idxs], axis =0) 
    else:
        alignment_idxs = pdb.top.select('protein and not element H')
        ref_centroid = np.average(ref_pos[alignment_idxs], axis =0)

    ref_centered_pos = ref_pos - ref_centroid

    return ref_centered_pos


def grouped_centered_frames(coords, unitcell_length, alignment_idxs, pair_idx1, pair_idx2):
    

    """For a set of coordinates/frames from a trajectory this first moves pair_idx2 
    coordinates to the image of the periodic unitcell that minimizes the difference between 
    the centers of geometry between the pair_idx2 and pair_idx1 (e.g. a protein and ligand).
    Then, centers the set of coordinates around a set of alingment atom indices.
    
    This uses the geomm group pair function, followed by centering of grouped snapshots. 
    Hence, it carries out the identical transformations in geomm, 
    but instead of a single snapshot, it does it for the entire set of snapshots in a trajectory.

    Parameters
    ----------

    coords : arraylike
        The coordinates array of all frames of the particles you will be transforming.
        (group pairing followed by centering)

    unitcell_length : arraylike 
        The lengths of the sides of a rectangular unitcell.

    alignment_idxs : arraylike 
        Collection of the indices which will be used to center the snapshot.

    pair_idx1 : arraylike 
        Collection of the indices that define that member of the pair.

    pair_idxs2 : arraylike
        Collection of the indices that define that member of the pair.

    Returns
    -------

    all_frames_grouped : list of arrays
        A list containing the transformed coordinates from all the frames.


    """
    assert coords.shape[2] == 3, "coordinates are not of 3 dimensions"
    
    n_frames = coords.shape[0]
    
    all_frames_grouped  = [] 
    for frame in range(n_frames):
        
        grouped_pos = group_pair(coords[frame], unitcell_len[frame], pair_idx1, pair_idx2)
        
        centroid = np.average(grouped_pos[alignment_idxs], axis =0)
        
        grouped_centered_pos = grouped_pos - centroid

        all_frames_grouped.append(grouped_centered_pos)

    return all_frames_grouped

def aligned_frames(coords, ref_coords,unitcell_length, alignment_idxs, pair_idx1,pair_idx2):

    """For a set of coordinates/frames from a trajectory this function does a bunch of operations
    systematically:
    (i) First moves pair_idx2 coordinates to the image of the periodic unitcell that minimizes the 
    difference between the centers of geometry between the pair_idx2 and pair_idx1 (e.g. a protein and ligand).
    (ii) Then, centers the set of coordinates around a set of alingment atom indices.
    (iii) Finally superimpose all the frames on top of a reference frame. (alignment)


    This uses the geomm group pair function, followed by centering of grouped snapshots and then again
    geomm superimpose function. It 
    Instead of a single snapshot, it does the alignment for the entire set of snapshots in a trajectory.

    Parameters
    ----------

    coords : arraylike
        The coordinates array of all frames of the particles you will be transforming.
        (group pairing followed by centering followed by superimpose/alignment)

    ref_coords: arraylike
        Output of the ref_centered_pose function
        Centered reference frame coordinates to be used in superimpose

    unitcell_length : arraylike 
        The lengths of the sides of a rectangular unitcell.

    alignment_idxs : arraylike 
        Collection of the indices which will be used to center the snapshot
        and align them against the ref pose.
        Please ensure that the same indices are used while centering the reference pose too.

    pair_idx1 : arraylike 
        Collection of the indices that define that member of the pair.

    pair_idxs2 : arraylike
        Collection of the indices that define that member of the pair.

    Returns
    -------

    all_frames_aligned : list of arrays
        A list containing the transformed coordinates from all the frames.
        Transformation: group pairing followed by centering followed by superimpose/alignment.

    """



    assert coords.shape[2] == 3, "coordinates are not of 3 dimensions"
    assert coords.shape[1] == ref_coords.shape[0], "Number of atoms does not match between reference and provided coords"
    n_frames = coords.shape[0]

    all_frames_aligned = []
    for frame in range(n_frames):
        
        grouped_pos = group_pair(coords[frame], unitcell_length[frame], pair_idx1, pair_idx2)
        
        #print(grouped_pos.shape, ref_pos.shape)
        centroid = np.average(grouped_pos[alignment_idxs], axis =0)
        grouped_centered_pos = grouped_pos - centroid
        
        superimposed_pos, rotation_matrix , _ = superimpose(ref_coords,grouped_centered_pos, idxs=alignment_idxs)

        all_frames_aligned.append(superimposed_pos)
    
    return all_frames_aligned


def rmsd_all_frames(coords, ref_coords,unitcell_length, alignment_idxs, pair_idx1,pair_idx2 , important_idxs):

    """For a set of coordinates/frames from a trajectory this function does a bunch of operations
    systematically:
    (i) First moves pair_idx2 coordinates to the image of the periodic unitcell that minimizes the
    difference between the centers of geometry between the pair_idx2 and pair_idx1 (e.g. a protein and ligand).
    (ii) Then, centers the set of coordinates around a set of alingment atom indices.
    (iii) Then, superimpose all the frames on top of a reference frame. (alignment)
    (iv) Finally, compute rmsd of the important region

    This uses the geomm group pair function, followed by centering of grouped snapshots and then again
    geomm superimpose function and calc_rmsd function.
    Instead of a single snapshot, it does the alignment for the entire set of snapshots in a trajectory.

    Parameters
    ----------

    coords : arraylike
        The coordinates array of all frames of the particles you will be transforming.
        (group pairing followed by centering followed by superimpose/alignment)

    ref_coords: arraylike
        Output of the ref_centered_pose function
        Centered reference frame coordinates to be used in superimpose

    unitcell_length : arraylike
        The lengths of the sides of a rectangular unitcell.

    alignment_idxs : arraylike
        Collection of the indices which will be used to center the snapshot
        and align them against the ref pose.
        Please ensure that the same indices are used while centering the reference pose too.

    pair_idx1 : arraylike
        Collection of the indices that define that member of the pair.

    pair_idxs2 : arraylike
        Collection of the indices that define that member of the pair.
    
    important_idxs: arraylike
        Collection of indices to be used for the RMSD calculation.

    Returns
    -------

    rmsd_list : list 
        A list containing the RMSD (in \AA) of the important_idxs atom from all the frames.

    """



    assert coords.shape[2] == 3, "coordinates are not of 3 dimensions"
    assert coords.shape[1] == ref_coords.shape[0], "Number of atoms does not match between reference and provided coords"
    n_frames = coords.shape[0]

    rmsd_list = []
    for frame in range(n_frames):

        grouped_pos = group_pair(coords[frame], unitcell_length[frame], pair_idx1, pair_idx2)

        #print(grouped_pos.shape, ref_pos.shape)
        centroid = np.average(grouped_pos[alignment_idxs], axis =0)
        grouped_centered_pos = grouped_pos - centroid

        superimposed_pos, rotation_matrix , _ = superimpose(ref_coords,grouped_centered_pos, idxs=alignment_idxs)
        rmsd = calc_rmsd(ref_coords, superimposed_pos, idxs=important_idxs)*10 
        rmsd_list.append(rmsd)

    return rmsd_list

