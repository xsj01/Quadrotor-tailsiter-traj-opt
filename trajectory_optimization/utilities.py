# Contains helpful tools
import numpy as np


def save_traj(filename, traj, u_traj, time_array):
    """ Save the trajectory arrays into file
    :param filename: filename to save trajectory at
    :param traj: array of states
    :param u_traj: array of control inputs
    :param time_array: array of times
    """
    file = open(filename, 'w+')
    np.savez(file, traj, u_traj, time_array)
    file.close()


def load_traj(filename):
    """ Load the trajectories from file
    :param filename: filename to load trajectory from
    :return: traj, u_traj, time_array
    """
    infile = open(filename, 'r')
    npzfile = np.load(infile)

    traj = npzfile["arr_0"]
    u_traj = npzfile["arr_1"]
    time_array = npzfile["arr_2"]

    return traj, u_traj, time_array