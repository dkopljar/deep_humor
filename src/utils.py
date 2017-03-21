import os
import constants


def dir_creator(dirs_list):
    """
    Creates directories if they don't exist.
    :param dirs_list: List of absolute directory paths
    :return:
    """
    for d in dirs_list:
        if not os.path.exists(d):
            os.makedirs(d)
            print("Created directory", d)


if __name__ == "__main__":
    dir_list = [constants.DATA, constants.RSC]
    dir_creator(dir_list)
