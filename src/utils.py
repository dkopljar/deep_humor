import os
import zipfile
import constants
import wget


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


def extract_zip(file, ext_dir):
    """
    Extracts the zip file to a chosen directroy
    :param file: Zip file path
    :param ext_dir: Extraction directory
    :return:
    """
    print("Extracting", file, "to", ext_dir)
    with zipfile.ZipFile(file, "r") as zip_ref:
        zip_ref.extractall(ext_dir)
    print("Extraction finished!\n")


def download_data(download_dir="/tmp"):
    """
    Downloads the dataset and resources required for the project.
    NOTE: You should have at least 5GB of disk space available.
    :return:
    """
    glove_dir = os.path.join(constants.RSC, "glove")
    tf_weights = os.path.join(constants.RSC, "tf_weights")

    dir_creator(
        [constants.DATA, constants.RSC, glove_dir, tf_weights])

    # Twitter glove vectors
    glove_name = os.path.join(download_dir, "glove.zip")
    if not os.path.exists(glove_name):
        print("Downloading Twitter Glove vector from", constants.GLOVE_TWITTER)
        print("This may take a while because the file size is 1.4GB")
        wget.download(constants.GLOVE_TWITTER, glove_name)
        print("Downloaded to", glove_name, "\n")
    extract_zip(glove_name, glove_dir)

    # Dataset
    print("Downloading dataset")
    for link, file in zip([constants.TRAIN, constants.VALIDATION,
                           constants.VALIDATION_NO_LABELS],
                          ["train", "valid", "eval"]):
        file = os.path.join(download_dir, file)
        if not os.path.exists(file):
            print("Downloading", link)
            wget.download(link, file)

        extract_zip(file, constants.DATA)

        # TODO Add trained weights download
