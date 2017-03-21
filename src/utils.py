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
    print("Extracting", file, "to", ext_dir)
    with zipfile.ZipFile(file, "r") as zip_ref:
        zip_ref.extractall(ext_dir)
    print("Extraction finished!\n")

    # Cleanup
    try:
        os.remove(file)
    except Exception as e:
        print("Could not delete file", file, e)


def download_data():
    """
    Downloads datasets and resources required for the project
    :return:
    """
    dir_creator([constants.DATA, constants.RSC])

    # glove
    glove_name = os.path.join(constants.RSC, "glove.zip")
    if not os.path.exists(glove_name):
        print("Downloading Twitter Glove vector from", constants.GLOVE_TWITTER)
        print("This may take a while because the file size is 1.4GB")
        wget.download(constants.GLOVE_TWITTER, glove_name)
        print("Downloaded to", glove_name, "\n")

    if os.path.emp
    extract_zip(glove_name, constants.RSC)

    # Dataset
    print("Downloading dataset")
    for link, file in zip([constants.TRAIN, constants.VALIDATION,
                           constants.VALIDATION_NO_LABELS],
                          ["train", "valid", "eval"]):
        file = os.path.join(constants.DATA, file)
        if not os.path.exists(file):
            print("Downloading", link)
            wget.download(link, file)

        extract_zip(file, constants.DATA)
