import os

def fileExtension(path):
    """

    :param path:
    :return:
    """
    filename,ext = os.path.split(path)
    return ext