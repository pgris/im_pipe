import os


def checkDir(outDir):
    """
    function to check whether a directory exist
    and create it if necessary
    """
    if not os.path.isdir(outDir):
        os.makedirs(outDir)
