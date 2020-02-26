"""This module's sole purpose is to get the path of this module and the root folder directory in order to be able to work with the project structure
author: rohue"""
from os.path import dirname, abspath, split

def getPath():
    return dirname(abspath(__file__))
    
def getRootPath():
    return split(dirname(abspath(__file__)))[0]
