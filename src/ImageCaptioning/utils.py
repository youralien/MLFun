import os
import numpy as np
import json
import glob

def vStackMatrices(x, new_x):
    return stackMatrices(x, new_x, np.vstack)

def hStackMatrices(x, new_x):
    return stackMatrices(x, new_x, np.hstack)

def colStackMatrices(x, new_x):
    return stackMatrices(x, new_x, np.column_stack)

def stackMatrices(x, new_x, fun):
    if x is None:
        x = new_x
    else: 
        x = fun((x, new_x))

    return x

def ensure_dir(f):
    """ Ensures the directory exists within the filesystem.
    If it does not currently exists, the directory is made

    Parameters
    ----------
    f: file which you might want to create
    """
    d = os.path.dirname(f)
    if not os.path.exists(d):
        os.makedirs(d)

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in xrange(0, len(l), n):
        yield l[i:i+n]

def json2dict(jsonpath):
    f = open(jsonpath, 'r')
    dictionary = json.load(f)
    f.close()
    return dictionary

def dict2json(dictionary, jsonpath):
    f = open(jsonpath, 'w')
    json.dump(dictionary, f)
    f.close()

def listdir_valid_extensions(pathname, valid_extensions):
    """Inspired by os.listdir, except it filters for a number of valid file extensions

    Parameters
    ----------
    pathname: string path

    valid_extensions: list of string extensions
        e.g. ```["jpg", "png"]```

    Returns
    -------
    filenames: a list of string filenames
    """
    pathlist = [os.path.join(pathname, "*.%s" % ext) for ext in valid_extensions]
    filelist_batches = [glob.glob(path) for path in pathlist]
    fns = [os.path.basename(full_fn) for filelist_batch in filelist_batches for full_fn in filelist_batch]
    return fns