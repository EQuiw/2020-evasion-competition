import lief
import numpy as np
from os import listdir
import os
from os.path import isfile, join

checkpath = "/<PATH-TO-DIR>/pe-malware/"

onlyfiles = np.sort([f for f in listdir(checkpath) if isfile(join(checkpath, f))])

nopefile = []

for i, curfile in enumerate(onlyfiles):
    file_data = open(os.path.join(checkpath, curfile), "rb").read()
    try:
        lief_binary = lief.PE.parse(list(file_data))
    except Exception:  # everything else (KeyboardInterrupt, SystemExit, ValueError):
        print("No Pe file:", i, curfile)
        nopefile.append(i)



for i in nopefile:
    print("I remove:", i, onlyfiles[i])
    os.remove(os.path.join(checkpath, onlyfiles[i]))

