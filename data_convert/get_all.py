import os
import numpy as np
all_file=[]
### Open and read in the txt file
for f in  os.listdir('/disk_pool/juno_data/dataset_txt/detsim/neutrino/txt2/'):
  #print("reading file {}".format(f))
  #filename = os.listdir('/disk1/liteng/work/JUNO/dataset_nu_elec_all/txt'):
  filename = "/disk_pool/juno_data/dataset_txt/detsim/neutrino/txt2/"+f
  all_file.append(filename)
np.save("all_file.npy",all_file)
