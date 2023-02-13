import os
import numpy as np
import pandas as pd
for f in os.listdir('./dataset_nu_elec_all_32/'):
  filename = "./dataset_nu_elec_all_32/" + f
  data = pd.read_csv(filename, dtype=np.float,delimiter=',')
  np.save("./dataset_nu_elec_all_32/"+f, data)





