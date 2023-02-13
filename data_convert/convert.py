import healpy as hp
import numpy as np
import sys
q=np.array([0.0483486, 0.0713992, 0.0714013, 0.0993545, 0.126781 , 0.154207 ,
       0.181633 , 0.209059 , 0.236485 , 0.263911 , 0.291338 , 0.318764 ,
       0.34214  , 0.369566 , 0.369567 , 0.393    , 0.420426 , 0.443926 ,
       0.471352 , 0.494919 , 0.522345 , 0.545912 , 0.569145 , 0.596571 ,
       0.620143 , 0.643427 , 0.670853 , 0.694466 , 0.717838 , 0.740967 ,
       0.768393 , 0.792037 , 0.815486 , 0.838739 , 0.866165 , 0.889815 ,
       0.913304 , 0.936635 , 0.95981  , 0.987237 , 1.01092  , 1.03448  ,
       1.05793  , 1.08125  , 1.10868  , 1.13238  , 1.156    , 1.17952  ,
       1.20297  , 1.22633  , 1.24962  , 1.27705  , 1.30076  , 1.32442  ,
       1.34804  , 1.3716   , 1.39514  , 1.41863  , 1.44209  , 1.46553  ,
       1.48894  , 1.51234  , 1.53573  , 1.55911  , 1.58249  , 1.60586  ,
       1.62925  , 1.65265  , 1.67606  , 1.6995   , 1.72296  , 1.74646  ,
       1.76999  , 1.79356  , 1.81717  , 1.84083  , 1.86454  , 1.89197  ,
       1.91526  , 1.93863  , 1.96207  , 1.9856   , 2.00921  , 2.03291  ,
       2.06034  , 2.08367  , 2.10711  , 2.13067  , 2.15436  , 2.18178  ,
       2.20496  , 2.22829  , 2.25178  , 2.27543  , 2.30285  , 2.32611  ,
       2.34956  , 2.3732   , 2.40063  , 2.42376  , 2.44713  , 2.47074  ,
       2.49817  , 2.52145  , 2.54502  , 2.57245  , 2.59568  , 2.61925  ,
       2.64667  , 2.67024  , 2.69767  , 2.72117  , 2.74859  , 2.77203  ,
       2.79945  , 2.82283  , 2.85026  , 2.87768  , 2.90511  , 2.93253  ,
       2.95996  , 2.98739  , 3.01481  , 3.04224  , 3.07019  , 3.09324  ])
### Create empty arrays to save all temp data
### These are 2D arrays for features. Shape: (number of events)*(number of pmts)
fht_p =np.zeros(126,240)
npe_p =np.zeros(126,240) 
mediantime_p =np.zeros(126,240)
slope_p=np.zeros(126,240)
### 2D array for direction label. Shape: (number of events)*2 (theta, phi)
y_p = []
N=240
graph =np.zeros(126,240)

### Open and read in the txt file
f = open(sys.argv[1])
fl = f.readlines()
i = 0
while True:
  if i>= len(fl): break
  ### For each event, create empty arrays to save data
  graph_fht =np.zeros(126,240)
  graph_npe =np.zeros(126,240)
  graph_mediantime =np.zeros(126,240)
  graph_slope=np.zeros(126,240)
  ### Each array is filled with an empty array. This is because
  ### each pixel may contain more than one PMT. Following code
  ### will merge them.
  ### First two lines are theta, phi and vertex
  event_theta = float(fl[i].split('\t')[0])
  event_phi = float(fl[i].split('\t')[1].strip())
  y_p.append(event_theta)
  i = i + 3
  ### Loop over all PMTs
  while True:
    ### Each event ends with a '####'. Break.
    if fl[i] == "####\n": 
      i = i + 1
      break
    ### Read in data for each PMT, and save them to arrays.
    this_pmt = fl[i].split('\t')
    nPE = float(this_pmt[0])
    fht = float(this_pmt[1])
    phi = float(this_pmt[2])
    theta = float(this_pmt[3])
    slope = float(this_pmt[4])
    mediantime = float(this_pmt[8])
    
    Neff=round(N*np.sin(theta))
    row_i = round(Neff*((np.pi/2-phi)/np.pi))+N/2
    col_i=np.where(ar==theta)[0][0]
    graph_fht[col_i][row_i]=fht
    graph_npe[col_i][row_i]=nPE
    graph_slope[col_i][row_i]=slope
    graph_mediantime[col_i][row_i]=mediantime
    

  fht_p =np.concatenate((fht_p,graph_fht),axis=0)
  npe_p =np.concatenate((npe_p,graph_npe),axis=0)
  mediantime_p =np.concatenate((mediantime_p,graph_mediantime),axis=0)
  slope_p =np.concatenate((slope_p,graph_slope),axis=0)

f.close()
del fl
np.save("fht.npy",fht_p)


