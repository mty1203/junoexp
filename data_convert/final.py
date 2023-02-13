import numpy as np
from multiprocessing import Pool
import warnings
import os
import sys
from tqdm import tqdm
import re
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
n=np.array([  7,   7,   7,  20,  28,  35,  41,  45,  53,  59,  65,  51,  71,
         14,  68,  82,  60,  90, 104, 104, 114, 114,  84, 128, 128, 128,
        142, 112, 142, 142, 159, 159, 159, 129, 174, 174, 174, 174, 174,
        157, 191, 191, 191, 191, 205, 175, 205, 205, 205, 205, 205, 183,
        219, 219, 219, 219, 219, 189, 219, 219, 219, 219, 219, 195, 195,
        219, 219, 219, 219, 219, 189, 219, 219, 219, 219, 219, 183, 205,
        205, 205, 205, 205, 175, 205, 191, 191, 191, 191, 158, 174, 174,
        174, 174, 174, 129, 159, 159, 159, 142, 142, 114, 142, 128, 128,
        128,  84, 114, 114, 104, 104,  90,  60,  82,  82,  71,  52,  65,
         59,  53,  45,  41,  35,  28,  20,  14,   7])
### Create empty arrays to save all temp data
### These are 2D arrays for features. Shape: (number of events)*(number of pmts)
N=228
rows=224


def process(list):
  label2=[]
  fht2 =np.zeros((rows,N))
  npe2 =np.zeros((rows,N)) 
  peaktime2 =np.zeros((rows,N))
  slope2=np.zeros((rows,N))
  fht2 =fht2[np.newaxis,:,:]
  npe2 =npe2[np.newaxis,:,:]
  peaktime2 =peaktime2[np.newaxis,:,:]
  slope2 =slope2[np.newaxis,:,:]
  y2=np.zeros((rows,N))
  y2=y2[np.newaxis,:,:]
  marker=0
  for filename in list:
    f = open(filename)
    print(filename)
    #number = filename.format("output_{}.txt")
    name= re.findall(r"\d+\.?\d*",filename)
    print(name)
    number = eval(name[1])
    #print(number)
    fl = f.readlines()
    i = 0
    while True:
      if i>= len(fl):
        break
    ### For each event, create empty arrays to save data
      label=[]
      y_p=np.zeros((rows,N))
      graph_fht =np.zeros((rows,N))
      graph_npe =np.zeros((rows,N))
      graph_peaktime =np.zeros((rows,N))
      graph_slope=np.zeros((rows,N))
      graph_fht[0][0]=number+marker
  ### Each array is filled with an empty array. This is because
  ### each pixel may contain more than one PMT. Following code
  ### will merge them.
  ### First two lines are theta, phi and vertex
      event_theta = float(fl[i].split('\t')[0])
      event_phi = float(fl[i].split('\t')[1].strip())
      i = i + 1
  ### Lepton theta and phi
      lepton_theta = float(fl[i].split('\t')[0])
      lepton_phi = float(fl[i].split('\t')[1].strip())
      i = i + 1

  ### PID
      pid = int(fl[i].split('\t')[0])
      cc = int(fl[i].split('\t')[1].strip())
      i = i + 1

  ### Energy
      energy = float(fl[i].split('\t')[0])
      i = i + 1

  ### Vertex
      vertex_x = float(fl[i].split('\t')[0])
      vertex_y = float(fl[i].split('\t')[1])
      vertex_z = float(fl[i].split('\t')[2].strip())
      i = i + 1

  ### Exit point
      exit_x = float(fl[i].split('\t')[0])
      exit_y = float(fl[i].split('\t')[1])
      exit_z = float(fl[i].split('\t')[2].strip())
      i = i + 1
      y_p[rows-1][0]=event_theta
      y_p[rows-1][1]=event_phi
      y_p[rows-1][2]=lepton_theta
      y_p[rows-1][3]=lepton_phi
      y_p[rows-1][4]=pid
      y_p[rows-1][5]=cc
      y_p[rows-1][6]=energy
      y_p[rows-1][7]=vertex_x
      y_p[rows-1][8]=vertex_y
      y_p[rows-1][9]=vertex_z
      y_p[rows-1][10]=exit_x
      y_p[rows-1][11]=exit_y
      y_p[rows-1][12]=exit_z
      y_p[rows-1][13]=number+marker
  ### Loop over all PMTs
      n_pmt=0
      while True:
    ### Each event ends with a '####'. Break.
        if fl[i] == "####\n": 
          marker=marker+1
          i = i + 1
          break
    ### Read in data for each PMT, and save them to arrays.
        if n_pmt <=17611:
          this_pmt = fl[i].split('\t')
          nPE = float(this_pmt[0])
          fht = float(this_pmt[1])
          phi = float(this_pmt[2])
          theta = float(this_pmt[3])
          slope = float(this_pmt[4])
          peaktime = float(this_pmt[6])
     # print("{} {} ***************".format(theta,phi)) 

          #Neff=round(238*np.sin(theta))
          #row_i = round(Neff/2*(phi/np.pi)+N/2)
          #row_i = int(row_i)
          row_i=round((phi+np.pi)/(2*np.pi)*226)
      #print("{}".format(row_i))
          col_i=round(theta/np.pi*224)
     # print("{}".format(col_i))
         # col_i=np.where(q==theta)[0][0]
          graph_fht[col_i][row_i]=fht
          graph_npe[col_i][row_i]=nPE
          graph_slope[col_i][row_i]=slope
          graph_peaktime[col_i][row_i]=peaktime
          i=i+1
          n_pmt=n_pmt+1
        else:
          i=i+1
      graph_fht =graph_fht[np.newaxis,:,:]
      graph_npe =graph_npe[np.newaxis,:,:]
      graph_peaktime =graph_peaktime[np.newaxis,:,:]
      graph_slope =graph_slope[np.newaxis,:,:]
      y_p=y_p[np.newaxis,:,:]
      fht2 =np.concatenate((fht2,graph_fht),axis=0)
      npe2 =np.concatenate((npe2,graph_npe),axis=0)
      peaktime2 =np.concatenate((peaktime2,graph_peaktime),axis=0)
      slope2 =np.concatenate((slope2,graph_slope),axis=0)
      y2 =np.concatenate((y2,y_p),axis=0)
    f.close()
    del fl
  
  return np.stack((fht2[1:,:,:,np.newaxis],npe2[1:,:,:,np.newaxis],peaktime2[1:,:,:,np.newaxis],slope2[1:,:,:,np.newaxis],y2[1:,:,:,np.newaxis]),axis=3)



### 2D array for direction label. Shape: (number of events)*2 (theta, phi)
all_file=[]
### Open and read in the txt file
for f in  os.listdir('/disk_pool/juno_data/dataset_txt/detsim/neutrino/txt2/'):
  #print("reading file {}".format(f))
  #filename = os.listdir('/disk1/liteng/work/JUNO/dataset_nu_elec_all/txt'):
  filename = "/disk_pool/juno_data/dataset_txt/detsim/neutrino/txt2/"+f
  all_file.append(filename)
  if(len(all_file)==10):
    break
#file_train= all_file[0:7001]
#file_test = all_file[7001:]
chunkNum=10
splitedfilename = np.array_split(all_file, chunkNum)
with warnings.catch_warnings():
  warnings.simplefilter("ignore")
  with Pool(40) as pool:
    res= np.concatenate(
                list(
                    tqdm(
                        pool.imap(
                            process,
                            splitedfilename
                        ),
                        total=chunkNum
                    )
                ),
                axis=0
            )

fht_f,npe_f,peaktime_f,slope_f,y_f= res[:,:,:,0,0],res[:,:,:,1,0],res[:,:,:,2,0],res[:,:,:,3,0],res[:,rows-1,0:14,4,0]
print(np.count_nonzero(fht_f[0,:,:]))
np.save("fht_t.npy",fht_f)
np.save("npe_t.npy",npe_f)
np.save("peaktime_t.npy",peaktime_f)
np.save("slope_t.npy",slope_f)
np.save("y_t.npy",y_f)
chunkNum=10

