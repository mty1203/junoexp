import healpy as hp
import numpy as np
import sys

nside = 32
npix = hp.nside2npix(nside)

### Create empty arrays to save all temp data
### These are 2D arrays for features. Shape: (number of events)*(number of pmts)
fht_p = []
npe_p = []
fht_spmt = []
npe_spmt = []
nperatio_p = []
nperatio4_p = []
peak_p = []
peaktime_p = []
slope_p = []
slope4_p = []
mediantime_p = []
meantime_p = []
stdtime_p = []
skewtime_p = []
kurtime_p = []
### 2D array for direction label. Shape: (number of events)*2 (theta, phi)
y_p = []

### Open and read in the txt file
print(sys.argv[1])
f = open(sys.argv[1])
fl = f.readlines()
i = 0
while True:
  if i>= len(fl): break

  ### For each event, create empty arrays to save data
  pix_hits_group_fht = []
  pix_hits_group_npe = []
  pix_hits_group_fht_spmt = []
  pix_hits_group_npe_spmt = []
  pix_hits_group_slope = []
  pix_hits_group_slope4 = []
  pix_hits_group_peak = []
  pix_hits_group_peaktime = []
  pix_hits_group_nperatio = []
  pix_hits_group_nperatio4 = []
  pix_hits_group_mediantime = []
  pix_hits_group_meantime = []
  pix_hits_group_stdtime = []
  pix_hits_group_skewtime = []
  pix_hits_group_kurtime = []

  ### Each array is filled with an empty array. This is because
  ### each pixel may contain more than one PMT. Following code
  ### will merge them.
  for j in range(npix):
    pix_hits_group_fht.append([])
    pix_hits_group_npe.append([])
    pix_hits_group_fht_spmt.append([])
    pix_hits_group_npe_spmt.append([])
    pix_hits_group_nperatio.append([])
    pix_hits_group_nperatio4.append([])
    pix_hits_group_peak.append([])
    pix_hits_group_peaktime.append([])
    pix_hits_group_slope.append([])
    pix_hits_group_slope4.append([])
    pix_hits_group_mediantime.append([])
    pix_hits_group_meantime.append([])
    pix_hits_group_stdtime.append([])
    pix_hits_group_skewtime.append([])
    pix_hits_group_kurtime.append([])

  ### Neutrino theta and phi
  event_theta = float(fl[i].split('\t')[0])
  event_phi = float(fl[i].split('\t')[1].strip())
  i = i + 1

  ### Lepton theta and phi
  lepton_theta = float(fl[i].split('\t')[0])
  lepton_phi = float(fl[i].split('\t')[1].strip())
  i = i + 1

  ### Visible theta and phi
  visible_theta = float(fl[i].split('\t')[0])
  visible_phi = float(fl[i].split('\t')[1].strip())
  i = i + 1

  ### PID
  pid = int(fl[i].split('\t')[0])
  cc = int(fl[i].split('\t')[1].strip())
  i = i + 1

  ### Energy
  energy = float(fl[i].split('\t')[0])
  energy_lepton = float(fl[i].split('\t')[1])
  energy_visible = float(fl[i].split('\t')[2].strip())
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
  

  ### Loop over all PMTs
  n_pmt = 0
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
    slope4 = float(this_pmt[5])
    peak = float(this_pmt[6])
    peaktime = float(this_pmt[7])
    nperatio = float(this_pmt[8])
    nperatio4 = float(this_pmt[9])
    mediantime = float(this_pmt[10])
    meantime = float(this_pmt[11])
    stdtime = float(this_pmt[12])
    skewtime = float(this_pmt[13])
    kurtime = float(this_pmt[14].strip())

    # Calculate the pixel number based on theta and phi
    this_pix = hp.pixelfunc.ang2pix(nside,theta,phi,nest=True)

    if n_pmt <=17611:
      # LPMT features
      pix_hits_group_fht[this_pix].append(fht)
      pix_hits_group_npe[this_pix].append(nPE)
      pix_hits_group_nperatio[this_pix].append(nperatio)
      pix_hits_group_nperatio4[this_pix].append(nperatio4)
      pix_hits_group_peak[this_pix].append(peak)
      pix_hits_group_peaktime[this_pix].append(peaktime)
      pix_hits_group_slope[this_pix].append(slope)
      pix_hits_group_slope4[this_pix].append(slope4)
      pix_hits_group_mediantime[this_pix].append(mediantime)
      pix_hits_group_meantime[this_pix].append(meantime)
      pix_hits_group_stdtime[this_pix].append(stdtime)
      pix_hits_group_skewtime[this_pix].append(skewtime)
      pix_hits_group_kurtime[this_pix].append(kurtime)
    else:
      # SPMT features
      pix_hits_group_fht_spmt[this_pix].append(fht)
      pix_hits_group_npe_spmt[this_pix].append(nPE)
    i = i + 1
    n_pmt = n_pmt + 1

  ### Merge each pixel
  for j in range(npix):
    min_fht = 1250
    min_fht_spmt = 1250
    sum_npe = 0
    sum_npe_spmt = 0
    sum_npe_ratio = 0.0
    sum_npe_ratio4 = 0.0
    sum_peak = 0.0
    sum_peaktime = 0.0
    sum_slope = 0.0
    sum_slope4 = 0.0
    sum_mediantime = 0.0
    sum_meantime = 0.0
    sum_stdtime = 0.0
    sum_skewtime = 0.0
    sum_kurtime = 0.0
    num_pmt = 0
    for k in range(len(pix_hits_group_fht[j])):
      if pix_hits_group_fht[j][k] < min_fht and pix_hits_group_fht[j][k] > 0: 
        # Find smaller fht. Negative fht is not valid and skipped.
        min_fht = pix_hits_group_fht[j][k]
      sum_npe = sum_npe + pix_hits_group_npe[j][k]
      sum_npe_ratio = sum_npe_ratio + pix_hits_group_nperatio[j][k]
      sum_npe_ratio4 = sum_npe_ratio4 + pix_hits_group_nperatio4[j][k]
      sum_peak = sum_peak + pix_hits_group_peak[j][k]
      sum_peaktime = sum_peaktime + pix_hits_group_peaktime[j][k]
      sum_slope = sum_slope + pix_hits_group_slope[j][k]
      sum_slope4 = sum_slope4 + pix_hits_group_slope4[j][k]
      sum_mediantime = sum_mediantime + pix_hits_group_mediantime[j][k]
      sum_meantime = sum_meantime + pix_hits_group_meantime[j][k]
      sum_stdtime = sum_stdtime + pix_hits_group_stdtime[j][k]
      sum_skewtime = sum_skewtime + pix_hits_group_skewtime[j][k]
      sum_kurtime = sum_kurtime + pix_hits_group_kurtime[j][k]
      num_pmt = num_pmt + 1

    for k in range(len(pix_hits_group_fht_spmt[j])):
      if pix_hits_group_fht_spmt[j][k] < min_fht and pix_hits_group_fht_spmt[j][k] > 0:
        # Find smaller fht. Negative fht is not valid and skipped.
        min_fht_spmt = pix_hits_group_fht_spmt[j][k]
      sum_npe_spmt = sum_npe_spmt + pix_hits_group_npe_spmt[j][k]

    if num_pmt > 0:
      pix_hits_group_nperatio[j] = sum_npe_ratio/num_pmt
      pix_hits_group_nperatio4[j] = sum_npe_ratio4/num_pmt
      pix_hits_group_peak[j] = sum_peak/num_pmt
      pix_hits_group_peaktime[j] = sum_peaktime/num_pmt
      pix_hits_group_slope[j] = sum_slope/num_pmt
      pix_hits_group_slope4[j] = sum_slope4/num_pmt
      pix_hits_group_mediantime[j] = sum_mediantime/num_pmt
      pix_hits_group_meantime[j] = sum_meantime/num_pmt
      pix_hits_group_stdtime[j] = sum_stdtime/num_pmt
      pix_hits_group_skewtime[j] = sum_skewtime/num_pmt
      pix_hits_group_kurtime[j] = sum_kurtime/num_pmt
    else:
      pix_hits_group_nperatio[j] = 0
      pix_hits_group_nperatio4[j] = 0
      pix_hits_group_peak[j] = 0
      pix_hits_group_peaktime[j] = 0
      pix_hits_group_slope[j] = 0
      pix_hits_group_slope4[j] = 0
      pix_hits_group_mediantime[j] = 0
      pix_hits_group_meantime[j] = 0
      pix_hits_group_stdtime[j] = 0
      pix_hits_group_skewtime[j] = 0
      pix_hits_group_kurtime[j] = 0
    pix_hits_group_fht[j] = min_fht
    pix_hits_group_npe[j] = sum_npe
    pix_hits_group_fht_spmt[j] = min_fht_spmt
    pix_hits_group_npe_spmt[j] = sum_npe_spmt

  ### Append merged data to the 2D array.
  fht_p.append(pix_hits_group_fht)
  npe_p.append(pix_hits_group_npe)
  fht_spmt.append(pix_hits_group_fht_spmt)
  npe_spmt.append(pix_hits_group_npe_spmt)
  nperatio_p.append(pix_hits_group_nperatio)
  nperatio4_p.append(pix_hits_group_nperatio4)
  peak_p.append(pix_hits_group_peak)
  peaktime_p.append(pix_hits_group_peaktime)
  slope_p.append(pix_hits_group_slope)
  slope4_p.append(pix_hits_group_slope4)
  mediantime_p.append(pix_hits_group_mediantime)
  meantime_p.append(pix_hits_group_meantime)
  stdtime_p.append(pix_hits_group_stdtime)
  skewtime_p.append(pix_hits_group_skewtime)
  kurtime_p.append(pix_hits_group_kurtime)
  y_p.append([event_theta,event_phi,lepton_theta,lepton_phi,visible_theta,visible_phi,pid,cc,energy,energy_lepton,energy_visible,vertex_x,vertex_y,vertex_z,exit_x,exit_y,exit_z])

f.close()
del fl

### Save all 2D arrays to CSV files
with open("ElecNu_FHT.csv", "a") as f:
  np.savetxt(f,np.array(fht_p), delimiter=",")
  del fht_p

with open("ElecNu_NPE.csv", "a") as f:
  np.savetxt(f,np.array(npe_p), delimiter=",")
  del npe_p

with open("ElecNu_FHT_SPMT.csv", "a") as f:
  np.savetxt(f,np.array(fht_spmt), delimiter=",")
  del fht_spmt

with open("ElecNu_NPE_SPMT.csv", "a") as f:
  np.savetxt(f,np.array(npe_spmt), delimiter=",")
  del npe_spmt

with open("ElecNu_NPERATIO.csv", "a") as f:
  np.savetxt(f,np.array(nperatio_p), delimiter=",")
  del nperatio_p

with open("ElecNu_NPERATIO4.csv", "a") as f:
  np.savetxt(f,np.array(nperatio4_p), delimiter=",")
  del nperatio4_p

with open("ElecNu_PEAK.csv", "a") as f:
  np.savetxt(f,np.array(peak_p), delimiter=",")
  del peak_p

with open("ElecNu_PEAKTIME.csv", "a") as f:
  np.savetxt(f,np.array(peaktime_p), delimiter=",")
  del peaktime_p

with open("ElecNu_SLOPE.csv", "a") as f:
  np.savetxt(f,np.array(slope_p), delimiter=",")
  del slope_p

with open("ElecNu_SLOPE4.csv", "a") as f:
  np.savetxt(f,np.array(slope4_p), delimiter=",")
  del slope4_p

with open("ElecNu_SKEWTIME.csv", "a") as f:
  np.savetxt(f,np.array(skewtime_p), delimiter=",")
  del skewtime_p

with open("ElecNu_KURTIME.csv", "a") as f:
  np.savetxt(f,np.array(kurtime_p), delimiter=",")
  del kurtime_p

with open("ElecNu_MEDIANTIME.csv", "a") as f:
  np.savetxt(f,np.array(mediantime_p), delimiter=",")
  del mediantime_p

with open("ElecNu_MEANTIME.csv", "a") as f:
  np.savetxt(f,np.array(meantime_p), delimiter=",")
  del meantime_p

with open("ElecNu_RMSTIME.csv", "a") as f:
  np.savetxt(f,np.array(stdtime_p), delimiter=",")
  del stdtime_p

with open("ElecNu_Y.csv", "a") as f:
  np.savetxt(f,np.array(y_p), delimiter=",")
  del y_p
