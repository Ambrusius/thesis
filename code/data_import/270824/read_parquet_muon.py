import numpy as np
import pandas as pd
import uproot
import pyarrow.parquet as pq
import awkward_pandas as akpd
import itertools
import mplhep as hep
hep.style.use([hep.style.ATLAS])
import matplotlib.pyplot as plt
import time



if True:
    path = '/groups/hep/kinch/H_Zg/samples_processed/HZmmg_ggF_MC_reduced_11sep_157var.parquet'
else:
    path = 'c:/Users/Jens/Documents/Github/Thesis/samples_processed/HZeeg_ggF_MC_reduced_28august_131var.parquet'

sample = pd.read_parquet(path) 
print(sample.shape)



sample_small = sample.head(100)
# print(sample_small.shape)

headers = sample.columns
print("Headers:")
for header in headers:
    # print(header)
    print(f'{header}, {sample[header].iloc[113]}, {type(sample[header].iloc[113])}')
    # first_index_shape = sample[header].iloc[0].shape if isinstance(sample[header].iloc[0], (pd.Series, np.ndarray)) else None
    #print("Shape of the first index:", first_index_shape)



def InDet_Track_finder(row):
    num_muons = len(row["mu_truthOrigin"])  # Number of muons instead of electrons
    dR_min = np.ones(num_muons) * 999  # Initialize minimum delta R values
    Track_index = np.ones(num_muons, dtype=int) * -1  # Initialize track indices
    
    for i_mu in range(num_muons):
        NTracks = len(row["InDetTrack_theta"])  # Number of tracks from detector
        for i_Track in range(NTracks):
            eta_Track = -np.log(np.tan(row['InDetTrack_theta'][i_Track] / 2))
            deta = row['mu_eta'][i_mu] - eta_Track
            dphi = row['mu_phi'][i_mu] - row['InDetTrack_phi'][i_Track]
            if dphi > np.pi: 
                dphi -= 2 * np.pi  # Adjust phi to be in the range [-pi, pi]
            elif dphi < -np.pi: 
                dphi += 2 * np.pi  # Adjust phi to be in the range [-pi, pi]
            dR_muT = np.sqrt(deta**2 + dphi**2)  # Calculate delta R
            
            if dR_muT < dR_min[i_mu]:  # Update minimum delta R and track index
                dR_min[i_mu] = dR_muT
                Track_index[i_mu] = i_Track
    
    return Track_index, dR_min  # Return track indices and minimum delta R values



sample_small = sample


# Create new dataframe containing all the pair of electrons
# Create an empty list to store the new rows
new_rows = []

# Iterate through each row of the original DataFrame
# for index, row in sample.iterrows():

time2_array = []
start_loop = time.time()

for z in range(len(sample_small['mu_truthOrigin'])):

    num_muons = len(sample_small.iloc[z]['mu_truthOrigin'])  # Number of muons instead of electrons
    Track_index, dR_min = InDet_Track_finder(sample_small.iloc[z])

    for i, j in itertools.combinations(range(num_muons), 2):
        
        mass = np.sqrt(2 * sample_small.iloc[z]['mu_pt'][i] * sample_small.iloc[z]['mu_pt'][j] * (np.cosh(sample_small.iloc[z]['mu_eta'][i] - sample_small.iloc[z]['mu_eta'][j]) - np.cos(sample_small.iloc[z]['mu_phi'][i] - sample_small.iloc[z]['mu_phi'][j])))

        if sample_small.iloc[z]['mu_charge'][i] != sample_small.iloc[z]['mu_charge'][j] and sample_small.iloc[z]['mu_truthOrigin'][i] == 13 and sample_small.iloc[z]['mu_truthOrigin'][j] == 13:
            isZ = 1
        else:
            isZ = 0
        new_row = {
            'event_index': z,
            'mu_index1': [i,j],
            'runNumber': sample_small.iloc[z]['runNumber'],
            'eventNumber': sample_small.iloc[z]['eventNumber'],
            'M_mm': mass,
            'isZ': isZ,
            
            'mu1_pt': sample_small.iloc[z]['mu_pt'][i],
            'mu2_pt': sample_small.iloc[z]['mu_pt'][j],
            'mu1_eta': sample_small.iloc[z]['mu_eta'][i],
            'mu2_eta': sample_small.iloc[z]['mu_eta'][j],
            'mu1_phi': sample_small.iloc[z]['mu_phi'][i],
            'mu2_phi': sample_small.iloc[z]['mu_phi'][j],
            'mu1_charge': sample_small.iloc[z]['mu_charge'][i],
            'mu2_charge': sample_small.iloc[z]['mu_charge'][j],
            'mu1_truthOrigin': sample_small.iloc[z]['mu_truthOrigin'][i],
            'mu2_truthOrigin': sample_small.iloc[z]['mu_truthOrigin'][j],
            'mu1_ptcone20': sample_small.iloc[z]['mu_ptcone20'][i],
            'mu2_ptcone20': sample_small.iloc[z]['mu_ptcone20'][j],
            'mu1_ptcone30': sample_small.iloc[z]['mu_ptcone30'][i],
            'mu2_ptcone30': sample_small.iloc[z]['mu_ptcone30'][j],
            'mu1_ptvarcone20': sample_small.iloc[z]['mu_ptvarcone20'][i],
            'mu2_ptvarcone20': sample_small.iloc[z]['mu_ptvarcone20'][j],
            'mu1_ptvarcone30': sample_small.iloc[z]['mu_ptvarcone30'][i],
            'mu2_ptvarcone30': sample_small.iloc[z]['mu_ptvarcone30'][j],
            'mu1_topetcone20': sample_small.iloc[z]['mu_topoetcone20'][i],
            'mu2_topetcone20': sample_small.iloc[z]['mu_topoetcone20'][j],
            'mu1_topetcone40': sample_small.iloc[z]['mu_topoetcone40'][i],
            'mu2_topetcone40': sample_small.iloc[z]['mu_topoetcone40'][j],
            # 'mu1_IsoCloseByCorr_eta': sample_small.iloc[z]['mu_IsoCloseByCorr_assocClustEta'][i],
            # 'mu2_IsoCloseByCorr_eta': sample_small.iloc[z]['mu_IsoCloseByCorr_assocClustEta'][j],
            # 'mu1_IsoCloseByCorr_phi': sample_small.iloc[z]['mu_IsoCloseByCorr_assocClustPhi'][i],
            # 'mu2_IsoCloseByCorr_phi': sample_small.iloc[z]['mu_IsoCloseByCorr_assocClustPhi'][j],
            # 'mu1_isoCloseByCorr_clustdecorr': sample_small.iloc[z]['mu_IsoCloseByCorr_assocClustDecor'][i],
            # 'mu2_isoCloseByCorr_clustdecorr': sample_small.iloc[z]['mu_IsoCloseByCorr_assocClustDecor'][j],
            'mu1_author': sample_small.iloc[z]['mu_author'][i],
            'mu2_author': sample_small.iloc[z]['mu_author'][j],
            'mu1_type': sample_small.iloc[z]['mu_type'][i],
            'mu2_type': sample_small.iloc[z]['mu_type'][j],
            'mu1_energyLossType': sample_small.iloc[z]['mu_enegylosstype'][i],
            'mu2_energyLossType': sample_small.iloc[z]['mu_enegylosstype'][j],
            'mu1_quality': sample_small.iloc[z]['mu_quality'][i],
            'mu2_quality': sample_small.iloc[z]['mu_quality'][j],
            'mu1_ptcone40': sample_small.iloc[z]['mu_ptcone40'][i],
            'mu2_ptcone40': sample_small.iloc[z]['mu_ptcone40'][j],
            'mu1_ptvarcone40': sample_small.iloc[z]['mu_ptvarcone40'][i],
            'mu2_ptvarcone40': sample_small.iloc[z]['mu_ptvarcone40'][j],
            'mu1_DFCommonMuonPassIDCuts': sample_small.iloc[z]['mu_DFCommonMuonPassIDCuts'][i],
            'mu2_DFCommonMuonPassIDCuts': sample_small.iloc[z]['mu_DFCommonMuonPassIDCuts'][j],
            'mu1_DFCommonJetDr': sample_small.iloc[z]['mu_DFCommonJetDr'][i],
            'mu2_DFCommonJetDr': sample_small.iloc[z]['mu_DFCommonJetDr'][j],
            'mu1_numprecisionlayers': sample_small.iloc[z]['mu_numprecisionlayers'][i],
            'mu2_numprecisionlayers': sample_small.iloc[z]['mu_numprecisionlayers'][j],
            'mu1_numprecisionholelayers': sample_small.iloc[z]['mu_numprecisionholelayers'][i],
            'mu2_numprecisionholelayers': sample_small.iloc[z]['mu_numprecisionholelayers'][j],
            'mu1_caloLRlikelihood': sample_small.iloc[z]['mu_caloLRlikelihood'][i],
            'mu2_caloLRlikelihood': sample_small.iloc[z]['mu_caloLRlikelihood'][j],
            'mu1_CaloMuonIDTag' : sample_small.iloc[z]['mu_CaloMuonIDTag'][i],
            'mu2_CaloMuonIDTag' : sample_small.iloc[z]['mu_CaloMuonIDTag'][j],

            'mu1_InDetTrack_d0': sample_small.iloc[z]['InDetTrack_d0'][Track_index[i]],
            'mu2_InDetTrack_d0': sample_small.iloc[z]['InDetTrack_d0'][Track_index[j]],
            'mu1_InDetTrack_z0': sample_small.iloc[z]['InDetTrack_z0'][Track_index[i]],
            'mu2_InDetTrack_z0': sample_small.iloc[z]['InDetTrack_z0'][Track_index[j]],
            'mu1_InDetTrack_phi': sample_small.iloc[z]['InDetTrack_phi'][Track_index[i]],
            'mu2_InDetTrack_phi': sample_small.iloc[z]['InDetTrack_phi'][Track_index[j]],
            'mu1_InDetTrack_theta': sample_small.iloc[z]['InDetTrack_theta'][Track_index[i]],
            'mu2_InDetTrack_theta': sample_small.iloc[z]['InDetTrack_theta'][Track_index[j]],
            # 'mu1_InDetTrack_qoverp': sample_small.iloc[z]['InDetTrack_qoverp'][Track_index[i]],
            # 'mu2_InDetTrack_qoverp': sample_small.iloc[z]['InDetTrack_qoverp'][Track_index[j]],

            'mu1_IDT_var0': sample_small.iloc[z]['InDetTrack_covdiag'][Track_index[i]][0],
            'mu2_IDT_var0': sample_small.iloc[z]['InDetTrack_covdiag'][Track_index[j]][0],
            'mu1_IDT_var1': sample_small.iloc[z]['InDetTrack_covdiag'][Track_index[i]][1],
            'mu2_IDT_var1': sample_small.iloc[z]['InDetTrack_covdiag'][Track_index[j]][1],
            'mu1_IDT_var2': sample_small.iloc[z]['InDetTrack_covdiag'][Track_index[i]][2],
            'mu2_IDT_var2': sample_small.iloc[z]['InDetTrack_covdiag'][Track_index[j]][2],
            'mu1_IDT_var3': sample_small.iloc[z]['InDetTrack_covdiag'][Track_index[i]][3],
            'mu2_IDT_var3': sample_small.iloc[z]['InDetTrack_covdiag'][Track_index[j]][3],
            'mu1_IDT_var4': sample_small.iloc[z]['InDetTrack_covdiag'][Track_index[i]][4],
            'mu2_IDT_var4': sample_small.iloc[z]['InDetTrack_covdiag'][Track_index[j]][4],

            'mu1_dRmin': dR_min[i],
            'mu2_dRmin': dR_min[j],
        }
        new_rows.append(new_row)
    if z % 200 == 0 and z != 0:
        end_loop = time.time()
        time2_array.append(end_loop - start_loop)
        print(f'z: {z}, {z/len(sample_small["el_truthOrigin"])*100:.2f}%, Time for 200 loops: {end_loop - start_loop:.2f}, total time: {np.sum(time2_array)/60:.2f} min')
        # print(f'Estimated time left: {(np.sum(time2_array)*(1-1/(z/len(sample_small["el_truthOrigin"]))))/60:.2f} min')
        start_loop = time.time()
    if z%1000 == 0 and z != 0:
        print('saving')
        new_df = pd.DataFrame(new_rows)
        new_df.to_parquet(f'/groups/hep/kinch/H_Zg/muons/muonpairs_{z}.parquet')
        new_rows = []
