import numpy as np
import pandas as pd
import uproot
import pyarrow.parquet as pq
import awkward_pandas as akpd
import itertools
import mplhep as hep
import multiprocessing as mp
import time
import matplotlib.pyplot as plt
from pathlib import Path

# Set up matplotlib with ATLAS style
hep.style.use([hep.style.ATLAS])


def GSF_Track_finder(row):
    num_electrons = len(row["el_truthOrigin"])
    dR_min = np.ones(num_electrons) * 999
    GSF_index = np.ones(num_electrons, dtype=int) * -1
    for i_el in range(num_electrons):
        NGSF = len(row["GSFTrackParticlesAuxDyn_theta"])
        for i_GSF in range(NGSF):
            eta_GSF = -np.log(np.tan(row['GSFTrackParticlesAuxDyn_theta'][i_GSF] / 2))
            deta = row['el_eta'][i_el] - eta_GSF
            dphi = row['el_phi'][i_el] - row['GSFTrackParticlesAuxDyn_phi'][i_GSF]
            # Correct dphi for periodic boundary conditions
            if dphi > np.pi:
                dphi -= 2 * np.pi
            elif dphi < -np.pi:
                dphi += 2 * np.pi
            # Compute deltaR
            dR_eG = np.sqrt(deta ** 2 + dphi ** 2)
            if dR_eG < dR_min[i_el]:
                dR_min[i_el] = dR_eG
                GSF_index[i_el] = i_GSF
    dR_min[dR_min == 999] = -1  # Handle cases where no match was found
    return GSF_index, dR_min


# Worker function to process a chunk of data
def process_chunk(sample_chunk, chunk_id):
    new_rows = []
    for z in range(len(sample_chunk['el_truthOrigin'])):
        row = sample_chunk.iloc[z]
        num_electrons = len(row["el_truthOrigin"])

        # Find the best match between electrons and GSF tracks
        GSF_index, GSF_dR = GSF_Track_finder(row)

        # Loop over all electron pairs
        for i in range(num_electrons):
            for j in range(i + 1, num_electrons):
                mass = np.sqrt(
                    2 * row['el_pt'][i] * row['el_pt'][j] *
                    (np.cosh(row['el_eta'][i] - row['el_eta'][j]) - np.cos(row['el_phi'][i] - row['el_phi'][j]))
                ) / 1000.  # Convert mass to GeV

                # Check if the pair originates from a Z boson decay
                isZ = 1 if (row["el_truthOrigin"][i] == 13 and 
                            row["el_truthOrigin"][j] == 13 and 
                            row['el_charge'][i] != row['el_charge'][j]) else 0

                new_row = {
                    'event_index': row.name,
                    'el1_index': i,
                    'el2_index': j,
                    'runNumber': row['runNumber'],
                    'eventNumber': row['eventNumber'],
                    'actualInteractionsPerCrossing': row['actualInteractionsPerCrossing'],
                    'averageInteractionsPerCrossing': row['averageInteractionsPerCrossing'],
                    'm_ee': mass,
                    'isZ': isZ,
                    'el1_pt': row['el_pt'][i],
                    'el1_eta': row['el_eta'][i],
                    'el1_phi': row['el_phi'][i],
                    'el1_m': row['el_m'][i],
                    'el1_charge': row['el_charge'][i],
                    'el1_ptvarcone20': row['el_ptvarcone20'][i],
                    'el1_topoetcone20': row['el_topoetcone20'][i],
                    'el1_topoetcone40': row['el_topoetcone40'][i],
                    'el1_f1': row['el_f1'][i],
                    'el1_neflowisol20': row['el_neflowisol20'][i],
                    'el1_truthPdgId': row['el_truthPdgId'][i],
                    'el1_truthType': row['el_truthType'][i],
                    'el1_truthOrigin': row['el_truthOrigin'][i],
                    'el1_DFCommonElectronsECIDS': row['el_DFCommonElectronsECIDS'][i],
                    'el1_DFCommonElectronsECIDSResult': row['el_DFCommonElectronsECIDSResult'][i],
                    'el1_DFCommonElectrons_pel': row['el_DFCommonElectronsDNN_pel'][i],
                    'el1_DFCommonElectrons_LHLoose': row['el_DFCommonElectronsLHLoose'][i],
                    'el1_GSFTrack_d0': row['GSFTrackParticlesAuxDyn_d0'][GSF_index[i]],
                    'el1_GSFTrack_z0': row['GSFTrackParticlesAuxDyn_z0'][GSF_index[i]],
                    'el1_GSFTrack_theta': row['GSFTrackParticlesAuxDyn_theta'][GSF_index[i]],
                    'el1_GSFTrack_phi': row['GSFTrackParticlesAuxDyn_phi'][GSF_index[i]],
                    'el1_GSFTrack_qOverP': row['GSFTrackParticlesAuxDyn_qOverP'][GSF_index[i]],
                    'el1_GSF_dR': GSF_dR[i],
                    'el1_GSF_Track_Var0': row['GSFTrackParticlesAuxDyn_definingParametersCovMatrixDiag'][GSF_index[i]][0],
                    'el1_GSF_Track_Var1': row['GSFTrackParticlesAuxDyn_definingParametersCovMatrixDiag'][GSF_index[i]][1],
                    'el1_GSF_Track_Var2': row['GSFTrackParticlesAuxDyn_definingParametersCovMatrixDiag'][GSF_index[i]][2],
                    'el1_GSF_Track_Var3': row['GSFTrackParticlesAuxDyn_definingParametersCovMatrixDiag'][GSF_index[i]][3],
                    'el1_GSF_Track_Var4': row['GSFTrackParticlesAuxDyn_definingParametersCovMatrixDiag'][GSF_index[i]][4],

                    'el2_pt': row['el_pt'][j],
                    'el2_eta': row['el_eta'][j],
                    'el2_phi': row['el_phi'][j],
                    'el2_m': row['el_m'][j],
                    'el2_charge': row['el_charge'][j],
                    'el2_ptvarcone20': row['el_ptvarcone20'][j],
                    'el2_topoetcone20': row['el_topoetcone20'][j],
                    'el2_topoetcone40': row['el_topoetcone40'][j],
                    'el2_f1': row['el_f1'][j],
                    'el2_neflowisol20': row['el_neflowisol20'][j],
                    'el2_truthPdgId': row['el_truthPdgId'][j],
                    'el2_truthType': row['el_truthType'][j],
                    'el2_truthOrigin': row['el_truthOrigin'][j],
                    'el2_DFCommonElectronsECIDS': row['el_DFCommonElectronsECIDS'][j],
                    'el2_DFCommonElectronsECIDSResult': row['el_DFCommonElectronsECIDSResult'][j],
                    'el2_DFCommonElectrons_pel': row['el_DFCommonElectronsDNN_pel'][j],
                    'el2_DFCommonElectrons_LHLoose': row['el_DFCommonElectronsLHLoose'][j],
                    'el2_GSFTrack_d0': row['GSFTrackParticlesAuxDyn_d0'][GSF_index[j]],
                    'el2_GSFTrack_z0': row['GSFTrackParticlesAuxDyn_z0'][GSF_index[j]],
                    'el2_GSFTrack_theta': row['GSFTrackParticlesAuxDyn_theta'][GSF_index[j]],
                    'el2_GSFTrack_phi': row['GSFTrackParticlesAuxDyn_phi'][GSF_index[j]],
                    'el2_GSFTrack_qOverP': row['GSFTrackParticlesAuxDyn_qOverP'][GSF_index[j]],
                    'el2_GSF_dR': GSF_dR[j],
                    'el2_GSF_Track_Var0': row['GSFTrackParticlesAuxDyn_definingParametersCovMatrixDiag'][GSF_index[j]][0],
                    'el2_GSF_Track_Var1': row['GSFTrackParticlesAuxDyn_definingParametersCovMatrixDiag'][GSF_index[j]][1],
                    'el2_GSF_Track_Var2': row['GSFTrackParticlesAuxDyn_definingParametersCovMatrixDiag'][GSF_index[j]][2],
                    'el2_GSF_Track_Var3': row['GSFTrackParticlesAuxDyn_definingParametersCovMatrixDiag'][GSF_index[j]][3],
                    'el2_GSF_Track_Var4': row['GSFTrackParticlesAuxDyn_definingParametersCovMatrixDiag'][GSF_index[j]][4],
                }
                new_rows.append(new_row)
    return new_rows


def split_dataframe(df, num_chunks):
    chunk_size = len(df) // num_chunks
    return [df.iloc[i*chunk_size:(i+1)*chunk_size] for i in range(num_chunks)]

# Parallel processing function with worker-based logic
def parallel_process(sample_small):
    num_workers = 10  # Use all available cores
    chunks = split_dataframe(sample_small, num_workers)
    with mp.Pool(num_workers) as pool:
        results = pool.starmap(process_chunk, zip(chunks, range(num_workers)))
    
    all_new_rows = list(itertools.chain(*results))
    new_df = pd.DataFrame(all_new_rows)
    return new_df


# Main processing code
if __name__ == "__main__":
    start_time = time.time()

    Num_events_to_save = 'all'
    file_path = '/groups/hep/kinch/H_Zg/samples_processed/ttbar'
    pathlist = Path(file_path).rglob('*.parquet')

    i = 0
    for path in pathlist:
        # Read sample data
        sample = pd.read_parquet(path)
        sample_small = sample#.head(Num_events_to_save)

        # Run the parallel processing
        new_df = parallel_process(sample_small)

        # Save the result to a parquet file
        save_path = path + f'/ttbar_pairs{i}.parquet'
        new_df.to_parquet(save_path)
        i += 1
        print(f"Saved {len(new_df)} pairs to {save_path}")

    end_time = time.time()
    print(f"Total processing time: {(end_time - start_time) / 60:.2f} minutes")
