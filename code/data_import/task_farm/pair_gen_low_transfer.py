from mpi4py import MPI



import numpy as np
import pandas as pd
# import uproot
import pyarrow.parquet as pq
import awkward_pandas as akpd
import itertools
import mplhep as hep
hep.style.use([hep.style.ATLAS])
import matplotlib.pyplot as plt
import time

time0 = time.time()
N = 1000

path = '/groups/hep/kinch/H_Zg/samples_processed/HZeeg_ggF_MC_reduced_28august_131var_small.parquet'

# path = 'c:/Users/Jens/Documents/Github/Thesis/samples_processed/HZeeg_ggF_MC_reduced_28august_131var.parquet'

sample = pd.read_parquet(path) 
# print(f'sample shape: {sample.shape}')

sample_small = sample.head(N)
# print(f'current sample shape: {sample_small.shape}')
load_time = time.time() - time0





def GSF_Track_finder(row):
    num_electrons = len(row["el_truthOrigin"])
    dR_min = np.ones(num_electrons)*999
    GSF_index = np.ones(num_electrons,dtype=int)*-1
    for i_el in range(num_electrons):
        NGSF = len(row["GSFTrackParticlesAuxDyn_theta"])
        for i_GSF in range(NGSF):
            eta_GSF = -np.log(np.tan(row['GSFTrackParticlesAuxDyn_theta'][i_GSF]/2))
            deta = row['el_eta'][i_el] - eta_GSF
            dphi = row['el_phi'][i_el] - row['GSFTrackParticlesAuxDyn_phi'][i_GSF]
            if (dphi > np.pi) : dphi = dphi - np.pi
            if (dphi < -np.pi) : dphi = dphi + np.pi
            dR_eG = np.sqrt(deta**2 + dphi**2)
            if (dR_eG < dR_min[i_el]):
                dR_min[i_el] = dR_eG
                GSF_index[i_el] = i_GSF
    return GSF_index, dR_min

def job(sample_small):
    new_rows = []
    for z in range(len(sample_small["el_truthOrigin"])):
        # Get the number of electrons in this event
        num_electrons = len(sample_small.iloc[z]["el_truthOrigin"])
        
        #first, find the best match between electrons and GSF tracks
        GSF_index, GSF_dR = GSF_Track_finder(sample_small.iloc[z])
        
        # print(z)
        # Generate all possible pairs of electrons
        for i, j in itertools.combinations(range(num_electrons), 2):
            # Create a new row with the combined variables of the two electrons
            
            # mass_old = invariant_mass_ee(sample_small.iloc[z],i,j)/1000.

            mass = np.sqrt(2*sample_small.iloc[z]['el_pt'][i]*sample_small.iloc[z]['el_pt'][j]*(np.cosh(sample_small.iloc[z]['el_eta'][i]-sample_small.iloc[z]['el_eta'][j]) - np.cos(sample_small.iloc[z]['el_phi'][i]-sample_small.iloc[z]['el_phi'][j])))/1000.
            # deltaM=(np.abs(mass-mass_old)) order of 10^-7
            if sample_small.iloc[z]["el_truthOrigin"][i] == 13 and sample_small.iloc[z]["el_truthOrigin"][j] == 13 and (sample_small.iloc[z]['el_charge'][i] != sample_small.iloc[z]['el_charge'][j]):
                isZ = 1
                # print(sample_small['truthel_pdgId'][z][i], sample_small['truthel_pdgId'][z][j])
            else:
                isZ = 0
            # print(i,j)
            new_row = {
                'event_index': z,
                'el_index':[i,j],
                'runNumber': sample_small.iloc[z]['runNumber'],
                'eventNumber': sample_small.iloc[z]['eventNumber'],
                'actualInteractionsPerCrossing': sample_small.iloc[z]['actualInteractionsPerCrossing'],
                'averageInteractionsPerCrossing': sample_small.iloc[z]['averageInteractionsPerCrossing'],
                'm_ee': mass,
                'isZ':isZ,
                
                'el1_pt':           sample_small.iloc[z]['el_pt'][i],
                'el1_eta':          sample_small.iloc[z]['el_eta'][i],
                'el1_phi':          sample_small.iloc[z]['el_phi'][i],
                'el1_m':            sample_small.iloc[z]['el_m'][i],
                'el1_charge':       sample_small.iloc[z]['el_charge'][i],
                'el1_ptvarcone20':  sample_small.iloc[z]['el_ptvarcone20'][i],
                'el1_topoetcone20': sample_small.iloc[z]['el_topoetcone20'][i],
                'el1_topoetcone40': sample_small.iloc[z]['el_topoetcone40'][i],
                'el1_f1':           sample_small.iloc[z]['el_f1'][i],
                'el1_neflowisol20': sample_small.iloc[z]['el_neflowisol20'][i],
                'el1_truthPdgId':   sample_small.iloc[z]['el_truthPdgId'][i],
                'el1_truthType':    sample_small.iloc[z]['el_truthType'][i],
                'el1_truthOrigin':  sample_small.iloc[z]['el_truthOrigin'][i],
                'el1_DFCommonElectronsECIDS': sample_small.iloc[z]['el_DFCommonElectronsECIDS'][i],
                'el1_DFCommonElectronsECIDSResult': sample_small.iloc[z]['el_DFCommonElectronsECIDSResult'][i],            
                'el1_DFCommonElectrons_pel': sample_small.iloc[z]['el_DFCommonElectronsDNN_pel'][i],
                'el1_DFcommonElectrons_LHLoose': sample_small.iloc[z]['el_DFCommonElectronsLHLoose'][i],

                'el1_GSFTrack_d0': sample_small.iloc[z]['GSFTrackParticlesAuxDyn_d0'][GSF_index[i]],
                'el1_GSFTrack_z0': sample_small.iloc[z]['GSFTrackParticlesAuxDyn_z0'][GSF_index[i]],
                'el1_GSFTrack_theta': sample_small.iloc[z]['GSFTrackParticlesAuxDyn_theta'][GSF_index[i]],
                'el1_GSFTrack_phi': sample_small.iloc[z]['GSFTrackParticlesAuxDyn_phi'][GSF_index[i]],
                'el1_GSFTrack_qOverP': sample_small.iloc[z]['GSFTrackParticlesAuxDyn_qOverP'][GSF_index[i]],
                'el1_GSF_dR': GSF_dR[i],

                'el1_GSF_Track_Var0': sample_small.iloc[z]['GSFTrackParticlesAuxDyn_definingParametersCovMatrixDiag'][GSF_index[i]][0],
                'el1_GSF_Track_Var1': sample_small.iloc[z]['GSFTrackParticlesAuxDyn_definingParametersCovMatrixDiag'][GSF_index[i]][1],
                'el1_GSF_Track_Var2': sample_small.iloc[z]['GSFTrackParticlesAuxDyn_definingParametersCovMatrixDiag'][GSF_index[i]][2],
                'el1_GSF_Track_Var3': sample_small.iloc[z]['GSFTrackParticlesAuxDyn_definingParametersCovMatrixDiag'][GSF_index[i]][3],
                'el1_GSF_Track_Var4': sample_small.iloc[z]['GSFTrackParticlesAuxDyn_definingParametersCovMatrixDiag'][GSF_index[i]][4],            

                'el2_pt':           sample_small.iloc[z]['el_pt'][j],
                'el2_eta':          sample_small.iloc[z]['el_eta'][j],
                'el2_phi':          sample_small.iloc[z]['el_phi'][j],
                'el2_m':            sample_small.iloc[z]['el_m'][j],
                'el2_charge':       sample_small.iloc[z]['el_charge'][j],
                'el2_ptvarcone20':  sample_small.iloc[z]['el_ptvarcone20'][j],
                'el2_topoetcone20': sample_small.iloc[z]['el_topoetcone20'][j],
                'el2_topoetcone40': sample_small.iloc[z]['el_topoetcone40'][j],
                'el2_f1':           sample_small.iloc[z]['el_f1'][j],
                'el2_neflowisol20': sample_small.iloc[z]['el_neflowisol20'][j],
                'el2_truthPdgId':   sample_small.iloc[z]['el_truthPdgId'][j],
                'el2_truthType':    sample_small.iloc[z]['el_truthType'][j],
                'el2_truthOrigin':  sample_small.iloc[z]['el_truthOrigin'][j],
                'el2_DFCommonElectronsECIDS': sample_small.iloc[z]['el_DFCommonElectronsECIDS'][j],
                'el2_DFCommonElectronsECIDSResult': sample_small.iloc[z]['el_DFCommonElectronsECIDSResult'][j],
                'el2_DFCommonElectrons_pel': sample_small.iloc[z]['el_DFCommonElectronsDNN_pel'][j],
                'el2_DFcommonElectrons_LHLoose': sample_small.iloc[z]['el_DFCommonElectronsLHLoose'][j],

                'el2_GSFTrack_d0': sample_small.iloc[z]['GSFTrackParticlesAuxDyn_d0'][GSF_index[j]],
                'el2_GSFTrack_z0': sample_small.iloc[z]['GSFTrackParticlesAuxDyn_z0'][GSF_index[j]],
                'el2_GSFTrack_theta': sample_small.iloc[z]['GSFTrackParticlesAuxDyn_theta'][GSF_index[j]],
                'el2_GSFTrack_phi': sample_small.iloc[z]['GSFTrackParticlesAuxDyn_phi'][GSF_index[j]],
                'el2_GSFTrack_qOverP': sample_small.iloc[z]['GSFTrackParticlesAuxDyn_qOverP'][GSF_index[j]],
                'el2_GSF_dR': GSF_dR[j],

                'el2_GSF_Track_Var0': sample_small.iloc[z]['GSFTrackParticlesAuxDyn_definingParametersCovMatrixDiag'][GSF_index[j]][0],
                'el2_GSF_Track_Var1': sample_small.iloc[z]['GSFTrackParticlesAuxDyn_definingParametersCovMatrixDiag'][GSF_index[j]][1],
                'el2_GSF_Track_Var2': sample_small.iloc[z]['GSFTrackParticlesAuxDyn_definingParametersCovMatrixDiag'][GSF_index[j]][2],
                'el2_GSF_Track_Var3': sample_small.iloc[z]['GSFTrackParticlesAuxDyn_definingParametersCovMatrixDiag'][GSF_index[j]][3],
                'el2_GSF_Track_Var4': sample_small.iloc[z]['GSFTrackParticlesAuxDyn_definingParametersCovMatrixDiag'][GSF_index[j]][4],

                #truth variables:
                # 'el_truth_m': sample_small['truthel_m'][z],
                # 'el_truth_px': sample_small['truthel_px'][z],
                # 'el_truth_py': sample_small['truthel_py'][z],
                # 'el_truth_pz': sample_small['truthel_pz'][z],
                # 'el_truth_E': sample_small['truthel_E'][z],
                # 'el_truth_pdgId': sample_small['truthel_pdgId'][z],
                # 'el_truth_Type:': sample_small['el_truthType'][z],
                # 'el_truth_Origin': sample_small['el_truthOrigin'][z],


                # 'el2_truth_m': sample_small['truthel_m'][z][j],
                # 'el2_truth_px': sample_small['truthel_px'][z][j],
                # 'el2_truth_py': sample_small['truthel_py'][z][j],
                # 'el2_truth_pz': sample_small['truthel_pz'][z][j],
                # 'el2_truth_E': sample_small['truthel_E'][z][j],
                # 'el2_truth_pdgId': sample_small['truthel_pdgId'][z][j],
                # 'el2_truth_Type:': sample_small['el_truthType'][z][j],
                # 'el2_truth_Origin': sample_small['el_truthOrigin'][z][j],

                #wtf, these indexes?
                
                # 'event_GSFTrack_cov' : sample_small['GSFTrackParticlesAuxDyn_definingParametersCovMatrixDiag'][z]
            }
            # Append the new row to the list
            new_rows.append(new_row)
    
    return new_rows




def main(rank, ws):
    print("Hello from rank", rank, "of", ws)
    
    # import data
    # send N events to each worker
    # receive results from workers
    # repeat until all events are processed
    # save results
    
    

    bunch_size = 5
    num_bunches = len(sample_small)/bunch_size
    result = []

    avail_tasks = list(range(len(sample_small)))
    avail_ws = list(range(1,comm.Get_size()))
    task_ws = {}
    count_ws = [0]*comm.Get_size()

    print('Starting main loop')
    state = MPI.Status()
    while len(avail_ws) > 0:
        job_index = avail_tasks[-bunch_size:]
        comm.send(job_index, dest=avail_ws[-1], tag=11)
        task_ws[avail_ws[-1]] = job_index
        avail_tasks = avail_tasks[:-bunch_size]
        avail_ws.pop()
    print('Sent initial jobs')
    while len(avail_tasks) > 0:
        t = comm.recv(source=MPI.ANY_SOURCE, tag=11, status=state)
        w = state.Get_source()
        
        count_ws[w] += bunch_size

        if len(avail_tasks) > 0:
            job_index = avail_tasks[-bunch_size:]
            comm.send(job_index, dest=w, tag=11)
            task_ws[w] = job_index
            avail_tasks = avail_tasks[:-bunch_size]
        else:
            job_index = avail_tasks
            comm.send(job_index, dest=w, tag=11)
            task_ws[w] = job_index
            avail_tasks = []
    print('Sent all jobs')
    while len(avail_ws) < ws:
        t = comm.recv(source=MPI.ANY_SOURCE, tag=11, status=state)
        w = state.Get_source()
        count_ws[w] += bunch_size

        comm.send(0, dest=w, tag=10)
        avail_ws.append(w)
    
    print('Received all results')
    print(f'count_ws: {count_ws}')

    


    



def worker(rank, ws):
    # print("Hello from rank", rank, "of", ws)
    time_noting = 0
    time_working = 0
    time_append = 0

    # receive N events
    # process events
    # send results to main
    state = MPI.Status()
    worker_rows = []
    while True:
        time_start_n = time.time()
        indexes = comm.recv(source=0, tag=MPI.ANY_TAG, status= state)
        if state.Get_tag() == 10:
            break
        time_start_w = time.time()
        new_rows = job(sample_small.iloc[indexes])
        time_append0 = time.time()
        worker_rows.extend(new_rows)
        time_append1 = time.time()
        time_end_w = time.time()
        comm.send(indexes, dest=0, tag=11)
        time_end_n = time.time()
        time_noting += time_end_n - time_start_n - (time_end_w - time_start_w)
        time_working += time_end_w - time_start_w
        time_append += time_append1 - time_append0
    
    df = pd.DataFrame(worker_rows)
    filename = f'pair_gen_low_transfer_{rank}.parquet'
    df.to_parquet(filename)
    # print(df.shape)
    print(f'Worker {rank} finished, time working: {time_working}, time noting: {time_noting}, time loading: {load_time}, time appending: {time_append}')



comm = MPI.COMM_WORLD
rank = comm.Get_rank()
ws = comm.Get_size() - 1

if rank == 0:
    time_start = time.time()
    main(rank, ws)
    time_finish = time.time()
    print(f'Time taken: {time_finish - time_start} seconds, using {ws} workers')
else:
    worker(rank, ws)