# Import relevant modules

import numpy as np
import datetime
import pytz
import h5py
import hdf5storage
import os
from pynwb import NWBFile, NWBHDF5IO
from pynwb.device import Device
from pynwb.ecephys import SpikeEventSeries
from pynwb.behavior import Position, SpatialSeries


# Load the .mat files containing sorted spikes

# Open info file
fpath0 = 'indy_20160407_02.mat'
f_info = hdf5storage.loadmat(fpath0)
#f_info = h5py.File(fpath0)
info = f_info.keys()


# Create a new NWB file instance

session_start_time = datetime.datetime(2016,4,7,tzinfo=pytz.timezone("America/Los_Angeles"))
experiment_description = 'The behavioral task was to make self-paced reaches to targets arranged in a grid (e.g. 8x8) without gaps or pre-movement delay intervals.'

nwb = NWBFile(session_description='Multichannel Sensorimotor Cortex Electrophysiology', 
              identifier='indy_20160407_02', 
              session_start_time=session_start_time,
              experimenter='Joseph E. ODoherty',
              lab='Sabes lab',
              institution='University of California, San Francisco',
              experiment_description=experiment_description,
              session_id='indy_20160407_02')


# Create Device and ElectrodeGroup and adding electrode information to nwb.


# M1

#Create device
device_M1 = Device('Recording_Device_M1')
nwb.add_device(device_M1)

# Create electrode group
electrode_group_M1 = nwb.create_electrode_group(name='ElectrodeArrayM1', description="96 Channels Electrode Array", 
                                    location="Motor Cortex", 
                                    device=device_M1)

# Add metadata about each electrode in the group
for idx in np.arange(96):
    nwb.add_electrode(x=np.nan, y=np.nan, z=np.nan,
                      imp=np.nan,
                      location='M1', filtering='none',
                      group=electrode_group_M1)


# S1

# Create device
device_S1 = Device('Recording_Device_S1')
nwb.add_device(device_S1)

# Create electrode group
electrode_group_S1 = nwb.create_electrode_group(name='ElectrodeArrayS1', description="96 Channels Electrode Array", 
                                    location="Somatosensory Cortex", 
                                    device=device_S1)

# Add metadata about each electrode in the group
for idx in np.arange(96):
    nwb.add_electrode(x=np.nan, y=np.nan, z=np.nan,
                      imp=np.nan,
                      location='S1', filtering='none',
                      group=electrode_group_S1)


#Store spike waveforms data in acquisition group


# M1

description = 'Spike event waveform "snippets" of M1. Each waveform corresponds to a timestamp in "spikes".'
comments = 'Waveform samples are in microvolts.'

# For each electrode i
for i in np.arange(96):
    
    # Create electrode table region for each electrode
    electrode_table_region_M1 = nwb.create_electrode_table_region([i], 'electrode i in array M1')
    
    # For each unit k
    for k in np.arange(3):
        
        data = f_info['wf'][i,k]
        timestamps = np.ravel(f_info['spikes'][i,k])
        
        # For units with no spike, the data array shape is saved as (48,0).
        # So, we transpose it
        if timestamps.shape==(0,):
            data = data.T
        
        # Create SpikeEventSeries container
        ephys_ts_M1 = SpikeEventSeries(name='M1 Spike Events electrode {0} and unit {1}'.format(i,k),
                                    data=data,
                                    timestamps=timestamps,
                                    electrodes=electrode_table_region_M1,
                                    resolution=4.096e-05,
                                    conversion=1e-6,
                                    description=description,
                                    comments=comments)
        
        # Store spike waveform data
        nwb.add_acquisition(ephys_ts_M1)


# S1

description = 'Spike event waveform "snippets" of S1. Each waveform corresponds to a timestamp in "spikes".'
comments = 'Waveform samples are in microvolts.'

# For each electrode i
for i in np.arange(96,192):
    
    # Create electrode table region for each electrode
    electrode_table_region_S1 = nwb.create_electrode_table_region([i], 'electrode i in array S1')
    
    # For each unit k
    for k in np.arange(3):
        
        data = f_info['wf'][i,k]
        timestamps = np.ravel(f_info['spikes'][i,k])
        
        # For units with no spike, the data array shape is saved as (48,0).
        # So, we transpose it
        if timestamps.shape==(0,):
            data = data.T
        # Create SpikeEventSeries container
        ephys_ts_S1 = SpikeEventSeries(name='S1 Spike Events electrode {0} and unit {1}'.format(i,k),
                                    data=data,
                                    timestamps=timestamps,
                                    electrodes=electrode_table_region_S1,
                                    resolution=4.096e-05,
                                    conversion=1e-6,
                                    description=description,
                                    comments=comments)
        
        # Store spike waveform data
        nwb.add_acquisition(ephys_ts_S1)


# Check the stored data
print(nwb.acquisition)


# Associate electrodes with units

# M1
for j in np.arange(96):
    nwb.add_unit(electrodes=[j],spike_times=np.ravel(f_info['spikes'][j,1]),electrode_group=electrode_group_M1)
    nwb.add_unit(electrodes=[j],spike_times=np.ravel(f_info['spikes'][j,2]),electrode_group=electrode_group_M1)

# S1
for j in np.arange(96,192):
    nwb.add_unit(electrodes=[j],spike_times=np.ravel(f_info['spikes'][j,1]),electrode_group=electrode_group_S1)
    nwb.add_unit(electrodes=[j],spike_times=np.ravel(f_info['spikes'][j,2]),electrode_group=electrode_group_S1)



# Add behavioral information


# SpatialSeries and Position data interfaces to store cursor_pos
cursor_pos = SpatialSeries(name='cursor_position', data=f_info['cursor_pos'], 
                           reference_frame='0,0', conversion=1e-3, resolution=1e-17, 
                           timestamps=np.ravel(f_info['t']), 
                           description='The position of the cursor in Cartesian coordinates (x, y) in mm')

cursor_position = Position(name='CursorPosition',spatial_series=cursor_pos)


# SpatialSeries and Position data interfaces to store finger_pos
finger_pos = SpatialSeries(name='finger_position', data=f_info['finger_pos'],
                           reference_frame='0,0', conversion=1e-2, resolution=1e-17, 
                           timestamps=np.ravel(f_info['t']), 
                           description='The position of the working fingertip in Cartesian coordinates (z, -x, -y), as reported by the hand tracker in cm')

finger_position = Position(name='FingerPosition',spatial_series=finger_pos)


# SpatialSeries and Position data interfaces to store target_pos
target_pos = SpatialSeries(name='target_position', data=f_info['target_pos'],
                           reference_frame='0,0', conversion=1e-3, resolution=1e-17, 
                           timestamps=np.ravel(f_info['t']), 
                           description='The position of the target in Cartesian coordinates (x, y) in mm')

target_position = Position(name='TargetPosition',spatial_series=target_pos)



# Create ProcessingModule add it to the nwb file
behavior_module = nwb.create_processing_module(name='behavior',
                                                   description='preprocessed position data')

# Add data interfaces to the ProcessingModule
nwb.processing['behavior'].add(cursor_position)
nwb.processing['behavior'].add(finger_position)
nwb.processing['behavior'].add(target_position)

print(nwb.processing)

# # Save NWB to file:

fname_nwb = 'indy_20160630_01.nwb'
fpath_nwb = os.path.join('./', fname_nwb)
with NWBHDF5IO(fpath_nwb, mode='w') as io:
    io.write(nwb)
print('File saved with size: ', os.stat(fpath_nwb).st_size/1e6, ' mb')


# # Load NWB:

# io = NWBHDF5IO(fpath_nwb, mode='r')
# nwbfile = io.read()


