import numpy as np
import datetime
from netCDF4 import Dataset,num2date,date2num

ny, nx = (41, 161)
nz = 4
data_npz = np.load('data_viz_TC_nowcasting.npz')
#pred_frames, target_frames, pred_labels, target_labels = data_npz['pred_frames'], data_npz['target_frames'], data_npz['pred_labels'], data_npz['target_labels']
pred_frames, target_frames = data_npz['pred_frames'], data_npz['target_frames']
print(pred_frames.shape)
print(target_frames.shape)
for i in range(pred_frames.shape[0]):
    #target frames shape (B, time steps = 2, num_channels, height, width)
    uu_vector = pred_frames[i, 0]
    uu_vector = uu_vector.reshape(1, uu_vector.shape[0], uu_vector.shape[1])
    vv_vector = pred_frames[i, 2]
    vv_vector = vv_vector.reshape(1, vv_vector.shape[0], vv_vector.shape[1])
    vorticity = pred_frames[i, 9]
    vorticity = vorticity.reshape(1, vorticity.shape[0], vorticity.shape[1])
    #print(dataout.shape)
    ncout = Dataset(f'nc_files/pred_frames/pred_frame_{i}.nc','w',format = 'NETCDF4')
    time = ncout.createDimension('time',None)
    lat = ncout.createDimension('lat',ny)
    lon = ncout.createDimension('lon',nx)
    #ncout.createDimension('lev',nz)
    times = ncout.createVariable('time','f4',('time'))
    latvar = ncout.createVariable('lat','f4',('lat'))
    lonvar = ncout.createVariable('lon','f4',('lon'))
    
    #levvar = ncout.createVariable('lev','float32',('lev'))
    uu = ncout.createVariable('uu','f4',('time','lat','lon'))
    vv = ncout.createVariable('vv','f4',('time','lat','lon'))
    vo = ncout.createVariable('vo','f4',('time','lat','lon'))
    #vo.setncattr('units','mm')
    vo.units = 'Unknown'

    latvar[:] = np.arange(5.0, 46.0, 1.0)
    lonvar[:] = np.arange(100, 261, 1.0)
    uu[:] = uu_vector
    vv[:] = vv_vector
    vo[:] = vorticity
    ncout.close()

for i in range(target_frames.shape[0]):
    #target frames shape (B, time steps = 2, num_channels, height, width)
    uu_vector = target_frames[i, 0]
    uu_vector = uu_vector.reshape(1, uu_vector.shape[0], uu_vector.shape[1])
    vv_vector = target_frames[i, 2]
    vv_vector = vv_vector.reshape(1, vv_vector.shape[0], vv_vector.shape[1])
    vorticity = target_frames[i, 9]
    vorticity = vorticity.reshape(1, vorticity.shape[0], vorticity.shape[1])
    #print(dataout.shape)
    ncout = Dataset(f'nc_files/target_frames/target_frame_{i}.nc','w',format = 'NETCDF4')
    time = ncout.createDimension('time',None)
    lat = ncout.createDimension('lat',ny)
    lon = ncout.createDimension('lon',nx)
    #ncout.createDimension('lev',nz)
    times = ncout.createVariable('time','f4',('time'))
    latvar = ncout.createVariable('lat','f4',('lat'))
    lonvar = ncout.createVariable('lon','f4',('lon'))
    
    #levvar = ncout.createVariable('lev','float32',('lev'))
    uu = ncout.createVariable('uu','f4',('time','lat','lon'))
    vv = ncout.createVariable('vv','f4',('time','lat','lon'))
    vo = ncout.createVariable('vo','f4',('time','lat','lon'))
    #vo.setncattr('units','mm')
    vo.units = 'Unknown'

    latvar[:] = np.arange(5.0, 46.0, 1.0)
    lonvar[:] = np.arange(100, 261, 1.0)
    uu[:] = uu_vector
    vv[:] = vv_vector
    vo[:] = vorticity
    ncout.close()
