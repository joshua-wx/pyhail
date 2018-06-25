import os
import glob
import pyart
import logging
import argparse
import h5py
import numpy as np
from pyhail import hsda, hdr, mesh, common
from cpol_processing import processing as cpol_prc
from datetime import datetime
from multiprocessing import Pool

import warnings

#TODO

def calc_beam_blocking(radar_ffn, srtm_ffn):
    #load radar object
    try:
        if ".h5" in radar_ffn:
            radar = pyart.aux_io.read_odim_h5(radar_ffn)
            #read in beamwidth info manually
            h5file = h5py.File(radar_ffn, 'r')
            bw     = h5file['how'].attrs['beamwH']
            h5file.close()
            ip_dict = radar.instrument_parameters
            if ip_dict is None:
                ip_dict = {}
            ip_dict['radar_beam_width_h'] = {'data':np.array([bw]), 'units':'degrees','standard_name':'beam_width'}
            radar.instrument_parameters = ip_dict
        elif ".nc" or ".mdv" in radar_ffn:
            radar      = pyart.io.read(radar_ffn)
    except Exception as e:
        print('CBB processing failed for: ',radar_ffn)
        print(e)
        return None

    cbb_meta = common.beam_blocking(radar, srtm_ffn)
    
    return cbb_meta
    
    
def chunks(l, n):
    """
    Yield successive n-sized chunks from l.
    From http://stackoverflow.com/a/312464
    """
    for i in range(0, len(l), n):
        yield l[i:i + n]
        
        
def worker(radar_file_name, out_path, cbb_meta):
    ###########################################################
    # Load file
    ###########################################################
    #load radar object
    try:
        if ".h5" in radar_file_name:
            radar = pyart.aux_io.read_odim_h5(radar_file_name)
            radar_name = radar.metadata['source'][6:8]
        elif ".nc" or ".mdv" in radar_file_name:
            radar      = pyart.io.read(radar_file_name)
            radar_name = radar.metadata['instrument_name'][0:3]
    except:
        print('worker failed on: ',radar_file_name)
        return None
    
    #extract date    
    date_str = radar.time['units'][-20:]
    dt       = datetime.strptime(date_str, '%Y-%m-%dT%H:%M:%SZ')    

    #fix field names
    radar.add_field(FIELDN['dbzh'], radar.fields.pop('reflectivity'))
    radar.add_field(FIELDN['zdr'], radar.fields.pop('differential_reflectivity'))
    radar.add_field(FIELDN['phi'], radar.fields.pop('differential_phase'))
    radar.add_field(FIELDN['kdp'], radar.fields.pop('specific_differential_phase'))
    radar.add_field(FIELDN['rhv'], radar.fields.pop('cross_correlation_ratio'))
    try:
        radar.add_field(FIELDN['ncp'], radar.fields.pop('normalized_coherent_power'))
    except:
        pass
    
    #add cbb data
    try:
        radar.add_field(FIELDN['cbb'], cbb_meta, replace_existing=True)
    except:
        #recalculate cbb, radar sampling must have changes
        cbb_meta = calc_beam_blocking(radar_file_name, SRTM_FFN)
        radar.add_field(FIELDN['cbb'], cbb_meta, replace_existing=True)
    
    ###########################################################
    # Filtering
    ###########################################################
    
    #rhohv gatefilter
    gatefilter = pyart.filters.GateFilter(radar)
    gatefilter.exclude_below(FIELDN['rhv'], 0.7)
    
    #rhohv texture filtering
    #gatefilter = pyart.filters.moment_and_texture_based_gate_filter(
    
    ###########################################################
    # Correction
    ###########################################################
    
    #build temp information
    height, temperature, snr = cpol_prc.radar_codes.snr_and_sounding(radar, SONDE_FFN, refl_field_name=FIELDN['dbzh'], 
                                                                     temp_field_name = 'temp') #temp from radiosonde nc
    radar.add_field(FIELDN['temp'], temperature, replace_existing=True)
    radar.add_field(FIELDN['alt'], height, replace_existing=True)
    radar.add_field(FIELDN['snr'], snr, replace_existing=True)
    
    #add NCP if it doesn't exist
    try:
        radar.fields[FIELDN['ncp']]
        fake_ncp = False
    except KeyError:
        # Creating a fake NCP field.
        ncp = pyart.config.get_metadata('normalized_coherent_power')
        emr2 = np.zeros_like(snr['data'])
        emr2[snr['data'] > 7.5] = 1
        ncp['data'] = emr2
        ncp['description'] = "THIS FIELD IS FAKE. SHOULD BE REMOVED!"
        radar.add_field(FIELDN['ncp'], ncp)
        fake_ncp = True
    
    #RHOHV Noise correct
    rho_corr = cpol_prc.radar_codes.correct_rhohv(radar, rhohv_name=FIELDN['rhv'], snr_name=FIELDN['snr'])
    radar.add_field_like(FIELDN['rhv'], FIELDN['rhv_corr'], rho_corr, replace_existing=True)
    
    #ZDR Noise Correct
    corr_zdr = cpol_prc.radar_codes.correct_zdr(radar, zdr_name=FIELDN['zdr'], snr_name=FIELDN['snr'])
    radar.add_field_like(FIELDN['zdr'], FIELDN['zdr_corr'], corr_zdr, replace_existing=True)
    
    #unfold phidp
    phi_unfold = cpol_prc.phase.unfold_raw_phidp(radar, refl_field=FIELDN['dbzh'], ncp_field=FIELDN['ncp'], 
                                                 rhv_field=FIELDN['rhv_corr'], phi_name=FIELDN['phi'])
    radar.add_field(FIELDN['phi_unfold'], phi_unfold, replace_existing=True)

    #recalculate phidp
    phimeta, kdpmeta = cpol_prc.phase.phidp_bringi(radar, gatefilter, refl_field=FIELDN['dbzh'], ncp_name=FIELDN['ncp'], 
                                                   rhohv_name=FIELDN['rhv_corr'], unfold_phidp_name=FIELDN['phi_unfold'])
    radar.add_field(FIELDN['phi_bringi'], phimeta, replace_existing=True)
    radar.add_field(FIELDN['kdp_bringi'], kdpmeta, replace_existing=True)
    radar.fields[FIELDN['phi_bringi']]['long_name'] = "corrected_differential_phase"
    radar.fields[FIELDN['kdp_bringi']]['long_name'] = "corrected_specific_differential_phase"

    ###########################################################
    # Attenuation
    ###########################################################
    
    #ZH attenuation correction
    atten_spec, zh_corr = cpol_prc.attenuation.correct_attenuation_zh_pyart(radar, refl_field=FIELDN['dbzh'], ncp_field=FIELDN['ncp'], 
                                                                            rhv_field=FIELDN['rhv_corr'], phidp_field=FIELDN['kdp_bringi'])
    radar.add_field(FIELDN['dbzh_corr'], zh_corr, replace_existing=True)
    radar.add_field(FIELDN['a_dbz'], atten_spec, replace_existing=True)    
    
    #ZDR attenuation correction
    atten_spec_zdr, zdr_corr = cpol_prc.attenuation.correct_attenuation_zdr(radar, zdr_name=FIELDN['zdr_corr'], kdp_name=FIELDN['kdp_bringi'], 
                                                                            alpha=0.016)
    radar.add_field_like(FIELDN['zdr'], FIELDN['zdr_corr'], zdr_corr, replace_existing=True)
    radar.add_field(FIELDN['a_zdr'], atten_spec_zdr,
                    replace_existing=True)
    
    ###########################################################
    # Apply filter
    ###########################################################
    
    #apply rhohv filter
    radar.fields[FIELDN['dbzh_corr']]['data']   = cpol_prc.filtering.filter_hardcoding(radar.fields[FIELDN['dbzh_corr']]['data'], gatefilter)
    radar.fields[FIELDN['zdr_corr']]['data']   = cpol_prc.filtering.filter_hardcoding(radar.fields[FIELDN['zdr_corr']]['data'], gatefilter)
    radar.fields[FIELDN['kdp_bringi']]['data'] = cpol_prc.filtering.filter_hardcoding(radar.fields[FIELDN['kdp_bringi']]['data'], gatefilter)
    radar.fields[FIELDN['rhv_corr']]['data']      = cpol_prc.filtering.filter_hardcoding(radar.fields[FIELDN['rhv_corr']]['data'], gatefilter)
    
    ###########################################################
    # Classifications
    ###########################################################
    
    #CSU HCA
    hydro_class = cpol_prc.hydrometeors.hydrometeor_classification(radar, refl_name=FIELDN['dbzh_corr'], zdr_name=FIELDN['zdr_corr'], 
                                                                   kdp_name=FIELDN['kdp_bringi'], rhohv_name=FIELDN['rhv_corr'], 
                                                                   height_name=FIELDN['alt'], temperature_name=FIELDN['temp'])
    radar.add_field(FIELDN['hca'], hydro_class, replace_existing=True)    
    
    #HSDA
    hsda_meta = hsda.main(radar, SONDE_FFN, FIELDN, HCA_HAIL_IDX, DZDR)
    radar.add_field(FIELDN['hsda'], hsda_meta, replace_existing=True) 
    
    #HDR
    hdr_meta = hdr.main(radar,FIELDN)
    radar.add_field(FIELDN['hdr'], hdr_meta, replace_existing=True)
    
    ###########################################################
    # CFradial output
    ###########################################################
    
    # Removing fake and useless fields.
    if fake_ncp:
        radar.fields.pop(FIELDN['ncp'])
    
    #write radar object to file
    out_fn  = '_'.join([radar_name, dt.strftime('%Y%m%d_%H%M%S'), 'processed']) + '.nc'
    out_ffn = '/'.join([out_path, out_fn])
    try:
        os.remove(out_ffn)
    except OSError:
        pass  
    pyart.io.write_cfradial(out_ffn, radar)
    
    print('completed volume ' + out_ffn)
    
    ###########################################################
    # Gridded Processing and Output
    ###########################################################
    
    out_fn  = '_'.join([radar_name, dt.strftime('%Y%m%d_%H%M%S'), 'meshgrids']) + '.nc'
    out_ffn = '/'.join([out_path,out_fn])
    try:
        os.remove(out_ffn)
    except OSError:
        pass
    
    #genreate grid object
    grid = pyart.map.grid_from_radars(
        radar,
        grid_shape = GRID_SHAPE,
        grid_limits = GRID_LIMITS,
        roi_func='constant', constant_roi = GRID_ROI)
    #MESH
    mesh.main(grid, FIELDN, out_ffn, SONDE_FFN)
    
    print('completed grid ' + out_ffn)
    
    return None


def main():
    
    #radar in/out data
    data_in_path   = '/'.join([VOL_PATH, RADAR_FOLDER])
    data_out_path  = '/'.join([OUT_PATH, RADAR_FOLDER])
    if not os.path.exists(data_out_path):
            os.makedirs(data_out_path)

    #index vol files
    vol_filelist = sorted(glob.glob(data_in_path + '/*.mdv'))
    #if no mdv, check for h5
    if not vol_filelist:
        vol_filelist = sorted(glob.glob(data_in_path + '/*.h5'))

    #calculate beam blocking using first radar file
    cbb_meta     = calc_beam_blocking(vol_filelist[0], SRTM_FFN)

    # Cutting the file list into smaller chunks. (The multiprocessing.Pool instance
    # is freed from memory, at each iteration of the main for loop).
    chunked_list = chunks(vol_filelist, NCPU)
    i            = 0
    n_files      = len(vol_filelist)
    #loop through chunks
    for one_slice in chunked_list:
        args_list = [(onefile, data_out_path, cbb_meta) for onefile in one_slice]
        with Pool(NCPU) as pool:
            pool.starmap(worker, args_list)
            #update user
            i += NCPU
            print('processed: ' + str(round(i/n_files*100,2)))

    
if __name__ == '__main__':
    """
    Global vars
    """
    #setup vars
    LOG_PATH     = '/home/548/jss548/dev/tmp/logs/pyhail'
    #hsda vars
    HCA_HAIL_IDX = [9] #list of hail classe(s) indices in HCA
    #grid
    GRID_SHAPE  = (41, 301, 301)
    GRID_LIMITS = ((0, 20000), (-150000.0, 150000.0), (-150000.0, 150000.0))
    GRID_ROI    = 2000
    #paths
    SRTM_FFN         = '/g/data1a/kl02/jss548/hail-research/srtm/srtm_67_18_Brisbane/srtm_67_18.tif'
    #field names (used to map to radar object fields)
    FIELDN      = {'dbzh':'DBZH',
                   'dbzh_corr':'DBZH_CORR',
                   'zdr':'ZDR',
                   'zdr_corr':'ZDR_CORR',
                   'phi':'PHIDP',
                   'phi_unfold':'PHI_UNF',
                   'phi_bringi':'PHIDP_BRINGI',
                   'kdp':'KDP',
                   'kdp_bringi':'KDP_BRINGI',
                   'rhv':'RHOHV',
                   'ncp':'NCP',
                   'a_dbz':'SPEC_ATT_REFL',
                   'a_zdr':'SPEC_ATT_DIFF',
                   'rhv_corr':'RHOHV_CORR',
                   'temp':'TEMPERATURE',
                   'alt':'HEIGHT',
                   'snr':'SNR',
                   'cbb':'CBB',
                   'hca':'HCA',
                   'hail_ke':'HAIL_KE',
                   'shi':'SHI',
                   'posh':'POSH',
                   'mesh':'MESH',
                   'hdr':'HDR',
                   'hsda': 'HSDA'}
    
    # Parse arguments
    parser_description = "Hail retrievals from dual pol radar data"
    parser = argparse.ArgumentParser(description=parser_description)
    parser.add_argument(
        '-j',
        '--cpu',
        dest='ncpu',
        default=16,
        type=int,
        help='Number of process')
    parser.add_argument(
        '-vp',
        '--vol_path',
        dest='vol_path',
        default=None,
        type=str,
        help='root path of radar data to process',
        required=True)
    parser.add_argument(
        '-rf',
        '--radar_folder',
        dest='radar_folder',
        default=None,
        type=str,
        help='radar folder to process',
        required=True)
    parser.add_argument(
        '-s',
        '--sonde_ffn',
        dest='sonde_ffn',
        default=None,
        type=str,
        help='sonde fill file name',
        required=True)
    parser.add_argument(
        '-o',
        '--out_path',
        dest='out_path',
        default=None,
        type=str,
        help='output path',
        required=True)    
    parser.add_argument(
        '-z',
        dest='dzdr',
        default=None,
        type=int,
        help='zdr offset',
        required=True)
    args = parser.parse_args()
    NCPU         = args.ncpu
    VOL_PATH     = args.vol_path
    RADAR_FOLDER = args.radar_folder
    OUT_PATH     = args.out_path
    SONDE_FFN    = args.sonde_ffn
    DZDR         = args.dzdr
    
    # # Creating the general log file.
    # now_dt            = datetime.now()
    # log_fn            = '_'.join([now_dt.strftime('%Y%m%d_%H%M%S'), RADAR_FOLDER, 'pyhail.log'])
    # log_ffn           = os.path.join(LOG_PATH, log_fn)
    # logging.basicConfig(
    #     level    = logging.DEBUG,
    #     format   = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    #     filename = log_ffn,
    #     filemode = 'w+')
    # logger = logging.getLogger(__name__)

    
    with warnings.catch_warnings():
        # Just ignoring warning messages.
        warnings.simplefilter("ignore")
        main()