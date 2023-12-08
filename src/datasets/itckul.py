import os
import sys
import glob
import torch
import shutil
import logging
import pandas as pd
from src.datasets import BaseDataset
from src.data import Data, Batch
from src.datasets.itckul_config import *
from torch_geometric.data import extract_zip
from src.utils import available_cpu_count, starmap_with_kwargs, \
    rodrigues_rotation_matrix, to_float_rgb
from src.transforms import RoomPosition
import laspy


DIR = osp.dirname(osp.realpath(__file__))
log = logging.getLogger(__name__)


__all__ = ['ITCKUL']


########################################################################
#                                 Utils                                #
########################################################################

def read_itckul_building(
        input_dir, xyz=True, rgb=True, semantic=True, instance=True,
        verbose=True, processes=-1):
    """Read all ITCKUL object-wise annotations in a given building directory.
    All building-wise data from one epoch are accumulated into a single cloud.

    :param building_dir: str
        Absolute path to the Area directory, eg: '/some/path/ITC_BUILDING'
    :param xyz: bool
        Whether XYZ coordinates should be saved in the output Data.pos
    :param rgb: bool
        Whether RGB colors should be saved in the output Data.rgb
    :param semantic: bool
        Whether semantic labels should be saved in the output Data.y
    :param instance: bool
        Whether instance labels should be saved in the output Data.y
    :param verbose: bool
        Verbosity
    :param processes: int
        Number of processes to use when reading rooms. `processes < 1`
        will use all CPUs available
    :return:
        Batch of accumulated points clouds
    """
    # List the object-wise las files in subfolders ## TODO: temporary, only KUL
    dir_split = input_dir.split('/')
    epoch_dir = dir_split[-1]
    building_dir = EPOCHS[epoch_dir]    
    base_dir = '/'.join(dir_split[0:-1])
    search_dir = osp.join(base_dir, building_dir, epoch_dir, 'PC*')
    directories = sorted([x for x in glob.glob(search_dir) if osp.isdir(x)])

    # Read all epochs of the Building and concatenate point clouds in a Batch (TODO: check if this makes sense!)
    print('Directories', directories) #debug
    #print('building dir', building_dir, 'Search dir:', search_dir) #debug
    processes = available_cpu_count() if processes < 1 else processes
    args_iter = [[r] for r in directories]
    kwargs_iter = {
        'xyz': xyz, 'rgb': rgb, 'semantic': semantic, 'instance': instance,
        'verbose': verbose}
    batch = Batch.from_data_list(starmap_with_kwargs(
        read_itckul_epoch, args_iter, kwargs_iter, processes=processes))

    # Convert from Batch to Data
    data_dict = batch.to_dict()
    del data_dict['batch']
    del data_dict['ptr']
    data = Data(**data_dict)

    return data


def read_itckul_epoch(
        epoch_dir, xyz=True, rgb=True, semantic=True, instance=True,
        verbose=True):
    """Read all object-wise annotations in a given epoch directory.

    :param epoch_dir: str
        Absolute path to the epoch directory, eg:
        '/some/path/ITC_BUILDING/2019/'
    :param xyz: bool
        Whether XYZ coordinates should be saved in the output `Data.pos`
    :param rgb: bool
        Whether RGB colors should be saved in the output `Data.rgb`
    :param semantic: bool
        Whether semantic labels should be saved in the output `Data.y`
    :param instance: bool
        Whether instance labels should be saved in the output `Data.y`
    :param verbose: bool
        Verbosity
    :return: Data
    """
    if verbose:
        log.debug(f"Reading epoch: {epoch_dir}")

    # Initialize accumulators for xyz, RGB, semantic label and instance
    # label
    xyz_list = [] if xyz else None
    rgb_list = [] if rgb else None
    y_list = [] if semantic else None
    o_list = [] if instance else None

    lasfiles = sorted(glob.glob(osp.join(epoch_dir, '*.las')))
    if(len(lasfiles) == 0):
    	log.error(f"Error: {epoch_dir}, no las files")
    	sys.exit(1)
    lasfilename = lasfiles[0] # assume only one las file, TODO

    # List the object-wise annotation files in the epoch
    with laspy.open(lasfilename) as fh:
        las = fh.read()
        tmp = las.header.vlrs[0]
        #print('Points from Header:', fh.header.point_count, 'Extra dimensions:', tmp.extra_bytes_structs)
        # NOTE: extra dimensions are accessible through their field names
        o_list= np.array(las.object_labels.copy(), dtype='int64') if instance else None
        y_list= np.array(las.segmentation_labels.copy(), dtype='int64') if semantic else None
        if xyz:
            xyz_list.append(np.ascontiguousarray(las.xyz, dtype='float32'))
        if rgb:
            rgbVal = np.ascontiguousarray([ las.red.copy(), las.green.copy(), las.blue.copy() ])
            rgbVal = rgbVal.transpose()
            rgbVal >>= 8  # 8 bit shift to right to correct color values from LAS
            rgb_list.append( np.array(rgbVal, dtype='int16'))
            # TypeError: can't convert np.ndarray of type numpy.uint16. The only supported types are: float64, float32, float16, complex64, complex128, int64, int32, int16, int8, uint8, and bool.


    # Concatenate and convert to torch
    xyz_data = torch.from_numpy(np.concatenate(xyz_list, 0)) if xyz else None
    rgb_data = to_float_rgb(torch.from_numpy(np.concatenate(rgb_list, 0))) if rgb else None
    y_data = torch.from_numpy(y_list) if semantic else None
    o_data = torch.from_numpy(o_list) if instance else None

    y_data = y_data.clamp(max=ITCKUL_NUM_CLASSES)
    print("sizes:", xyz_data.size(), rgb_data.size(), y_data.size(), o_data.size()) 
    # Store into a Data object
    data = Data(pos=xyz_data, rgb=rgb_data, y=y_data, o=o_data)
    return data


########################################################################
#                               ITCKUL                               #
########################################################################

class ITCKUL(BaseDataset):
    """ dataset, for Area-wise prediction.

    Parameters
    ----------
    root : `str`
        Root directory where the dataset should be saved.
    fold : `int`
        Integer in [1, ..., 6] indicating the Test Area
    stage : {'train', 'val', 'test', 'trainval'}, optional
    transform : `callable`, optional
        transform function operating on data.
    pre_transform : `callable`, optional
        pre_transform function operating on data.
    pre_filter : `callable`, optional
        pre_filter function operating on data.
    on_device_transform: `callable`, optional
        on_device_transform function operating on data, in the
        'on_after_batch_transfer' hook. This is where GPU-based
        augmentations should be, as well as any Transform you do not
        want to run in CPU-based DataLoaders
    """

    _form_url = FORM_URL
    _zip_name = ZIP_NAME
    _unzip_name = UNZIP_NAME

    def __init__(self, *args, fold=5, **kwargs):
        self.fold = fold
        super().__init__(*args, val_mixed_in_train=True, test_mixed_in_val=True, **kwargs)

    @property
    def class_names(self):
        """List of string names for dataset classes. This list may be
        one-item larger than `self.num_classes` if the last label
        corresponds to 'unlabelled' or 'ignored' indices, indicated as
        `-1` in the dataset labels.
        """
        return CLASS_NAMES

    @property
    def num_classes(self):
        """Number of classes in the dataset. May be one-item smaller
        than `self.class_names`, to account for the last class name
        being optionally used for 'unlabelled' or 'ignored' classes,
        indicated as `-1` in the dataset labels.
        """
        return ITCKUL_NUM_CLASSES

    @property
    def all_base_cloud_ids(self):
        """Dictionary holding lists of clouds ids, for each
        stage.

        The following structure is expected:
            `{'train': [...], 'val': [...], 'test': [...]}`
        """
        # Note! these set also the main directories for computation
        return {
            'train': TRAIN_EPOCHS,
            'val': VALIDATION_EPOCHS,
            'test': TEST_EPOCHS}

    def download_dataset(self):
        # Manually download the dataset
        if not osp.exists(osp.join(self.root, self._zip_name)):
            log.error(
                f"\nNot supported: automatic download.\n File missing:"+ osp.join(self.root, self._zip_name)+"\n Continue run...\n")
            sys.exit(1)
            
        print("skip extracting..")
        return
        # Unzip the file and rename it into the `root/raw/` directory. This
        # directory contains the raw Area folders from the zip
        extract_zip(osp.join(self.root, self._zip_name), self.root)
        shutil.rmtree(self.raw_dir)
        os.rename(osp.join(self.root, self._unzip_name), self.raw_dir)

    def read_single_raw_cloud(self, raw_cloud_path):
        """Read a single raw cloud and return a Data object, ready to
        be passed to `self.pre_transform`.
        """
        return read_itckul_building(
            raw_cloud_path, xyz=True, rgb=True, semantic=True, instance=True,
            verbose=True)

# This is optional
#    @property
#    def raw_file_structure(self):
#        return f"""
#    {self.root}/
#        └── {self._zip_name}
#        └── raw/
#            └── ITC_BUILDING
#                └── 2019
#                    └── ...
#            """

    @property
    def raw_file_names(self):
        """The file paths to find in order to skip the download."""
        area_folders = super().raw_file_names
        return area_folders

    def id_to_relative_raw_path(self, id):
        """Given a cloud id as stored in `self.cloud_ids`, return the
        path (relative to `self.raw_dir`) of the corresponding raw
        cloud.
        """
        return self.id_to_base_id(id)


