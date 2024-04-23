"""

CODEX nuclei segmentation and feature extraction plugin

    - Select frame to use for nuclei segmentation
    - Iterate through tiles? (Remove border nuclei)
    - Segment nuclei using user-specified parameters
        - Threshold, minimum size, something for splitting adjacent nuclei
    - Create nuclei annotations
    - Iterate through all frames
    - Extract frame intensity statistics
        - Mean, median, standard deviation, maximum, minimum, etc.
    - Add to annotation user metadata
    - Post entire thing to slide annotations?

Find some way to leverage GPUs or something to make this more efficient.
Can annotations be appended to or just posted all at once?

"""

import os
import sys
from math import ceil, floor
import numpy as np
import json

from ctk_cli import CLIArgumentParser

sys.path.append('..')
from deepcell.applications import NuclearSegmentation
#from deepcell.utils._auth import extract_archive
from skimage.morphology import remove_small_objects
from scipy import ndimage as ndi

import requests
from PIL import Image, UnidentifiedImageError
from io import BytesIO

import wsi_annotations_kit.wsi_annotations_kit as wak
from shapely.geometry import Polygon
from skimage.draw import polygon

from pathlib import Path
import tensorflow as tf

import girder_client
import logging
import tarfile
import zipfile
from tqdm import tqdm

def extract_archive(file_path, path="."):
    """Extracts an archive if it matches tar, tar.gz, tar.bz, or zip formats.

    Args:
        file_path: Path to the archive file.
        path: Where to extract the archive file.

    Returns:
        True if a match was found and an archive extraction was completed,
        False otherwise.
    """
    logging.basicConfig(level=logging.INFO)

    file_path = os.fspath(file_path) if isinstance(file_path, os.PathLike) else file_path
    path = os.fspath(path) if isinstance(path, os.PathLike) else path

    logging.info(f'Extracting {file_path}')

    status = False

    if tarfile.is_tarfile(file_path):
        with tarfile.open(file_path) as archive:
            archive.extractall(path)
        status = True
    elif zipfile.is_zipfile(file_path):
        with zipfile.ZipFile(file_path) as archive:
            archive.extractall(path)
        status = True

    if status:
        logging.info(f'Successfully extracted {file_path} into {path}')
    else:
        logging.info(f'Failed to extract {file_path} into {path}')


class DeepCellHandler:
    def __init__(self, gc, image_id: str, user_token: str):

        # Loading model (list contains EC2 model id and athena model id)
        self.nuclear_segmentation_model_id = ["65e0c399adb89a58fea1152b","65f857bfd2f45e99a914a26c"]
        # Attempting to download the model:
        self.model = None
        self.model_path = Path.home() / ".deepcell/models"
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        for n in self.nuclear_segmentation_model_id:
            try:
                _ = gc.downloadItem(
                    itemId = n,
                    dest = self.model_path,
                    name = 'NuclearSegmentation-75.tar.gz'
                )

                # Extracting files from archive
                extract_archive(self.model_path / 'NuclearSegmentation-75.tar.gz',self.model_path)

                # loading model
                model_weights = self.model_path / 'NuclearSegmentation'
                model_weights = tf.keras.models.load_model(model_weights)

                self.model = NuclearSegmentation(model = model_weights)

                print('Using server-hosted model version')
            except girder_client.HttpError:
                print(f'File not found at: {n}')
                continue
        
        if self.model is None:
            print('File not found at provided id')
            
            # Downloading model
            # Initializing NuclearSegmentation application with default parameters
            self.model = NuclearSegmentation()

        self.gc = gc
        self.image_id = image_id
        self.user_token = user_token

    def predict(self,region_coords:list, frame_index:int):

        # This expects an input image with channels XY (grayscale)
        # Step 1: Expanding image dimensions to expected rank (4)
        image = self.get_image_region(region_coords,frame_index)
        if not image is None:        
            image = image[None,:,:,None]

            # Step 2: Generate labeled image
            labeled_image = self.model.predict(image)

            # Getting rid of the extra dimensions
            labeled_image = np.squeeze(labeled_image)
            # one-hot labeling
            processed_nuclei = labeled_image[:,:,None]

            #TODO: There should be a way to add this post processing function or modify the existing deepwatershed postprocessing function
            # Step 3: Removing small pieces
            #processed_nuclei = remove_small_objects(labeled_image>0,self.min_size)

            return processed_nuclei
        else:
            print('No image, some PIL.UnidentifiedImageError thing')
            return None

    def get_image_region(self,coords_list,frame_index):
        
        try:
            if isinstance(frame_index,list):
                image_region = np.zeros((int(coords_list[3]-coords_list[1]),int(coords_list[2]-coords_list[0])))
                for f in frame_index:
                    image_region += np.array(Image.open(BytesIO(requests.get(self.gc.urlBase+f'/item/{self.image_id}/tiles/region?token={self.user_token}&frame={frame_index}&left={coords_list[0]}&top={coords_list[1]}&right={coords_list[2]}&bottom={coords_list[3]}').content)))
                
                # Return mean of intersecting channels
                image_region /= len(frame_index)
                image_region = np.uint8(image_region)
            else:
                image_region = np.array(Image.open(BytesIO(requests.get(self.gc.urlBase+f'/item/{self.image_id}/tiles/region?token={self.user_token}&frame={frame_index}&left={coords_list[0]}&top={coords_list[1]}&right={coords_list[2]}&bottom={coords_list[3]}').content)))
            
            return image_region
        except UnidentifiedImageError:
            print('Error found :(')
            print(self.gc.urlBase+f'/item/{self.image_id}/tiles/region?token={self.user_token}&frame={frame_index}&left={coords_list[0]}&top={coords_list[1]}&right={coords_list[2]}&bottom={coords_list[3]}')

            response = requests.get(self.gc.urlBase+f'/item/{self.image_id}/tiles/region?token={self.user_token}&frame={frame_index}&left={coords_list[0]}&top={coords_list[1]}&right={coords_list[2]}&bottom={coords_list[3]}')
            print(f'Status Code: {response.status_code}')

            return None


class FeatureExtractor:
    def __init__(self, n_frames, gc, user_token):
        
        self.n_frames = n_frames
        self.gc = gc
        self.user_token = user_token
        self.cyto_pixels = 5

    def get_image_region(self,coords_list:list):
        
        try:
            image_region = np.zeros((int(coords_list[3]-coords_list[1]),int(coords_list[2]-coords_list[0]),int(self.n_frames)))
            for f in range(self.n_frames):
                image_region[:,:,f] += np.array(Image.open(BytesIO(requests.get(self.gc.urlBase+f'/item/{self.image_id}/tiles/region?token={self.user_token}&frame={f}&left={coords_list[0]}&top={coords_list[1]}&right={coords_list[2]}&bottom={coords_list[3]}').content)))
            
            image_region = np.uint8(image_region)

            return image_region
        except UnidentifiedImageError:
            print('Error found :(')

            return None
    
    def make_mask(self,nuc_poly):
        """
        Making a mask for the current polygon
        """

        poly_bbox = list(nuc_poly.bounds)
        poly_mask = np.zeros((int(poly_bbox[3]-poly_bbox[1]),int(poly_bbox[2]-poly_bbox[0])))

        poly_exterior = list(nuc_poly.exterior.coords)
        # x = cols, y = rows
        x_coords = [i[0] for i in poly_exterior]
        y_coords = [i[1] for i in poly_exterior]

        rows, cols = polygon(r = y_coords, c = x_coords, shape = (np.shape(poly_mask)[0],np.shape(poly_mask)[1]))
        poly_mask[rows,cols] = 1

        return poly_mask      

    def get_features(self, coords_list:list):
        """
        Finding features for masked image region
        """
        nuc_poly = Polygon(coords_list).buffer(self.cyto_pixels)
        nuc_bbox = list(nuc_poly.bounds)

        nuc_mask = self.make_mask(nuc_poly)
        # multi-frame image array
        nuc_image = self.get_image_region(nuc_bbox)

        masked_image_pixels = nuc_image[nuc_mask>0,:]
        print(np.shape(masked_image_pixels))
        mean_vals = np.nanmean(masked_image_pixels,axis=0)
        print(np.shape(mean_vals))

        std_vals = np.nanstd(masked_image_pixels,axis = 0)
        print(np.shape(std_vals))

        feature_dict = {
            'Channel Means': [float(i) for i in mean_vals.tolist()],
            'Channel Stds': [float(i) for i in std_vals.tolist()]
        }

        return feature_dict

        
    

def main(args):

    sys.stdout.flush()

    # Initialize girder client
    gc = girder_client.GirderClient(apiUrl = args.girderApiUrl)
    gc.setToken(args.girderToken)

    print('Input arguments:')
    for a in vars(args):
        print(f'{a}: {getattr(args,a)}')

    # Getting image information (image id)
    image_id = gc.get(f'/file/{args.input_image}')['itemId']
    image_info = gc.get(f'/item/{image_id}')
    print(f'Working on: {image_info["name"]}')

    # Checking if DeepCell API is provided
    if args.deepCellApi:
        os.environ['DEEPCELL_ACCESS_TOKEN'] = args.deepCellApi

    # Copying it over to the plugin filesystem
    image_tiles_info = gc.get(f'/item/{image_id}/tiles')
    print(f'Image has {len(image_tiles_info["frames"])} Channels!')
    print(f'Image is {image_tiles_info["sizeY"]} x {image_tiles_info["sizeX"]}')

    # Testing if nuclei frame is provided as comma separated list
    if ',' in args.nuclei_frame:
        args.nuclei_frame = [int(float(i)) for i in args.nuclei_frame.split(',')]
    else:
        args.nuclei_frame = int(float(args.nuclei_frame))

    # Creating patch iterator 
    # Initializing deepcell object
    cell_finder = DeepCellHandler(
        gc,
        image_id,
        args.girderToken
    )

    patch_annotations = wak.AnnotationPatches()
    patch_annotations.define_patches(
        region_crs = args.input_region[0:2],
        height = args.input_region[3],
        width = args.input_region[2],
        patch_height = args.patch_size,
        patch_width = args.patch_size,
        overlap_pct = 0.1 
    )

    patch_annotations = iter(patch_annotations)

    more_patches = True
    while more_patches:
        try:
            # Getting the next patch region
            new_patch = next(patch_annotations)
            print(f'On patch: {patch_annotations.patch_idx} of {len(patch_annotations.patch_list)}')
            next_region = [new_patch.left, new_patch.top, new_patch.right,new_patch.bottom]
            # Getting features and annotations within that region
            region_annotations = cell_finder.predict(next_region,args.nuclei_frame)

            patch_annotations.add_patch_mask(
                mask = region_annotations,
                patch_obj = new_patch,
                mask_type = 'one-hot-labeled',
                structure = ['CODEX Nuclei']
            )

        except StopIteration:
            more_patches = False
    
    print('--------------------------------------------------')
    print('Cleaning up annotations')
    patch_annotations.clean_patches()
    print('Annotation patches cleaned up!')

    all_nuc_annotations = wak.Histomics(patch_annotations).json

    if args.get_features:
        # Extracting channel-level statistics and adding as user metadata
        n_nuclei = len(all_nuc_annotations[0]["annotation"]["elements"])
        print(f'{n_nuclei} nuclei found!')
        print('Calculating features now!')

        feature_extractor = FeatureExtractor(
            n_frames = len(image_tiles_info["frames"]),
            gc = gc,
            user_token = args.girderToken
        )

        with tqdm(all_nuc_annotations[0]["annotation"]["elements"],total = n_nuclei) as pbar:
            for el_idx, el in enumerate(all_nuc_annotations[0]["annotation"]["elements"]):
                
                pbar.set_description(f'Working on nucleus: {el_idx}/{n_nuclei}')
                pbar.update(1)

                # X, Y, Z vertices
                nuc_coords = np.array(el['points']).tolist()
                nuc_coords = [(i[0],i[1]) for i in nuc_coords]

                nuc_features = feature_extractor.get_features(nuc_coords)
                
                el['user'] = nuc_features


    print('Posting annotations')
    # Posting annotations to item
    gc.post(f'/annotation/item/{image_id}?token={args.girderToken}',
            data = json.dumps(all_nuc_annotations),
            headers = {
                'X-HTTP-Method': 'POST',
                'Content-Type': 'application/json'
                }
            )



if __name__=='__main__':

    main(CLIArgumentParser().parse_args())

