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
from skimage.morphology import remove_small_objects, remove_small_holes, binary_erosion
from skimage.filters import threshold_otsu
from skimage.measure import find_contours, label
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from scipy import ndimage as ndi

import requests
from PIL import Image, UnidentifiedImageError
from io import BytesIO

import wsi_annotations_kit.wsi_annotations_kit as wak
from shapely.geometry import Polygon, box, Point, MultiPoint, shape
from shapely.validation import make_valid
from shapely.ops import unary_union, voronoi_diagram
from skimage.draw import polygon

import rasterio.features

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

        self.model_path = Path.home() / ".deepcell/models"

        # loading model
        model_weights = self.model_path / 'NuclearSegmentation'
        model_weights = tf.keras.models.load_model(model_weights)

        self.model = NuclearSegmentation(model = model_weights)

        self.gc = gc
        self.image_id = image_id
        self.user_token = user_token

    def predict(self,region_coords:list, frame_index:int):

        # This expects an input image with channels XY (grayscale)
        # Step 1: Expanding image dimensions to expected rank (4)
        image = self.get_image_region(region_coords,frame_index)
        if not image is None:    
            # Expected input is one batch of 2D images    
            image = image[:,:,None]

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
    def __init__(self, n_frames, image_id, gc, user_token):
        
        self.n_frames = n_frames
        self.image_id = image_id
        self.gc = gc
        self.user_token = user_token
        self.cyto_pixels = 5

        image_meta = self.gc.get(f'/item/{self.image_id}/tiles')
        self.image_bbox = box(0,0,image_meta['sizeX'],image_meta['sizeY'])

        self.annotations = wak.Annotation()

    def get_image_region(self,coords_list:list):
        
        coords_list = [round(i) for i in coords_list]
        try:
            image_region = np.zeros((round(coords_list[3]-coords_list[1]),round(coords_list[2]-coords_list[0]),int(self.n_frames)))
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
        poly_bbox = [round(i) for i in list(nuc_poly.bounds)]
        poly_mask = np.zeros((int(poly_bbox[3]-poly_bbox[1]),int(poly_bbox[2]-poly_bbox[0])))

        poly_exterior = list(nuc_poly.exterior.coords)
        # x = cols, y = rows
        x_coords = [int(i[0]-poly_bbox[0]) for i in poly_exterior]
        y_coords = [int(i[1]-poly_bbox[1]) for i in poly_exterior]

        rows, cols = polygon(r = y_coords, c = x_coords, shape = (np.shape(poly_mask)[0],np.shape(poly_mask)[1]))
        poly_mask[rows,cols] = 1

        return poly_mask      

    def voronoi_process(self,pre_mask):
        """
        Generate a voronoi diagram from peaks within the clumped nucleus segmentation
        """
        # Finding peaks within mask:
        distance = ndi.distance_transform_edt(pre_mask)

        coords = peak_local_max(distance, num_peaks=1+np.sum(pre_mask)//1000)

        voronoi_islands_mask = np.zeros(distance.shape,dtype=bool)
        voronoi_islands_mask[tuple(coords.T)] = True
        voronoi_islands_mask = np.uint8(voronoi_islands_mask)

        geom_list = []
        for c,v in rasterio.features.shapes(voronoi_islands_mask, connectivity=8):
            geom_list.append(list(shape(c).centroid.coords))
        
        nuc_voronoi = voronoi_diagram(MultiPoint(geom_list), envelope= box(0,0,pre_mask.shape[1],pre_mask.shape[0]))
        new_nuc_mask = np.zeros_like(pre_mask).astype(np.uint8)
        for new_nuc in nuc_voronoi.geoms:
            if new_nuc.geom_type=='Polygon' and new_nuc.is_valid and new_nuc.area>800:
                try:
                    nuc_mask = rasterio.features.rasterize([new_nuc.buffer(-1)],out_shape = new_nuc_mask.shape)
                    nuc_mask = np.uint8(np.bitwise_and(nuc_mask,pre_mask))

                    new_nuc_mask += nuc_mask
                except ValueError:
                    print(f'ValueError encountered with object: {new_nuc.area}')

        return new_nuc_mask

    def post_process_mask(self, nuc_mask):
        """
        Post-process individual nucleus mask
        """

        sub_mask = nuc_mask>0
        sub_mask = remove_small_objects(sub_mask,10) if np.sum(sub_mask)>1 else sub_mask

        # Checking for clumps:
        if np.sum(sub_mask)>2000:
            
            new_nuc_mask = self.voronoi_process(sub_mask)

            return new_nuc_mask
        else:
            return sub_mask

    def get_features(self, coords_list:list):
        """
        Finding features for masked image region
        """
        nuc_poly = Polygon(coords_list).buffer(self.cyto_pixels+1)
        nuc_poly = nuc_poly.intersection(self.image_bbox)
        nuc_bbox = list(nuc_poly.bounds)

        nuc_mask = self.make_mask(nuc_poly)
        nuc_mask = np.pad(nuc_mask,(1,1),constant_values=0)
        nuc_mask = np.squeeze(self.post_process_mask(nuc_mask))

        # multi-frame image array
        nuc_image = self.get_image_region(nuc_bbox)
        nuc_image = np.pad(nuc_image,(1,1),constant_values=0)[:,:,1:-1]

        labeled_nuc_mask = label(nuc_mask)
        for i in np.unique(labeled_nuc_mask).tolist()[1:]:
            masked_image_pixels = nuc_image * (np.uint8(labeled_nuc_mask==i))[:,:,None]

            mean_vals = np.nanmean(masked_image_pixels,axis= (0,1))
            std_vals = np.nanstd(masked_image_pixels,axis = (0,1))

            feature_dict = {
                'Channel Means': [float(i) if not np.isnan(i) else 0 for i in mean_vals.tolist()],
                'Channel Stds': [float(i) if not np.isnan(i) else 0 for i in std_vals.tolist()],
                'Area': np.sum(labeled_nuc_mask==i)
            }

            for c,v in rasterio.features.shapes(labeled_nuc_mask==i,connectivity=8):
                
                nuc_poly = shape(c)
                bbox = list(nuc_poly.bounds)

                self.annotations.add_shape(
                    poly = nuc_poly,
                    box_crs = [nuc_bbox[0]+bbox[0],nuc_bbox[1]+bbox[1]],
                    structure = 'CODEX Nuclei',
                    properties = feature_dict
                )

        
def get_tissue_mask(gc,token,image_item_id):
    """
    Extract tissue mask and use to filter out false positive cell segmentations outside of the main tissue
    """
    image_metadata = gc.get(f'/item/{image_item_id}/tiles')

    thumb_frame_list = []
    for f in range(len(image_metadata['frames'])):
        thumb = np.array(
            Image.open(
                BytesIO(
                    requests.get(
                        f'{gc.urlBase}/item/{image_item_id}/tiles/thumbnail?frame={f}&token={token}'
                    ).content
                )
            )
        )
        thumb_frame_list.append(np.max(thumb,axis=-1)[:,:,None])
    
    thumb_array = np.concatenate(tuple(thumb_frame_list),axis=-1)
    thumbX, thumbY = np.shape(thumb_array)[1], np.shape(thumb_array)[0]
    scale_x = image_metadata['sizeX']/thumbX
    scale_y = image_metadata['sizeY']/thumbY

    # Mean of all channels/frames to make grayscale mask
    gray_mask = np.squeeze(np.mean(thumb_array,axis=-1))

    threshold_val = threshold_otsu(gray_mask)
    tissue_mask = gray_mask <= threshold_val

    tissue_mask = remove_small_holes(tissue_mask,area_threshold=150)

    print(f'shape of tissue mask {np.shape(tissue_mask)}')
    print(f'sum of tissue_mask: {np.sum(tissue_mask)}')
    labeled_mask = label(tissue_mask)
    tissue_pieces = np.unique(labeled_mask).tolist()

    tissue_shape_list = []
    for piece in tissue_pieces[1:]:
        tissue_contours = find_contours(labeled_mask==piece)

        for contour in tissue_contours:

            poly_list = [(i[1]*scale_x,i[0]*scale_y) for i in contour]
            if len(poly_list)>2:
                obj_polygon = Polygon(poly_list)

                if not obj_polygon.is_valid:
                    made_valid = make_valid(obj_polygon)

                    if made_valid.geom_type == 'Polygon':
                        tissue_shape_list.append(made_valid)
                    elif made_valid.geom_type in ['MultiPolygon','GeometryCollection']:
                        for g in made_valid.geoms:
                            if g.geom_type=='Polygon':
                                tissue_shape_list.append(g)
                
                else:
                    tissue_shape_list.append(obj_polygon)

    # Merging shapes together to remove holes
    merged_tissue = unary_union(tissue_shape_list)
    if merged_tissue.geom_type == 'Polygon':
        merged_tissue = [merged_tissue]
    elif merged_tissue.geom_type in ['MultiPolygon','GeometryCollection']:
        merged_tissue = merged_tissue.geoms

    # Creating mask from the exterior coordinates
    #wsi_tissue_mask = np.zeros((image_metadata['sizeY'],image_metadata['sizeX']))
    """
    for t in merged_tissue:

        poly_exterior = list(t.exterior.coords)
        # x = cols, y = rows
        x_coords = [int(i[0]) for i in poly_exterior]
        y_coords = [int(i[1]) for i in poly_exterior]

        rows, cols = polygon(r = y_coords, c = x_coords, shape = (image_metadata['sizeY'],image_metadata['sizeX']))
        wsi_tissue_mask[rows,cols] = 1

    return wsi_tissue_mask
    """
    return merged_tissue
    

def make_patch_filter(intersect_regions, test_patch):
    """
    Making a mask just for a single patch and it's intersecting regions
    """

    test_patch_bounds = list(test_patch.bounds)
    intersect_mask = np.zeros((int(test_patch_bounds[3]-test_patch_bounds[1]),int(test_patch_bounds[2]-test_patch_bounds[0])))
    for t in intersect_regions:

        poly_exterior = list(t.exterior.coords)
        # x = cols, y = rows
        x_coords = [int(i[0]-test_patch_bounds[0]) for i in poly_exterior]
        y_coords = [int(i[1]-test_patch_bounds[1]) for i in poly_exterior]

        rows, cols = polygon(r = y_coords, c = x_coords, shape = (np.shape(intersect_mask)[0],np.shape(intersect_mask)[1]))
        intersect_mask[rows,cols] = 1

    return intersect_mask
    




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

    if args.input_region == [-1,-1,-1,-1]:
        args.input_region = [0,0,image_tiles_info['sizeX'],image_tiles_info['sizeY']]

    patch_annotations = wak.AnnotationPatches()
    patch_annotations.define_patches(
        region_crs = args.input_region[0:2],
        height = args.input_region[3],
        width = args.input_region[2],
        patch_height = args.patch_size,
        patch_width = args.patch_size,
        overlap_pct = 0.1 
    )

    batch_size = 16

    print(f'Total # of patches: {len(patch_annotations.patch_list)}')
    print(f'Number of batches with batch size: {batch_size}: {ceil(len(patch_annotations.patch_list)/batch_size)}')
    patch_annotations = iter(patch_annotations)

    # Get tissue mask for the whole slide and use that to mask predictions at the end
    tissue_mask = get_tissue_mask(gc,args.girderToken,image_id)

    # Total number of iterations needed with batch_size
    batch_number = ceil(len(patch_annotations.patch_list)/batch_size)

    for batch in range(batch_number):
        if not (batch * batch_size)>=len(patch_annotations.patch_list):

            patch_batch = []
            for x in range(batch_size):
                # Getting the next patch regions
                new_patch = next(patch_annotations)
                print(f'On patch: {patch_annotations.patch_idx} of {len(patch_annotations.patch_list)}')
                next_region = [new_patch.left, new_patch.top, new_patch.right,new_patch.bottom]

                patch_box = box(new_patch.left,new_patch.top,new_patch.right,new_patch.bottom)

                if any([patch_box.intersects(i) for i in tissue_mask]):
                    patch_batch.append(next_region)

        else:
            patch_batch = []
            print(f'Remainder patches: {len(patch_annotations.patch_list)-((batch-1)*batch_size)}')
            for x in range(len(patch_annotations.patch_list)-((batch-1)*batch_size)):

                # Getting the next patch regions
                new_patch = next(patch_annotations)
                print(f'On patch: {patch_annotations.patch_idx} of {len(patch_annotations.patch_list)}')
                next_region = [new_patch.left, new_patch.top, new_patch.right,new_patch.bottom]

                patch_box = box(new_patch.left,new_patch.top,new_patch.right,new_patch.bottom)

                if any([patch_box.intersects(i) for i in tissue_mask]):
                    patch_batch.append(next_region)

        region_annotations = cell_finder.predict(patch_batch,args.nuclei_frame)

        for y in range(batch_size):

            y_patch = patch_batch[y]
            patch_box = box(y_patch.left,y_patch.top,y_patch.right,y_patch.bottom)

            y_annotations = region_annotations[y,:,:,:]

            # Creating the region filter
            region_filter = make_patch_filter(
                intersect_regions = [i for i in tissue_mask if patch_box.intersects(i)],
                test_patch = patch_box
            )

            # Getting the same region from the tissue mask
            y_annotations = y_annotations[:,:,0] * region_filter
            y_annotations = label(y_annotations)[:,:,None]

            patch_annotations.add_patch_mask(
                mask = y_annotations,
                patch_obj = y_patch,
                mask_type = 'one-hot-labeled',
                structure = ['CODEX Nuclei']
            )



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
            image_id=image_id,
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

                feature_extractor.get_features(nuc_coords)
                
        all_nuc_annotations = wak.Histomics(feature_extractor.annotations).json

        print(type(all_nuc_annotations))
        if isinstance(all_nuc_annotations,list):
            print(len(all_nuc_annotations))
            if len(all_nuc_annotations)>0:
                print(type(all_nuc_annotations[0]))
                if isinstance(all_nuc_annotations[0],dict):
                    print(list(all_nuc_annotations[0].keys()))
                    if 'annotation' in all_nuc_annotations[0]:
                        if 'elements' in all_nuc_annotations[0]['annotation']:
                            print(f'Number of nuclei: {len(all_nuc_annotations[0]["annotation"]["elements"])}')
                        else:
                            print('elements not in all_nuc_annotations[0]["annotation"]')
                    else:
                        print('annotation not in all_nuc_annotations[0]')
                else:
                    print('Not a dictionary')
            else:
                print('No annotations present')
        elif isinstance(all_nuc_annotations,dict):
            print(list(all_nuc_annotations.keys()))
            if 'annotation' in list(all_nuc_annotations.keys()):
                if 'elements' in list(all_nuc_annotations['annotation'].keys()):
                    print(f'Number of nuclei: {len(all_nuc_annotations["annotation"]["elements"])}')
                else:
                    print('elements not in all_nuc_annotations["annotation"]')
            else:
                print('annotation not in all_nuc_annotations')
        else:
            print('all_nuc_annotations not a list or dict')

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

