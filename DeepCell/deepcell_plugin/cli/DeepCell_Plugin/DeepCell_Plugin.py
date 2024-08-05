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

from cellpose import models

from skimage.morphology import remove_small_objects, remove_small_holes
from skimage.filters import threshold_otsu
from skimage.measure import find_contours, label
from skimage.exposure import rescale_intensity
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
from math import pi

import rasterio.features

from typing_extensions import Union

from pathlib import Path
import tensorflow as tf

import girder_client
from tqdm import tqdm


class CellSegmenter:
    def __init__(self,gc,image_id:str, user_token:str):

        self.gc = gc
        self.image_id = image_id
        self.user_token = user_token

    def get_image_region(self,coords_list: list, frame_index: Union[int,list]):
        """
        Grabbing image region based on set of bounding box coordinates and frame 
        """
        try:
            if isinstance(frame_index,list):
                image_region = np.zeros((int(coords_list[3]-coords_list[1]),int(coords_list[2]-coords_list[0])))
                for f in frame_index:
                    image_region += np.array(
                        Image.open(
                            BytesIO(
                                requests.get(
                                    self.gc.urlBase+f'/item/{self.image_id}/tiles/region?token={self.user_token}&frame={frame_index}&left={coords_list[0]}&top={coords_list[1]}&right={coords_list[2]}&bottom={coords_list[3]}'
                                    ).content
                                )
                            )
                        )
                
                # Return mean of intersecting channels
                image_region /= len(frame_index)
                image_region = np.uint8(image_region)
            else:
                image_region = np.array(
                    Image.open(
                        BytesIO(
                            requests.get(
                                self.gc.urlBase+f'/item/{self.image_id}/tiles/region?token={self.user_token}&frame={frame_index}&left={coords_list[0]}&top={coords_list[1]}&right={coords_list[2]}&bottom={coords_list[3]}'
                                ).content
                            )
                        )
                    )
            
            return image_region
        except UnidentifiedImageError:
            print('Error found :(')
            print(self.gc.urlBase+f'/item/{self.image_id}/tiles/region?token={self.user_token}&frame={frame_index}&left={coords_list[0]}&top={coords_list[1]}&right={coords_list[2]}&bottom={coords_list[3]}')

            response = requests.get(
                self.gc.urlBase+f'/item/{self.image_id}/tiles/region?token={self.user_token}&frame={frame_index}&left={coords_list[0]}&top={coords_list[1]}&right={coords_list[2]}&bottom={coords_list[3]}'
                )
            print(f'Status Code: {response.status_code}')

            return None


class DeepCellHandler(CellSegmenter):
    """
    https://github.com/vanvalenlab/deepcell-tf
    
    Greenwald, N.F., Miller, G., Moen, E. et al. Whole-cell segmentation of tissue images with human-level performance using large-scale data annotation and deep learning. Nat Biotechnol 40, 555–565 (2022). https://doi.org/10.1038/s41587-021-01094-0
    
    """
    def __init__(self, gc, image_id: str, user_token: str):
        super().__init__(gc,image_id,user_token)
        self.model_path = Path("../.deepcell/models")

        # loading model
        model_weights = self.model_path / 'NuclearSegmentation'
        model_weights = tf.keras.models.load_model(model_weights)

        self.model = NuclearSegmentation(model = model_weights)

    def predict(self,region_patches:list, frame_index:Union[int,list]):

        # This expects an input image with channels XY (grayscale)
        non_nones = [i for i in region_patches if not i is None]

        batch_size = len(non_nones)
        patch_size = int(non_nones[0].bottom-non_nones[0].top)
        image_batch = np.zeros((batch_size,patch_size,patch_size))
        for idx,r in enumerate(non_nones):
            image = self.get_image_region([r.left, r.top, r.right, r.bottom],frame_index)
            image_batch[idx,:,:] += image

        # Image batch has to have rank=4
        image_batch = image_batch[:,:,:,None]
        if not image_batch is None:    
            # Step 2: Generate labeled image
            labeled_image = self.model.predict(image_batch)

            # Getting rid of the extra dimensions (BXYC, C will be removed)
            processed_nuclei = [labeled_image[i,:,:,:] for i in range(batch_size)]
            return processed_nuclei
        else:
            print('No image, some PIL.UnidentifiedImageError thing')
            return None

class CellPoseHandler(CellSegmenter):
    """
    https://github.com/MouseLand/cellpose

    Stringer, C., Wang, T., Michaelos, M., & Pachitariu, M. (2021). Cellpose: a generalist algorithm for cellular segmentation. Nature methods, 18(1), 100-106. 

    """
    def __init__(self, gc, image_id:str, user_token:str):
        super().__init__(gc,image_id,user_token)

        self.model = models.Cellpose(gpu = True, model_type = 'cyto3')

    def predict(self, region_patches:list, frame_index:Union[int,list]):
        
        non_nones = [i for i in region_patches if not i is None]
        batch_size = len(non_nones)
        patch_size = int(non_nones[0].bottom-non_nones[1].top)
        image_batch = np.zeros((batch_size,patch_size,patch_size))

        for idx,r in enumerate(non_nones):
            image = self.get_image_region([r.left,r.top,r.right,r.bottom],frame_index)
            image_batch[idx,:,:] += image

        # image_batch has rank 4 with dimensions BYXC
        image_batch = image_batch[:,:,:,None]
        
        pred_masks, _, _, _ = self.model.eval(image_batch, diameter = None, channels = [0,0])

        return pred_masks
        

class OtsuCellHandler(CellSegmenter):
    """
    Cell segmentation using classical image analysis
    """
    def __init__(self,gc,image_id:str,user_token:str):
        super().__init__(gc,image_id,user_token)

    def predict(self, region_patches: list, frame_index:Union[int,list]):
        
        non_nones = [i for i in region_patches if not i is None]

        processed_nuclei = []
        for idx,r in enumerate(non_nones):
            image = self.get_image_region([r.left,r.top,r.right,r.bottom])
            
            # Increasing contrast by clipping lowest 0.5% to 0 and highest 0.5% to 1
            vmin, vmax = np.percentile(image, q=(0.5, 99.5))

            clipped_data = rescale_intensity(
                image, in_range=(vmin, vmax), out_range=np.float32
            )
            image_thresh = threshold_otsu(clipped_data)
            threshed_image = clipped_data <= image_thresh

            small_removed = remove_small_objects(threshed_image, min_size = 100)
            holes_removed = ndi.binary_fill_holes(small_removed)

            processed_nuclei.append(holes_removed)

        return processed_nuclei



class FeatureExtractor:
    def __init__(self,
                 n_frames,
                 image_id,
                 gc,
                 user_token,
                 post_processor):
        
        self.n_frames = n_frames
        self.image_id = image_id
        self.gc = gc
        self.user_token = user_token
        self.cyto_pixels = 5

        self.post_processor = post_processor

        self.approx_cell_radius = 15

        self.approx_cell_area = pi*(self.approx_cell_radius**2)

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
        approx_n_cells = 1+np.sum(pre_mask)//self.approx_cell_area
        
        remaining_distance = distance.copy()
        voronoi_point_list = []
        for n in range(approx_n_cells):
            max_dist = np.max(remaining_distance)

            mean_point = np.mean(np.argwhere(remaining_distance==max_dist),axis=0)

            mask_y_min = int(np.minimum(0,mean_point[0]-self.approx_cell_radius))
            mask_x_min = int(np.minimum(0,mean_point[1]-self.approx_cell_radius))
            mask_y_max = int(np.minimum(pre_mask.shape[0],mean_point[0]+self.approx_cell_radius))
            mask_x_max = int(np.minimum(pre_mask.shape[1],mean_point[1]+self.approx_cell_radius))

            remaining_distance[mask_y_min:mask_y_max,mask_x_min:mask_x_max] = 0
            voronoi_point_list.append(Point(mean_point[1],mean_point[0]))

        nuc_voronoi = voronoi_diagram(MultiPoint(voronoi_point_list),envelope=box(0,0,pre_mask.shape[1],pre_mask.shape[0]))
        new_nuc_mask = np.zeros_like(pre_mask).astype(np.uint8)
        for new_nuc in nuc_voronoi.geoms:
            if new_nuc.geom_type=='Polygon' and new_nuc.is_valid and new_nuc.area>(0.5*self.approx_cell_area):
                try:
                    nuc_mask = rasterio.features.rasterize([new_nuc.buffer(-1)],out_shape = pre_mask.shape)
                    nuc_mask = np.uint8(np.bitwise_and(nuc_mask,pre_mask))

                    new_nuc_mask += nuc_mask
                except ValueError:
                    continue

        return new_nuc_mask

    def post_process_mask(self, nuc_mask):
        """
        Post-process individual nucleus mask
        """

        sub_mask = nuc_mask>0
        sub_mask = remove_small_objects(sub_mask,10) if np.sum(sub_mask)>1 else sub_mask

        # Checking for clumps:
        if np.sum(sub_mask)>2*self.approx_cell_area:
            
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
                'Area': float(np.sum(labeled_nuc_mask==i))
            }

            nuc_contours = find_contours(labeled_nuc_mask==i)
            nuc_polys = []
            for n in nuc_contours:
                if len(n)>3:
                    nuc_polys.append(
                        Polygon([(i[1],i[0]) for i in n])
                    )

            if len(nuc_polys)>0:
                largest_idx = np.argmax([i.area for i in nuc_polys])
                if nuc_polys[largest_idx].area > 500:
                    nuc_poly = nuc_polys[largest_idx]

                    self.annotations.add_shape(
                        poly = nuc_poly,
                        box_crs = [nuc_bbox[0],nuc_bbox[1]],
                        structure = 'CODEX Nuclei',
                        properties = feature_dict
                    )
            

        
def get_tissue_mask(gc,token,image_item_id):
    """
    Extract tissue mask and use to filter out false positive cell segmentations outside of the main tissue
    """
    image_metadata = gc.get(f'/item/{image_item_id}/tiles')
    if not 'frames' in image_metadata:
        # Grabbing the thumbnail of the image (RGB)
        thumbnail_img = Image.open(BytesIO(requests.get(f'{gc.urlBase}/item/{image_item_id}/tiles/thumbnail?token={token}').content))
        thumb_array = np.array(thumbnail_img)

    else:
        # Getting the max projection of the thumbnail
        thumb_frame_list = []
        for f in range(len(image_metadata['frames'])):
            thumb = np.array(Image.open(BytesIO(requests.get(f'{gc.urlBase}/item/{image_item_id}/tiles/thumbnail?frame={f}&token={token}').content)))
            thumb_frame_list.append(np.max(thumb,axis=-1)[:,:,None])

        thumb_array = np.concatenate(tuple(thumb_frame_list),axis=-1)

    # Getting scale factors for thumbnail image to full-size image
    thumbX, thumbY = np.shape(thumb_array)[1],np.shape(thumb_array)[0]
    scale_x = image_metadata['sizeX']/thumbX
    scale_y = image_metadata['sizeY']/thumbY

    # Mean of all channels/frames to make grayscale mask
    gray_mask = np.squeeze(np.mean(thumb_array,axis=-1))

    threshold_val = threshold_otsu(gray_mask)
    tissue_mask = gray_mask <= threshold_val

    tissue_mask = remove_small_holes(tissue_mask,area_threshold=150)

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

                    if made_valid.geom_type=='Polygon':
                        tissue_shape_list.append(made_valid)
                    elif made_valid.geom_type in ['MultiPolygon','GeometryCollection']:
                        for g in made_valid.geoms:
                            if g.geom_type=='Polygon':
                                tissue_shape_list.append(g)
                else:
                    tissue_shape_list.append(obj_polygon)

    # Merging shapes together to remove holes
    merged_tissue = unary_union(tissue_shape_list)
    if merged_tissue.geom_type=='Polygon':
        merged_tissue = [merged_tissue]
    elif merged_tissue.geom_type in ['MultiPolygon','GeometryCollection']:
        merged_tissue = merged_tissue.geoms

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
    if args.method == 'DeepCell':
        cell_finder = DeepCellHandler(
            gc,
            image_id,
            args.girderToken
        )
    elif args.method == 'Otsu':
        cell_finder = OtsuCellHandler(
            gc,
            image_id,
            args.girderToken
        )
    elif args.method == 'CellPose':
        cell_finder = CellPoseHandler(
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

        patch_batch = []
        try:
            for x in range(batch_size):
                # Getting the next patch regions
                new_patch = next(patch_annotations)
                print(f'On patch: {patch_annotations.patch_idx} of {len(patch_annotations.patch_list)}')

                patch_box = box(new_patch.left,new_patch.top,new_patch.right,new_patch.bottom)
                if any([patch_box.intersects(i) for i in tissue_mask]):
                    patch_batch.append(new_patch)

        except StopIteration:
            print('All out!')

        if len(patch_batch)>0:
            region_annotations = cell_finder.predict(patch_batch,args.nuclei_frame)

            for y_patch,y_annotations in zip(patch_batch,region_annotations):

                patch_box = box(y_patch.left,y_patch.top,y_patch.right,y_patch.bottom)

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
            user_token = args.girderToken,
            post_processor=args.post_processor
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

