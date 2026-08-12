# Mellanie Martin, Tristan Tucker
# July 2025, August 2026


### IMPORTS ###
from skimage.draw import polygon2mask
from skimage.measure import regionprops, label
from skimage.color import rgb2gray
from skimage import io
from shapely import Polygon, Point
import numpy as np
import pandas as pd
import json
from collections import defaultdict
import os
import statistics as st



# CLASSES = {
#     0:"root",                
#     1:"root_hair",            
#     2:"bump",   
#     3:"background_root_hair", 
#     4:"tangled_root_hair",                 
#     5:"edge_root_hair",                 
#     6:"embedded_root_hair",               
#     7:"bubble",   
#     8:"dirt"    
# }


### FUNCTIONS ###

def segmentation_to_mask(segmentation, height, width):
    mask = np.zeros((height, width), dtype=bool)
    for poly in segmentation:  # handles multi-part polygons
        pts = np.array(poly).reshape(-1, 2)      # (x, y) pairs
        pts_rc = pts[:, ::-1]                      # swap to (row, col) = (y, x)
        mask |= polygon2mask((height, width), pts_rc)
    return mask

def get_stats(categories, json_data="/home/mellanie/Desktop/Tristan_2026_code/detectron2/model2.5_data/train.json"):
    '''for bumps and embedded root hairs; get eccentricity, aspect ratio, area, and num_pixels'''

    stats = []

    print("in get_stats function")

    # Build annotation lists keyed by file_name
    segmentations_by_filename = defaultdict(list)

    # Open the json file
    with open(json_data) as f: 
        data = json.load(f)

    # data is a dist with "images" and "annotations"
    image_meta_by_filename = {
        img["file_name"]: {
            "gt_image_id":   img["id"],
            "height":        img["height"],
            "width":         img["width"],
            "ecotype":       img.get("ecotype", img["file_name"]),
        } for img in data["images"]}


    id_to_filename = {img["id"]: img["file_name"] for img in data["images"]}

    for ann in data["annotations"]:
        fname = id_to_filename.get(ann["image_id"])
        if fname:
            segmentations_by_filename[fname].append(ann)


    # so we have
    #   image_meta_by_filename[img_file_path] -> img meta
    #   segmentations_by_filename[img_file_path] -> annotations

    filenames = set(image_meta_by_filename.keys())
    for img_file_path in filenames:
        # print("img_path", img_file_path)

        height = image_meta_by_filename[img_file_path]["height"]
        width = image_meta_by_filename[img_file_path]["width"]
        ecotype = image_meta_by_filename[img_file_path]["ecotype"]

        for annotation in segmentations_by_filename[img_file_path]:

            points = annotation["segmentation"]
            category = annotation["old_category"]

            # print(category)
        
            if category not in categories: continue

            #     points = np.array(polygon["points"])
            #     height = data["imageHeight"]
            #     width = data["imageWidth"]

            mask = segmentation_to_mask(points, height, width)
            labeled = label(mask)

            #     img_num = os.path.basename(ann) # get img for intensity
            #     img_num = img_num.removesuffix(".json")
            #     nitrate_conc = {0: 128, 1: 4, 2: 16, 3: 32, 4: 64}.get(i % 5)
            #     image_path = f"/home/mellanie/Desktop/annotations/Images/20250409_N{nitrate_conc}_Plate{i}_DAG6_1000_L/{img_num}.tif"

            image = io.imread(img_file_path)
            gray_image = rgb2gray(image) if image.ndim == 3 else image

            props = regionprops(labeled, intensity_image = gray_image)


            if props:
                largest = max(props, key=lambda p: p.area)
                e = largest.eccentricity
                ar = largest.axis_major_length/props[0].axis_minor_length
                area = largest.area
                intensity = largest.intensity_mean
                aml = largest.axis_major_length
                pixels = largest.num_pixels
                stats.append({"class": category, "ecotype": ecotype, "pixels": pixels, "area": area, "eccentricity": e,
                                "intensity": intensity, "axis_major_length": aml, "aspect_ratio": ar})

              


    if stats:
        results = pd.DataFrame(stats)
        results.to_csv(f"/home/mellanie/Desktop/Tristan_2026_code/src/classifier_remake/stats_in_csv/{categories[0]}_{categories[1]}_stats.csv", index=False)


# def get_dist_from_root(data, object):

#     for polygon in data["shapes"]:
#         if polygon["label"] == "root":
#             root = Polygon(polygon["points"])
#             break  # only get the first root

#     pt_coords = object["points"][0]  # Point[0] should be base of hair...
#     pt = Point(pt_coords)
#     dist = root.distance(pt)

#     return dist

# if __name__ == "__main__":

get_stats(["embedded_root_hair", "bump"])

