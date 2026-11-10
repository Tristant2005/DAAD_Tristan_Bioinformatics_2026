from src.segmentation_pipeline import segmentation_pipeline

# these modules rely on the detectron2 engine
from src.model_run import detectron2_instances
from src.fusion import *

from src.classic_instances import instance_regions
from src.classifier import classify_instances
from src.coco_json_functions import *

import cv2
import numpy as np


    

# go through each 5 major setps. Look at documentation pdf or hold Ctr key when left-clicking for more info
def instance_segmentation_pipeline(img_path, 
                                    predictor,
                                    classifier,
                                    max_hole_size=2000, 
                                    min_area=100, 
                                    max_distance=20, 
                                    circularity_threshold=0.8, 
                                    tangled=False,
                                    verbose=False,
                                    fusion_iou_threshold = 0.1,
                                    detectron_confidence_threshold = 0.0, 
                                    is_stitched=False,
                                    show_overlay=True):
    
    #1. Classic Segmentation
    #   function returns the mask for the root and the mask for the hairs as seperate variables
    root, hairs = segmentation_pipeline(img_path, max_hole_size=max_hole_size, min_area=min_area, max_distance=max_distance, 
                                        circularity_threshold=circularity_threshold, verbose=verbose, 
                                        is_stitched=is_stitched, show_overlay=show_overlay) 


    # #2. Separate Classic Segments by Instance
    # #   get a list of each unique instance mask of root hairs
    classic_masks = instance_regions(hairs, root, tangled=tangled)
    
    # 3. Get the Model outputs
    #   Instances is a list of all found instances in the image, including bounding box, mask, label, and label id
    model_instances = detectron2_instances(predictor, img_path, verbose=verbose)

    #4. Fusion
    fused_instances = fusion_of_masks(classic_masks, model_instances, iou_threshold=fusion_iou_threshold,
                                      detectron_confidence_threshold=detectron_confidence_threshold)

    # 5. Classify
    classified_instances = classify_instances(fused_instances, root, img_path, classifier, tangled)
    

    return classified_instances, model_instances, classic_masks

