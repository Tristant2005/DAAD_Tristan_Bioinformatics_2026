# Imports
from scipy.optimize import linear_sum_assignment
import numpy as np
import cv2
from matplotlib import pyplot as plt
from src.segmentation_pipeline import dilate_mask
from scipy.ndimage import binary_dilation



# These are the final classes in order
CLASSES = [
    "root",                
    "root_hair",            
    "bump",   
    "background_root_hair", 
    "tangled_root_hair",                 
    "edge_root_hair",                 
    "embedded_root_hair",               
    "bubble",   
    "dirt"    
]

# Take the instances output by the model and place them in a custom json-like format
def load_model_instances(instances: list[dict], detectron_confidence_threshold: float = 0.5) -> dict:

    # define variables
    masks = []
    scores = []
    categories = []
    boxes = []

    if instances == []:
        return {"masks": masks, "scores": scores, "classes": categories, "box": boxes}
    
    # iterate through instances from the Detectron2 model 
    for mask, score, category, box in zip(instances.pred_masks, instances.scores, instances.pred_classes, instances.pred_boxes.tensor):
        if float(score) < detectron_confidence_threshold:
            continue

        mask = mask.numpy().astype(bool)

        # keep these 2 lines only if you want to dilate mask
        # dilation_radius = 2
        # mask = dilate_mask(mask, dilation_radius)


        masks.append(mask)
        scores.append(float(score))

        # shift all indexes by one to account for the root class being slotted in
        # Basically because the model was trained without the root class, the indexing is like this:
        # 0: "root_hair",            
        # 1: "bump",   
        # 2: "background_root_hair", 
        # 3: "tangled_root_hair",                 
        # 4: "edge_root_hair",                 
        # 5: "embedded_root_hair",               
        # 6: "bubble",   
        # 7: "dirt"

        categories.append(int(category) + 1)
            
        boxes.append(box.tolist())

    return {"masks": masks, "scores": scores, "classes": categories, "box": boxes}


# Compute the IOU of the model instances and the classic instances
def compute_mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """IoU between two binary masks. Returns 0.0 if union is empty."""
    intersection = np.logical_and(mask_a, mask_b).sum()
    union        = np.logical_or(mask_a, mask_b).sum()
    return float(intersection) / float(union) if union > 0 else 0.0

# Compute IoU matrix between N classical and M DL instance masks.
def build_iou_matrix(classical_masks: list[np.ndarray], dl_masks: list[np.ndarray]) -> np.ndarray:
    """Entry [i, j] = IoU(classical_masks[i], dl_masks[j])."""
    n, m = len(classical_masks), len(dl_masks)
    iou_matrix = np.zeros((n, m), dtype=np.float32)
    for i, cm in enumerate(classical_masks):
        for j, dm in enumerate(dl_masks):
            iou_matrix[i, j] = compute_mask_iou(cm, dm)
    return iou_matrix

def dilate_all(mask_list, dilate_px):
    if dilate_px <= 0:
        return list(mask_list)
    return [binary_dilation(t, iterations=dilate_px) for t in mask_list]

# Fraction of `small` that lies inside `big` (IoMin when small is the smaller mask)
def coverage(small: np.ndarray, big: np.ndarray) -> float:
    area = np.count_nonzero(small)
    if area:
        return np.count_nonzero(small & big) / area 
    else:
        return 0.0


# checks if fragments attached to one target lie in two non-collinear directions
def has_noncollinear_spread(target: np.ndarray, fragments: list, max_angle_deg: float,
                             min_offset_px: float = 3.0) -> bool: 
    """Direction is taken as an AXIS (angle mod 180 deg), so fragments sitting at
    opposite ends of the same hair count as collinear (correct), while a fragment
    sticking out sideways from the middle counts as a different direction
    (suggests the target is really two hairs fused into one mask)."""

    def centroid(m):
        ys, xs = np.nonzero(m)
        return np.array([ys.mean(), xs.mean()])
 
    t_c = centroid(target)
    axes = []
    for f in fragments:
        d = centroid(f) - t_c
        # fragment centred on the target: direction is noise
        if np.hypot(*d) < min_offset_px: 
            continue
        axes.append(np.arctan2(d[0], d[1]) % np.pi)
 
    # Largest pairwise difference between axes (wrapping at 180 deg).
    for a in range(len(axes)):
        for b in range(a + 1, len(axes)):
            diff = abs(axes[a] - axes[b])
            diff = min(diff, np.pi - diff)
            if np.degrees(diff) > max_angle_deg:
                return True
            
    return False




# Pair classical and DL instances using the Hungarian algorithm.
def pair_instances(classical_masks: list[np.ndarray], dl_masks: list[np.ndarray], iou_threshold: float = 0.1, verbose=False) -> dict:
    """

    The IoU threshold here is intentionally LOW (default 0.1) compared
    to the F1 evaluation threshold (0.5). The reason: a DL hair whose
    base originates inside the root polygon will have very low overlap
    with the classical mask (which was cropped at the root boundary),
    but they still represent the same physical hair and should be merged.
    Use 0.1-0.2 to catch these partial overlaps.

    If two masks genuinely represent different hairs, their IoU will be
    close to 0.0 even with this lenient threshold.

    Args:
        classical_masks: list of bool masks from Part 1 splitting
        dl_masks:        list of bool masks from DL model
        iou_threshold:   minimum IoU to consider two masks the same hair

    Returns:
        dict with keys:
            matched_pairs — list of (classical_idx, dl_idx, iou) for Case A
            classical_only — list of classical_idx for Case B
            dl_only        — list of dl_idx for Case C
            iou_matrix     — full float array (useful for debugging)
    """

    if len(classical_masks) == 0 and len(dl_masks) == 0:
        return {"matched_pairs": [],
                "classical_only": [],
                "dl_only": [],
                "iou_matrix": np.zeros((0, 0))}

    if len(classical_masks) == 0:
        return {"matched_pairs": [],
                "classical_only": [],
                "dl_only": list(range(len(dl_masks))),
                "iou_matrix": np.zeros((0, len(dl_masks)))}

    if len(dl_masks) == 0:
        return {"matched_pairs": [],
                "classical_only": list(range(len(classical_masks))),
                "dl_only": [],
                "iou_matrix": np.zeros((len(classical_masks), 0))}

    iou_matrix = build_iou_matrix(classical_masks, dl_masks)

    # Hungarian algorithm: find globally optimal one-to-one assignment
    classical_idxs, dl_idxs = linear_sum_assignment(-iou_matrix)

    matched_pairs = []
    matched_classical = set()
    matched_dl = set()

    for c_idx, d_idx in zip(classical_idxs, dl_idxs):
        iou = iou_matrix[c_idx, d_idx]
        if iou >= iou_threshold:
            matched_pairs.append((c_idx, d_idx, float(iou)))
            matched_classical.add(c_idx)
            matched_dl.add(d_idx)

    classical_only = [i for i in range(len(classical_masks)) if i not in matched_classical]
    dl_only = [j for j in range(len(dl_masks)) if j not in matched_dl]

    if verbose:
        print(f"Pairing results:")
        print(f"  Case A (matched):  {len(matched_pairs)}")
        print(f"  Case B (classical only): {len(classical_only)}")
        print(f"  Case C (DL only): {len(dl_only)}")

    return {
        "matched_pairs":  matched_pairs,
        "classical_only": classical_only,
        "dl_only": dl_only,
        "iou_matrix": iou_matrix,
    }



def pair_instances_v2(classical_masks: list[np.ndarray],
                      dl_masks: list[np.ndarray],
                      iou_threshold: float = 0.5,
                      min_cov: float = 0.6,
                      dilate_px: int = 2,
                      min_area: int = 30,
                      check_merge_errors: bool = True,
                      max_angle_deg: float = 45.0,
                      verbose: bool = False) -> dict:
    """Two-stage pairing of classical and DL instance masks.
 
    Stage 1  strict one-to-one: Hungarian matching on mask IoU, keep IoU >= iou_threshold.
    Stage 2  coverage-based many-to-one: every instance NOT paired in Stage 1 (from either
             segmentation) is assigned to the instance of the OTHER segmentation that covers
             the largest fraction of it, if that coverage >= min_cov. Targets include
             instances already paired in Stage 1, so a locked pair can absorb extra fragments.
 
    Returns the same dict as pair_instances():
        matched_pairs  - list of (classical_idx, dl_idx, score). score is IoU for Stage 1
                         pairs and coverage for Stage 2 pairs. Because Stage 2 is many-to-one,
                         the same classical_idx or dl_idx can now appear in several tuples:
                         all tuples sharing an index belong to ONE fused instance.
        classical_only - classical indices with no partner (Case B)
        dl_only        - DL indices with no partner (Case C)
        iou_matrix     - full N x M Stage 1 IoU matrix

    two extra keys ignored by old code:
        pair_stage     - list parallel to matched_pairs, 1 or 2
        qc_flags       - list of dicts for targets whose fragments spread in non-collinear
                         directions (possible merged hairs). Those fragments are NOT fused.
    """
 
    # trivial cases: nothing to pair 
    if len(classical_masks) == 0 or len(dl_masks) == 0:
        return {"matched_pairs": [],
                "classical_only": list(range(len(classical_masks))),
                "dl_only": list(range(len(dl_masks))),
                "iou_matrix": np.zeros((len(classical_masks), len(dl_masks))),
                "pair_stage": [],
                "qc_flags": []}

    # STAGE 1: strict one-to-one pairing on IoU

    iou_matrix = build_iou_matrix(classical_masks, dl_masks)
    classic_idxs, deeplearning_idxs = linear_sum_assignment(1.0 - iou_matrix)
 
    matched_pairs, pair_stage = [], []
    paired_c, paired_d = set(), set()
    for c, d in zip(classic_idxs, deeplearning_idxs):
        # keep confident pairs only
        if iou_matrix[c, d] >= iou_threshold:  
            matched_pairs.append((int(c), int(d), float(iou_matrix[c, d])))
            pair_stage.append(1)
            paired_c.add(int(c))
            paired_d.add(int(d))
 
    # Everything else is a leftover, in either segmentation. Stage 1 pairs stay fixed.
    unpaired_classic = [i for i in range(len(classical_masks)) if i not in paired_c]
    unpaired_dl = [j for j in range(len(dl_masks)) if j not in paired_d]


     # STAGE 2: coverage-based many-to-one assignment

    # Keys identify an instance as ("c", idx) or ("d", idx)
    masks = {"c": classical_masks, "d": dl_masks}
    area = lambda key: int(np.count_nonzero(masks[key[0]][key[1]]))
 
    # Dilate the masks to get more matches
    targets_dil = {"c": None, "d": None}
    if unpaired_classic:
        # classical leftovers look for DL targets
        targets_dil["d"] = dilate_all(dl_masks, dilate_px)
    if unpaired_dl:
        # DL leftovers look for classical targets
        targets_dil["c"] = dilate_all(classical_masks, dilate_px)





    # 2A) each leftover picks the target with the highest coverage (argmax)
    # candidates: fragment_key -> (target_key, coverage)
    candidates = {}
    for side, leftovers in (("c", unpaired_classic), ("d", unpaired_dl)):
        other = "c"
        if side == "c":
            other = "d"

        for idx in leftovers:
            frag = masks[side][idx]

            # if less than min area dont do anything
            if np.count_nonzero(frag) < min_area:
                continue

            covs = [coverage(frag, t) for t in targets_dil[other]]
            # exactly one target per fragment
            j = int(np.argmax(covs))                    
            if covs[j] >= min_cov:
                candidates[(side, idx)] = ((other, j), float(covs[j]))
 
    # 2B. Avoid chains so nothing is merged transitively i.e stop (A -> B -> C) 
    for a, (b, _) in list(candidates.items()):
        if a in candidates and b in candidates and candidates[b][0] == a and a < b:
            # mutual pair: drop the candidate of the larger mask
            candidates.pop(a if area(a) >= area(b) else b, None)

    claimed = {tgt for tgt, _ in candidates.values()}
    for key in [k for k in candidates if k in claimed]:
        candidates.pop(key)
 
    # 2C. Group fragments by target and run the merge-error check
    # target_key -> [(fragment_key, cov)]
    groups = {}                                         
    for frag_key, (tgt_key, cov) in candidates.items():
        groups.setdefault(tgt_key, []).append((frag_key, cov))

    qc_flags = []
    for tgt_key, members in list(groups.items()):
        if check_merge_errors and len(members) >= 2:
            frags = [masks[k[0]][k[1]] for k, _ in members]
            if has_noncollinear_spread(masks[tgt_key[0]][tgt_key[1]], frags, max_angle_deg):
                # Possible two hairs merged into one mask: flag for QC instead of fusing.
                # The fragments fall back to single-source instances.
                qc_flags.append({"target_side": "classical" if tgt_key[0] == "c" else "dl",
                                 "target_idx": tgt_key[1],
                                 "fragment_idxs": [k[1] for k, _ in members],
                                 "reason": "fragments in non-collinear directions"})
                del groups[tgt_key]
 
    # 2D). Turn the accepted assignments into (classical_idx, dl_idx, coverage)
    for tgt_key, members in groups.items():
        for frag_key, cov in members:
            # frag_key / tgt_key are on opposite sides, so order them as (classical, dl)
            c_idx = frag_key[1] if frag_key[0] == "c" else tgt_key[1]
            d_idx = frag_key[1] if frag_key[0] == "d" else tgt_key[1]
            matched_pairs.append((int(c_idx), int(d_idx), float(cov)))
            pair_stage.append(2)
            paired_c.add(int(c_idx))
            paired_d.add(int(d_idx))


    # Anything still without a partner stays a single-source instance.
    classical_only = [i for i in range(len(classical_masks)) if i not in paired_c]
    dl_only = [j for j in range(len(dl_masks)) if j not in paired_d]
 
    if verbose:
        print("Pairing results (v2):")
        print(f"  Stage 1 pairs (IoU >= {iou_threshold}):  {pair_stage.count(1)}")
        print(f"  Stage 2 assignments (cov >= {min_cov}): {pair_stage.count(2)}")
        print(f"  QC-flagged targets:                {len(qc_flags)}")
        print(f"  Classical only: {len(classical_only)} | DL only: {len(dl_only)}")
 
    return {"matched_pairs": matched_pairs,
            "classical_only": classical_only,
            "dl_only": dl_only,
            "iou_matrix": iou_matrix,
            "pair_stage": pair_stage,
            "qc_flags": qc_flags}


 




# Bitwise or between two paired instances to create the combined instance
def fuse_instance_pair(classical_mask: np.ndarray, dl_mask: np.ndarray, ) -> np.ndarray:
    return (classical_mask | dl_mask).astype(bool)



# Apply the four-case fusion logic to produce the final instance list.
def fuse_all_instances(classical_masks: list[np.ndarray], model_dict: list[np.ndarray], pairing: dict) -> list[dict]:
    """
    Args:
        classical_masks: list of bool masks
        dl_masks:        list of bool masks
        pairing:         output of pair_instances()

    Returns:
        List of dicts, each representing one final root hair instance:
        {
            "mask":   bool array,
            "case":   "A", "B", or "C",
            "source": human-readable description,
            "iou":    float IoU of the pair (Case A only, else None),
            "area_px": int pixel area of the fused mask,
        }
    """
    final_instances = []
    dl_masks = model_dict["masks"]
    index = 0

    # --- Case A: matched pairs ---
    for (c_idx, d_idx, iou) in pairing["matched_pairs"]:
        fused = fuse_instance_pair(classical_masks[c_idx], dl_masks[d_idx])
        final_instances.append({
            "mask":             fused,
            "case":             "A",
            "bbox":             model_dict["box"][d_idx],
            "category_id":      model_dict["classes"][d_idx],
            "iou":              iou,
            "area_px":          int(fused.sum()),
            "id":               index,
            "model_instance":   d_idx,
            "classic_instance":   c_idx,
        })

        index += 1

    # --- Case B: classical only  ---
    for c_idx in pairing["classical_only"]:
        mask = classical_masks[c_idx].astype(bool)
        final_instances.append({
            "mask":             mask,
            "case":             "B",
            "bbox":             [],
            "category_id":      1,
            "iou":              None,
            "area_px":          int(mask.sum()),
            "id":               index,
            "model_instance":   None,
            "classic_instance":   c_idx,
        })

        index += 1

    # --- Case C: DL only  ---
    for d_idx in pairing["dl_only"]:
        mask = dl_masks[d_idx].astype(bool)
        final_instances.append({
            "mask":             mask,
            "case":             "C",
            "bbox":             model_dict["box"][d_idx],
            "category_id":      model_dict["classes"][d_idx],
            "iou":              None,
            "area_px":          int(mask.sum()),
            "id":               index,
            "model_instance":   d_idx,
            "classic_instance":   None,
        })

        index += 1

    return final_instances


# Function to run through all fusion logic
def fusion_of_masks(classic_instances: list[np.ndarray], model_instances: list[dict], iou_threshold = 0.1, detectron_confidence_threshold=0.5, version=1, verbose= False) -> list[dict]:
    # First turn the model instances into the right data format
    model_dict = load_model_instances(model_instances, detectron_confidence_threshold=detectron_confidence_threshold)

    # Get results of pairing algorithm
    if version == 1:
        results = pair_instances(classic_instances, model_dict["masks"], iou_threshold=iou_threshold, verbose=verbose)

    else:
        results = pair_instances_v2(classic_instances, model_dict["masks"], iou_threshold=iou_threshold, verbose=verbose)
    
    # fuse instances together
    final_instances = fuse_all_instances(classic_instances, model_dict, results)

    # visualize the pairing
    if verbose:
        for index, tuple_pair in enumerate(results["matched_pairs"]):

            classic_instance_idx = int(tuple_pair[0])
            model_instance_idx = int(tuple_pair[1])

            # print(classic_instance_idx, model_instance_idx)


            classic_mask = classic_instances[classic_instance_idx]
            model_mask = model_dict["masks"][model_instance_idx]

            # Convert bool array to uint8 image
            classic_mask_image = (classic_mask * 255).astype(np.uint8)
            model_mask_image = (model_mask * 255).astype(np.uint8)

            union = cv2.add(model_mask_image, classic_mask_image)

            # Display
            fig, axes = plt.subplots(1, 3, figsize=(13, 10))


            axes[0].imshow(classic_mask_image, cmap="gray")
            axes[0].axis("off")
            axes[0].set_title(f"Classic Instance {classic_instance_idx}")

            axes[1].imshow(model_mask_image, cmap="gray")
            axes[1].axis("off")
            axes[1].set_title(f"Model Instance {model_instance_idx}")

            axes[2].imshow(union, cmap="gray")
            axes[2].axis("off")
            axes[2].set_title("Union of Instances")


            plt.show()


            if index == 3:
                break


    # This is a list of dicts containing all segmented instances
    return final_instances
