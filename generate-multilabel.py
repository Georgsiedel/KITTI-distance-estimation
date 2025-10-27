#!/usr/bin/env python3
"""
generate-multilabel.py

Create multilabel image-level annotations from `annotations.csv` produced by
generate-csv.py in the KITTI-distance-estimation repo.

Labels created (binary 0/1):
 - non_vulnerable_present     : any Car/Van/Truck/Tram present (passes KITTI moderate filters)
 - non_vulnerable_nearby     : any Car/Van/Truck/Tram with zloc <= threshold (and passes filters)
 - vulnerable_present        : any Pedestrian/Person_sitting/Cyclist present (passes filters)
 - vulnerable_nearby        : any Pedestrian/Person_sitting/Cyclist with zloc <= threshold (and passes filters)
 - crowded_critical          : 3 or more objects (any of the considered classes) with zloc <= threshold

Moderate KITTI filters applied to each object (same as KITTI moderate/standard):
 - bbox height (ymax - ymin) >= 25 px
 - occluded <= 1  (0 = fully visible, 1 = partly occluded)
 - truncated <= 0.3

Produces: CSV with one row per image (filename) and the above labels + counts.

Author: ChatGPT (assistant)
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd


# object groupings (match generate-csv.py naming)
NON_VULNERABLE = {'Car', 'Van', 'Truck', 'Tram'}
VULNERABLE = {'Pedestrian', 'Person_sitting', 'Cyclist'}
# ignore boxes with these types (optional, mirrors generate-csv behavior)
IGNORE_TYPES = {'DontCare', 'Misc'}


def _safe_col(df, names):
    """Return first matching column name from list 'names' that exists in df, or None."""
    for n in names:
        if n in df.columns:
            return n
    return None


def prepare_dataframe(path):
    """Load CSV and sanitize column names / types. Returns DataFrame."""
    df = pd.read_csv(path)

    # find expected columns (tolerant)
    col_filename = _safe_col(df, ['filename', 'file', 'img'])
    col_class = _safe_col(df, ['class', 'type', 'label'])
    col_trunc = _safe_col(df, ['truncated', 'truncation', 'trunc'])
    col_occ = _safe_col(df, ['occluded', 'occlusion', 'occluded_level'])
    col_xmin = _safe_col(df, ['xmin', 'x_min', 'left'])
    col_ymin = _safe_col(df, ['ymin', 'y_min', 'top'])
    col_xmax = _safe_col(df, ['xmax', 'x_max', 'right'])
    col_ymax = _safe_col(df, ['ymax', 'y_max', 'bottom'])
    col_z = _safe_col(df, ['zloc', 'z', 'z_loc', 'z_location'])
    # quick existence check
    missing = []
    for name, col in (('filename', col_filename), ('class', col_class),
                      ('truncated', col_trunc), ('occluded', col_occ),
                      ('ymin', col_ymin), ('ymax', col_ymax), ('zloc', col_z)):
        if col is None:
            missing.append(name)
    if missing:
        raise RuntimeError(f"Missing required columns in annotations CSV: {missing}. "
                           "Expected columns like filename,class,truncated,occluded,ymin,ymax,zloc")

    # rename to canonical names
    df = df.rename(columns={
        col_filename: 'filename',
        col_class: 'class',
        col_trunc: 'truncated',
        col_occ: 'occluded',
        col_xmin: 'xmin' if col_xmin else None,
        col_ymin: 'ymin',
        col_xmax: 'xmax' if col_xmax else None,
        col_ymax: 'ymax',
        col_z: 'zloc'
    })

    # convert types (coerce invalid -> NaN)
    df['truncated'] = pd.to_numeric(df['truncated'], errors='coerce').fillna(0.0).astype(float)
    df['occluded'] = pd.to_numeric(df['occluded'], errors='coerce').fillna(3).astype(int)  # unknown -> 3
    df['ymin'] = pd.to_numeric(df['ymin'], errors='coerce')
    df['ymax'] = pd.to_numeric(df['ymax'], errors='coerce')
    df['zloc'] = pd.to_numeric(df['zloc'], errors='coerce')

    # drop rows with no bbox or no zloc
    df = df.dropna(subset=['ymin', 'ymax', 'zloc', 'class'])

    # strip whitespace from class names
    df['class'] = df['class'].astype(str).str.strip()

    return df


def object_passes_filters(row, min_bbox_h=25, max_trunc=0.5, max_occluded=3):
    """Return True if object passes moderate KITTI criteria."""
    bbox_h = float(row['ymax']) - float(row['ymin'])
    if bbox_h < min_bbox_h:
        return False
    if float(row.get('truncated', 0.0)) > max_trunc:
        return False
    # occluded is int where 0=visible,1=partly,2=largely,3=unknown
    if int(row.get('occluded', 3)) > max_occluded:
        return False
    return True


def build_multilabel(df, z_threshold=15.0, min_bbox_h=25,
                     max_trunc=0.5, max_occluded=3):
    """
    Build multilabel DataFrame, one row per filename.

    Returns: DataFrame with columns:
      filename,
      non_vulnerable_present,
      non_vulnerable_nearby,
      vulnerable_present,
      vulnerable_nearby,
      crowded_critical,
      non_vulnerable_nearby_count,
      vulnerable_nearby_count,
      nearby_total_count
    """
    grouped = df.groupby('filename')
    rows = []
    for fname, group in grouped:
        # filter out ignored classes (DontCare/Misc)
        group = group[~group['class'].isin(IGNORE_TYPES)].copy()

        # apply moderate KITTI filters per object
        passes_mask = group.apply(
            lambda r: object_passes_filters(r, min_bbox_h=min_bbox_h,
                                            max_trunc=max_trunc,
                                            max_occluded=max_occluded), axis=1)
        group = group[passes_mask]

        # initialization
        nv_present = 0
        nv_nearby = 0
        v_present = 0
        v_nearby = 0
        nearby_total = 0

        # iterate objects
        for _, obj in group.iterrows():
            cls = obj['class']
            z = float(obj['zloc']) if not np.isnan(obj['zloc']) else np.inf

            if cls in NON_VULNERABLE:
                nv_present = 1 if nv_present == 0 else nv_present
                if z <= z_threshold:
                    nv_nearby += 1
                    nearby_total += 1
            elif cls in VULNERABLE:
                v_present = 1 if v_present == 0 else v_present
                if z <= z_threshold:
                    v_nearby += 1
                    nearby_total += 1
            else:
                # if type is not in either group but passed filters, count toward nearby_total if within threshold
                if z <= z_threshold:
                    nearby_total += 1

        crowded_critical = 1 if nearby_total >= 3 else 0

        rows.append({
            'filename': fname,
            'non_vulnerable_present': int(nv_present > 0),
            'non_vulnerable_nearby': int(nv_nearby > 0),
            'vulnerable_present': int(v_present > 0),
            'vulnerable_nearby': int(v_nearby > 0),
            'crowded_critical': int(crowded_critical),
            #'non_vulnerable_nearby_count': int(nv_nearby),
            #'vulnerable_nearby_count': int(v_nearby),
            #'nearby_total_count': int(nearby_total),
        })

    out_df = pd.DataFrame(rows)
    # ensure deterministic ordering
    out_df = out_df.sort_values('filename').reset_index(drop=True)
    return out_df


def main():
    parser = argparse.ArgumentParser(
        description="Generate multilabel dataset CSV from KITTI annotations.csv")
    parser.add_argument('--input', '-i', required=True,
                        help='input annotations.csv file (the one produced by generate-csv.py)')
    parser.add_argument('--output', '-o', default='multilabel_annotations.csv',
                        help='output multilabel CSV file (default: multilabel_annotations.csv)')
    parser.add_argument('--z_threshold', '-z', type=float, default=15.0,
                        help='distance threshold in meters for "nearby" (default: 15.0)')
    parser.add_argument('--min_bbox_h', type=float, default=25.0,
                        help='minimum bbox height in pixels (default: 25)')
    parser.add_argument('--max_trunc', type=float, default=0.5,
                        help='maximum truncation (0..1) allowed (default: 0.5)')
    parser.add_argument('--max_occluded', type=int, default=3,
                        help='maximum occlusion allowed (0..3) (default: 1)')
    parser.add_argument('--summary', default='multilabel_summary.json',
                        help='output JSON summary file (default: multilabel_summary.json)')

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Input file not found: {args.input}", file=sys.stderr)
        sys.exit(2)

    print(f"Loading annotations from {args.input} ...")
    df = prepare_dataframe(args.input)

    print("Building multilabel annotations ...")
    out_df = build_multilabel(df, z_threshold=args.z_threshold,
                              min_bbox_h=args.min_bbox_h,
                              max_trunc=args.max_trunc,
                              max_occluded=args.max_occluded)

    print(f"Saving multilabel CSV to {args.output} ...")
    out_df.to_csv(args.output, index=False)

    # make a quick summary
    summary = {
        'input': os.path.abspath(args.input),
        'output': os.path.abspath(args.output),
        'n_images': int(out_df.shape[0]),
        'n_non_vulnerable_present': int(out_df['non_vulnerable_present'].sum()),
        'n_non_vulnerable_nearby': int(out_df['non_vulnerable_nearby'].sum()),
        'n_vulnerable_present': int(out_df['vulnerable_present'].sum()),
        'n_vulnerable_nearby': int(out_df['vulnerable_nearby'].sum()),
        'n_crowded_critical': int(out_df['crowded_critical'].sum()),
        'z_threshold': float(args.z_threshold),
        'bbox_height_threshold': float(args.min_bbox_h),
        'max_truncation': float(args.max_trunc),
        'max_occluded': int(args.max_occluded)
    }

    with open(args.summary, 'w') as fh:
        json.dump(summary, fh, indent=2)

    print("Done. Summary:")
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
