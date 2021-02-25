'''
Author Nick Weir, 2020
'''

import os
import solaris as sol
import geopandas as gpd
import pandas as pd
from tqdm import tqdm


top_dir = '???'
pretrain_im_paths = [os.path.join(top_dir, 'train', 'pre_train', 'images', f)
                  for f in os.listdir(os.path.join(top_dir, 'train', 'pre_train', 'images'))
                  if f.endswith('.tif')]
finetune_im_paths = [os.path.join(top_dir, 'train', 'fine_tune', 'images', f)
                  for f in os.listdir(os.path.join(top_dir, 'train', 'fine_tune', 'images'))
                  if f.endswith('.tif')]
pretrain_gj_paths = [os.path.join(top_dir, 'train', 'pre_train', 'geojsons', f)
                  for f in os.listdir(os.path.join(top_dir, 'train', 'pre_train', 'geojsons'))
                  if f.endswith('.geojson')]
finetune_gj_paths = [os.path.join(top_dir, 'train', 'fine_tune', 'geojsons', f)
                  for f in os.listdir(os.path.join(top_dir, 'train', 'fine_tune', 'geojsons'))
                  if f.endswith('.geojson')]
test_im_paths = [os.path.join(top_dir, 'test', 'images', f)
                  for f in os.listdir(os.path.join(top_dir, 'test', 'images'))
                  if f.endswith('.tif')]
test_gj_paths = [os.path.join(top_dir, 'test', 'geojsons', f)
                  for f in os.listdir(os.path.join(top_dir, 'test', 'geojsons'))
                  if f.endswith('.geojson')]

print(f'Number of pre-train set collect images: {len(pretrain_im_paths)}')
print(f'Number of pre-train set geojsons: {len(pretrain_gj_paths)}')
print(f'Number of fine-tuning set collect images: {len(finetune_im_paths)}')
print(f'Number of fine-tuning set geojsons: {len(finetune_gj_paths)}')
print(f'Number of test set collect images: {len(test_im_paths)}')
print(f'Number of test set geojsons: {len(test_gj_paths)}')

desired_makes = [
    'a',
    'b',
    'c'
]

make_to_id = {
    'a': 0,
    'b': 1,
    'c': 2
}


def convert_gj_to_px_csv(dest_dir, gj_path):
    """Convert a geojson to pixel-coordinate CSV, dropping undesired columns and re-labeling makes."""
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    # get the matching image path from the geojson path
    im_path = gj_path.replace('geojsons', 'images').replace('.geojson', '.tif')
    # convert to pixel coordinates, reducing precision of geometry to 2 decimal places
    gdf = sol.vector.polygon.geojson_to_px_gdf(gj, im_path, precision=2)
    # save the original make in case we end up wanting to know this later for "Other" aircraft
    gdf['original_make'] = gdf['make'].copy(deep=True)
    # convert make to "Other" for everything that's not in the desired subset
    gdf.loc[~gdf.make.isin(desired_makes), 'make'] = 'Other'
    # create a numerical ID for each new make category
    gdf['da_make_id'] = gdf['make'].apply(lambda x: make_to_id[x])
    # drop undesired columns
    gdf = gdf[['image_fname', 'cat_id', 'loc_id', 'location',
               'original_make', 'make', 'da_make_id', 'pnp_id', 'geometry']]
    # convert to a regular dataframe since it's not georeferenced anymore
    df = pd.DataFrame(gdf)
    # save
    df.to_csv(os.path.join(dest_dir, os.path.splitext(os.path.split(gj_path)[1])[0] + '.csv'))
    
    
dest_dir = '???'
#os.mkdir(dest_dir)

for gj in tqdm(pretrain_gj_paths):
    convert_gj_to_px_csv(os.path.join(top_dir, 'train', 'pre_train', 'csvs'),
                         gj)

for gj in tqdm(finetune_gj_paths):
    convert_gj_to_px_csv(os.path.join(top_dir, 'train', 'fine_tune', 'csvs'),
                         gj)

for gj in tqdm(test_gj_paths):
    convert_gj_to_px_csv(os.path.join(top_dir, 'test', 'csvs'),
                         gj)