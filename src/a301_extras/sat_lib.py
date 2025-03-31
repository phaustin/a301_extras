import xarray
import rioxarray
from numpy.typing import NDArray
import numpy as np
from skimage import exposure, img_as_ubyte
from copy import deepcopy
import pyproj
import affine




def make_dataset(
      scene_dict: dict)-> xarray.Dataset:
    """
    given a dictionary with landsat bands stored as rioxarray, keyed by
    the band name, return an rioxarray dataset containing all the bands
    plus metadata

    Parameters
    ----------

    scene_dict: dictionary with keys like 'B03'

    Returns
    -------

    ds_allbands: xarray dataset with all bands from the dictionary stored as variables
    
    """
    the_keys=list(scene_dict.keys())
    first_band=the_keys[0]
    ds_allbands = xarray.Dataset(data_vars=scene_dict,
               coords=scene_dict[first_band].coords,attrs=scene_dict[first_band].attrs)
    return ds_allbands


def make_bool_mask(
      da_fmask:xarray.DataArray
     ) -> NDArray[np.uint8]:
    """'
    turn a Landsat fmask into a boolean 1/0 array where
    cloud-free land pixels are 1 and all other pixels are 0
    For use by skimage.exposure.equalize_hist

    Parameters
    ----------

    da_fmask: the fmask DataArray

    Returns: bool_mask with the same shape
    """
    scene_mask = da_fmask.data
    mask_select =  0b00100011  #find water (bit 5), cloud (bit 1) , cirrus (bit 0)
    ref_mask = np.zeros_like(scene_mask)
    ref_mask[...] = mask_select
    masked_values = np.bitwise_and(scene_mask,ref_mask)
    masked_values[masked_values>0]=1  #cloud or water
    masked_values[masked_values==0]=0 #rest of scene
    #
    # now invert this, writing 1 for 0 and 0 for 1
    # use 9 as a placeholder value
    bool_mask = masked_values[...]
    bool_mask[masked_values==1] = 9
    bool_mask[masked_values==0] = 1
    bool_mask[masked_values==9] = 0
    return bool_mask

def make_false_color(
        the_ds: xarray.Dataset,
        band_names: list[str]) -> xarray.DataArray:
    """
    given a landsat dataset created with at least an fmask and 3 bands,
    return a histogram-equalized false color image with rgb mapped
    to the bands in the order they appear in the list band_names

    example usage:

    landsat_654 = make_false_color(the_ds,['B06','B05','B04'])

    Parameters
    ----------

    the_ds:
       created by make_dataset -- must contain at least 3 bands and Fmask
    band_names: 
       list of strings with the names of the bands to be mapped to red, green and blue

    Returns
    -------

    false_color: rioxarray with shape [3,nrows,ncols] that can be converted to png
    """
    the_ds = the_ds.squeeze()
    rgb_names = ["band_red", "band_green", "band_blue"]
    #
    # dictionary to hold the 3 rgb bands
    #
    scene_dict = dict()
    for the_rgb, the_band in zip(rgb_names, band_names):
        # print(f"{the_rgb=}, {the_band=}")
        scene_dict[the_rgb] = the_ds[the_band]
    crs = the_ds.rio.crs
    transform = the_ds.rio.transform()
    fmask = the_ds["fmask"].data
    bool_mask = make_bool_mask(fmask)
    #
    # histogram equalize the 3 bands
    #
    for key, image in scene_dict.items():
        scene_dict[key] = exposure.equalize_hist(image.data, mask=bool_mask)
    nrows, ncols = bool_mask.shape
    band_values = np.empty([3, nrows, ncols], dtype=np.uint8)
    #
    # convert to 0-255
    #
    for index, key in enumerate(rgb_names):
        stretched = scene_dict[key]
        band_values[index, :, :] = img_as_ubyte(stretched)
    #
    # only keep a subset of the attributes
    #
    keep_attrs = ["cloud_cover", "date", "day", "target_lat", "target_lon"]
    all_attrs = the_ds.attrs
    attr_dict = {key: value for key, value in all_attrs.items() if key in keep_attrs}
    attr_dict["history"] = "written by make_false_color"
    attr_dict["landsat_rgb_bands"] = band_names
    band_nums = [int(item[-1]) for item in band_names]
    coords = {"band": band_nums, "y": the_ds["y"], "x": the_ds["x"]}
    dims = ["band", "y", "x"]
    # print(f"{dims=}")
    # print(f"{band_values.shape=}")
    false_color = xarray.DataArray(
        band_values, coords=coords, dims=dims, attrs=attr_dict
    )
    false_color.rio.write_crs(crs, inplace=True)
    false_color.rio.write_transform(transform, inplace=True)
    return false_color

def mask_image(
    image_da:xarray.DataArray,
    fmask_da:xarray.DataArray,
    mask_value:np.uint8) -> xarray.DataArray:
    """
    given an image, a bit mask, and a mask value, 
    return the modified image with all masked values set to np.nan

    Parameters
    ----------

    image_da: a rioxarray data array with a single band image
    fmask_da: landsat fmask with the same bounding box
    mask_value: bits to mask, like  0b00100011

    Returns
    -------

    masked_da: a copy of image_da array 
    with masked values set to np.nan
    """
    scene_mask = fmask_da.data
    ref_mask = np.zeros_like(fmask_da.data)
    ref_mask[...] = mask_value
    masked_values = np.bitwise_and(scene_mask,ref_mask)
    #
    # change all masked values to placeholder
    #
    hit = masked_values > 0
    masked_values[hit]= 40  #placeholder
    fmask_float = masked_values.astype(np.float32)
    #
    # bad pixels are all 40
    #
    fmask_float[fmask_float == 40] = np.nan
    #
    # good pixels are all zero
    #
    fmask_float[fmask_float == 0] = 1
    #
    # return the masked image
    #
    image_copy = deepcopy(image_da)
    image_copy.data = image_da.data*fmask_float
    return image_copy
    


def make_new_rioxarray(
    rawdata: np.ndarray,
    coords: dict,
    dims: tuple,
    crs: pyproj.CRS,
    transform: affine.Affine,
    attrs: dict | None = None,
    missing: float | None = None,
    name: str | None = "name_here") -> xarray.DataArray:
    """
    create a new rioxarray from an ndarray plus components

    Parameters
    ----------

    rawdata: numpy array
    crs: pyproj crs for scene
    coords: xarray coordinate dict
    dims: x and y dimension names from coorcds
    transform: scene affine transform
    attrs: optional attribute dictionary
    missing: optional missing value
    name: optional variable name, defaults to "name_here"

    Returns
    -------

    rio_da: a new rioxarray
    """
    rio_da=xarray.DataArray(rawdata.data,coords=coords,
                            dims=dims,name=name)
    rio_da.rio.write_crs(crs, inplace=True)
    rio_da.rio.write_transform(transform, inplace=True)
    if attrs is not None:
        rio_da=rio_da.assign_attrs(attrs)
    if missing is not None:
        rio_da = rio_da.rio.set_nodata(missing)
    return rio_da

