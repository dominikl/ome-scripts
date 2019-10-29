from omero.gateway import BlitzGateway, ColorHolder
from omero.model import MaskI
from omero.rtypes import rint, rlong, rstring, rdouble
import omero
import omero.scripts as scripts
import omero.util.script_utils as script_utils

import os
import tempfile
import subprocess
import shutil
import math

from PIL import Image

import numpy as np
from skimage.io import imread


ilastik_path = '/home/omero/ilastik/run_ilastik.sh '\
              +'--headless '\
              +'--output_format png '\
              +'--output_filename_format {dataset_dir}/{nickname}_probmap.png '

def export_image(image, path):
  """
  Export an OMERO image to file as png

  Args:
  image (ImageWrapper): The image
  path (string): The directory where to save the image

  Returns:
  string: The full path of the exported image
  """
  plane = image.renderImage(0, 0, compression=0)
  img_path = path+"/"+str(image.getId())+".png"
  plane.save(img_path, "PNG")
  return img_path


def run_cmd(cmd, cwd = '/tmp'):
  """
  Run an external command

  Args:
  cmd (string): The command to run
  cwd (string): The working directory (default: /tmp)

  Returns:
  bool: True if successfull, False otherwise
  """
  try:
    print "Running command: [%s]$ %s" % (cwd, cmd)
    subprocess.check_output(cmd.split(), cwd=cwd, stdin=None, stderr=subprocess.STDOUT, shell=False)
    return True
  except Exception as ex:
    print("An error occurred:\n"+str(ex))
    return False


def load_images(conn, script_params):
  """
  Load images specified by the script parameters

  Args:
  conn (BlitzGateway): The OMERO connection
  script_params (dict): The script parameters

  Returns:
  list(ImageWrapper): The images
  """
  objects, log_message = script_utils.get_objects(conn, script_params)
  data_type = script_params["Data_Type"]
  images = []
  if data_type == 'Dataset':
    for ds in objects:
      images.extend(list(ds.listChildren()))
  elif data_type == 'Project':
    for p in objects:
      for ds in p.listChildren():
        images.extend(list(ds.listChildren()))
  else:
    images = objects
  return images


def find_fileannotation(conn, script_params):
  """
  Find the ilastik project file annotations either attached 
  to the image, dataset or project

  Args:
  conn (BlitzGateway): The OMERO connection
  script_params (dict): The script parameters

  Returns:
  The first ilastik project file annotation found

  Raises:
  Exception: If no ilastik file annotation was found
  """
  objects, log_message = script_utils.get_objects(conn, script_params)

  for obj in objects:
    for ann in obj.listAnnotations():
      if ann.OMERO_TYPE == omero.model.FileAnnotationI:
        if ann.getFile().getName().endswith(".ilp"):
          return ann

  for obj in objects:
    if type(obj) == omero.gateway._ImageWrapper:
      ds = obj.getParent()
      for ann in ds.listAnnotations():
        if ann.OMERO_TYPE == omero.model.FileAnnotationI:
          if ann.getFile().getName().endswith(".ilp"):
            return ann
      pr = obj.getProject()
      for ann in pr.listAnnotations():
        if ann.OMERO_TYPE == omero.model.FileAnnotationI:
          if ann.getFile().getName().endswith(".ilp"):
            return ann

  raise Exception("No ilastik project found!")


def export_fileannotation(conn, script_params, path):
  """
  Export the first found ilastik project file annotation
  into the given directory

  Args:
  conn (BlitzGateway): The OMERO connection
  script_params (dict): The script parameters
  path (string): The directory where to save the ilastik project

  Returns:
  string: The full path to the ilastik project
  """
  ann = find_fileannotation(conn, script_params)
  if ann is None:
    return
  file_path = os.path.join(path, ann.getFile().getName())
  with open(str(file_path), 'w') as f:
    for chunk in ann.getFileInChunks():
      f.write(chunk)
  return file_path


def prop_map_threshold(pixs, fg_channel=0, fg_threshold=127, bg_channel=None, bg_threshold=None):
  """
  Thresholds an ilastik probability map image

  Args:
  pixs (numpy array): The probability map image [x,y,c], uint8
  fg_channel (int): The foreground channel index
  fg_threshold (int): The threshold for the foreground channel
  bg_channel (int): The background channel index (or None (default))
  bg_threshold (int): The threshold for the background channel (or None (default))

  Returns:
  numpy array: The binary image [x,y] bool
  """
  fg = pixs[:, :, fg_channel]
  fg = fg > fg_threshold
  if bg_channel is not None:
    bg = pixs[:, :, bg_channel]
    bg = bg > bg_threshold
  else:
    bg = np.empty(fg.shape)
    bg.fill(False)
  res = np.zeros(fg.shape)
  for f,b,r in np.nditer([fg,bg,res], op_flags=['readwrite']):
    r[...] = f and not b
  print("fg_channel: %s, fg_threshold: %s, segmented mean: %f" % (str(fg_channel), str(fg_threshold), np.mean(fg)))
  print("bg_channel: %s, bg_threshold: %s, segmented mean: %f" % (str(bg_channel), str(bg_threshold), np.mean(bg)))
  print("result mean: %f" % np.mean(res))
  return res

def create_mask(
      binim, rgba=None, z=None, c=None, t=None, text=None,
      raise_on_no_mask=False):
  """
  Create a mask shape from a binary image (background=0)

  :param numpy.array binim: Binary 2D array, must contain values [0, 1] only
  :param rgba int-4-tuple: Optional (red, green, blue, alpha) colour
  :param z: Optional Z-index for the mask
  :param c: Optional C-index for the mask
  :param t: Optional T-index for the mask
  :param text: Optional text for the mask
  :param raise_on_no_mask: If True (default) throw an exception if no mask
         found, otherwise return an empty Mask
  :return: An OMERO mask
  :raises NoMaskFound: If no labels were found
  :raises InvalidBinaryImage: If the maximum labels is greater than 1
  """

  # Find bounding box to minimise size of mask
  xmask = binim.sum(0).nonzero()[0]
  ymask = binim.sum(1).nonzero()[0]
  if any(xmask) and any(ymask):
    x0 = min(xmask)
    w = max(xmask) - x0 + 1
    y0 = min(ymask)
    h = max(ymask) - y0 + 1
    submask = binim[y0:(y0 + h), x0:(x0 + w)]
    if (not np.array_equal(np.unique(submask), [0, 1]) and not
            np.array_equal(np.unique(submask), [1])):
        raise InvalidBinaryImage()
  else:
    if raise_on_no_mask:
        raise NoMaskFound()
    x0 = 0
    w = 0
    y0 = 0
    h = 0
    submask = []

  mask = MaskI()
  mask.setBytes(np.packbits(np.asarray(submask, dtype=int)))
  mask.setWidth(rdouble(w))
  mask.setHeight(rdouble(h))
  mask.setX(rdouble(x0))
  mask.setY(rdouble(y0))

  if rgba is not None:
    ch = ColorHolder.fromRGBA(*rgba)
    mask.setFillColor(rint(ch.getInt()))
  if z is not None:
    mask.setTheZ(rint(z))
  if c is not None:
    mask.setTheC(rint(c))
  if t is not None:
   mask.setTheT(rint(t))
  if text is not None:
   mask.setTextValue(rstring(text))

  return mask

def add_mask(conn, image, path, fg_threshold=127, fg_channel=0,\
               bg_threshold=None, bg_channel=None, rgba=(255,255,0,100)):
  """
  Segments the ilastik probabilty map for the given image and adds the 
  result as mask to the image

  Args:
  conn (BlitzGateway): The OMERO connection
  image (ImageWrapper): The image
  fg_channel (int): The foreground channel index
  fg_threshold (int): The threshold for the foreground channel
  bg_channel (int): The background channel index (or None (default))
  bg_threshold (int): The threshold for the background channel (or None (default))
  rgba (int tuple): The RGBA value of the mask color

  Returns:
  MaskWrapper: The mask
  """
  p = path+"/"+str(image.getId())+"_probmap.png"
  prop_image = imread(p)

  binary_image = prop_map_threshold(prop_image, fg_channel=fg_channel, fg_threshold=fg_threshold,\
                                    bg_channel=bg_channel, bg_threshold=bg_threshold)

  mask = create_mask(binary_image, rgba=rgba)
  roi = omero.model.RoiI()
  roi.setImage(image._obj)
  roi.addShape(mask)
  return conn.getUpdateService().saveAndReturnObject(roi)

def delete_rois(conn, image):
  """
  Delets all mask ROIs of the given image.

  Args:
  conn (BlitzGateway): The OMERO connection
  image (ImageWrapper): The image
  """
  roi_service = conn.getRoiService()
  result = roi_service.findByImage(image.getId(), None)
  ids = []
  for roi in result.rois:
    for s in roi.copyShapes():
      if type(s) == omero.model.MaskI:
        ids.append(roi.getId().getValue())
        break
  if ids:
    conn.deleteObjects("Roi", ids, deleteAnns=True, deleteChildren=True, wait=True)

def run_script(conn, script_params):
  fg_threshold = script_params["Foreground_Threshold"]
  del_rois = script_params["Delete_previous_Masks"]
  fg_channel_string = script_params["Foreground_Channel"]
  if fg_channel_string == 'Green':
    fg_channel = 1
  elif fg_channel_string == 'Blue':
    fg_channel = 2
  else:
    fg_channel = 0

  if script_params["Use_Background_Channel"]:
    bg_threshold = script_params["Background_Threshold"]
    bg_channel_string = script_params["Background_Channel"]
    if bg_channel_string == 'Green':
      bg_channel = 1
    elif bg_channel_string == 'Blue':
      bg_channel = 2
    else:
      bg_channel = 0
  else:
    bg_channel = None
    bg_threshold = None

  if script_params["Custom_Mask_Color"]:
    r = int(script_params["Mask_color_Red"])
    g = int(script_params["Mask_color_Green"])
    b = int(script_params["Mask_color_Blue"])
    a = int(script_params["Mask_color_Alpha"])
    rgba = (r,g,b,a)
  else:
    rgba = (255,255,0,100)

  tmp = tempfile.mkdtemp()

  images = load_images(conn, script_params)
  img_paths = []
  for img in images:
    exp = export_image(img, tmp)
    img_paths.append(exp)
  img_paths = " ".join(img_paths)

  ilp_path = export_fileannotation(conn, script_params, tmp)

  cmd = "bash "+ilastik_path+" --project "+ilp_path+" "+img_paths
  status = run_cmd(cmd, cwd=tmp)
  if status:
    for img in images:
      if del_rois:
        delete_rois(conn, img)
      add_mask(conn, img, tmp, fg_threshold=fg_threshold, fg_channel=fg_channel,\
               bg_threshold=bg_threshold, bg_channel=bg_channel, rgba=rgba)
  shutil.rmtree(tmp)
  return status

if __name__ == "__main__":

    dataTypes = [rstring('Image'),rstring('Dataset'),rstring('Project')]
    rgb_channels = ['Red','Green','Blue']

    client = scripts.client(
        'Run_Ilastik.py', """
        Runs an Ilastik pixel classification project on the given images.
        It expects that the Ilastik project exports a probability map image
        with three channels (RGB) and unit8 pixel values, where one channel
        marks the features and one channel the background.
        The Ilastik project has to be attached to either the image or the
        project/dataset. If there are several Ilastik projects attached, a
        random one will be selected!
        """,

        scripts.String(
            "Data_Type", optional=False, grouping="1",
            description="Choose source of images",
            values=dataTypes, default="Dataset"),

        scripts.List(
            "IDs", optional=False, grouping="1",
            description="List of IDs to process.").ofType(rlong(0)),

        scripts.String(
            "Foreground_Channel", optional=False, grouping="2.1",
            description="The target channel of the RGB probability map",
            values=rgb_channels, default="Red"),
        scripts.Int("Foreground_Threshold", grouping="2.2",
          description="Pixel value threshold of the RGB probability map",
          min=0, max=255, default=127),
        scripts.Bool("Use_Background_Channel", grouping="3", default=True,\
                     description="Take background channel information into account"),
        scripts.String(
            "Background_Channel", optional=False, grouping="3.1",
            description="The background channel of the RGB probability map",
            values=rgb_channels, default="Green"),
        scripts.Int("Background_Threshold", grouping="3.2",
          description="Pixel value threshold of the RGB probability map",
          min=0, max=255, default=127),

        scripts.Bool("Custom_Mask_Color", grouping="4", default=True),
        scripts.Int("Mask_color_Red", grouping="4.1",
          description="Display color of the mask",
          min=0, max=255, default=255),
        scripts.Int("Mask_color_Green", grouping="4.2",
          description="Display color of the mask",
          min=0, max=255, default=255),
        scripts.Int("Mask_color_Blue", grouping="4.3",
          description="Display color of the mask",
          min=0, max=255, default=0),
        scripts.Int("Mask_color_Alpha", grouping="4.4",
          description="Display color of the mask",
          min=0, max=255, default=100),

        scripts.Bool("Delete_previous_Masks",
          description="Delete all previous Masks of the images",
          grouping="5", default=True),

        version="1.0",
        authors=["Dominik Lindner"],
        contact="d.lindner@dundee.ac.uk",
    )

    try:
        conn = BlitzGateway(client_obj=client)
        script_params = client.getInputs(unwrap=True)
        success = run_script(conn, script_params)
        if success:
          client.setOutput("Message", rstring("Finished."))
        else:
          client.setOutput("Message", rstring("An error occurred."))

    finally:
        client.closeSession()
