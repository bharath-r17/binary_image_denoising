# Importing Libraries
import cv2 as cv
import numpy as np
import maxflow 
from multiprocessing.pool import ThreadPool
from scipy import ndimage

# FILTERS
k1 = np.array([[0,0,0],[0,1,0],[0,0,0]],  np.uint8)  # Identity Filter
k2 = np.array([[0,-1,0],[-1,3,1],[0,-1,0]],  np.uint8) #  Sharpen Filter

# Median Blur
def remove_noise(image, ksize=3):
    return cv.medianBlur(image, ksize)

# It will help after the blur operation
def thresholding(image):
    return cv.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


def Scaling(img, scale_percent=75):
  # scale_percent = 75 # percent of original size

  width = int(img.shape[1] * scale_percent / 100)
  height = int(img.shape[0] * scale_percent / 100)
  dim = (width, height)

  resized = cv.resize(img, dim, interpolation = cv.INTER_AREA)
  return resized



def Descaling(img, dim):

  resized = cv.resize(img, dim, interpolation = cv.INTER_AREA)
  return resized



# Our Approcah/Algorithm
def Algorithm(img, ksize=3, scale_percent=75, smoothing = 10):
  
  img = Scaling(img,scale_percent)

  # Create the graph.
  g = maxflow.Graph[int]()
  ### Add the nodes. nodeids has the identifiers of the nodes in the grid.
  nodeids = g.add_grid_nodes(img.shape)
  ### Add non-terminal edges with the same capacity.
  g.add_grid_edges(nodeids, smoothing)
  ### Add the terminal edges. The image pixels are the capacities
  # of the edges from the source node.
  g.add_grid_tedges(nodeids, img, 255-img)
  ### Find the maximum flow.
  g.maxflow()
  # Get the segments of the nodes in the grid.
  sgm = g.get_grid_segments(nodeids)

  img_denoised = np.logical_not(sgm).astype(np.uint8) * 255

  
  morph = img_denoised
  # Convolve the identity filer
  morph = ndimage.convolve(morph , k1, mode='constant', cval=10.0)
  # # Convolve the sharpen filer
  # # morph = ndimage.convolve(morph , k2, mode='constant', cval=0.0)

  # # Remove noise by median blur
  morph = remove_noise(morph, ksize=1)

  # # morph = ndimage.convolve(morph , k1, mode='constant', cval=10.0)
  morph = ndimage.convolve(morph , k1, mode='constant', cval=10.0)

  # # Threashold to balance the bluring 
  # #morph = thresholding(morph)
  _,morph= cv.threshold(morph, 127, 255, cv.THRESH_BINARY)

  return morph



if __name__ == '__main__':
    total_time = 0.0
    start_timer = 0.0
    end_timer = 0.0

    # START TIMER
    start_timer = cv.getTickCount()

    image_binary = cv.imread("test_input_image.png", cv.IMREAD_GRAYSCALE); # Reading the original image
    image_output = None                                              # This should contain the final denoised image

    pool = ThreadPool(processes=128)

    # Starting the timer

    # HYPER PARAMETERS
    '''
    -> Scaling of the image
    -> smoothing of the graph
    -> k size for median blur
    '''

    scale_percent = 40
    k_size = 3
    smoothing = 10

    # Parellel threading 
    async_result = pool.apply_async(Algorithm, (image_binary,k_size,scale_percent,smoothing,))
    image_output = async_result.get()

    # Resizing Back
    org_w = int(image_binary.shape[1])
    org_h = int(image_binary.shape[0])
    width = org_w
    height = org_h
    dim = (width, height)
    image_output = Descaling(image_output, dim)

    # END TIMER
    end_timer = cv.getTickCount()

    # if (image_output != None):
    #     cv.imshow("Output", image_output)
    
    try:
      cv.imwrite('test_img.png',image_output)
    except:
      pass
    message = 'Time taken: {:0.4f} ms'
    total_time = (end_timer-start_timer)*1000/cv.getTickFrequency()
    print(message.format(total_time))

