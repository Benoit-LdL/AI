from keras.datasets import mnist
from matplotlib import pyplot
import random as rnd
import matplotlib.pyplot as plt
import numpy as np


rnd.seed()

pixels_wdth = 28                        # # pixels of image width
num_pixels  = pixels_wdth*pixels_wdth   # # pixels of image = # neurons in entry layer
num_hidden  = 100                       # # neurons in hidden layer
num_out     = 10                        # # neurons in exit layer

theta       = 0.5
epsilon     = 0.01

### VECTOR / MTRX DELCARATION

#""" VEC / MTRX for testing layers propagation
## FOR TESTING HIDDEN LAYER
DEBUG_hidden_vec_img_ones    = [1]*num_pixels

DEBUG_hidden_vec_img_one_s     = [0]*num_pixels
DEBUG_hidden_vec_img_one_s[0]  = 1

DEBUG_hidden_vec_img_one_m     = [0]*num_pixels
DEBUG_hidden_vec_img_one_m[int(num_pixels/2)]  = 1

DEBUG_hidden_vec_img_one_l     = [0]*num_pixels
DEBUG_hidden_vec_img_one_l[num_pixels-1]  = 1

DEBUG_hidden_vec_img_zeros   = [0]*num_pixels

DEBUG_mtrx_weights_hidden_ones = [[1]*num_pixels]*num_hidden

## FOR TESTING OUT LAYER
DEBUG_out_vec_img_ones    = [1]*num_hidden

DEBUG_out_vec_img_one_s     = [0]*num_hidden
DEBUG_out_vec_img_one_s[0]  = 1

DEBUG_out_vec_img_one_m     = [0]*num_hidden
DEBUG_out_vec_img_one_m[int(num_hidden/2)]  = 1

DEBUG_out_vec_img_one_l     = [0]*num_hidden
DEBUG_out_vec_img_one_l[num_hidden-1]  = 1

DEBUG_out_vec_img_zeros   = [0]*num_hidden

DEBUG_mtrx_weights_out_ones = [[1]*num_hidden]*num_out
#"""


def getRndmImg(inImgs, inLbls):
    rndInt = rnd.randint(0,len(inLbls))
    #print(inY[rndInt])
    #print(f"len y = {len(inY)}")
    #plt.imshow(inX[rndInt], cmap=plt.get_cmap('gray'))
    #plt.title(f"image of number {inY[rndInt]}")
    #plt.show()
    return inImgs[rndInt], inLbls[rndInt]

def normaliseConvert2D(inImg,inPixels=num_pixels,norm=255):
    nLst = [None]*inPixels
    counter = 0
    for row in inImg:
        for col in row:
            #print(f"r={row} | c={col}")
            nLst[counter] = col/norm
            counter+=1
    return nLst
    #print(f"rows={rows} | cols={cols/rows}")

def normaliseLbl(inLbl):
    nLbl = [0]*10
    nLbl[inLbl] = 1
    return nLbl

def printImg(inImg,inPixel=pixels_wdth):
    spacing = 6
    counter = 0
    line = ""
    
    #print(f"=== print matrix ===")
    print("="*inPixel*spacing)
    
    for elem in inImg:
        counter+=1
        line += str(round(elem,spacing-3)).ljust(spacing) #+"\t"
        if counter%inPixel == 0:
            print(line)
            line = ""
    print("="*inPixel*spacing)


def getWeights(numNeuronIn, numNeuronOut):
    nVec_weights = []
    for neuron in range(numNeuronIn):
        nVec_neuron = []
        for weight in range(numNeuronOut):
            nVec_neuron.append(rnd.random()/numNeuronOut)
        nVec_weights.append(nVec_neuron)
    return nVec_weights

def calcPotential(inImg, inWeights):
    sum = 0
    for i in range(len(inWeights)):
        sum += inWeights[i] * inImg[i]
    return sum

def activationFunc(inX):
    return  1 / (1 + np.e**(-inX))

def activationFuncDerived(inX):
   return activationFunc(inX) * ( 1 - activationFunc(inX) )

def calcErrorLayerOut(inVecPotsOut, inLblNorm, inVecLayerOut):
    vec_error_out = [None]*num_out
    for n in range(num_out):
        vec_error_out[n] = activationFuncDerived(inVecPotsOut[n]) * inLblNorm[n] - inVecLayerOut[n]
    return vec_error_out

def updateWeights(inMtrx, inWeights, inError,inEpsilon=0.01):
    #print(f"weights len: {len(inWeights)}")
    #print(f"mtrx len: {len(inMtrx)}")
    #print(f"error : {inError}")   
    for i in range(len(inWeights)):
        inWeights[i] = inWeights[i] + inEpsilon * inError * inMtrx[i]
    return inWeights

def calcNeuronOut(inNorm,inWeights,inNumLayer):
    vec_pots    = [None]*inNumLayer
    for n in range(inNumLayer):
        vec_pots[n] = calcPotential(inNorm, inWeights[n])
    #print(f"vec pots: {vec_pots}")
    
    vec_layer   = [None]*inNumLayer
    for n in range(inNumLayer):
        vec_layer[n] = activationFunc(vec_pots[n])
    #print(f"vec layer:  {vec_layer}")

    return vec_layer, vec_pots

############################################################
##################### CODE #################################
############################################################

#  X = image
#  Y = correct value of number on image

# IMPORT MNSIT DATA FROM LIBRARY
(train_X, train_y), (test_X, test_y) = mnist.load_data()


# DEFINE NETWORK MATRICES
vec_layer_in     = [0] * num_pixels
vec_layer_hidden = [0] * num_hidden
vec_layer_out    = [0] * num_out
#print(f"netw entry length  = {len(vec_layer_in)}")
#print(f"netw hidden length = {len(vec_layer_hidden)}")
#print(f"netw exit length   = {len(vec_layer_out)}")

### 1. Select random image and corresponding Y value
img_sel, lbl_sel  = getRndmImg(train_X,train_y)
#print(f"Selected Y = {lbl_sel}\nSelected img: {img_sel}")

### 2. Normalise image and label
img_norm        = normaliseConvert2D(img_sel)
lbl_norm        = normaliseLbl(lbl_sel)
print(f"selected img label: {lbl_sel}  | normalised: {lbl_norm}")
printImg(img_norm)

### X. Init weight vectors
vec_weight_hidden    = getWeights(num_pixels, num_hidden)
vec_weight_out       = getWeights(num_hidden, num_out)
#print(f"rows weight hidden: {len(vec_weight_hidden)} | col weight hidden: {len(vec_weight_hidden[0])} ")
#print(f"some random weights:\n{vec_weight_hidden[5][20:25]}\n{vec_weight_hidden[25][250:255]}\n{vec_weight_hidden[80][560:565]}\n")

### 3. Calc neuron output for HIDDEN layer
vec_pots_hidden, vec_layer_hidden = calcNeuronOut(img_norm, vec_weight_hidden, num_hidden)

### 4. Calc neuron output for OUT layer
vec_pots_out, vec_layer_out = calcNeuronOut(vec_layer_hidden, vec_weight_out,num_out)


""" DEBUG TEST PROPAGATION
print("test hidden layer")
#print(f"DEBUG only ones:    {DEBUG_hidden_vec_img_ones}")                  # TESTED = OK
#print(f"DEBUG only zeros:  {DEBUG_hidden_vec_img_zeros}")                  # TESTED = OK
#print(f"DEBUG 1 start:      {DEBUG_hidden_vec_img_one_s}")                 # TESTED = OK
#print(f"DEBUG 1 middle:     {DEBUG_hidden_vec_img_one_m}")                 # TESTED = OK
#print(f"DEBUG 1 end:        {DEBUG_hidden_vec_img_one_l}")                 # TESTED = OK
print(f"DEBUG weights hid:  {DEBUG_mtrx_weights_hidden_ones}")
DEBUG_vec_pots_hidden = [None]*num_hidden
for n in range(num_hidden):
    DEBUG_vec_pots_hidden[n] = calcPotential(DEBUG_hidden_vec_img_one_l, DEBUG_mtrx_weights_hidden_ones[n])
print(f"\n{DEBUG_vec_pots_hidden}")

print("test out layer")
#print(f"DEBUG only ones:    {DEBUG_out_vec_img_ones}")                  # TESTED = OK
#print(f"DEBUG only zeeros:  {DEBUG_out_vec_img_zeros}")                 # TESTED = 
#print(f"DEBUG 1 start:      {DEBUG_out_vec_img_one_s}")                 # TESTED = 
#print(f"DEBUG 1 middle:     {DEBUG_out_vec_img_one_m}")                 # TESTED = 
print(f"DEBUG 1 end:        {DEBUG_out_vec_img_one_l}")                 # TESTED = 
#print(f"DEBUG weights hid:  {DEBUG_mtrx_weights_out_ones}")


DEBUG_vec_pots_out = [None]*num_out
for n in range(num_out):
    DEBUG_vec_pots_out[n] = calcPotential(DEBUG_out_vec_img_one_l, DEBUG_mtrx_weights_out_ones[n])
print(f"\n{DEBUG_vec_pots_out}")

"""


### 5. Calc error of each neuron in OUT layer       
vec_error_out = calcErrorLayerOut(vec_pots_out, lbl_norm, vec_layer_out)    
#print(f"vec error out: {vec_error_out}")


### 6. Calc error of each neuron in HIDDEN layer
## MATTIS EST CONVAINCU, MOI BOF

#DEBUG_error_out = [1]*num_out
#DEBUG_weight_out = [ [1]*num_out for i in range(num_hidden)]
#print(f"debug error out: {DEBUG_error_out}")
#print(f"debug weight out len: {DEBUG_weight_out}")
#
vec_error_hidden = [None]*num_hidden

print(f"weights out:    # rows:{len(vec_weight_out)}  # cols:{len(vec_weight_out[0])}")
print(f"weights hidden: # rows:{len(vec_weight_hidden)}  # cols:{len(vec_weight_hidden[0])}")

for n_h in range(num_hidden):
    sum_err = 0                                 # sum of each error in out layer * weight between current FOR neuron in hidden layer and error in out layer
    for n_o in range(num_out):
        #print(f" calc:{vec_error_out[n_o]} * {vec_weight_out[n_o][n_h]}")
        #print(f" calc result:{vec_error_out[n_o] * vec_weight_out[n_o][n_h]}")
        sum_err +=  vec_error_out[n_o] * vec_weight_out[n_h][n_o]
    #print(f"sum errout: {sum_errOut_wOut}")
    vec_error_hidden[n_h] = activationFuncDerived( vec_pots_hidden[n_h] ) * sum_err
#print(f"\nvec error hidden: {vec_error_hidden}")


### 7. Learn: re-calc weights from Hidden and out layer


## re-calc Hidden weights
#print(f"\nweights hidden = {vec_weight_hidden}")
for i in range(len(vec_error_hidden)):
    vec_weight_hidden[i] = updateWeights(vec_layer_in, vec_weight_hidden[i], vec_error_hidden[i], 0.01)
#print(f"\nweights hidden = {vec_weight_hidden}")


## re-calc out weights
for i in range(len(vec_error_out)):
    vec_weight_out[i] = updateWeights(vec_layer_hidden, vec_weight_out[i], vec_error_out[i], 0.01)

### 8. Calc error percentage on random sample of num_imgs imgs
"""
num_imgs = 100
mtrx_norm_imgs   = [None]*num_imgs
vec_norm_lbl    = [None]*num_imgs
for i in range(num_imgs):
    img_sel, lbl_sel    = getRndmImg(train_X,train_y)
    mtrx_norm_imgs[i]   = normaliseConvert2D(img_sel)
    vec_norm_lbl[i]     = normaliseLbl(lbl_sel)
"""




#DEBUG
"""printing the shapes of the vectors 
print('X_train: ' + str(train_X.shape))
print('Y_train: ' + str(train_y.shape))
print('X_test:  '  + str(test_X.shape))
print('Y_test:  '  + str(test_y.shape))
print('X_test:  '  + str(test_X.shape))
print('Y_test:  '  + str(test_y.shape))
"""

""" print 9 images
for i in range(9):  
    plt.subplot(330 + 1 + i)
    plt.imshow(train_X[i], cmap=plt.get_cmap('gray'))
    print(train_y[i])
plt.show()
"""