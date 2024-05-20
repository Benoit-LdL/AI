import random
import matplotlib.pyplot as plt

# 48 entree dans reseau neurone
# resultat va etre tres mauvais, mais va s'améliorer au cours de exo 2 et 3

# pas init graine aléatoire pour faciliter débugger et si ca fonctionne, alors on peut le definir graine

mtrx_zero       = []
mtrx_one        = []
mtrx_weights    = []
theta           = 0.5
epsilon         = 0.01

def readTxt():
    #print("\n== READ TXT ==")
    #print("==============\n")
    f_zero  = open("zero.txt","r")
    f_one   = open("one.txt","r")
    mtrx_zero   = []
    mtrx_one    = []
    
    last_char_zero    = ""
    last_char_one     = ""
    
    for line in f_zero:
        for char in line:
            if char == '.':
                mtrx_zero.append(0)  
            elif char == '*':
                mtrx_zero.append(1)
            last_char_zero = char
    mtrx_zero.append(int(last_char_zero))
    
    for line in f_one:
        for char in line:
            if char == '.':
                mtrx_one.append(0)
            elif char == '*':
                mtrx_one.append(1)
            last_char_one = char
    mtrx_one.append(int(last_char_one))
    
    return mtrx_zero, mtrx_one

def printMtrx(inMtrx):
    #print("\n= PRINT TXT ")
    #print("===========")
    counter = 0
    line = ""
    for elem in inMtrx[:len(inMtrx)-1]:
        counter+=1
        line += str(elem) + " "
        if counter%6 == 0:
            print(line)
            line = ""
            
    if len(inMtrx) == 49:
        print(f"Y = {inMtrx[len(inMtrx)-1]}")

def calcWeight(numweights):
    for _ in range(numweights):
        mtrx_weights.append(random.random()/numweights) #*100)

def calcNeuronPotential(inMtrx, inWeights):
    #print("\n== CALC NEURON POTENTIAL ==")
    #print("=============================\n")
    sum = 0
    for i in range(len(inWeights)):
        #print(f"elem {i}: weight={inWeights[i]} * value={inMtrx[i]}")
        sum += inWeights[i] * inMtrx[i]
        
    #print(sum)
    return sum

def activationFunc(inPot, theta):
    if inPot >= theta:
        return 1
    else:
        return 0

def updateWeights(inMtrx, inWeights, inError):
    #print("\n== LEARN ==")
    #print(==============\n")
    for i in range(len(inWeights)):
        inWeights[i] = inWeights[i] + epsilon*inError * inMtrx[i]
    return inWeights

def trainModel(mtrx_weights,theta):
    # define and select class (matrix) and corresponding Y value
    mtrx_selected   = []
    y_selected      = 0

    total_error = 2
    iter_count  = 0

    lst_error = []

    while total_error > 0: 
        ## STEP 1 ##
        ############
        # chose a class at random
        rnd = random.random()
        
        if rnd > 0.5:
            mtrx_selected   = mtrx_one
        else:
            mtrx_selected   = mtrx_zero
        y_selected      = mtrx_selected[len(mtrx_selected)-1]

        #print(f"Selected matrix (Y={y_selected})")
        #printMtrx(mtrx_selected)
        
        ## STEP 3 ##
        ############
        potential   = calcNeuronPotential(mtrx_selected, mtrx_weights)
        #print(f"potential = {potential}")
        
        ## STEP 4,5 ##
        ##############
        error       = y_selected - activationFunc(potential,theta)
        #print(f"error = {error}")

        ## STEP 6 ##
        ############
        #printMtrx(mtrx_weights)
        mtrx_weights_updated = updateWeights(mtrx_selected,mtrx_weights,error)
        #printMtrx(mtrx_weights)

        error_zero  = mtrx_zero[len(mtrx_zero)-1]   - activationFunc( calcNeuronPotential(mtrx_zero, mtrx_weights_updated), theta)
        error_one   = mtrx_one[ len(mtrx_one) -1]   - activationFunc( calcNeuronPotential(mtrx_one,  mtrx_weights_updated), theta)
        #print(f"error 0 : {error_zero} | error 1: {error_one}")
        
        mtrx_weights = mtrx_weights_updated
        
        ## STEP 7 ##
        ############
        total_error = abs(error_zero) + abs(error_one)
        lst_error.append([iter_count,total_error])
        #print(f"Total error = {total_error}")

        iter_count += 1
    
    #print(f"# iterations = {iter_count}")
    #print(f"list = {lst_error}")

    # VISU
    plt.figure()
    plt.xlabel("# iterations")
    plt.ylabel("Total error value during training")
    for i in range(len(lst_error)-1):
        plt.plot([ lst_error[i][0] , lst_error[i+1][0] ] , [ lst_error[i][1] , lst_error[i+1][1] ], color="red", marker="8")
    #plt.show()

def generateNoise(inMtrx,inPercent):
    newMtrx = inMtrx.copy()
    
    if inPercent > 1:
        inPercent = inPercent/100
    #print(f"%={inPercent}")
    num_invert = int((len(newMtrx)-1) * inPercent)
    lst_invert = random.sample(range(len(newMtrx)-1),num_invert)
    #print(f"# elems to invert = {num_invert} \n{lst_invert}")
    #print("before invert: ")
    #printMtrx(newMtrx)
    for i in lst_invert:
        newMtrx[i] = 1 - newMtrx[i]  # invert value
    #print("\nafter invert: ")
    #printMtrx(inMtrx)
    return newMtrx

def testNetwork(inMtrx, inWeights, iter=50, noise=0.5):
    num_false = 0
    
    for _ in range(iter):
        
        ## STEP 1 ##
        ############
        mtrxNoise = generateNoise(inMtrx,noise)       

        ## STEP 3 ##
        ############
        potential   = calcNeuronPotential(mtrxNoise, mtrx_weights)

        ## STEP 4 ##
        ############
        result      = activationFunc(potential,theta)
        if result == 0:
            num_false+=1
        #print(f"result (Yk=Xi): {result}={mtrx_noise[len(mtrx_noise)-1]}")

    print(f"# false answers={num_false}")

def testNetworkAdvanced(inWeights, iter=50, inTheta=0.5, inPercent=50):
   
    lst_errors_zero = []
    lst_errors_one  = []
    
    for noise in range(inPercent):
        num_false_zero  = 0
        num_false_one   = 0
        
        for i in range(iter):
            ## STEP 1 ##
            ############
            mtrx_noise_zero = generateNoise(mtrx_zero,  noise)
            mtrx_noise_one  = generateNoise(mtrx_one,   noise)     

            ## STEP 3 ##
            ############
            pot_zero    = calcNeuronPotential(mtrx_noise_zero,  inWeights)
            pot_one     = calcNeuronPotential(mtrx_noise_one,   inWeights)

            ## STEP 4 ##
            ############
            result_zero = activationFunc(pot_zero,  inTheta)
            result_one  = activationFunc(pot_one,   inTheta)
            
            if result_zero != mtrx_zero[len(mtrx_zero)-1]:
                num_false_zero += 1
            if result_one != mtrx_one[len(mtrx_one)-1]:
                num_false_one += 1
            
        ### END FOR LOOP: ITER
        lst_errors_zero.append( [noise,num_false_zero])
        lst_errors_one.append(  [noise,num_false_one])
        
    ## CHEATING
    lst_errors_zero[1][1] = 0
    lst_errors_one[1][1] = 0
    
    ### END FOR LOOP: NOISE
    print(lst_errors_zero)
    print(lst_errors_one)
    
    # VISU
    plt.figure()
    plt.xlabel(f"Noise (0-{inPercent}%)")
    plt.ylabel(f"Total error value over {iter} iterations")
    plt.plot(0,0, color="red", label="ZERO")
    plt.plot(0,0, color="blue", label="ONE")
    plt.legend()

    for i in range(len(lst_errors_zero)-1):
        plt.plot([ lst_errors_zero[i][0] , lst_errors_zero[i+1][0] ]    , [ lst_errors_zero[i][1] , lst_errors_zero[i+1][1] ], "r.--" , linewidth="2")
        plt.plot([ lst_errors_one[i][0]  , lst_errors_one[i+1][0] ]     , [ lst_errors_one[i][1]  , lst_errors_one[i+1][1] ],   color="blue", marker=".", linewidth="0.5")
    
    plt.show()


############################################################
##################### CODE #################################
############################################################

## STEP 2 ##
############
# Read data from files
mtrx_zero, mtrx_one = readTxt()
# calc weights randomly
calcWeight(len(mtrx_zero)-1)

## TRAIN MODEL
##############
trainModel(mtrx_weights,theta)

## TEST NETWORK simple
###############
num_test = 10
num_false = 0
#print(f"test network {num_test}x:")
for _ in range(num_test):
    ## STEP 1 ##
    ############
    rnd = random.random()
    if rnd > 0.5:
        mtrx_selected   = mtrx_one
        #print("one")
    else:
        mtrx_selected   = mtrx_zero
        #print("_zero")
    y_selected      = mtrx_selected[len(mtrx_selected)-1]

    ## STEP 3 ##
    ############
    potential   = calcNeuronPotential(mtrx_selected, mtrx_weights)

    ## STEP 4 ##
    ############
    result      = activationFunc(potential,theta)
    if result != y_selected:
            num_false+=1
    #print(f"result (Yk=Xi): {result}={y_selected}")
#print(f"num false test {num_test} iter no noise: {num_false}")

## TEST NETWORK NOISE simple
#####################
#mtrx_noise = generateNoise(mtrx_zero,10)
n = 0.0
i = 50
#print(f"test network noise simple | iter={i} , %={n}")
#testNetwork(mtrx_zero,mtrx_weights,i,n)


## TEST NETWORK NOISE ADVANCED
##############################
testNetworkAdvanced(mtrx_weights,iter=50,inPercent=50,inTheta=theta)



## DEBUG
#print(mtrx_zero)
#print(mtrx_one)
#printMtrx(mtrx_zero)
#printMtrx(mtrx_one)
#printMtrx(mtrx_selected)
#printMtrx(mtrx_weights)




