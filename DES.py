import cv2
import numpy as np
import math
from skimage.measure import compare_ssim as ssim
from GA_key import gakey 
from psoga import psogakey
from psodwcnga import psodwcngakey
from pso import psokey
from psodwcn import psodwcnkey
import csv
import matplotlib.pyplot as plt
img1 = cv2.imread("lena1.jpg")
img= img1.flatten()
enc_img = np.zeros(img.shape[0])
key=psodwcngakey()

print(key)
with open(r'resultkeys.csv','a') as f:
    writer=csv.writer(f)
    writer.writerow(key)


nor=16
shift_per_round = 2*np.ones(nor)
shift_per_round[0]=1
shift_per_round[1]=1
shift_per_round[8]=1
shift_per_round[15]=1
def MSE(enc_image,img):
    error=0
    for i in range(enc_image.shape[0]):
        for j in range(enc_image.shape[1]):
            error += (enc_image[i][j]-img[i][j])**2
    return error/(enc_image.shape[0]*enc_image.shape[1])
def MSE_coloured(enc_image,img):
    error=0
    for i in range(enc_image.shape[0]):
        for j in range(enc_image.shape[1]):
            for k in range(enc_image.shape[2]):
                error += (enc_image[i][j][k]-img[i][j][k])**2
    return error/(enc_image.shape[0]*enc_image.shape[1]*enc_image.shape[2])
def PSNR(enc_image,img):
    error=0
    for i in range(enc_image.shape[0]):
        for j in range(enc_image.shape[1]):
            error += (enc_image[i][j]-img[i][j])**2
    return 20*(math.log(255,10))-10*(math.log(error/(enc_image.shape[0]*enc_image.shape[1]),10))
def PSNR_coloured(enc_image,img):
    error=0
    for i in range(enc_image.shape[0]):
        for j in range(enc_image.shape[1]):
            for k in range(enc_image.shape[2]):
                error += (enc_image[i][j][k]-img[i][j][k])**2
                
    return 30*(math.log(255,10))-10*(math.log(error/(enc_image.shape[0]*enc_image.shape[1]*enc_image.shape[2]),10))
#----------------------------Generate key-----------------------
"""
for i in range(8):
    temp = format(np.random.randint(256),'#010b')
    key += temp[2:]
print("key=",key)
"""
#----------------------------Remove parity positions in key--------------------------
parity_drop = np.array([57,49,41,33,25,17,9,1,58,50,42,34,26,18,10,2,59,51,43,35,27,19,11,3,60,52,44,36,63,55,47,39,31,23,15,7,62,54,46,38,30,22,14,6,61,53,45,37,29,21,13,5,28,20,12,4])
new_key = ''
for i in range(parity_drop.shape[0]):
    new_key += key[parity_drop[i]-1]
compress_key = np.array([14,17,11,24,1,5,3,28,15,6,21,10,23,19,12,4,26,8,16,7,27,20,13,2,41,52,31,37,47,55,30,40,51,45,33,48,44,49,39,56,34,53,46,42,50,36,29,32])

    
#-----------------Initial Permutation of plain text----------------------------------
init_per = np.array([58,50,42,34,26,18,10,2,60,52,44,36,28,20,12,4,62,54,46,38,30,22,14,6,64,56,48,40,32,24,16,8,57,49,41,33,25,17,9,1,59,51,43,35,27,19,11,3,61,53,45,37,29,21,13,5,63,55,47,39,31,23,15,7])
rev_per = np.array([40,8,48,16,56,24,64,32,39,7,47,15,55,23,63,31,38,6,46,14,54,22,62,30,37,5,45,13,53,21,61,29,36,4,44,12,52,20,60,28,35,3,43,11,51,19,59,27,34,2,42,10,50,18,58,26,33,1,41,9,49,17,57,25])
#-----------------Right side expansion-----------------------------------------------
right_side_expansion= np.array([32,1,2,3,4,5,4,5,6,7,8,9,8,9,10,11,12,13,12,13,14,15,16,17,16,17,18,19,20,21,20,21,22,23,24,25,24,25,26,27,28,29,28,29,30,31,32,1])
s_box_comp = np.array([[ 
        [14,4,13,1,2,15,11,8,3,10,6,12,5,9,0,7], 
        [0,15,7,4,14,2,13,1,10,6,12,11,9,5,3,8], 
        [4,1,14,8,13,6,2,11,15,12,9,7,3,10,5,0], 
        [15,12,8,2,4,9,1,7,5,11,3,14,10,0,6,13] 
    ], 
    [ 
        [15,1,8,14,6,11,3,4,9,7,2,13,12,0,5,10], 
        [3,13,4,7,15,2,8,14,12,0,1,10,6,9,11,5], 
        [0,14,7,11,10,4,13,1,5,8,12,6,9,3,2,15], 
        [13,8,10,1,3,15,4,2,11,6,7,12,0,5,14,9] 
    ], 
  
  
    [ 
        [10,0,9,14,6,3,15,5,1,13,12,7,11,4,2,8], 
        [13,7,0,9,3,4,6,10,2,8,5,14,12,11,15,1], 
        [13,6,4,9,8,15,3,0,11,1,2,12,5,10,14,7], 
        [1,10,13,0,6,9,8,7,4,15,14,3,11,5,2,12] 
    ], 
    [ 
        [7,13,14,3,0,6,9,10,1,2,8,5,11,12,4,15], 
        [13,8,11,5,6,15,0,3,4,7,2,12,1,10,14,9], 
        [10,6,9,0,12,11,7,13,15,1,3,14,5,2,8,4], 
        [3,15,0,6,10,1,13,8,9,4,5,11,12,7,2,14] 
    ], 
    [ 
        [2,12,4,1,7,10,11,6,8,5,3,15,13,0,14,9], 
        [14,11,2,12,4,7,13,1,5,0,15,10,3,9,8,6], 
        [4,2,1,11,10,13,7,8,15,9,12,5,6,3,0,14], 
        [11,8,12,7,1,14,2,13,6,15,0,9,10,4,5,3] 
    ], 
    [ 
        [12,1,10,15,9,2,6,8,0,13,3,4,14,7,5,11], 
        [10,15,4,2,7,12,9,5,6,1,13,14,0,11,3,8], 
        [9,14,15,5,2,8,12,3,7,0,4,10,1,13,11,6], 
        [4,3,2,12,9,5,15,10,11,14,1,7,6,0,8,13] 
    ], 
    [ 
        [4,11,2,14,15,0,8,13,3,12,9,7,5,10,6,1], 
        [13,0,11,7,4,9,1,10,14,3,5,12,2,15,8,6], 
        [1,4,11,13,12,3,7,14,10,15,6,8,0,5,9,2], 
        [6,11,13,8,1,4,10,7,9,5,0,15,14,2,3,12] 
    ], 
    [ 
        [13,2,8,4,6,15,11,1,10,9,3,14,5,0,12,7], 
        [1,15,13,8,10,3,7,4,12,5,6,11,0,14,9,2], 
        [7,11,4,1,9,12,14,2,0,6,10,13,15,3,5,8], 
        [2,1,14,7,4,10,8,13,15,12,9,0,3,5,6,11] 
    ]])

for i in range(0,img.shape[0],8):
    sub_array = img[i:i+8]
    plain_text = ''
    for k in range(8):
        temp = format(sub_array[k],'#010b')
        plain_text += temp[2:]
       
    #---------perform initial permutation---------------------------------
    plain_text1 = ''
    for k in range(64):
        plain_text1 += plain_text[init_per[k]-1]
    plain_text = plain_text1
    #---------------- Split the plain_text into two halves-----------------
    left_half = plain_text[0:32]
    right_half = plain_text[32:]
    #-----------------Generate key-----------------------------------------
    key_left = new_key[0:28]
    key_right = new_key[28:]
    #-----------------for each round---------------------------------------
    for k in range(nor):
        shifts = int(shift_per_round[k])
        key_left = key_left[shifts:] + key_left[0:shifts]
        key_right = key_right[28-shifts:] + key_right[0:28-shifts]
        new_key1 = key_left+ key_right
        com_key = ''
        for l in range(compress_key.shape[0]):
            com_key += new_key1[compress_key[l]-1]
          
        exp_right_half = ''
        for l in range(right_side_expansion.shape[0]):  
            exp_right_half += right_half[right_side_expansion[l]-1]
            
        #-----------------XOR between right side and key-------------------
        var = int(exp_right_half,2)^int(com_key,2)
        var = "{:048b}".format(var)
        s_box = ''
        for l1 in range(8):
            l = 6*l1
            num1 = int(var[l]+var[l+5],2)
            num2 = int(var[l+1:l+5],2)
            new_str = s_box_comp[l1][num1][num2]        
            s_box += "{:04b}".format(new_str)
        var = int(s_box,2)^int(left_half,2)
        var = "{:032b}".format(var)
        left_half = right_half
        right_half = var
        
    cipher_text = left_half + right_half
    #-----------------Perform final permutation----------------------------
    cipher_text1=''
    for k in range(64):
        cipher_text1 += cipher_text[rev_per[k]-1]
    cipher_text = cipher_text1
       
    for k in range(8):
        num1 = cipher_text[8*k:8*k+8]
        num1 = int(num1,2)
        enc_img[i+k] = num1
enc_img = np.reshape(enc_img,(img1.shape[0],img1.shape[1],img1.shape[2]))
mse = MSE_coloured(enc_img,img1)
psnr = PSNR_coloured(enc_img,img1)
sim=ssim(enc_img,img1,multichannel=True)
print("mse =",mse)
print("psnr =",psnr)
ans=[mse,psnr,sim]
plt.hist(enc_img[0])

plt.savefig("lena_.jpg_encr(r).png")
plt.hist(enc_img[1])

plt.savefig("lena.jpg_encr(g).png")
plt.hist(enc_img[2])

plt.savefig("lena.jpg_encr(b).png")


with open(r'results.csv','a') as f:
    writer=csv.writer(f)
    writer.writerow(ans)

np.savetxt('test.csv',enc_img.flatten(), delimiter=',')
cv2.imwrite("lena_enc.jpg",enc_img)


        

        
        
 

