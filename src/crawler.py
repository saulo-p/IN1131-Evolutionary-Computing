import urllib as ul

base_url = 'http://www.ux.uis.no/~tranden/brodatz/D'
dataset_size = 112

for i in range(1,dataset_size+1):
    ul.urlretrieve(base_url+str(i)+'.gif', './../data/brodatz/D'+str(i)+'.gif')

#>Data preparation script: ----------------------------------------------

#It is necessary to convert the images to .bmp before running this script
# OpenCV does not support .gif images

# # Resample Brodatz input images
# data_path = './../data/brodatz/'

# WS = 64
# for i in range(1,112+1):
#     if (i == 14):
#         continue

#     img = cv2.imread(data_path + 'D' + str(i) +'.bmp', cv2.IMREAD_GRAYSCALE)

#     for rb in range(0, 10):
#         for cb in range(0, 10):
#             img_block = img[rb*WS:(rb+1)*WS, cb*WS:(cb+1)*WS]
#             # cv2.imshow('Test Image', img_block)
#             # cv2.waitKey(0)

#             cv2.imwrite(data_path + 'resampled/D' +
#               str(i) + '_' + str(rb) + '_' + str(cb) + '.bmp' , img_block)