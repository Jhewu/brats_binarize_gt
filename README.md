This script binarizes the ground truth data
which is numpy file with 0-3 values, by binarizing on 
values greater than 0, thus creating a mask of the whole tumor. 
This mask data is then used to train YOLOv8-seg model 