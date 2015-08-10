# posys
Multi Camera Positioning System based on Convolution Net

Importance Class
camera.h   handle serialize and store  camera len distortion 
ConvNet.h  CNN stuff , prefix par_ mean that it's used in parallel processing

Importance Function in main.cpp
calibrateCamera() calibrate len distortion
cm_223()   convert ( 2D point + Camera Info ) -> 3D point
cm_322()   convert 3D point - project on z=0 plane -> 2D point
cm_plot_its_obeam()
cm_plot_obeam()
cm_plot_tdp()
computerIntersect()
count_white()
cross2d()
cvt2bw()
cvt2clr()
distance()
do_bgsub()
do_bgsub_edge()
do_bgsub_gabor()
do_bgsub_hsv()
do_conv()
do_detect()
do_dilate()
do_edge()
do_erode()
do_filter()
do_get_cammat()
do_kmean_corner()
do_label()
do_pyramid()
do_rotate()
do_sortCorners()
do_validate_corners()
dot()
draw_triangle()
error()
find_moving_obj()
foundCorners()
getConfigPath()

getTrainingSetPath()
get_aabb_lists()
getframe()
help()
kalman_filter()
load_cifar10()
load_cifar100()
load_cifar100_label()
load_cifar10_label()
load_testimage()
pl_draw_mark()
pl_get_bk()
printMat()
readMat()
read_file_binary()
renderCube()
selfcalib()
selfcalib_dummy()
sortCorners()
str_concat()
task1()
unDist()
writeMat()
write_file_binary()





In Progress


Done
- init code from opencv sample  ( opencv + qt + opengl )
- multi-camera calibrate & configure 
- cifar training set cifar100
- port subset of ConvNetJS into c++ version
- 

TODO
- room information parser
- room model
- camera model
- construct pixel into mesh
- mesh intersection
- normalize image to fit ConvNet
- visualizer






