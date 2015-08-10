# posys
Multi Camera Positioning System based on Convolution Net

Importance Class
 - camera.h   handle serialize and store  camera len distortion 
 - ConvNet.h  CNN stuff , prefix par_ mean that it's used in parallel processing
 - cammat  store calibration configuration
 - obeam   object frustum

Importance Function in main.cpp
 - calibrateCamera() calibrate len distortion
 - cm_223()   convert ( 2D point + Camera Info ) -> 3D point
 - cm_322()   convert 3D point - project on z=0 plane -> 2D point
 - cm_plot_its_obeam()  plot intersect 2 obeams
 - cm_plot_obeam()  plot obeams
 - cm_plot_tdp()  plot 3D point
 - computerIntersect() compute intersect
 - count_white()  count white pixels
 - cross2d()  mathematic function
 - cvt2bw()  convert cv:Mat to black-white image
 - cvt2clr()  convert cv:Mat to color
 - distance()  perform euler distance
 - do_bgsub()  background subtract << ***
 - do_bgsub_edge()  filter apply before do bgsub
 - do_bgsub_gabor()  filter apply before do bgsub
 - do_bgsub_hsv()  filter apply before do bgsub
 - do_conv()  find convolution
 - do_dilate() normal dilate
 - do_erode() normal erode
 - do_get_cammat()  find cammat
 - do_kmean_corner()  find corner by kmean
 - do_pyramid()  resize image to multi size
 - do_rotate()  roate image
 - do_sortCorners()  sort corner CW order
 - do_validate_corners()  check corners
 - draw_triangle()  draw triangle
 - find_moving_obj()  find moving object
 - getConfigPath()  return configuration path
 - getTrainingSetPath()  return training set path
 - load_cifar10()  load training set
 - load_cifar100()  load training set
 - load_cifar100_label()  load training label
 - load_cifar10_label()  load training label
 - load_testimage()  load test image
 - pl_draw_mark() draw center calibration mark
 - pl_get_bk()  draw background
 - printMat()  print Matrix
 - readMat() read Matrix
 - read_file_binary()  
 - renderCube()
 - selfcalib()
 - selfcalib_dummy()
 - sortCorners()
 - str_concat()
 - task1()
 - unDist()
 - writeMat()
 - write_file_binary()





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






