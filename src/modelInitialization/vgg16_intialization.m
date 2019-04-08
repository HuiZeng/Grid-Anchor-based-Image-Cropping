function net = vgg16_intialization(downsample,warpsize,cdim,ndim,RoIRoD)

switch downsample
    case 3 
        net = vgg16_intialization_down3(warpsize, cdim, ndim, RoIRoD);
    case 4
        net = vgg16_intialization_down4(warpsize, cdim, ndim, RoIRoD);
    case 5
        net = vgg16_intialization_down5(warpsize, cdim, ndim, RoIRoD);
end