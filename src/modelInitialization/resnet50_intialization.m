function net = resnet50_intialization(downsample,warpsize,cdim,ndim,RoIRoD)

switch downsample
    case 3 
        net = resnet50_intialization_down3(warpsize,cdim,ndim,RoIRoD);
    case 4
        net = resnet50_intialization_down4(warpsize,cdim,ndim,RoIRoD);
    case 5
        net = resnet50_intialization_down5(warpsize,cdim,ndim,RoIRoD);
end