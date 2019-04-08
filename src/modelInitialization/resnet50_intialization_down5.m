function net = resnet50_intialization_down5(warpsize,cdim,ndim,RoIRoD)

netStruct = load('data/pretrained_models/imagenet-resnet-50-dag.mat') ;
net = dagnn.DagNN.loadobj(netStruct) ;
net.layers = net.layers(1:152);
net.renameVar('data','input');
net.rebuild();

% Dimension reduction
net.addLayer('dimred', convBlocknoPad(1,1,2048,cdim), {'res5ax'}, {'xRed'}, {'dimred_filter','dimred_bias'});
net = initLayerParams(net);

% RoIAlign
if strcmp(RoIRoD,'RoIRoD') || strcmp(RoIRoD,'RoIOnly')
    roiSize = [warpsize,warpsize]; 
    net.addLayer('roialign', dagnn.RoiAlign('subdivisions',roiSize,'transformation',1/32), {'xRed','rois'}, 'xRoi');
    net.addLayer('roipool', dagnn.Pooling('method', 'avg', 'poolSize', [2 2], 'stride', 1), {'xRoi'}, {'pool_xRoi'}, {});
end

% RoDAlign
if strcmp(RoIRoD,'RoIRoD') || strcmp(RoIRoD,'RoDOnly')
    rodSize = [warpsize+1,warpsize+1]; 
    net.addLayer('rodalign', dagnn.RodAlign('subdivisions',rodSize,'transformation',1/32), {'xRed','rois'}, 'xRod');
    net.addLayer('rodpool', dagnn.Pooling('method', 'avg', 'poolSize', [2 2], 'stride', 1), {'xRod'}, {'pool_xRod'}, {});
end


switch RoIRoD
    case 'RoIRoD'
        net.addLayer('concat', dagnn.Concat('dim', 3), {'pool_xRoi', 'pool_xRod'}, {'xConc'}, {});
        net.addLayer('fc1', convBlocknoPad(warpsize,warpsize,cdim*2,ndim), {'xConc'}, {'fc1'}, {'fc1_filter', 'fc1_bias'});
    case 'RoIOnly'
        net.addLayer('fc1', convBlocknoPad(warpsize,warpsize,cdim,ndim), {'pool_xRoi'}, {'fc1'}, {'fc1_filter', 'fc1_bias'});
    case 'RoDOnly'
        net.addLayer('fc1', convBlocknoPad(warpsize,warpsize,cdim,ndim), {'pool_xRod'}, {'fc1'}, {'fc1_filter', 'fc1_bias'});
    otherwise
        error('Unknown RoIRoD choice.');
end
net = initLayerParams(net);
net.addLayer('relu_fc1', dagnn.ReLU(), {'fc1'}, {'relu_fc1'}, {});
net.addLayer('fc2', convBlocknoPad(1,1,ndim,ndim), {'relu_fc1'}, {'fc2'}, {'fc2_filter', 'fc2_bias'});
net = initLayerParams(net);
net.addLayer('relu_fc2', dagnn.ReLU(), {'fc2'}, {'relu_fc2'}, {});

net.addLayer('dropout',dagnn.DropOut('rate',0.5),'relu_fc2','dropout',{});

net.addLayer('predcls',convBlocknoPad(1,1,ndim,1), 'dropout','predcls',{'predcls_filter','predcls_bias'});
net = initLayerParams(net);

% net.print({'input', [256 256 3]}, 'all', true);
net.addLayer('losscls',dagnn.RegressionLoss('lossType','Huber'), {'predcls','label'}, 'losscls',{});

% Meta parameters
net.meta.trainOpts.learningRate = logspace(-4,-4,25);%logspace(-4,-5,10)
net.meta.trainOpts.weightDecay = 0.0001;
net.meta.trainOpts.batchSize = 1 ;
net.meta.trainOpts.solver = @solver.adam ;
net.meta.trainOpts.numEpochs = numel(net.meta.trainOpts.learningRate) ;
net.meta.normalization.averageImage = reshape([122.7717 102.9801 115.9465],[1 1 3]);


end

function convObj = convBlocknoPad(fh,fw,fc,k)
    convObj = dagnn.Conv('size', [fh fw fc k], 'hasBias', true);
end

function net = initLayerParams(net)
    p = net.getParamIndex(net.layers(end).params) ;
    params = net.layers(end).block.initParams();
    [net.params(p).value] = deal(params{:}) ;
end
