function net = resnet50_intialization_down3()

netStruct = load('data/pretrained_models/imagenet-resnet-50-dag.mat') ;
net = dagnn.DagNN.loadobj(netStruct) ;
net.layers = net.layers(1:58);
net.renameVar('data','input');
net.rebuild();

ch = 32;
net.addLayer('dimred', convBlocknoPad(1,1,512,ch), {'res3bx'}, {'xRed'}, {'dimred_filter','dimred_bias'});
net = initLayerParams(net);
    
roiSize = [8,8]; 
net.addLayer('roiresize', dagnn.RoiResize('subdivisions',roiSize,'transformation',1/8), {'xRed','rois'}, 'xRoi');

rodSize = [8,8]; 
net.addLayer('rodresize', dagnn.RodResize('subdivisions',rodSize,'transformation',1/8), {'xRed','rois'}, 'xRod');

net.addLayer('concat', dagnn.Concat('dim', 3), {'xRoi', 'xRod'}, {'xConc'}, {});
 
net.addLayer('fc1', convBlocknoPad(8,8,ch*2,1024), {'xConc'}, {'fc1'}, {'fc1_filter', 'fc1_bias'});
net = initLayerParams(net);
net.addLayer('relu_fc1', dagnn.ReLU(), {'fc1'}, {'relu_fc1'}, {});

net.addLayer('fc2', convBlocknoPad(1,1,1024,128), {'relu_fc1'}, {'fc2'}, {'fc2_filter', 'fc2_bias'});
net = initLayerParams(net);
net.addLayer('relu_fc2', dagnn.ReLU(), {'fc2'}, {'relu_fc2'}, {});

net.addLayer('dropout',dagnn.DropOut('rate',0.5),'relu_fc2','dropout',{});

net.addLayer('predcls',convBlocknoPad(1,1,128,1), 'dropout','predcls',{'predcls_filter','predcls_bias'});
net = initLayerParams(net);

% net.print({'input', [256 256 3]}, 'all', true);
net.addLayer('losscls',dagnn.RegressionLoss('lossType','Huber'), {'predcls','label'}, 'losscls',{});

[s1,s2] = paramSize(net);

% Meta parameters
net.meta.trainOpts.learningRate = logspace(-4,-4,25);%logspace(-4,-5,10)
net.meta.trainOpts.weightDecay = 0.0001;
net.meta.trainOpts.batchSize = 1 ;
net.meta.trainOpts.solver = @solver.adam ;
net.meta.trainOpts.numEpochs = numel(net.meta.trainOpts.learningRate) ;
net.meta.normalization.averageImage = reshape([122.7717 102.9801 115.9465],[1 1 3]);
% net.meta.transformation = 1/16;

end

function convObj = convBlocknoPad(fh,fw,fc,k)
    convObj = dagnn.Conv('size', [fh fw fc k], 'hasBias', true);
end

function net = initLayerParams(net)
    p = net.getParamIndex(net.layers(end).params) ;
    params = net.layers(end).block.initParams();
    [net.params(p).value] = deal(params{:}) ;
end

