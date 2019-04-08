function [net, info] = trainModel(varargin)


opts.batchNormalization = false ;
opts.network = [];
opts.networkType = 'dagnn' ;
opts.database = 'polyu1000';
opts.model = 'vgg16';
opts.downsample = 5;
opts.cdim = 4;
opts.ndim = 1024;
opts.warpsize = 8;
opts.trainNum = 1000;
opts.RoIRoD = 'RoIRoD';
opts.seed = 1;

[opts, varargin] = vl_argparse(opts, varargin) ;

sfx = opts.networkType ;
if opts.batchNormalization, sfx = [sfx '-bnorm'] ; end
[opts, varargin] = vl_argparse(opts, varargin) ;

opts.expDir = fullfile('data',opts.database) ;
opts.train = struct() ;
opts.train.derOutputs = {'losscls', 1} ;
opts = vl_argparse(opts, varargin) ;
if ~isfield(opts.train, 'gpus'), opts.train.gpus = [1]; end;

switch opts.RoIRoD
    case 'RoIOnly'
        opts.expDir = fullfile(opts.expDir,opts.model,'RoIOnly');
    case 'RoDOnly'
        opts.expDir = fullfile(opts.expDir,opts.model,'RoDOnly');
    case 'RoIRoD'
        opts.expDir = fullfile(opts.expDir,opts.model,'RoIRoD');
    otherwise
        error('Unknown RoIRoD choice.');    
end

%% prepare the model and data
switch opts.model
    case 'vgg16'
        net = vgg16_intialization(opts.downsample,opts.warpsize,opts.cdim,opts.ndim, opts.RoIRoD);
    case 'resnet50'
        net = resnet50_intialization(opts.downsample,opts.warpsize,opts.cdim,opts.ndim, opts.RoIRoD);
end

opts.minScale = 256;
opts.imdbPath = fullfile(['imdb_' opts.database num2str(opts.minScale) '.mat']);
opts.expDir = fullfile(opts.expDir,[num2str(opts.trainNum) '_down' num2str(opts.downsample)...
              '_warp' num2str(opts.warpsize) 'cdim' num2str(opts.cdim) 'ndim' num2str(opts.ndim) 'seed' num2str(opts.seed)]);


if strcmp(opts.database,'GAIC')
    imdb = setup_database_GAIC(opts.trainNum,opts.seed);
end


gt_scores = cat(2,imdb.bbox.gt_scores{imdb.images.set==1});
gt_scores_means = mean(gt_scores);
gt_scores_stds = std(gt_scores);
for i = 1:length(imdb.bbox.gt_scores)
    imdb.bbox.gt_scores{i} = (imdb.bbox.gt_scores{i} - gt_scores_means) / gt_scores_stds;
end
imdb.gt_scores.means = gt_scores_means;
imdb.gt_scores.stds = gt_scores_stds;

imdb.images.dataMean = net.meta.normalization.averageImage;
net.meta.minScale = opts.minScale;

% --------------------------------------------------------------------
%  Train
% --------------------------------------------------------------------

[net, info] = cnn_train_dag(net, imdb, @(i,b) getBatch_rois(opts,i,b),...
  'expDir', opts.expDir, net.meta.trainOpts, opts.train, ...
  'val', find(imdb.images.set == 3),'solver',@solver.adam);


% --------------------------------------------------------------------
function inputs = getBatch_rois(opts, imdb, batch)
% --------------------------------------------------------------------

imo = vl_imreadjpeg(imdb.meta.img_path(batch),'numThreads',1,'Contrast',0.5,'Saturation',0.5);
num_samples = 64; 
for i = 1:length(batch)
    
    [x1,x2,x3] = size(imo{i}); 
    imre = imresize(imo{i},opts.minScale/min([x1,x2]),'bilinear');
    
    [r1,r2,r3] = size(imre); 
    r1 = 32*round(r1/32);
    r2 = 32*round(r2/32);
    imre = imresize(imre,[r1,r2],'bilinear');
    
    imre = bsxfun(@minus,imre,imdb.images.dataMean);

    sel = randperm(numel(imdb.bbox.gt_scores{batch(i)}),min(num_samples,numel(imdb.bbox.gt_scores{batch(i)})));
    rois{i} = imdb.bbox.boxes{batch(i)}(:,sel);
    
    scale1 = r1/x1;
    scale2 = r2/x2;
    rois{i}(1,:) = max(floor(rois{i}(1,:) * scale1),1);
    rois{i}(2,:) = max(floor(rois{i}(2,:) * scale2),1);
    rois{i}(3,:) = min(ceil(rois{i}(3,:) * scale1),r1);
    rois{i}(4,:) = min(ceil(rois{i}(4,:) * scale2),r2);

    rois{i} = [i*ones(1,length(sel));rois{i}];
    gt_scores{i} = imdb.bbox.gt_scores{batch(i)}(sel);
end

rois = cat(2,rois{:});
gt_scores = cat(2,gt_scores{:});


if numel(opts.train.gpus) > 0
  imre = gpuArray(imre) ;
  rois = gpuArray(single(rois)) ;
end

inputs = {'input', imre, 'label', gt_scores, 'rois', rois} ;
