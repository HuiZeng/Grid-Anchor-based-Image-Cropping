function testModel(downsample,model,trainNum,warpsize,cdim,ndim,seed,RoIRoD)

switch model
    case 'vgg16'
        Epoches = 40;
    case 'resnet50'
        Epoches = 20;
end

modelDir = fullfile('data','GAIC',model,RoIRoD,[num2str(trainNum) '_down' num2str(downsample)...
           '_warp' num2str(warpsize) 'cdim' num2str(cdim) 'ndim' num2str(ndim) 'seed' num2str(seed)]);

load(['imdb_GAIC',num2str(trainNum),'.mat'],'imdb');
gt_scores = cat(2,imdb.bbox.gt_scores{imdb.images.set==1});
gt_scores_means = mean(gt_scores);
gt_scores_stds = std(gt_scores);

for epoch = 1:Epoches
    acc5 = zeros(200,4);
    acc10 = zeros(200,4);
    fprintf('processing epoch %d\n',epoch);
    netStruct = load(fullfile(modelDir,['net-epoch-',num2str(epoch) '.mat']),'net');
    net = dagnn.DagNN.loadobj(netStruct.net) ;
    net.mode = 'test' ;
    net.move('gpu') ;
    probVarI = net.getVarIndex('predcls');
    net.vars(probVarI).precious = 1;
    minScale = net.meta.minScale;
    
    testID = imdb.images.testSet;
    tic;
    for i = 1:numel(testID)
        img = imread(imdb.meta.img_path{testID(i)});
        [x1,x2,x3] = size(img);
        boxes = imdb.bbox.boxes{testID(i)};
        ss = (boxes(3,:)-boxes(1,:)).*(boxes(4,:)-boxes(2,:));

        imre = single(imresize(img,minScale/min([x1,x2]),'bilinear'));
        imre = bsxfun(@minus,imre,net.meta.normalization.averageImage);
        
        [r1,r2,r3] = size(imre); 
        r1 = 32*round(r1/32);
        r2 = 32*round(r2/32);
        imre = imresize(imre,[r1,r2],'bilinear');

        scale1 = r1/x1;
        scale2 = r2/x2;
        boxes_s = [];
        boxes_s(1,:) = max(floor(boxes(1,:) * scale1),1);
        boxes_s(2,:) = max(floor(boxes(2,:) * scale2),1);
        boxes_s(3,:) = min(ceil(boxes(3,:) * scale1),r1);
        boxes_s(4,:) = min(ceil(boxes(4,:) * scale2),r2);
        
        
        inputs = {'input', gpuArray(imre), 'rois', gpuArray(single([ones(1,size(boxes_s,2));boxes_s]))} ;
        
        net.eval(inputs) ;
        preds{i} = squeeze(gather(net.vars(probVarI).value)) ;
        preds{i} = preds{i} * gt_scores_stds + gt_scores_means;
        gts = imdb.bbox.gt_scores{testID(i)};
        
        srcc(i) = corr(preds{i}, gts', 'type', 'Spearman');
        
        [gts_sorted,id_gts] = sort(gts,'descend');
        [~,id_preds] = sort(preds{i},'descend');
        [~,id_baseline] = sort(ss,'descend');

        for k = 1:4
            for j = 1:k
                if gts(id_preds(j)) >= gts_sorted(5)
                    acc5(i,k) = acc5(i,k) + 1;
                end
            end
            acc5(i,k) = acc5(i,k) / k;
        end
        
        for k = 1:4
            for j = 1:k
                if gts(id_preds(j)) >= gts_sorted(10)
                    acc10(i,k) = acc10(i,k) + 1;
                end
            end
            acc10(i,k) = acc10(i,k) / k;
        end
    end
    toc;
    Acc5(epoch,:) = sum(acc5)/numel(testID);
    Acc10(epoch,:) = sum(acc10)/numel(testID);
    SRCC(epoch) = mean(srcc);
end
AllRes = cat(2,Acc5,mean(Acc5,2),Acc10,mean(Acc10,2),SRCC');
save(fullfile(modelDir,'Result.mat'),'Acc5','Acc10','SRCC','AllRes');

end

