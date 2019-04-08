% This file is designed for qualitative testing on the GAIC data. Note that
% images in GAIC were preprocessed to have resolution no larger than 1024.


function testGAIC_qualitative()

load(['imdb_GAIC1000.mat'],'imdb');
gt_scores = cat(2,imdb.bbox.gt_scores{imdb.images.set==1});
gt_scores_means = mean(gt_scores);
gt_scores_stds = std(gt_scores);

netStruct = load(fullfile('data','pretrained_models','net-epoch-37.mat'),'net');
net = dagnn.DagNN.loadobj(netStruct.net) ;
net.mode = 'test' ;
net.move('gpu') ;
probVarI = net.getVarIndex('predcls');
net.vars(probVarI).precious = 1;
minScale = net.meta.minScale;

testID = imdb.images.testSet;

for i = 1:numel(testID)
    img = imread(imdb.meta.img_path{testID(i)});
    [x1,x2,x3] = size(img);

    %% choose different functions to generate candidates
    boxes = generateBoxes_b12(img);

    if isempty(boxes)
        continue;
    end

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
    boxes_s(3,:) = min(ceil(boxes(3,:) * scale1),size(imre,1));
    boxes_s(4,:) = min(ceil(boxes(4,:) * scale2),size(imre,2));
    inputs = {'input', gpuArray(imre), 'rois', gpuArray(single([ones(1,size(boxes_s,2));boxes_s]))} ;

    net.eval(inputs) ;
    preds = squeeze(gather(net.vars(probVarI).value)) ;
    preds = preds * gt_scores_stds + gt_scores_means;
    [~,id_preds] = sort(preds,'descend');

    predBox = boxes(:,id_preds(1));
    outDir = fullfile('dataset','GAIC','result','ours');
    mkdir(outDir);

%     copyfile(imdb.meta.img_path{testID(i)},fullfile(outDir,imdb.meta.img_path{testID(i)}(end-9:end)));
    imwrite(img(predBox(1):predBox(3),predBox(2):predBox(4),:),...
        fullfile(outDir,[imdb.meta.img_path{testID(i)}(end-9:end-4) '_preds_b12.jpg']));
%         imwrite(img(gtBox(1):gtBox(3),gtBox(2):gtBox(4),:),fullfile(outDir,[imdb.meta.img_path{testID(i)}(end-9:end-4) '_groundtruth.jpg']));

end




function boxes = generateBoxes_b12(im)

bins = 12;
[s1,s2,s3] = size(im);
step_x = single(s1) / bins;
step_y = single(s2) / bins;
cnt = 0;
for x1 = 0:3
    for y1 = 0:3
        for x2 = 8:11
            for y2 = 8:11
                if (x2-x1)*(y2-y1)<0.4999*bins*bins
                    continue;
                end
                if (y2-y1)*step_y/(x2-x1)/step_x<0.5 || (y2-y1)*step_y/(x2-x1)/step_x>2.0
                    continue;
                else
                    cnt = cnt + 1;
                    boxes(1,cnt) = step_x*(0.5+x1);
                    boxes(2,cnt) = step_y*(0.5+y1);
                    boxes(3,cnt) = step_x*(0.5+x2);
                    boxes(4,cnt) = step_y*(0.5+y2);
                end
            end
        end
    end
end
boxes = round(boxes);

function boxes = generateBoxes_b14(im)

bins = 14;
[s1,s2,s3] = size(im);
step_x = single(s1) / bins;
step_y = single(s2) / bins;
cnt = 0;
for x1 = 0:4
    for y1 = 0:4
        for x2 = 9:13
            for y2 = 9:13
                if (x2-x1)*(y2-y1)<0.4999*bins*bins
                    continue;
                end
                if (y2-y1)*step_y/(x2-x1)/step_x<0.5 || (y2-y1)*step_y/(x2-x1)/step_x>2.0
                    continue;
                else
                    cnt = cnt + 1;
                    boxes(1,cnt) = step_x*(0.5+x1);
                    boxes(2,cnt) = step_y*(0.5+y1);
                    boxes(3,cnt) = step_x*(0.5+x2);
                    boxes(4,cnt) = step_y*(0.5+y2);
                end
            end
        end
    end
end
boxes = round(boxes);

function boxes = generateBoxes_b16(im)

bins = 16;
[s1,s2,s3] = size(im);
step_x = single(s1) / bins;
step_y = single(s2) / bins;
cnt = 0;
for x1 = 0:5
    for y1 = 0:5
        for x2 = 10:15
            for y2 = 10:15
                if (x2-x1)*(y2-y1)<0.4999*bins*bins
                    continue;
                end
                if (y2-y1)*step_y/(x2-x1)/step_x<0.5 || (y2-y1)*step_y/(x2-x1)/step_x>2.0
                    continue;
                else
                    cnt = cnt + 1;
                    boxes(1,cnt) = step_x*(0.5+x1);
                    boxes(2,cnt) = step_y*(0.5+y1);
                    boxes(3,cnt) = step_x*(0.5+x2);
                    boxes(4,cnt) = step_y*(0.5+y2);
                end
            end
        end
    end
end
boxes = round(boxes);


function boxes = generateBoxes_16_9(im)

boxes = [];

%% this setting was designed for input images whose longer side caotains 1024 pixels
out_Dim = [16*(40:2:64)',9*(40:2:64)'];

[s1,s2,s3] = size(im);
cnt = 0;
for scale = 1:size(out_Dim,1)
    if out_Dim(1,1)*out_Dim(1,2)<0.4*s1*s2
        out_Dim = out_Dim(scale+1:end,:);
    end
end

for scale = 1:size(out_Dim,1)
    step_x = 16:32:(s1-out_Dim(scale,2)-32);
    step_y = 16:18:(s2-out_Dim(scale,1)-32);
    for x1 = step_x
        for y1 = step_y
            cnt = cnt + 1;
            boxes(1,cnt) = x1;
            boxes(2,cnt) = y1;
            boxes(3,cnt) = x1+out_Dim(scale,2)-1;
            boxes(4,cnt) = y1+out_Dim(scale,1)-1;
        end
    end
end
boxes = round(boxes);


function boxes = generateBoxes_4_3(im)

boxes = [];

%% this setting was designed for input images whose longer side caotains 1024 pixels
out_Dim = [16*(40:2:64)',12*(40:2:64)'];

[s1,s2,s3] = size(im);
cnt = 0;
for scale = 1:size(out_Dim,1)
    if out_Dim(1,1)*out_Dim(1,2)<0.4*s1*s2
        out_Dim = out_Dim(scale+1:end,:);
    end
end

for scale = 1:size(out_Dim,1)
    step_x = 16:32:(s1-out_Dim(scale,2)-32);
    step_y = 16:24:(s2-out_Dim(scale,1)-32);
    for x1 = step_x
        for y1 = step_y
            cnt = cnt + 1;
            boxes(1,cnt) = x1;
            boxes(2,cnt) = y1;
            boxes(3,cnt) = x1+out_Dim(scale,2)-1;
            boxes(4,cnt) = y1+out_Dim(scale,1)-1;
        end
    end
end
boxes = round(boxes);

function boxes = generateBoxes_1_1(im)

boxes = [];

%% this setting was designed for input images whose longer side caotains 1024 pixels
out_Dim = [16*(30:2:64)',16*(30:2:64)'];

[s1,s2,s3] = size(im);
cnt = 0;
% for scale = 1:size(out_Dim,1)
%     if out_Dim(1,1)*out_Dim(1,2)<0.4*s1*s2
%         out_Dim = out_Dim(scale+1:end,:);
%     end
% end

for scale = 1:size(out_Dim,1)
    step_x = 16:32:(s1-out_Dim(scale,2)-32);
    step_y = 16:32:(s2-out_Dim(scale,1)-32);
    for x1 = step_x
        for y1 = step_y
            cnt = cnt + 1;
            boxes(1,cnt) = x1;
            boxes(2,cnt) = y1;
            boxes(3,cnt) = x1+out_Dim(scale,2)-1;
            boxes(4,cnt) = y1+out_Dim(scale,1)-1;
        end
    end
end
boxes = round(boxes);