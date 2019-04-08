% This file is designed for customer testing. Users can put their target
% images into the folder "dataset/test/images". This file supports testing
% images with arbitrary resolution


function testGAIC_qualitative_customer()


imDir = dir(fullfile('dataset','test','images','*.*'));
imDir = imDir(3:end);
imList = fullfile('dataset','test','images',{imDir.name});

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

for aspect_ratios = {'1_1','4_3','16_9','2_3','4_5','5_7'}

    for i = 1:numel(imList)
        try
            img = imread(imList{i});
        catch
            continue
        end

        [x1,x2,x3] = size(img);

        imre = single(imresize(img,minScale/min([x1,x2]),'bilinear'));
        imre = bsxfun(@minus,imre,net.meta.normalization.averageImage);

        [r1,r2,r3] = size(imre); 
        r1 = 32*round(r1/32);
        r2 = 32*round(r2/32);
        imre = imresize(imre,[r1,r2],'bilinear');
        
        
        boxes = generateBoxes(imre,aspect_ratios{1});
        if isempty(boxes)
            continue;
        end
        
        inputs = {'input', gpuArray(imre), 'rois', gpuArray(single([ones(1,size(boxes,2));boxes]))} ;
        
        net.eval(inputs) ;
        preds = squeeze(gather(net.vars(probVarI).value)) ;
        preds = preds * gt_scores_stds + gt_scores_means;
        [~,id_preds] = sort(preds,'descend');
        
        predBox = boxes(:,id_preds(1));
        
        scale1 = x1/r1;
        scale2 = x2/r2;
        boxes_s(1,:) = max(floor(predBox(1,:) * scale1),1);
        boxes_s(2,:) = max(floor(predBox(2,:) * scale2),1);
        boxes_s(3,:) = min(ceil(predBox(3,:) * scale1),x1);
        boxes_s(4,:) = min(ceil(predBox(4,:) * scale2),x2);
        
        outDir = fullfile('dataset','test','result');
        mkdir(outDir);

        imwrite(img(boxes_s(1):boxes_s(3),boxes_s(2):boxes_s(4),:),...
            fullfile(outDir,[imDir(i).name(1:end-4) '_preds_' aspect_ratios{1} '.jpg']));

    end

end


function boxes = generateBoxes(im,aspect_ratios)

switch aspect_ratios
    case '1_1'
        boxes = generateBoxes_1_1(im);
    case '4_3'
        boxes = generateBoxes_4_3(im);
    case '16_9'
        boxes = generateBoxes_16_9(im);
    case '2_3'
        boxes = generateBoxes_2_3(im);
    case '4_5'
        boxes = generateBoxes_4_5(im);
    case '5_7'
        boxes = generateBoxes_5_7(im);
    case '1_2'
        boxes = generateBoxes_1_2(im);
    case '1_3'
        boxes = generateBoxes_1_3(im);
    case '1_4'
        boxes = generateBoxes_1_4(im);
    case '2_1'
        boxes = generateBoxes_2_1(im);
    case '3_1'
        boxes = generateBoxes_3_1(im);
    case '4_1'
        boxes = generateBoxes_4_1(im);
end
boxes = round(boxes);


function boxes = generateBoxes_16_9(im)

boxes = [];
[s1,s2,s3] = size(im);
id = floor(max(s1,s2)/8);
out_Dim = [16*(14:id)',9*(14:id)'];
out_Dim = out_Dim(out_Dim(:,1)<=s2,:);
out_Dim = out_Dim(out_Dim(:,2)<=s1,:);


cnt = 0;
if out_Dim(end,1)*out_Dim(end,2)>0.4*s1*s2
for scale = 1:size(out_Dim,1)
    if out_Dim(1,1)*out_Dim(1,2)<0.4*s1*s2
        out_Dim = out_Dim(2:end,:);
    end
end
end

for scale = 1:size(out_Dim,1)
    step_x = 1:16:(s1-out_Dim(scale,2)+1);
    step_y = 1:9:(s2-out_Dim(scale,1)+1);
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
[s1,s2,s3] = size(im);
id = floor(max(s1,s2)/8);
out_Dim = [8*(25:id)',6*(25:id)'];
out_Dim = out_Dim(out_Dim(:,1)<=s2,:);
out_Dim = out_Dim(out_Dim(:,2)<=s1,:);

cnt = 0;
if out_Dim(end,1)*out_Dim(end,2)>0.4*s1*s2
for scale = 1:size(out_Dim,1)
    if out_Dim(1,1)*out_Dim(1,2)<0.4*s1*s2
        out_Dim = out_Dim(2:end,:);
    end
end
end

for scale = 1:size(out_Dim,1)
    step_x = 1:16:(s1-out_Dim(scale,2)+1);
    step_y = 1:12:(s2-out_Dim(scale,1)+1);
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
[s1,s2,s3] = size(im);

out_Dim = [4*(40:2:64)',4*(40:2:64)'];

cnt = 0;
if out_Dim(end,1)*out_Dim(end,2)>0.4*s1*s2
for scale = 1:size(out_Dim,1)
    if out_Dim(1,1)*out_Dim(1,2)<0.4*s1*s2
        out_Dim = out_Dim(2:end,:);
    end
end
end

for scale = 1:size(out_Dim,1)
    step_x = 1:12:(s1-out_Dim(scale,2)+1);
    step_y = 1:12:(s2-out_Dim(scale,1)+1);
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

function boxes = generateBoxes_2_3(im)

boxes = [];

[s1,s2,s3] = size(im);
id = floor(max(s1,s2)/8);
out_Dim = [8*(18:id)',12*(18:id)'];
out_Dim = out_Dim(out_Dim(:,1)<=s2,:);
out_Dim = out_Dim(out_Dim(:,2)<=s1,:);


cnt = 0;
if out_Dim(end,1)*out_Dim(end,2)>0.4*s1*s2
for scale = 1:size(out_Dim,1)
    if out_Dim(1,1)*out_Dim(1,2)<0.4*s1*s2
        out_Dim = out_Dim(2:end,:);
    end
end
end

for scale = 1:size(out_Dim,1)
    step_x = 1:8:(s1-out_Dim(scale,2)+1);
    step_y = 1:12:(s2-out_Dim(scale,1)+1);
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

function boxes = generateBoxes_4_5(im)

boxes = [];

[s1,s2,s3] = size(im);
id = floor(max(s1,s2)/8);
out_Dim = [8*(20:id)',10*(20:id)'];
out_Dim = out_Dim(out_Dim(:,1)<=s2,:);
out_Dim = out_Dim(out_Dim(:,2)<=s1,:);

if out_Dim(end,1)*out_Dim(end,2)>0.4*s1*s2

for scale = 1:size(out_Dim,1)
    if out_Dim(1,1)*out_Dim(1,2)<0.4*s1*s2
        out_Dim = out_Dim(2:end,:);
    end
end

end

cnt = 0;
for scale = 1:size(out_Dim,1)
    step_x = 1:8:(s1-out_Dim(scale,2)+1);
    step_y = 1:10:(s2-out_Dim(scale,1)+1);
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

function boxes = generateBoxes_5_7(im)

boxes = [];

[s1,s2,s3] = size(im);
id = floor(max(s1,s2)/8);
out_Dim = [10*(15:id)',14*(15:id)'];
out_Dim = out_Dim(out_Dim(:,1)<=s2,:);
out_Dim = out_Dim(out_Dim(:,2)<=s1,:);
cnt = 0;

if out_Dim(end,1)*out_Dim(end,2)>0.4*s1*s2
for scale = 1:size(out_Dim,1)
    if out_Dim(1,1)*out_Dim(1,2)<0.4*s1*s2
        out_Dim = out_Dim(2:end,:);
    end
end
end

for scale = 1:size(out_Dim,1)
    step_x = 1:8:(s1-out_Dim(scale,2)+1);
    step_y = 1:14:(s2-out_Dim(scale,1)+1);
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

function boxes = generateBoxes_1_2(im)

boxes = [];

[s1,s2,s3] = size(im);
% id = floor(max(s1,s2)/8);
out_Dim = [6*(18:100)',12*(18:100)'];
out_Dim = out_Dim(out_Dim(:,1)<=s2,:);
out_Dim = out_Dim(out_Dim(:,2)<=s1,:);


cnt = 0;
if out_Dim(end,1)*out_Dim(end,2)>0.4*s1*s2
for scale = 1:size(out_Dim,1)
    if out_Dim(1,1)*out_Dim(1,2)<0.4*s1*s2
        out_Dim = out_Dim(2:end,:);
    end
end
end

for scale = 1:size(out_Dim,1)
    step_x = 1:8:(s1-out_Dim(scale,2)+1);
    step_y = 1:16:(s2-out_Dim(scale,1)+1);
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

function boxes = generateBoxes_1_3(im)

boxes = [];

[s1,s2,s3] = size(im);
out_Dim = [4*(18:100)',12*(18:100)'];
out_Dim = out_Dim(out_Dim(:,1)<=s2,:);
out_Dim = out_Dim(out_Dim(:,2)<=s1,:);


cnt = 0;
if out_Dim(end,1)*out_Dim(end,2)>0.4*s1*s2
for scale = 1:size(out_Dim,1)
    if out_Dim(1,1)*out_Dim(1,2)<0.4*s1*s2
        out_Dim = out_Dim(2:end,:);
    end
end
end

for scale = 1:size(out_Dim,1)
    step_x = 1:8:(s1-out_Dim(scale,2)+1);
    step_y = 1:16:(s2-out_Dim(scale,1)+1);
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


function boxes = generateBoxes_1_4(im)

boxes = [];

[s1,s2,s3] = size(im);
out_Dim = [3*(18:100)',12*(18:100)'];
out_Dim = out_Dim(out_Dim(:,1)<=s2,:);
out_Dim = out_Dim(out_Dim(:,2)<=s1,:);


cnt = 0;
if out_Dim(end,1)*out_Dim(end,2)>0.4*s1*s2
for scale = 1:size(out_Dim,1)
    if out_Dim(1,1)*out_Dim(1,2)<0.4*s1*s2
        out_Dim = out_Dim(2:end,:);
    end
end
end

for scale = 1:size(out_Dim,1)
    step_x = 1:8:(s1-out_Dim(scale,2)+1);
    step_y = 1:16:(s2-out_Dim(scale,1)+1);
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


function boxes = generateBoxes_2_1(im)

boxes = [];

[s1,s2,s3] = size(im);
out_Dim = [12*(18:100)',6*(18:100)'];
out_Dim = out_Dim(out_Dim(:,1)<=s2,:);
out_Dim = out_Dim(out_Dim(:,2)<=s1,:);


cnt = 0;
if out_Dim(end,1)*out_Dim(end,2)>0.4*s1*s2
for scale = 1:size(out_Dim,1)
    if out_Dim(1,1)*out_Dim(1,2)<0.4*s1*s2
        out_Dim = out_Dim(2:end,:);
    end
end
end

for scale = 1:size(out_Dim,1)
    step_x = 1:16:(s1-out_Dim(scale,2)+1);
    step_y = 1:8:(s2-out_Dim(scale,1)+1);
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


function boxes = generateBoxes_3_1(im)

boxes = [];

[s1,s2,s3] = size(im);
out_Dim = [12*(18:100)',4*(18:100)'];
out_Dim = out_Dim(out_Dim(:,1)<=s2,:);
out_Dim = out_Dim(out_Dim(:,2)<=s1,:);


cnt = 0;
if out_Dim(end,1)*out_Dim(end,2)>0.4*s1*s2
for scale = 1:size(out_Dim,1)
    if out_Dim(1,1)*out_Dim(1,2)<0.4*s1*s2
        out_Dim = out_Dim(2:end,:);
    end
end
end

for scale = 1:size(out_Dim,1)
    step_x = 1:16:(s1-out_Dim(scale,2)+1);
    step_y = 1:8:(s2-out_Dim(scale,1)+1);
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


function boxes = generateBoxes_4_1(im)

boxes = [];

[s1,s2,s3] = size(im);
out_Dim = [12*(18:100)',3*(18:100)'];
out_Dim = out_Dim(out_Dim(:,1)<=s2,:);
out_Dim = out_Dim(out_Dim(:,2)<=s1,:);


cnt = 0;
if out_Dim(end,1)*out_Dim(end,2)>0.4*s1*s2
for scale = 1:size(out_Dim,1)
    if out_Dim(1,1)*out_Dim(1,2)<0.4*s1*s2
        out_Dim = out_Dim(2:end,:);
    end
end
end

for scale = 1:size(out_Dim,1)
    step_x = 1:16:(s1-out_Dim(scale,2)+1);
    step_y = 1:8:(s2-out_Dim(scale,1)+1);
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