function imdb = setup_database_GAIC(trainNum,seed)

if ~exist(['imdb_GAIC' num2str(trainNum) '.mat'],'file')
    train_img_dir = fullfile('dataset','GAIC','images','train');
    train_img_path = dir(train_img_dir);
    train_img_path = fullfile(train_img_dir,{train_img_path(3:end).name});
    
    test_img_dir = fullfile('dataset','GAIC','images','test');
    test_img_path = dir(test_img_dir);
    test_img_path = fullfile(test_img_dir,{test_img_path(3:end).name});
    
    img_path = cat(2,train_img_path,test_img_path);
    
    for i = 1:numel(img_path)
        [~,imgname,~] = fileparts(img_path{i});
        anno = load(fullfile('dataset','GAIC','annotations',[imgname '.txt']));
        boxes{i} = anno(:,1:4)';
        avg_scores(i,:) = anno(:,5)';
        kill = (avg_scores(i,:)==-2);
        gt_scores{i} = avg_scores(i,~kill);
        boxes{i} = boxes{i}(:,~kill);
    end
    
    rng(seed)
    trainSet = randperm(numel(train_img_path),trainNum);
    valSet = setdiff(1:numel(train_img_path),trainSet);
    testSet = numel(train_img_path)+1:numel(train_img_path)+numel(test_img_path);

    set = ones(1,numel(img_path));
    set(valSet) = 3;
    set(testSet) = 2;
    imdb.images.trainSet = trainSet;
    imdb.images.valSet = valSet;
    imdb.images.testSet = testSet;
    imdb.bbox.boxes = boxes;
    imdb.bbox.gt_scores = gt_scores;
    imdb.images.set = set;
    imdb.meta.sets = {'train', 'val', 'test'} ;
    imdb.meta.img_path = img_path;
    save(['imdb_GAIC' num2str(trainNum) '.mat'],'imdb');
else
    load(['imdb_GAIC' num2str(trainNum) '.mat'],'imdb');
end

