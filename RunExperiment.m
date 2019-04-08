% clc;
clear;
warning off
addpath(genpath('tools'));
addpath(genpath('src'));
vl_setupnn;

database = 'GAIC'; 
model = 'vgg16'; % resnet50
trainNum = 1000;
cdim = 8;
downsample = 4;
warpsize = 9;
ndim = 512;
seed = 1;

RoIRoD = 'RoIRoD'; % RoIRoD RoIOnly RoDOnly


trainModel('database',database,'model',model,'downsample',downsample,'RoIRoD',RoIRoD,...
           'trainNum',trainNum,'warpsize',warpsize,'cdim',cdim,'ndim',ndim,'seed',seed);
testModel(downsample,model,trainNum,warpsize,cdim,ndim,seed,RoIRoD);


