classdef RodAlign < dagnn.Layer

    properties
        subdivisions = [6 6]
        transformation = 1/8
    end
  
    methods
        function outputs = forward(obj, inputs, params) 
          % calculation on cpu is faster for matlab
          inputs{1} = gather(inputs{1});
          inputs{2} = gather(inputs{2});
          
          feat_h = size(inputs{1},1);
          feat_w = size(inputs{1},2);
          num_roi = size(inputs{2},2);
          roi_h_start = max(round(inputs{2}(2,:) * obj.transformation),1);
          roi_w_start = max(round(inputs{2}(3,:) * obj.transformation),1);
          roi_h_end = min(round(inputs{2}(4,:) * obj.transformation),feat_h);
          roi_w_end = min(round(inputs{2}(5,:) * obj.transformation),feat_w);
          
          scale_h = feat_h / obj.subdivisions(1);
          scale_w = feat_w / obj.subdivisions(2);
          bin_h = ((1:obj.subdivisions(1))-0.5) * scale_h;
          bin_W = ((1:obj.subdivisions(2))-0.5) * scale_w;
          
          floor_bin_h = max(floor(bin_h),1);
          ceil_bin_h = floor_bin_h+1;
          floor_bin_w = max(floor(bin_W),1);
          ceil_bin_w = floor_bin_w+1;
          c11 = min(ceil_bin_h-bin_h,1)'*min(ceil_bin_w-bin_W,1);
          c12 = min(ceil_bin_h-bin_h,1)'*max(bin_W-floor_bin_w,0);
          c21 = max((bin_h-floor_bin_h),0)'*min(ceil_bin_w-bin_W,1);
          c22 = max(bin_h-floor_bin_h,0)'*max(bin_W-floor_bin_w,0);
          feats = repmat(inputs{1},1,1,1,num_roi);
          for i = 1:num_roi
              feats(roi_h_start(i):roi_h_end(i),roi_w_start(i):roi_w_end(i),:,i) = 0;
          end
          outputs = c11.*feats(floor_bin_h,floor_bin_w,:,:) ...
                  + c12.*feats(floor_bin_h,ceil_bin_w,:,:) ... 
                  + c21.*feats(ceil_bin_h,floor_bin_w,:,:) ...
                  + c22.*feats(ceil_bin_h,ceil_bin_w,:,:);
          outputs = {gpuArray(outputs)};

        end

        function [derInputs, derParams] = backward(obj, inputs, param, derOutputs)
          %When assigning into a GPUArray, the subscripts must contain unique values.
          inputs{1} = gather(inputs{1});
          inputs{2} = gather(inputs{2});
          derOutputs{1} = gather(derOutputs{1});
          
          feat_h = size(inputs{1},1);
          feat_w = size(inputs{1},2);
          ch = size(inputs{1},3);
          num_roi = size(inputs{2},2);
          roi_h_start = max(round(inputs{2}(2,:) * obj.transformation),1);
          roi_w_start = max(round(inputs{2}(3,:) * obj.transformation),1);
          roi_h_end = min(round(inputs{2}(4,:) * obj.transformation),feat_h);
          roi_w_end = min(round(inputs{2}(5,:) * obj.transformation),feat_w);
          
          scale_h = feat_h / obj.subdivisions(1);
          scale_w = feat_w / obj.subdivisions(2);
          bin_h = ((1:obj.subdivisions(1))-0.5) * scale_h;
          bin_W = ((1:obj.subdivisions(2))-0.5) * scale_w;
          
          floor_bin_h = max(floor(bin_h),1);
          ceil_bin_h = floor_bin_h+1;
          floor_bin_w = max(floor(bin_W),1);
          ceil_bin_w = floor_bin_w+1;
          c11 = min(ceil_bin_h-bin_h,1)'*min(ceil_bin_w-bin_W,1);
          c12 = min(ceil_bin_h-bin_h,1)'*max(bin_W-floor_bin_w,0);
          c21 = max((bin_h-floor_bin_h),0)'*min(ceil_bin_w-bin_W,1);
          c22 = max(bin_h-floor_bin_h,0)'*max(bin_W-floor_bin_w,0);
          
          derbilinear = zeros([feat_h,feat_w,ch,num_roi],'single');
          for i = 1:num_roi
              derbilinear(floor_bin_h,floor_bin_w,:,i) = derbilinear(floor_bin_h,floor_bin_w,:,i) + c11 .* derOutputs{1}(:,:,:,i);
              derbilinear(floor_bin_h,ceil_bin_w,:,i) = derbilinear(floor_bin_h,ceil_bin_w,:,i) + c12 .* derOutputs{1}(:,:,:,i);
              derbilinear(ceil_bin_h,floor_bin_w,:,i) = derbilinear(ceil_bin_h,floor_bin_w,:,i) + c21 .* derOutputs{1}(:,:,:,i);
              derbilinear(ceil_bin_h,ceil_bin_w,:,i) = derbilinear(ceil_bin_h,ceil_bin_w,:,i) + c22 .* derOutputs{1}(:,:,:,i);
              derbilinear(roi_h_start(i):roi_h_end(i),roi_w_start(i):roi_w_end(i),:,i) = 0;
          end
          derInputs{1} = gpuArray(sum(derbilinear,4));
              
          derInputs{2} = [];
          derParams = {} ;
        end

        function obj = RodAlign(varargin)
          obj.load(varargin);
        end
    end
end
