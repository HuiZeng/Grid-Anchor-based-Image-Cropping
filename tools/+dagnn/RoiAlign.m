classdef RoiAlign < dagnn.Layer

    properties
        subdivisions = [6 6]
        transformation = 1/8
    end
  
    methods
        function outputs = forward(obj, inputs, params)
          % calculation on cpu is faster for matlab
          inputs{1} = gather(inputs{1});
          inputs{2} = gather(inputs{2});
          h = size(inputs{1},1);
          w = size(inputs{1},2);
          ch = size(inputs{1},3);
          num_roi = size(inputs{2},2);
          roi_h_start = inputs{2}(2,:) * obj.transformation;
          roi_w_start = inputs{2}(3,:) * obj.transformation;
          roi_h_end = inputs{2}(4,:) * obj.transformation;
          roi_w_end = inputs{2}(5,:) * obj.transformation;
          roi_height = roi_h_end - roi_h_start;
          roi_width = roi_w_end - roi_w_start;
          bin_size_h = roi_height / obj.subdivisions(1);
          bin_size_w = roi_width / obj.subdivisions(2);
          
          outputs = zeros(obj.subdivisions(1)+1,obj.subdivisions(2)+1,ch,num_roi,'single');

          for i = 1:num_roi
              
              bin_h = (0:obj.subdivisions(1)) * bin_size_h(i) + roi_h_start(i);
              bin_w = (0:obj.subdivisions(2)) * bin_size_w(i) + roi_w_start(i);
              floor_bin_h = max(min(floor(bin_h),h-1),1);
              ceil_bin_h = floor_bin_h+1;
              floor_bin_w = max(min(floor(bin_w),w-1),1);
              ceil_bin_w = floor_bin_w+1;
              c11 = min(ceil_bin_h-bin_h,1)'*min(ceil_bin_w-bin_w,1);
              c12 = min(ceil_bin_h-bin_h,1)'*max(bin_w-floor_bin_w,0);
              c21 = max((bin_h-floor_bin_h),0)'*min(ceil_bin_w-bin_w,1);
              c22 = max(bin_h-floor_bin_h,0)'*max(bin_w-floor_bin_w,0);

              outputs(:,:,:,i) = c11.*inputs{1}(floor_bin_h,floor_bin_w,:) ...
                               + c12.*inputs{1}(floor_bin_h,ceil_bin_w,:) ...
                               + c21.*inputs{1}(ceil_bin_h,floor_bin_w,:) ...
                               + c22.*inputs{1}(ceil_bin_h,ceil_bin_w,:);
          end
          outputs = {gpuArray(outputs)};
        end

        function [derInputs, derParams] = backward(obj, inputs, param, derOutputs)
          %When assigning into a GPUArray, the subscripts must contain unique values.
          inputs{1} = gather(inputs{1});
          inputs{2} = gather(inputs{2});
          h = size(inputs{1},1);
          w = size(inputs{1},2);
          derOutputs{1} = gather(derOutputs{1});
          
          num_roi = size(inputs{2},2);
          roi_h_start = inputs{2}(2,:) * obj.transformation;
          roi_w_start = inputs{2}(3,:) * obj.transformation;
          roi_h_end = inputs{2}(4,:) * obj.transformation;
          roi_w_end = inputs{2}(5,:) * obj.transformation;
          roi_height = roi_h_end - roi_h_start;
          roi_width = roi_w_end - roi_w_start;
          bin_size_h = roi_height / obj.subdivisions(1);
          bin_size_w = roi_width / obj.subdivisions(2);
          
          derbilinear = zeros(size(inputs{1}),'single');
          for i = 1:num_roi
              
              bin_h = (0:obj.subdivisions(1)) * bin_size_h(i) + roi_h_start(i);
              bin_w = (0:obj.subdivisions(2)) * bin_size_w(i) + roi_w_start(i);
              floor_bin_h = max(min(floor(bin_h),h-1),1);
              ceil_bin_h = floor_bin_h+1;
              floor_bin_w = max(min(floor(bin_w),w-1),1);
              ceil_bin_w = floor_bin_w+1;
              c11 = min(ceil_bin_h-bin_h,1)'*min(ceil_bin_w-bin_w,1);
              c12 = min(ceil_bin_h-bin_h,1)'*max(bin_w-floor_bin_w,0);
              c21 = max((bin_h-floor_bin_h),0)'*min(ceil_bin_w-bin_w,1);
              c22 = max(bin_h-floor_bin_h,0)'*max(bin_w-floor_bin_w,0);
              
              derbilinear(floor_bin_h,floor_bin_w,:) = derbilinear(floor_bin_h,floor_bin_w,:) + c11 .* derOutputs{1}(:,:,:,i);
              derbilinear(floor_bin_h,ceil_bin_w,:) = derbilinear(floor_bin_h,ceil_bin_w,:) + c12 .* derOutputs{1}(:,:,:,i);
              derbilinear(ceil_bin_h,floor_bin_w,:) = derbilinear(ceil_bin_h,floor_bin_w,:) + c21 .* derOutputs{1}(:,:,:,i);
              derbilinear(ceil_bin_h,ceil_bin_w,:) = derbilinear(ceil_bin_h,ceil_bin_w,:) + c22 .* derOutputs{1}(:,:,:,i);                             

          end
          derInputs{1} = gpuArray(derbilinear);

          derInputs{2} = [];
          derParams = {} ;
        end

        function obj = RoiAlign(varargin)
          obj.load(varargin);
        end
    end
end
