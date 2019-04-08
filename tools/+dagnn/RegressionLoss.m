classdef RegressionLoss < dagnn.Loss

  properties
    lossType = 1.
  end
  
  methods
    function outputs = forward(obj, inputs, params)
        X = inputs{1};
        c = inputs{2};
        assert(numel(X) == numel(c));
        c = reshape(c,[1,1,size(c)]);
        switch obj.lossType
            case 'CE'
                X = vl_nnsigmoid(X);
                Y = sum(squeeze(-c.*log(X) - (1-c).*log(1.001-X)));
            case 'MSE'
                Y = sum(squeeze((X - c).^2));                  
            case 'Huber'
                delta = 1.0;
                a = abs(squeeze(X - c));
                y1 = 0.5*sum(a(a<=delta).^2);
                y2 = sum((a(a>delta)-0.5*delta)*delta);
                Y = y1 + y2;                    
        end
        outputs{1} = Y;
        if obj.ignoreAverage, return; end;
        n = obj.numAveraged ;
        m = n + size(inputs{2},2);
        obj.average = (n * obj.average + gather(outputs{1})) / m ;
        obj.numAveraged = m ;
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
        X = inputs{1};
        c = inputs{2};
        assert(numel(X) == numel(c));
        c = reshape(c,[1,1,size(c)]);
        switch obj.lossType
            case 'CE'
                X = vl_nnsigmoid(X);
                Y = X - c;
            case 'MSE'
                Y = X - c;
            case 'Huber'
                delta = 1;
                a = X - c;
                Y = X - c;
                Y(a>delta) = delta;
                Y(a<-delta) = - delta;
        end
        derInputs = {Y, []};
        derParams = {};  
    end

    function obj = RegressionLoss(varargin)
      obj.load(varargin) ;
    end
  end
end
