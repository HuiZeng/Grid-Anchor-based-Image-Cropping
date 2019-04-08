function [s1,s2] = paramSize(net)
    s1 = 0;s2=0;
    p = net.getParamIndex('dimred_filter') ;
    for i = 1:p-1
        s1 = s1 + prod(size(net.params(i).value));
    end
    s1 = s1 * 4/1024/1024;
    p2 = net.getParamIndex('predcls_filter');
    for i = p:p2
        s2 = s2 + prod(size(net.params(i).value));
    end
    s2 = s2 * 4/1024/1024;
end