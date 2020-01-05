function [trainK, testK] = cmpExpX2Kernel(trainD, testD, gamma)
        %% Train Kernel %%
        [n, ~] = size(trainD);
        for i=1:n
            trainK(i, 1) = i;
            for j=2:n+1
                [kernelVal] = computeKernel(trainD(i,:)', trainD(j-1, :)', gamma);
                trainK(i,j) = kernelVal;
            end
        end

        %% Test Kernel %%
        
        for i=1:size(testD, 1)
            testK(i, 1) = i;
            for j=2:n+1
                [kernelVal] = computeKernel(testD(i,:)', trainD(j-1, :)', gamma);
                testK(i, j) = kernelVal;
            end
        end 
        

end
       
function [kernelVal] = computeKernel(x, y, gamma)
    %x and y are 1 Dimensional vectors
    epsilon = 0.00001;
    nume = (x - y).^2;
    deno = x + y + epsilon;

    fraction = nume./deno;

    temp = sum(fraction);
    temp = temp/gamma;
    kernelVal = exp(-1*temp);
end