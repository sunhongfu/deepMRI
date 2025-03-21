classdef maeRegressionLayer < nnet.layer.RegressionLayer
    % Example custom regression layer with mean-absolute-error loss.
    
    methods
        function layer = maeRegressionLayer(name)
            % layer = maeRegressionLayer(name) creates a
            % mean-absolute-error regression layer and specifies the layer
            % name.
            
            % Set layer name.
            layer.Name = name;
            
            % Set layer description.
            layer.Description = 'My Mean absolute error';
        end
        
        function loss = forwardLoss(layer, Y, T)
            % Try to add the input 
            % loss = forwardLoss(layer, Y, T) returns the MAE loss between
            % the predictions Y and the training targets T.
            
            % Calculate MAE.
%             R = size(Y,3);
 %           meanAbsoluteError = sum(abs(Y-T),3)/R;
            meanAbsoluteError = sum(abs(Y-T),'all');
            % mse=sqrt(sum((a-b).^2))./2;
            % Take mean over mini-batch.
            N = size(Y,5);
            loss = meanAbsoluteError / N;
        end
        
        function dLdY = backwardLoss(layer, Y, T)
            % dLdY = backwardLoss(layer, Y, T) returns the derivatives of
            % the MAE loss with respect to the predictions Y.
            
 %           R = size(Y,3);
            N = size(Y,5);
            dLdY = sign(Y-T)/N;
        end
    end
end