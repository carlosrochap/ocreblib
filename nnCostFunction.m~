function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
                                   
    %NNCOSTFUNCTION Implements the neural network cost function for a two layer
    %neural network which performs classification
    %   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
    %   X, y, lambda) computes the cost and gradient of the neural network. The
    %   parameters for the neural network are "unrolled" into the vector
    %   nn_params and need to be converted back into the weight matrices. 
    % 
    %   The returned parameter grad should be a "unrolled" vector of the
    %   partial derivatives of the neural network.
    %

    % Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
    % for our 2 layer neural network
    Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                     hidden_layer_size, (input_layer_size + 1));

    Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                     num_labels, (hidden_layer_size + 1));

    % Setup some useful variables
    m = size(X, 1);
             
    % You need to return the following variables correctly 
    J = 0;
    Theta1_grad = zeros(size(Theta1));
    Theta2_grad = zeros(size(Theta2));


    % Add ones to the X data matrix
    X       = [ones(m, 1) X];
    a1      = sigmoid(X*Theta1');

    % Add ones to the a1 data matrix
    a1 = [ones(size(a1, 1), 1) a1];
    a2 = sigmoid(a1*Theta2');

    xlog    = log(a2);
    xlog_2  = log(1-a2);

    YRows = eye(num_labels);
    Y = YRows(y, :);

    pJ = -Y .* xlog - (1-Y) .* xlog_2;
    J  = (1/m) * sum(sum(pJ));

    if lambda > 0
        %Regularization
        regularizationTerm = (lambda/(2*m)) * (sum(sum(Theta1(:, 2:end) .^ 2)) + sum(sum(Theta2(:, 2:end) .^ 2)));
        J = J + regularizationTerm;
        %keyboard();
    end

    %--------------------------------------------------------------
    %Back Propagation
    D_1 = Theta1_grad; 
    D_2 = Theta2_grad;
    for t = 1:m
        a_1 = X(t, :);
        
        z_2 = a_1*Theta1';
        a_2 = sigmoid(z_2);
        a_2 = [1 a_2];
        
        a_3 = sigmoid(a_2*Theta2');
        
        d_3 = a_3 - Y(t, :);
        
        d_2 = (d_3 * Theta2);
        d_2 = d_2(2:end) .* sigmoidGradient(z_2);
        
        D_1 = D_1 + d_2' * a_1;
        D_2 = D_2 + d_3' * a_2;
        
    end

    Theta1_grad = (1/m) * D_1; 
    Theta2_grad = (1/m) * D_2;

    Theta1_grad(:, 2:end) += (lambda/m) * Theta1(:, 2:end);
    Theta2_grad(:, 2:end) += (lambda/m) * Theta2(:, 2:end);

    % -------------------------------------------------------------

    % =========================================================================


    % Unroll gradients
    grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
