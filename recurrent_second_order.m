clear all; close all; clc;

% Load in training and testing data
training_sequence = load('data\sequence_solarWind_train.mat');
testing_sequence = load('data\sequence_solarWind_test.mat');


% Create desired output sequence for neural network from training data
training_data = zeros(9, length(training_sequence.sequence));

for i = 1:length(training_sequence.sequence)
    training_data(training_sequence.sequence(i), i) = 1; 
end

% Create input sequence for neural network from training data
prior_obs = [1 ; training_sequence.sequence(1:end - 1)];
pprior_obs = [1 ; 1 ; training_sequence.sequence(1:end - 2)];
observations = [prior_obs pprior_obs];

% Initialize pattern neural network of size 10 x 1 with Bayesian Regularization
net = layrecnet(1:2,10,'trainbr');

% Train neural network with training data
net = train(net,observations', training_data);

% Test neural network on testing data
sequenceLength = initializeSymbolMachineF24('data\sequence_solarWind_test.mat',0);

% We can start with a uniform forecast for the first symbol
probs = [1/9 1/9 1/9 1/9 1/9 1/9 1/9 1/9 1/9];
[symbol,penalty] = symbolMachineF24(probs);
prev_symbol = 5;

% Log results to text file
diary results\recurrent_second_order_solarWind.txt;

for ii = 2:sequenceLength
    % Get prediction from neural network
    prediction = net([symbol; prev_symbol]).';

    % Normalize prediction between 0 and 1
    prediction = (prediction - min(prediction))/(max(prediction) - min(prediction));
    prediction = prediction / sum(prediction);

    % Increase and decrement min and max values to avoid 0 probabilities
    [M_min,I_min] = min(prediction);
    [M_max,I_max] = max(prediction);
    prediction(I_min) = prediction(I_min) + 0.0001;
    prediction(I_max) = prediction(I_max) - 0.0001;

    % Update previous symbol
    prev_symbol = symbol;

    % For each subsequent symbol, we can base our forecast on the preceding
    % symbol (which was given to us by the Symbol Machine)
    [symbol,penalty] = symbolMachineF24(prediction);
end
reportSymbolMachineF24;