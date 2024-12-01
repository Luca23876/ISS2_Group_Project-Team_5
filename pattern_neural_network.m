clear all; close all; clc;

training_sequence = load('sequence_DIAwind_train.mat');
testing_sequence = load('sequence_DIAwind_test.mat');

training_data = zeros(9, length(training_sequence.sequence));

for i = 1:length(training_sequence.sequence)
    training_data(training_sequence.sequence(i), i) = 1; 
end

training_sequence.sequence = [1 ; training_sequence.sequence(1:end - 1)];

net = patternnet(10:2,'trainbr');

net = train(net,training_sequence.sequence.', training_data);

sequenceLength = initializeSymbolMachineF24('sequence_DIAtemp_test.mat',0);
% We can start with a uniform forecast for the first symbol
probs = [1/9 1/9 1/9 1/9 1/9 1/9 1/9 1/9 1/9];
[symbol,penalty] = symbolMachineF24(probs);
for ii = 2:sequenceLength
    % For each subsequent symbol, we can base our forecast on the preceding
    % symbol (which was given to us by the Symbol Machine)
    [symbol,penalty] = symbolMachineF24(net(symbol).');
end
reportSymbolMachineF24;