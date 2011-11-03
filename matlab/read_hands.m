% read a file with hand data
% return labels as a cell array of the class labels
% return data as a matrix of samples
function [labels, data] = read_hands(hand_file)

    data_file = fopen(hand_file, 'r');
    C = textscan(data_file, '%s %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f');
    fclose(data_file);

    data = cell2mat(C(2:end));
    labels = C{1};
