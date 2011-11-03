% train a nearest neighbor classifier on the data in trainin
% test it on testin
% output training results to trainout
% output test results to testout

function classifier(trainin, testin, trainout, testout)
    
    [sample_labels, originals] = read_hands(trainin); % read training data
    samples = transform_hands(originals); % transform it to have useful features
    n_samples = size(samples, 1);

    per_class = 5;
    per_train_class = 4;
    correct = zeros(per_class, 1);

    judge = nn_classifier(); % make the training sample

    for offset = 1:5 % 5-fold cross-val

        % partition data into training and validation
        [train_samples, train_labels, validation_samples, test_labels] = split_samples(samples, sample_labels, per_class, per_train_class, offset);

        judge = judge.train(train_labels, train_samples);

        for i = 1:size(validation_samples, 1)
            prediction = judge.predict(validation_samples(i, :));
            if strcmp(prediction, test_labels{i})
                correct(offset) = correct(offset) + 1;
            end
        end

        correct(offset) = correct(offset)/size(validation_samples, 1);
    end

    % sort by distance from the mean
    [s, i] = sort(abs(correct(:) - mean(correct)))
    o = i(1)
    correct

    [train_samples, train_labels, validation_samples, test_labels] = split_samples(samples, sample_labels, per_class, per_train_class, o);

    % train on the fold of training data that was closest to the average
    judge = judge.train(train_labels, train_samples);
    correct = 0;

    % classify every training sample
    out_file = fopen(trainout, 'w');
    for i = 1:size(samples, 1)
        prediction = judge.predict(samples(i, :));
        if strcmp(prediction, sample_labels{i})
            correct = correct + 1;
        end
        fprintf(out_file, '%s', prediction);
        fprintf(out_file, ' %d', originals(i, :));
        fprintf(out_file, '\n');
    end
    fclose(out_file);

    [test_labels, originals] = read_hands(testin); % read test data
    test_samples = transform_hands(originals); % transform test data to have useful features
    out_file = fopen(testout, 'w');

    % classify test samples
    for i = 1:size(test_samples, 1)
        prediction = judge.predict(test_samples(i, :));
        fprintf(out_file, '%s', prediction);
        fprintf(out_file, ' %d', originals(i, :));
        fprintf(out_file, '\n');
    end
    fclose(out_file);

% split samples for cross validation
function [train_samples, train_labels, validation_samples, test_labels] = split_samples(samples, labels, per_class, per_train_class, offset)

    train_samples = [];
    train_labels = {};
    validation_samples = [];
    test_labels = {};

    for i = 1:size(samples, 1)
        if mod(i - offset, per_class) < per_train_class
            train_samples(end+1, :) = samples(i, :);
            train_labels{end+1} = labels{i};
        else
            validation_samples(end+1, :) = samples(i, :);
            test_labels{end+1} = labels{i};
        end
    end

