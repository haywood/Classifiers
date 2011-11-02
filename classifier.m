function classifier(trainin, testin, trainout, testout)
    
    [sample_labels, originals] = read_hands(trainin);
    samples = transform_hands(originals);
    n_samples = size(samples, 1);

    per_class = 5;
    per_train_class = floor(per_class/2);

    train_samples = [];
    train_labels = {};
    test_samples = [];
    test_labels = {};

    for i = 1:n_samples
        if mod(i - 1, per_class) < per_train_class
            train_samples(end+1, :) = samples(i, :);
            train_labels{end+1} = sample_labels{i};
        else
            test_samples(end+1, :) = samples(i, :);
            test_labels{end+1} = sample_labels{i};
        end
    end

    judge = nn_classifier();
    judge = judge.train(train_labels, train_samples);
    correct = 0;

    out_file = fopen(trainout, 'w');
    for i = 1:size(test_samples, 1)
        prediction = judge.predict(test_samples(i, :));
        if strcmp(prediction, test_labels{i})
            correct = correct + 1;
        end
        fprintf(out_file, '%s', prediction);
        fprintf(out_file, ' %f', test_samples(i, :));
        fprintf(out_file, '\n');
    end
    fclose(out_file);

    correct/size(test_samples, 1)

    [test_labels, test_samples] = read_hands(testin);
    out_file = fopen(testout, 'w');
    for i = 1:size(test_samples, 1)
        prediction = judge.predict(test_samples(i, :));
        fprintf(out_file, '%s', prediction);
        fprintf(out_file, ' %f', test_samples(i, :));
        fprintf(out_file, '\n');
    end
    fclose(out_file);


function [labels, data] = read_hands(hand_file)

    data_file = fopen(hand_file, 'r');
    C = textscan(data_file, '%s %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f');
    fclose(data_file);

    data = cell2mat(C(2:end));
    labels = C{1};

    data = transform_hands(data);
