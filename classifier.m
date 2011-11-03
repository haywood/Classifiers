function classifier(trainin, testin, trainout, testout)
    
    [sample_labels, originals] = read_hands(trainin);
    samples = transform_hands(originals);
    n_samples = size(samples, 1);

    per_class = 5;
    per_train_class = 4;
    correct = zeros(per_class, 1);

    judge = nn_classifier();
    for offset = 1:5
        [train_samples, train_labels, test_samples, test_labels] = split_samples(samples, sample_labels, per_class, per_train_class, offset);

        judge = judge.train(train_labels, train_samples);

        for i = 1:size(test_samples, 1)
            prediction = judge.predict(test_samples(i, :));
            if strcmp(prediction, test_labels{i})
                correct(offset) = correct(offset) + 1;
            end
        end

        correct(offset) = correct(offset)/size(test_samples, 1);
    end

    [s, i] = sort(abs(correct(:) - mean(correct)));
    o = i(1);

    [train_samples, train_labels, test_samples, test_labels] = split_samples(samples, sample_labels, per_class, per_train_class, o);
    judge = judge.train(train_labels, train_samples);
    correct = 0;

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

    correct/n_samples

    [test_labels, originals] = read_hands(testin);
    test_samples = transform_hands(originals);
    out_file = fopen(testout, 'w');
    for i = 1:size(test_samples, 1)
        prediction = judge.predict(test_samples(i, :));
        fprintf(out_file, '%s', prediction);
        fprintf(out_file, ' %d', originals(i, :));
        fprintf(out_file, '\n');
    end
    fclose(out_file);

function [train_samples, train_labels, test_samples, test_labels] = split_samples(samples, labels, per_class, per_train_class, offset)

    train_samples = [];
    train_labels = {};
    test_samples = [];
    test_labels = {};

    for i = 1:size(samples, 1)
        if mod(i - offset, per_class) < per_train_class
            train_samples(end+1, :) = samples(i, :);
            train_labels{end+1} = labels{i};
        else
            test_samples(end+1, :) = samples(i, :);
            test_labels{end+1} = labels{i};
        end
    end

