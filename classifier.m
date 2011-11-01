function classifier(trainin, testin, trainout, testout)
    
    train_file = fopen(trainin, 'r');
    C = textscan(train_file, '%s %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f');
    fclose(train_file);

    feature_file = fopen('best_hand_features_nn.txt', 'r');
    expected_acc = fscanf(feature_file, '%f', 1)
    features = fscanf(feature_file, '%d')'
    fclose(feature_file);

    samples = cell2mat(C(2:end));
    sample_labels = C{1};
    n_samples = size(samples, 1);

    X = zeros(n_samples, size(samples, 2)/2- 1);
    for i = 1:size(samples, 1);
        x = [];
        for j = 1:2:size(samples, 2)-3
            x(end+1) = norm(samples(i, j:j+1) - samples(i, j+2:j+3));
        end
        X(i, :) = x;
    end
    samples = X(:, features);

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

    for i = 1:size(test_samples, 1)
        prediction = judge.predict(test_samples(i, :));
        if strcmp(prediction, test_labels{i})
            correct = correct + 1;
        end
    end

    correct/size(test_samples, 1)
