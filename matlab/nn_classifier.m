% A nearest neighbor classifier class
classdef nn_classifier

    properties
        examples = {}; % training data
        mean_ = []; % mean of training data
        cov_ = []; % covariance of training data
    end

    methods
        % train on a given set of samples and their labels
        % this overrides old training data
        function self = train(self, sample_labels, samples)
            self.examples = {};
            self.mean_ = mean(samples);
            self.cov_ = cov(samples);
            for i = 1:size(samples, 1)
                self.examples{end+1} = {sample_labels{i}, (samples(i, :) - self.mean_)./sqrt(diag(self.cov_)')};
            end
        end

        % predict the class of a sample
        function prediction = predict(self, x)
            best = {inf, ''};
            x = (x - self.mean_)./sqrt(diag(self.cov_)');
            for i = 1:size(self.examples, 2)
                d = norm(x - self.examples{i}{2});
                if d < best{1}
                    best = {d, self.examples{i}{1}};
                end
            end
            prediction = best{2};
        end
    end
end
