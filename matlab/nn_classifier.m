classdef nn_classifier

    properties
        examples = {};
        mean_ = [];
        cov_ = [];
    end

    methods
        function self = train(self, sample_labels, samples)
            self.examples = {};
            self.mean_ = mean(samples);
            self.cov_ = cov(samples);
            for i = 1:size(samples, 1)
                self.examples{end+1} = {sample_labels{i}, (samples(i, :) - self.mean_)./sqrt(diag(self.cov_)')};
            end
        end

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
