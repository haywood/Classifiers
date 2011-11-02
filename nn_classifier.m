classdef nn_classifier

    properties
        examples = {};
        _mean = [];
        _cov = [];
    end

    methods
        function self = train(self, sample_labels, samples)
            self.examples = {};
            self._mean = mean(samples);
            self._cov = cov(samples);
            for i = 1:size(samples, 1)
                self.examples{end+1} = {sample_labels{i}, (samples(i, :) - self._mean)/sqrt(diag(self._cov))};
            end
        end

        function prediction = predict(self, x)
            best = {inf, ''};
            x = (x - self._mean)/sqrt(diag(self._cov));
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
