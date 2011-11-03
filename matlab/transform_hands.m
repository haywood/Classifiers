% transform data into a useful representation
% I tried to pick features that I felt were invariant under different hand poses

function image = transform_hands(preimage)

    n_points = size(preimage, 2)/2;
    p = zeros(n_points, 2);
    N = size(preimage, 1);
    image = zeros(N, 11);

    for i = 1:N % for each sample
        x = preimage(i, :);

        % extract the poitns from the sample and store in matrix p
        for j = 1:2:2*n_points
            p(1+floor(j/2), :) = x(j:j+1);
        end

        j = 1;
        image(i, j) = norm(p(1, :) - p(18, :)); % this is the width of the palm

        % this loop calculates the width of each finger
        for k = 3:4:15
            j = j + 1;
            image(i, j) = norm(p(k, :) - p(k+2, :));
        end

        % this loop estimates the perimeter of each finger
        for k = 2:4:14
            j = j + 1;
            for l = k:k+4
                image(i, j) = image(i, j) + norm(p(l, :) - p(l+1, :));
             end
        end

        % the length and width of the thumb
        image(i, j+1) = norm(p(19, :) - p(20, :)) + norm(p(20, :) - p(21, :)) + norm(p(21, :) - p(22, :)) + norm(p(22, :) - p(23, :));
        image(i, j+2) = norm(p(20, :) - p(22, :));
    end
