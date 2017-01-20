function [pi, fs] = rowsubsetselection(X)
    delta = 0.000001;
    alpha = 0.0001;
    beta = 0.9;
    X = X(:,1:end-1);
    size(X)
    [n, ~] = size(X);
    
    
    pi = 1/n * ones(n, 1);
    inner = @(pi) inv(X'*diag(pi)*X);
    f = @(pi) trace(inner(pi));
%     grad = @(pi) -sum((inner(pi)*X').^2)';
    grad = @(pi) -diag(X*inner(pi)*inner(pi)*X');
    fs = f(pi);
    Gt = @(pi,t) (pi-projection(pi-t*grad(pi)))/t;
    
    while true
        t = 1e-5;
        disp(f(projection(pi-alpha*t*grad(pi))));
        pi = projection(pi-t*grad(pi));
        fs = [fs f(pi)];
        if fs(end-1)-fs(end) < delta
            break;
        end
    end
end

function [projected_pi] = projection(pi)
    [n, ~] = size(pi);
    sorted_pi = sort(pi,'descend');
    K = 0;
    for k = 1:n
        if (sum(sorted_pi(1:k))-1)/k < sorted_pi(k)
            K = k;
        end
    end
    tao = (sum(sorted_pi(1:K))-1)/K;
    projected_pi = arrayfun(@(x) max(x-tao,0), pi);
end