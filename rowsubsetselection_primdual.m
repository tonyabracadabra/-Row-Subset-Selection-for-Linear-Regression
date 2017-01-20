function [pi, fs] = rowsubsetselection_primdual(X, alpha, beta)
    mu = 50;

    [n, ~] = size(X);
    
    inner = @(pi) inv(X'*diag(pi)*X);
    f = @(pi) trace(inner(pi));

    H = @(pi) 2*(X*(inner(pi)*inner(pi))*X').*(X*inner(pi)*X');
    grad = @(pi) -sum((inner(pi)*X').^2)';
    
    t = @(pi,u) -2*n*mu/(-pi'*u);
    
    A = @(pi,u,v) [H(pi),-eye(n),ones(n,1);...
                    -diag(u),-diag(pi),zeros(n,1);...
                    ones(1,n),zeros(1,n+1)];
                
    r_prim = @(pi,u,v) grad(pi)-u+v*ones(n,1);
    r_cent = @(pi,u) -diag(u)*pi+1/t(pi,u);
    r_dual = @(pi) ones(1,n)*pi-1;
    
    B = @(pi,u,v) [r_prim(pi,u,v);r_cent(pi,u);r_dual(pi)];
    
    deltaZ = @(pi,u,v) linsolve(A(pi,u,v),-B(pi,u,v));
    
    %% Initialize variables
    
    pi = 1/n * ones(n, 1);
    size(inner(pi)*inner(pi))
    size(X')
    2*(X*(inner(pi)*inner(pi))*X').*(X*inner(pi)*X');
    
    u = ones(n,1);
    v = 1;
    update = deltaZ(pi,u,v);
    
    tmp = update(n+1:2*n);
    tmp2 = -u./tmp;
    theta_max = min(1, min(tmp2(tmp<0)));
    theta = 0.99*theta_max;
    
    fs = [f(pi)];
    %% Start backtracking
    while true
        disp(1)
        update = deltaZ(pi,u,v);
        
        pi_new = pi+theta*update(1:n);
        u_new = u+theta*update(n+1:2*n);
        v_new = v+theta*update(end);
        
        while -pi_new >= 0
            theta = beta*theta;
%             disp(theta);
        end
                
        r_new = B(pi_new,u_new,v_new);
        r = B(pi,u,v);
        
        while norm(r_new) > (1-alpha*theta)*norm(r)
            theta = beta*theta;
            disp(1)
        end
        
        rp = r_prim(pi,u,v);
        rd = r_dual(pi);
        
        if sqrt(sum(rp.^2)+sum(rd.^2)) <= 10e-6 || pi'*u <= 10e-6
            disp(pi'*u);
            break
        end
        
        pi = pi_new;
        u = u_new;
        v = v_new;
        disp(theta);
        fs = [fs f(pi)];
        
    end

end