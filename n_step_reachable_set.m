for Nreach = 1:100

    e0 = x - x_ref;

    e_var = sdpvar(nx,Nreach+1);
    u_var = sdpvar(nu,Nreach);

    con = [e_var(:,1) == e0];

    for i = 1:Nreach
        con = [con, e_var(:,i+1) == Ad*e_var(:,i) + Bd*u_var(:,i)];
        con = [con, umin <= u_var(:,i) <= umax];
    end

    con = [con, e_var(:,Nreach+1) == zeros(nx,1)];

    sol = optimize(con, 0, sdpsettings('solver','OSQP','verbose',0));

    if sol.problem == 0
        fprintf('Initial state is in K_%d(0)\n', Nreach);
        break;
    end
end
