% import stock data (5 years of monthly price data, 61 prices)
stock_history = readmatrix("stock data.xlsx");
returns = stock_history(2:61,:) ./ stock_history(1:60,:) - 1;
cal_returns = returns(1:36, :);
test_prices = stock_history(37:61, :);

sds = zeros(1,27);
for i = 1:27
    sds(i) = std(cal_returns(:, i));
end

% importing benchmark data (S&P 500)
index_history = readmatrix("SP500.xlsx");
returns1 = index_history(2:61,:) ./ index_history(1:60,:) - 1;
cal_returns2 = returns1(1:36, :);
test_prices2 = index_history(37:61, :);

var_index = std(cal_returns2)^2;

covar = cov([cal_returns cal_returns2]);

% calculating betas
betas = zeros(1,27);
for i = 1:27
    cov_mat = cov(cal_returns(:,i), cal_returns2); 
    betas(i) = cov_mat(1,2) / var_index;
end

% calculating the alphas & mu
alphas = zeros(1,27);
mu = zeros(1,28);
index_return = mean(cal_returns2(30:36,:));
for i = 1:27
    mu(i) = mean(cal_returns(30:36,i)); % avg of last 6 months of returns
   alphas(i) = mu(i) - betas(i)*index_return;
end

x_plot = 0:1:24;
index_value = zeros(1,25);
initial_index_value = sum(test_prices2(1,:));
for i = 1:25
    index_value(i) = sum(test_prices2(i,:)) / initial_index_value;
end

%% Model #1 Long Only Portfolio
Aeq_0 = ones(1,27);

beq_0 = 1;

lb_0 = zeros(1,27);

x_max_LO = linprog(-alphas, [], [], Aeq_0, beq_0, lb_0);
ret_max_LO = alphas*x_max_LO;

x_min_LO = linprog(alphas, [], [], Aeq_0, beq_0, lb_0);
ret_min_LO = alphas*x_min_LO;

% Optimization to find portfolio
lb = zeros(1,28);
ub = [Inf.*ones(1,27) 0];

x_B = [zeros(27,1);
       1         ];

f = -(x_B' * covar)';

Aeq = [ones(1,27) 0;
        betas      0];

beq = ones(2,1);

A = [-alphas 0];
b = -0.005;
[x, fval] = quadprog(covar, f, A, b, Aeq, beq, lb, ub);

% Compute portfolio values for buy and hold
portfolio_value = zeros(1,25);
initial_portfolio_value = test_prices(1,:) * x(1:27,1);

for i = 1:25
    portfolio_value(i) = test_prices(i,:) * x(1:27,1) / initial_portfolio_value;
end

% Plot of portfolio (no rebalancing)
hold on
plot(x_plot, portfolio_value, 'r')
plot(x_plot, index_value, 'b')
title('Performance of Long Only Portfolio versus S&P 500')
legend({'Portfolio','Index'},'Location','southeast')
xlabel('Last 24 months')
ylabel('Value ($)')

%% Long Only Quarterly Rebalanced Portfolio
weights = ones(25,28);
portfolio_value_q = zeros(1,25);
for i = 1:25
    weights(i, :) = x';
    portfolio_value_q(i) = test_prices(i,:) * x(1:27,1) / initial_portfolio_value;
    if mod(i, 3) == 0 
        % recalulate var and covar
        cal_returns = returns(1 + i:36 + i, :);
        cal_returns2 = returns1(1 + i:36 + i, :);

        var_index_q = std(cal_returns2)^2;

        covar_q = cov([cal_returns cal_returns2]);
        
        % recalculate betas
        betas_q = zeros(1,27);
        for j = 1:27
            cov_mat = cov(cal_returns(:,j), cal_returns2); 
            betas_q(j) = cov_mat(1,2) / var_index_q;
        end

        % recalculate alphas
        alphas_q = zeros(1,27);
        mu_q = zeros(1,28);
        index_return_q = mean(returns1((30 + i):(36 + i),:));
        for k = 1:27
            mu_q(k) = mean(returns(30 + i:36 + i,k)); % avg of last 6 months of returns
           alphas_q(k) = mu_q(k) - betas_q(k)*index_return_q;
        end

        % recalculate x
        lb = zeros(1,28);
        ub = [Inf.*ones(1,27) 0];
        
        x_B = [zeros(27,1);
               1         ];
        
        f = -(x_B' * covar_q)';
        
        Aeq = [ones(1,27) 0;
               betas_q   0];
        
        beq = ones(2,1);
        
        A = [-alphas_q 0];
        b = -0.005;
        [x, fval] = quadprog(covar_q, f, A, b, Aeq, beq, lb, ub);
    end
end

% Plot of portfolio (quarterly rebalancing)
hold on
plot(x_plot, portfolio_value_q, 'r')
plot(x_plot, index_value, 'b')
title('Performance of Long Only Portfolio Rebalanced Quarterly versus S&P 500')
legend({'Portfolio','Index'},'Location','southeast')
xlabel('Last 24 months')
ylabel('Value ($)')

%% Model #2 Long Short Portfolio
% we decide on how much alpha per month we want
lb1 = [-Inf.*ones(1,27) 0];
ub1 = [Inf.*ones(1,27) 0];

x_B = [zeros(27,1);
       1         ];

f1 = -(x_B' * covar)';

Aeq1 = [ones(1,27) 0;
        betas      0];

beq1 = ones(2,1);

A1 = [-alphas 0];
b1 = -0.005;
[x1, fval1] = quadprog(covar, f1, A1, b1, Aeq1, beq1, lb1, ub1);

% Compute portfolio values
initial_portfolio_value1 = test_prices(1,:) * x1(1:27,1);
portfolio_value1 = zeros(1,25);
for i = 1:25
    portfolio_value1(i) = test_prices(i,:) * x1(1:27,1) / initial_portfolio_value1;
end

% Plot of no turnover portfolio
hold on
plot(x_plot, portfolio_value1, 'r')
plot(x_plot, index_value, 'b')
title('Performance of Long Short Portfolio versus S&P 500')
legend({'Portfolio','Index'},'Location','southeast')
xlabel('Last 24 months')
ylabel('Value ($)')

%% Long Short Quarterly Rebalanced Portfolio
weights = ones(25,28);
portfolio_value_q1 = zeros(1,25);
for i = 1:25
    weights(i, :) = x1';
    portfolio_value_q1(i) = test_prices(i,:) * x1(1:27,1) / initial_portfolio_value1;
    if mod(i, 3) == 0 
        % recalulate var and covar
        cal_returns = returns(1 + i:36 + i, :);
        cal_returns2 = returns1(1 + i:36 + i, :);

        var_index_q = std(cal_returns2)^2;

        covar_q = cov([cal_returns cal_returns2]);
        
        % recalculate betas
        betas_q = zeros(1,27);
        for j = 1:27
            cov_mat = cov(cal_returns(:,j), cal_returns2); 
            betas_q(j) = cov_mat(1,2) / var_index_q;
        end

        % recalculate alphas
        alphas_q = zeros(1,27);
        mu_q = zeros(1,28);
        index_return_q = mean(returns1(30 + i:36 + i,:));
        for k = 1:27
            mu_q(k) = mean(returns(30 + i:36 + i,k)); % avg of last 6 months of returns
           alphas_q(k) = mu_q(k) - betas_q(k)*index_return_q;
        end

        % recalculate x
        lb1 = [-Inf.*ones(1,27) 0];
        ub1 = [Inf.*ones(1,27) 0];
        
        x_B = [zeros(27,1);
               1         ];
        
        f1 = -(x_B' * covar_q)';
       
        Aeq1 = [ones(1,27) 0;
                betas_q   0];
        
        beq1 = ones(2,1);
        
        A1 = [-alphas_q 0];
        b1 = -0.005;
        [x1, fval1] = quadprog(covar_q, f1, A1, b1, Aeq1, beq1, lb1, ub1);
    end
end

% Plot of portfolio (quarterly rebalancing)
hold on
plot(x_plot, portfolio_value_q1, 'r')
plot(x_plot, index_value, 'b')
title('Performance of Long Short Portfolio Rebalanced Quarterly versus S&P 500')
legend({'Portfolio','Index'},'Location','southeast')
xlabel('Last 24 months')
ylabel('Value ($)')

%% Model #3 130/30 Portfolio
covar2 = [covar zeros(28);
          zeros(28, 56)];
lb3 = [-Inf.*ones(1,27) 0 -zeros(1,27) 0];
ub3 = [Inf.*ones(1,27) 0 Inf.*ones(1,27) 0];

x_B_1 = [zeros(27,1);
         1          ;
         zeros(28,1)];

f2 = -(x_B_1' * covar2)';

Aeq3 = [ones(1,27) zeros(1,29);
       betas      zeros(1,29)];

beq3 = ones(2,1);

A2_p1 = [-alphas 0 zeros(1,28)];
A2_p2 = [-eye(27) zeros(27,1) -eye(27) zeros(27,1)];
A2_p3 = [zeros(1,28) ones(1,27)  0];

A2 = [A2_p1 ;
      A2_p2 ;
      A2_p3];

b2 = [-0.005     ;
      zeros(27,1);
      0.3       ];

[x2, fval2] = quadprog(covar2, f2, A2, b2, Aeq3, beq3, lb3, ub3);

% Compute portfolio values
initial_portfolio_value2 = test_prices(1,:) * x2(1:27,1);
portfolio_value2 = zeros(1,25);
for i = 1:25
    portfolio_value2(i) = test_prices(i,:) * x2(1:27,1) / initial_portfolio_value2;
end

% Plot of no turnover portfolio
hold on
plot(x_plot, portfolio_value2, 'r')
plot(x_plot, index_value, 'b')
title('Performance of 130/30 Portfolio versus S&P 500')
legend({'Portfolio','Index'},'Location','southeast')
xlabel('Last 24 months')
ylabel('Value ($)')

%% 130/30 Quarterly Rebalanced Portfolio
weights = ones(25,56);
portfolio_value_q2 = zeros(1,25);
for i = 1:25
    weights(i, :) = x2';
    portfolio_value_q2(i) = test_prices(i,:) * x2(1:27,1) / initial_portfolio_value2;
    if mod(i, 3) == 0 
        % recalulate var and covar
        cal_returns = returns(1 + i:36 + i, :);
        cal_returns2 = returns1(1 + i:36 + i, :);

        var_index_q = std(cal_returns2)^2;

        covar_q = cov([cal_returns cal_returns2]);
        
        % recalculate betas
        betas_q = zeros(1,27);
        for j = 1:27
            cov_mat = cov(cal_returns(:,j), cal_returns2); 
            betas_q(j) = cov_mat(1,2) / var_index_q;
        end

        % recalculate alphas
        alphas_q = zeros(1,27);
        mu_q = zeros(1,28);
        index_return_q = mean(returns1(30 + i:36 + i,:));
        for k = 1:27
            mu_q(k) = mean(returns(30 + i:36 + i,k)); % avg of last 6 months of returns
           alphas_q(k) = mu_q(k) - betas_q(k)*index_return_q;
        end

        % recalculate x
        covar2 = [covar_q zeros(28);
                   zeros(28, 56)];
        lb3 = [-Inf.*ones(1,27) 0 -zeros(1,27) 0];
        ub3 = [Inf.*ones(1,27) 0 Inf.*ones(1,27) 0];
        
        x_B_1 = [zeros(27,1);
                 1          ;
                 zeros(28,1)];
        
        f2 = -(x_B_1' * covar2)';
        
        Aeq3 = [ones(1,27) zeros(1,29);
               betas_q    zeros(1,29)];
        
        beq3 = ones(2,1);
        
        A2_p1 = [-alphas_q 0 zeros(1,28)];
        A2_p2 = [-eye(27) zeros(27,1) -eye(27) zeros(27,1)];
        A2_p3 = [zeros(1,28) ones(1,27)  0];
        
        A2 = [A2_p1 ;
              A2_p2 ;
              A2_p3];
        
        b2 = [-0.005     ;
              zeros(27,1);
              0.3       ];
        
        [x2, fval2] = quadprog(covar2, f2, A2, b2, Aeq3, beq3, lb3, ub3);

    end
end

% Plot of portfolio (quarterly rebalancing)
hold on
plot(x_plot, portfolio_value_q2, 'r')
plot(x_plot, index_value, 'b')
title('Performance of 130/30 Portfolio Rebalanced Quarterly versus S&P 500')
legend({'Portfolio','Index'},'Location','southeast')
xlabel('Last 24 months')
ylabel('Value ($)')






