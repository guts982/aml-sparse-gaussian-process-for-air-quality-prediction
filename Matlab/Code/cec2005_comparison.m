%% cec2005_comparison.m
% Part 3: Compare GA vs PSO on Sphere & Rastrigin (CEC'2005 style)

clear; clc; close all;
rng(1); % reproducibility

%% -----------------------------
% Benchmark functions
sphere = @(x) sum(x.^2,2);
rastrigin = @(x) 10*size(x,2) + sum(x.^2 - 10*cos(2*pi*x),2);

funcs = {@(x) sphere(x), @(x) rastrigin(x)};
funcNames = {'Sphere','Rastrigin'};
bounds = {[-100 100], [-5.12 5.12]};

dims = [2, 10];   % dimensionalities
runs = 15;        % independent runs
maxIter = 200;    % iterations per run
popSize = 30;     % population/swarm size

%% -----------------------------
% Storage
results = struct();

for f=1:2
    for d = dims
        fprintf('\n=== %s function, D=%d ===\n', funcNames{f}, d);
        lb = bounds{f}(1); ub = bounds{f}(2);

        % GA runs
        gaHistAll = zeros(runs, maxIter);
        gaBest = zeros(runs,1);
        for r=1:runs
            [bestVal, hist] = runGA(funcs{f}, d, lb, ub, popSize, maxIter);
            gaHistAll(r,:) = hist;
            gaBest(r) = bestVal;
        end

        % PSO runs
        psoHistAll = zeros(runs, maxIter);
        psoBest = zeros(runs,1);
        for r=1:runs
            [bestVal, hist] = runPSO(funcs{f}, d, lb, ub, popSize, maxIter);
            psoHistAll(r,:) = hist;
            psoBest(r) = bestVal;
        end

        % Summaries
        statsGA = [min(gaBest), max(gaBest), mean(gaBest), std(gaBest)];
        statsPSO= [min(psoBest), max(psoBest), mean(psoBest), std(psoBest)];
        results.(funcNames{f}).(['D' num2str(d)]).GA = statsGA;
        results.(funcNames{f}).(['D' num2str(d)]).PSO= statsPSO;

        % Print
        fprintf('GA : Best %.4e | Worst %.4e | Mean %.4e | Std %.4e\n', statsGA);
        fprintf('PSO: Best %.4e | Worst %.4e | Mean %.4e | Std %.4e\n', statsPSO);

        % Convergence plot (mean across runs)
        figure('Name',sprintf('%s D=%d',funcNames{f},d));
        semilogy(mean(gaHistAll,1),'r','LineWidth',1.5); hold on;
        semilogy(mean(psoHistAll,1),'b','LineWidth',1.5);
        xlabel('Iteration'); ylabel('Mean Best Fitness');
        legend('GA','PSO'); grid on;
        title(sprintf('%s function (D=%d)',funcNames{f},d));
    end
end

%% -----------------------------
% Summary table
Func   = {};
Dim    = [];
Algo   = {};
Best   = [];
Worst  = [];
MeanV  = [];
StdV   = [];

fieldsF = fieldnames(results);
for i = 1:numel(fieldsF)
    fname = fieldsF{i};
    dimsF = fieldnames(results.(fname));
    for j = 1:numel(dimsF)
        dname = dimsF{j};   % e.g. 'D2' or 'D10'
        dimVal = sscanf(dname,'D%d');   % FIXED HERE
        algos = fieldnames(results.(fname).(dname));
        for k = 1:numel(algos)
            aname = algos{k};
            vals = results.(fname).(dname).(aname);
            Func{end+1,1} = fname; %#ok<AGROW>
            Dim(end+1,1)  = dimVal; %#ok<AGROW>
            Algo{end+1,1} = aname; %#ok<AGROW>
            Best(end+1,1) = vals(1); %#ok<AGROW>
            Worst(end+1,1)= vals(2); %#ok<AGROW>
            MeanV(end+1,1)= vals(3); %#ok<AGROW>
            StdV(end+1,1) = vals(4); %#ok<AGROW>
        end
    end
end

T = table(Func,Dim,Algo,Best,Worst,MeanV,StdV);
disp('=== Results Table ===');
disp(T);

% writetable(T,'cec2005_results.xlsx') % optional save

%% -------------------------------------------------
% Helper functions MUST be at the end

clc;
clear;

% -------------------------------
% Problem Definition
% -------------------------------
D = 10;                  % Dimension of the problem
lb = -5; ub = 5;         % Lower and upper bounds
popSize = 50;            % Population size
maxIter = 100;           % Number of iterations

% Fitness function: Sphere function (min at [0,...,0])
fitnessFcn = @(x) sum(x.^2, 2);

% -------------------------------
% Run Genetic Algorithm (GA)
% -------------------------------
[bestValGA, bestHistGA] = runGA(fitnessFcn, D, lb, ub, popSize, maxIter);

% -------------------------------
% Run Particle Swarm Optimization (PSO)
% -------------------------------
[bestValPSO, bestHistPSO] = runPSO(fitnessFcn, D, lb, ub, popSize, maxIter);

% -------------------------------
% Display Results
% -------------------------------
fprintf('Best GA Value: %.6f\n', bestValGA);
fprintf('Best PSO Value: %.6f\n', bestValPSO);

% -------------------------------
% Plot Convergence
% -------------------------------
figure;
plot(1:maxIter, bestHistGA, 'r-', 'LineWidth', 2); hold on;
plot(1:maxIter, bestHistPSO, 'b-', 'LineWidth', 2);
xlabel('Iteration');
ylabel('Best Fitness Value');
legend('GA', 'PSO');
title('GA vs PSO Convergence');
grid on;

% ====================================================
% Helper Functions MUST be at the end of the file
% ====================================================

function [bestVal, bestHist] = runGA(fitnessFcn, D, lb, ub, popSize, maxIter)
    pop = lb + (ub-lb).*rand(popSize,D);
    fit = fitnessFcn(pop);
    [bestVal,idx] = min(fit);
    best = pop(idx,:);
    bestHist = zeros(1,maxIter);
    for t=1:maxIter
        newPop = zeros(size(pop));
        for i=1:2:popSize
            p1 = tournament(pop,fit); p2 = tournament(pop,fit);
            crossPoint = randi(D);
            child1 = [p1(1:crossPoint), p2(crossPoint+1:end)];
            child2 = [p2(1:crossPoint), p1(crossPoint+1:end)];
            if rand<0.1, child1(randi(D)) = lb + (ub-lb)*rand; end
            if rand<0.1, child2(randi(D)) = lb + (ub-lb)*rand; end
            newPop(i,:) = child1;
            if i+1<=popSize, newPop(i+1,:) = child2; end
        end
        pop = newPop;
        fit = fitnessFcn(pop);
        [val,idx] = min(fit);
        if val < bestVal
            bestVal = val; best = pop(idx,:);
        end
        bestHist(t) = bestVal;
    end
end

function [bestVal, bestHist] = runPSO(fitnessFcn, D, lb, ub, popSize, maxIter)
    x = lb + (ub-lb).*rand(popSize,D);
    v = zeros(popSize,D);
    fit = fitnessFcn(x);
    pbest = x; pbestVal = fit;
    [gbestVal,idx] = min(pbestVal);
    gbest = x(idx,:);
    bestHist = zeros(1,maxIter);
    w=0.7; c1=1.5; c2=1.5;
    for t=1:maxIter
        r1=rand(popSize,D); r2=rand(popSize,D);
        v = w*v + c1*r1.*(pbest-x) + c2*r2.*(gbest-x);
        x = x+v;
        x = max(lb,min(ub,x));
        fit = fitnessFcn(x);
        better = fit < pbestVal;
        pbest(better,:) = x(better,:);
        pbestVal(better) = fit(better);
        [val,idx] = min(pbestVal);
        if val<gbestVal
            gbestVal=val; gbest=pbest(idx,:);
        end
        bestHist(t)=gbestVal;
    end
    bestVal=gbestVal;
end

function p = tournament(pop,fit)
    k=3;
    idx = randi(length(fit),k,1);
    [~,i] = min(fit(idx));
    p = pop(idx(i),:);
end
