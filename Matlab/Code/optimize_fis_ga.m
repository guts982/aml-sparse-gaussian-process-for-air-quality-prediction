
clear; clc; close all;

% ---------------------------
% 1) Build baseline FIS (from Part 1)
[baselineFIS, rules] = build_baseline_fis();  % (function defined below)

% ---------------------------
% 2) Load or generate training data
% If you have a training file flc_training.mat with data Nx7: [Temp Lux Occ Pref Heater Fan Dimmer]
if exist('flc_training.mat','file')
    s = load('flc_training.mat');
    data = s.data;
    if size(data,2) ~= 7
        error('flc_training.mat must contain variable data (N x 7): [Temp Lux Occup Pref Heater Fan Dimmer]');
    end
else
    % generate synthetic dataset (baseline outputs + small noise)
    rng(1);
    N = 300;
    Temp = 10 + 25*rand(N,1);       % 10..35
    Lux  = 0 + 1000*rand(N,1);     % 0..1000
    Occup = double(rand(N,1) > 0.4); % 0 or 1
    Pref  = rand(N,1);             % 0..1
    inputs = [Temp Lux Occup Pref];
    outputs = evalfis(baselineFIS, inputs);
    noise = 0.02*100*randn(size(outputs));
    targets = max(0,min(100, outputs + noise));
    data = [inputs targets];
    save('flc_training_generated.mat','data');
    fprintf('No training file found — generated synthetic training set and saved as flc_training_generated.mat\n');
end

X = data(:,1:4);
Y_target = data(:,5:7);

% ---------------------------
% 3) Chromosome encoding details
nT = 5; nL = 4; nOutPer = 4;
len = nT + nT + nL + nL + nOutPer*3; % 5 centers + 5 widths + 4 centers + 4 widths + 12 output centers = 30

% bounds
lb = [ repmat(10,1,nT), repmat(1,1,nT), repmat(0,1,nL), repmat(10,1,nL), repmat(0,1,12) ];
ub = [ repmat(35,1,nT), repmat(8,1,nT), repmat(1000,1,nL), repmat(400,1,nL), repmat(100,1,12) ];

% ---------------------------
% 4) initial population
popSize = 60;
maxGen = 120;
initPop = bsxfun(@plus, lb, bsxfun(@times, rand(popSize,len), (ub-lb)));

% ---------------------------
% 5) Fitness wrapper for built-in ga (if present)
fitnessFun = @(chrom) fitness_chrom(chrom, rules, X, Y_target);

% Try MATLAB ga first (if Global Optimization Toolbox available)
used_builtin = false;
if exist('ga','file') == 2
    try
        opts = optimoptions('ga','PopulationSize',popSize,'MaxGenerations',maxGen, ...
            'Display','iter','UseParallel',false,'EliteCount',2,'PlotFcn',{@gaplotbestf});
        fprintf('Running MATLAB built-in ga...\n');
        [xbest, fbest, exitflag, output] = ga(fitnessFun, len, [],[],[],[], lb, ub, [], opts);
        used_builtin = true;
    catch ME
        warning('Built-in ga failed (will use custom GA): %s', ME.message);
        used_builtin = false;
    end
end

% If built-in ga not used, run a simple custom GA
if ~used_builtin
    fprintf('Running custom GA ...\n');
    % GA params
    elitism = 2;
    mutRate = 0.12;
    sigma0 = 0.08*(ub-lb); % mutation scale
    pop = initPop;
    fitnessPop = zeros(popSize,1);
    for i=1:popSize
        fitnessPop(i) = fitness_chrom(pop(i,:), rules, X, Y_target);
    end
    [fitnessPop, idxs] = sort(fitnessPop);
    pop = pop(idxs,:);
    bestHist = zeros(maxGen,1);
    for gen=1:maxGen
        newPop = pop(1:elitism,:); % keep elites
        % produce children until population filled
        while size(newPop,1) < popSize
            % tournament selection (k=3)
            a = randi(popSize,3,1); b = randi(popSize,3,1);
            [~,ia] = min(fitnessPop(a)); [~,ib] = min(fitnessPop(b));
            parent1 = pop(a(ia),:); parent2 = pop(b(ib),:);
            % arithmetic crossover
            alpha = rand(1,len);
            child = alpha.*parent1 + (1-alpha).*parent2;
            % mutation
            mutMask = rand(1,len) < mutRate;
            child(mutMask) = child(mutMask) + sigma0(mutMask).*randn(1,sum(mutMask));
            % keep within bounds
            child = max(lb, min(ub, child));
            newPop = [newPop; child]; %#ok<AGROW>
        end
        pop = newPop;
        % evaluate
        for i=1:popSize
            fitnessPop(i) = fitness_chrom(pop(i,:), rules, X, Y_target);
        end
        [fitnessPop, idxs] = sort(fitnessPop);
        pop = pop(idxs,:);
        bestHist(gen) = fitnessPop(1);
        if mod(gen,10)==0
            fprintf('Gen %d: best fitness = %.6f\n', gen, fitnessPop(1));
        end
        % early stop small plateau
        if gen>30 && abs(mean(bestHist(gen-29:gen)) - bestHist(gen)) < 1e-8
            fprintf('Early stopping at gen %d\n', gen);
            bestHist = bestHist(1:gen);
            break;
        end
    end
    xbest = pop(1,:);
    fbest = fitnessPop(1);
    fprintf('Custom GA finished. Best MSE = %.6f\n', fbest);
end

% ---------------------------
% 6) Apply best params and compare
fis_opt = fis_from_chrom(xbest, rules);
Y_pred_before = evalfis(baselineFIS, X);
Y_pred_after  = evalfis(fis_opt, X);

mse_before = mean(sum((Y_pred_before - Y_target).^2,2));
mse_after  = mean(sum((Y_pred_after  - Y_target).^2,2));
fprintf('MSE before = %.6f, MSE after = %.6f\n', mse_before, mse_after);

% Save optimized FIS
save('fis_optimized_fixed.mat','fis_opt','xbest','fbest','mse_before','mse_after');

% ---------------------------
% 7) Plots for report
figure('Name','Heater Before vs After');
subplot(2,1,1); plot(Y_target(:,1),'k.'); hold on; plot(Y_pred_before(:,1),'b.'); title('Heater target vs before'); legend('Target','Before')
subplot(2,1,2); plot(Y_target(:,1),'k.'); hold on; plot(Y_pred_after(:,1),'r.'); title('Heater target vs after'); legend('Target','After')

figure('Name','Fan Before vs After');
subplot(2,1,1); plot(Y_target(:,2),'k.'); hold on; plot(Y_pred_before(:,2),'b.'); title('Fan target vs before'); legend('Target','Before')
subplot(2,1,2); plot(Y_target(:,2),'k.'); hold on; plot(Y_pred_after(:,2),'r.'); title('Fan target vs after'); legend('Target','After')

figure('Name','Dimmer Before vs After');
subplot(2,1,1); plot(Y_target(:,3),'k.'); hold on; plot(Y_pred_before(:,3),'b.'); title('Dimmer target vs before'); legend('Target','Before')
subplot(2,1,2); plot(Y_target(:,3),'k.'); hold on; plot(Y_pred_after(:,3),'r.'); title('Dimmer target vs after'); legend('Target','After')

% surface comparison (Temp & Pref -> Heater)
figure('Name','Surface comparison: Temp & Pref -> Heater');
subplot(1,2,1); gensurf(baselineFIS,[1 4],1); title('Baseline FIS');
subplot(1,2,2); gensurf(fis_opt,[1 4],1); title('Optimized FIS');

fprintf('Optimized FIS saved as fis_optimized_fixed.mat\n');

% End of main script ----------------------------------------------------

%% Local functions - MUST be at end of script

function mse = fitness_chrom(chrom, rules, X, Y_target)
    % Build fis from chrom and compute MSE vs targets
    fis_try = fis_from_chrom(chrom, rules);
    Y_pred = evalfis(fis_try, X);
    err = Y_pred - Y_target;
    mse = mean(sum(err.^2,2));
end

function fis_new = fis_from_chrom(chrom, rules)
    % Decode chromosome and build a new FIS whose MFs come from chrom.
    % The rules matrix is added exactly as provided (so MF index ordering must match).
    nT = 5; nL = 4; nOutPer = 4;
    idx = 1;
    T_centers = chrom(idx:idx+nT-1); idx = idx + nT;
    T_widths  = chrom(idx:idx+nT-1); idx = idx + nT;
    L_centers = chrom(idx:idx+nL-1); idx = idx + nL;
    L_widths  = chrom(idx:idx+nL-1); idx = idx + nL;
    Out_H = chrom(idx:idx+nOutPer-1)'; idx = idx + nOutPer;
    Out_F = chrom(idx:idx+nOutPer-1)'; idx = idx + nOutPer;
    Out_D = chrom(idx:idx+nOutPer-1)';

    % Build new FIS
    fis_new = mamfis("Name","AssistiveHomeFLC_optim", ...
        "AndMethod","min","OrMethod","max","ImplicationMethod","min","AggregationMethod","max","DefuzzificationMethod","centroid");

    % Temperature input [10 35]
    fis_new = addInput(fis_new,[10 35],"Name","Temperature");
    for i=1:nT
        c = T_centers(i);
        w = max(0.5, T_widths(i));
        a = c - w; b = c; d = c + w;
        if i==1
            params = [10 10 b d]; % left trap (clamped at 10)
            fis_new = addMF(fis_new,"Temperature","trapmf",params,"Name",sprintf("Cold%d",i));
        elseif i==nT
            params = [a b 35 35]; % right trap (clamped at 35)
            fis_new = addMF(fis_new,"Temperature","trapmf",params,"Name",sprintf("Hot%d",i));
        else
            params = [a b d];
            fis_new = addMF(fis_new,"Temperature","trimf",params,"Name",sprintf("Temp%d",i));
        end
    end

    % Light input [0 1000]
    fis_new = addInput(fis_new,[0 1000],"Name","Light");
    for i=1:nL
        c = L_centers(i);
        w = max(5, L_widths(i));
        a = c - w; b = c; d = c + w;
        if i==1
            params = [max(0,a) max(0,a) b d];
            fis_new = addMF(fis_new,"Light","trapmf",params,"Name",sprintf("Light%d",i));
        elseif i==nL
            params = [a b min(1000,d) min(1000,d)];
            fis_new = addMF(fis_new,"Light","trapmf",params,"Name",sprintf("Light%d",i));
        else
            params = [a b d];
            fis_new = addMF(fis_new,"Light","trimf",params,"Name",sprintf("Light%d",i));
        end
    end

    % Occupancy input [0 1]  (keep same as baseline)
    fis_new = addInput(fis_new,[0 1],"Name","Occupancy");
    fis_new = addMF(fis_new,"Occupancy","trapmf",[0 0 0.15 0.35],"Name","Vacant");
    fis_new = addMF(fis_new,"Occupancy","trapmf",[0.65 0.85 1 1],"Name","Present");

    % Preference input [0 1] (keep same)
    fis_new = addInput(fis_new,[0 1],"Name","Preference");
    fis_new = addMF(fis_new,"Preference","trapmf",[0 0 0.2 0.45],"Name","LikesCool");
    fis_new = addMF(fis_new,"Preference","trimf",[0.35 0.5 0.65],"Name","Neutral");
    fis_new = addMF(fis_new,"Preference","trapmf",[0.55 0.8 1 1],"Name","LikesWarm");

    % Outputs (0..100) built from chrom centers
    fis_new = addOutput(fis_new,[0 100],"Name","Heater");
    for i=1:nOutPer
        c = Out_H(i);
        w = 12;
        a = max(0,c-w); b = c; d = min(100,c+w);
        if i==1
            params = [0 0 b d];
            fis_new = addMF(fis_new,"Heater","trapmf",params,"Name",sprintf("H%d",i));
        elseif i==nOutPer
            params = [a b 100 100];
            fis_new = addMF(fis_new,"Heater","trapmf",params,"Name",sprintf("H%d",i));
        else
            params = [a b d];
            fis_new = addMF(fis_new,"Heater","trimf",params,"Name",sprintf("H%d",i));
        end
    end

    fis_new = addOutput(fis_new,[0 100],"Name","Fan");
    for i=1:nOutPer
        c = Out_F(i);
        w = 12;
        a = max(0,c-w); b = c; d = min(100,c+w);
        if i==1
            params = [0 0 b d];
            fis_new = addMF(fis_new,"Fan","trapmf",params,"Name",sprintf("F%d",i));
        elseif i==nOutPer
            params = [a b 100 100];
            fis_new = addMF(fis_new,"Fan","trapmf",params,"Name",sprintf("F%d",i));
        else
            params = [a b d];
            fis_new = addMF(fis_new,"Fan","trimf",params,"Name",sprintf("F%d",i));
        end
    end

    fis_new = addOutput(fis_new,[0 100],"Name","Dimmer");
    for i=1:nOutPer
        c = Out_D(i);
        w = 12;
        a = max(0,c-w); b = c; d = min(100,c+w);
        if i==1
            params = [0 0 b d];
            fis_new = addMF(fis_new,"Dimmer","trapmf",params,"Name",sprintf("D%d",i));
        elseif i==nOutPer
            params = [a b 100 100];
            fis_new = addMF(fis_new,"Dimmer","trapmf",params,"Name",sprintf("D%d",i));
        else
            params = [a b d];
            fis_new = addMF(fis_new,"Dimmer","trimf",params,"Name",sprintf("D%d",i));
        end
    end

    % Add rules (use provided numeric rules matrix)
    fis_new = addRule(fis_new, rules);
end

function [fis, rules] = build_baseline_fis()
    % Builds the baseline FIS (same MFs / rules as Part 1)
    fis = mamfis("Name","AssistiveHomeFLC", ...
        "AndMethod","min","OrMethod","max","ImplicationMethod","min","AggregationMethod","max","DefuzzificationMethod","centroid");

    % Temperature [10 35]
    fis = addInput(fis,[10 35],"Name","Temperature");
    fis = addMF(fis,"Temperature","trapmf",[10 10 12 16],"Name","Cold");
    fis = addMF(fis,"Temperature","trimf",[13 17 21],"Name","Cool");
    fis = addMF(fis,"Temperature","trimf",[19 22.5 26],"Name","Comfortable");
    fis = addMF(fis,"Temperature","trimf",[24 27.5 31],"Name","Warm");
    fis = addMF(fis,"Temperature","trapmf",[29 32 35 35],"Name","Hot");

    % Light [0 1000]
    fis = addInput(fis,[0 1000],"Name","Light");
    fis = addMF(fis,"Light","trapmf",[0 0 60 200],"Name","Dark");
    fis = addMF(fis,"Light","trimf",[120 280 440],"Name","Dim");
    fis = addMF(fis,"Light","trimf",[380 520 660],"Name","Moderate");
    fis = addMF(fis,"Light","trapmf",[600 800 1000 1000],"Name","Bright");

    % Occupancy [0 1]
    fis = addInput(fis,[0 1],"Name","Occupancy");
    fis = addMF(fis,"Occupancy","trapmf",[0 0 0.15 0.35],"Name","Vacant");
    fis = addMF(fis,"Occupancy","trapmf",[0.65 0.85 1 1],"Name","Present");

    % Preference [0 1]
    fis = addInput(fis,[0 1],"Name","Preference");
    fis = addMF(fis,"Preference","trapmf",[0 0 0.2 0.45],"Name","LikesCool");
    fis = addMF(fis,"Preference","trimf",[0.35 0.5 0.65],"Name","Neutral");
    fis = addMF(fis,"Preference","trapmf",[0.55 0.8 1 1],"Name","LikesWarm");

    % Outputs (Heater, Fan, Dimmer) 0..100
    fis = addOutput(fis,[0 100],"Name","Heater");
    fis = addMF(fis,"Heater","trapmf",[0 0 5 15],"Name","Zero");
    fis = addMF(fis,"Heater","trimf",[10 30 50],"Name","Low");
    fis = addMF(fis,"Heater","trimf",[40 60 80],"Name","Medium");
    fis = addMF(fis,"Heater","trapmf",[70 85 100 100],"Name","High");

    fis = addOutput(fis,[0 100],"Name","Fan");
    fis = addMF(fis,"Fan","trapmf",[0 0 5 15],"Name","Zero");
    fis = addMF(fis,"Fan","trimf",[10 30 50],"Name","Low");
    fis = addMF(fis,"Fan","trimf",[40 60 80],"Name","Medium");
    fis = addMF(fis,"Fan","trapmf",[70 85 100 100],"Name","High");

    fis = addOutput(fis,[0 100],"Name","Dimmer");
    fis = addMF(fis,"Dimmer","trapmf",[0 0 5 15],"Name","Off");
    fis = addMF(fis,"Dimmer","trimf",[10 30 50],"Name","Low");
    fis = addMF(fis,"Dimmer","trimf",[40 60 80],"Name","Medium");
    fis = addMF(fis,"Dimmer","trapmf",[70 85 100 100],"Name","High");

    % Rules (numeric): each row [Temp Light Occup Pref Heater Fan Dimmer weight conn]
    rules = [ ...
     1  0  2  3   4  1  0   1 1;
     2  0  2  3   3  1  0   1 1;
     1  0  2  2   3  1  0   1 1;
     2  0  2  2   2  1  0   1 1;
     3  0  2  1   1  2  0   1 1;
     4  0  2  1   1  3  0   1 1;
     5  0  2  1   1  4  0   1 1;
     5  0  2  2   1  4  0   1 1;
     4  0  2  2   1  3  0   1 1;
     1  0  1  0   2  1  0   1 1;
     0  0  1  0   1  1  0   1 1;
     0  1  2  0   0  0  4   1 1;
     0  2  2  0   0  0  3   1 1;
     0  3  2  0   0  0  2   1 1;
     0  4  2  0   0  0  1   1 1;
     0  0  1  0   0  0  1   1 1;
     3  1  2  3   2  1  4   1 1;
     3  1  2  1   1  2  4   1 1;
    ];

    fis = addRule(fis, rules);
end
