%% Assistive Home Fuzzy Logic Controller (Mamdani)
% Part 1 of assignment: design, rules, inference, plots
% Requires: Fuzzy Logic Toolbox

clear; clc; close all;

%% Create Mamdani FIS
fis = mamfis( ...
    "Name","AssistiveHomeFLC", ...
    "AndMethod","min", ...
    "OrMethod","max", ...
    "ImplicationMethod","min", ...
    "AggregationMethod","max", ...
    "DefuzzificationMethod","centroid");

%% -----------------------
%  INPUTS
% 1) Temperature [10..35] °C
fis = addInput(fis,[10 35],"Name","Temperature");
fis = addMF(fis,"Temperature","trapmf",[10 10 12 16],"Name","Cold");
fis = addMF(fis,"Temperature","trimf",[13 17 21],"Name","Cool");
fis = addMF(fis,"Temperature","trimf",[19 22.5 26],"Name","Comfortable");
fis = addMF(fis,"Temperature","trimf",[24 27.5 31],"Name","Warm");
fis = addMF(fis,"Temperature","trapmf",[29 32 35 35],"Name","Hot");

% 2) Ambient Light [0..1000] lux
fis = addInput(fis,[0 1000],"Name","Light");
fis = addMF(fis,"Light","trapmf",[0 0 60 200],"Name","Dark");
fis = addMF(fis,"Light","trimf",[120 280 440],"Name","Dim");
fis = addMF(fis,"Light","trimf",[380 520 660],"Name","Moderate");
fis = addMF(fis,"Light","trapmf",[600 800 1000 1000],"Name","Bright");

% 3) Occupancy [0..1]
fis = addInput(fis,[0 1],"Name","Occupancy");
fis = addMF(fis,"Occupancy","trapmf",[0 0 0.15 0.35],"Name","Vacant");
fis = addMF(fis,"Occupancy","trapmf",[0.65 0.85 1 1],"Name","Present");

% 4) User Temperature Preference [0..1]
fis = addInput(fis,[0 1],"Name","Preference");
fis = addMF(fis,"Preference","trapmf",[0 0 0.2 0.45],"Name","LikesCool");
fis = addMF(fis,"Preference","trimf",[0.35 0.5 0.65],"Name","Neutral");
fis = addMF(fis,"Preference","trapmf",[0.55 0.8 1 1],"Name","LikesWarm");

%% -----------------------
%  OUTPUTS  (all as 0..100 % duty)
% Heater
fis = addOutput(fis,[0 100],"Name","Heater");
fis = addMF(fis,"Heater","trapmf",[0 0 5 15],"Name","Zero");
fis = addMF(fis,"Heater","trimf",[10 30 50],"Name","Low");
fis = addMF(fis,"Heater","trimf",[40 60 80],"Name","Medium");
fis = addMF(fis,"Heater","trapmf",[70 85 100 100],"Name","High");

% Fan
fis = addOutput(fis,[0 100],"Name","Fan");
fis = addMF(fis,"Fan","trapmf",[0 0 5 15],"Name","Zero");
fis = addMF(fis,"Fan","trimf",[10 30 50],"Name","Low");
fis = addMF(fis,"Fan","trimf",[40 60 80],"Name","Medium");
fis = addMF(fis,"Fan","trapmf",[70 85 100 100],"Name","High");

% Light Dimmer
fis = addOutput(fis,[0 100],"Name","Dimmer");
fis = addMF(fis,"Dimmer","trapmf",[0 0 5 15],"Name","Off");
fis = addMF(fis,"Dimmer","trimf",[10 30 50],"Name","Low");
fis = addMF(fis,"Dimmer","trimf",[40 60 80],"Name","Medium");
fis = addMF(fis,"Dimmer","trapmf",[70 85 100 100],"Name","High");

%% -----------------------
% RULES (numeric form)
% MF index maps (for readability in comments):
% Temperature: 1 Cold, 2 Cool, 3 Comfortable, 4 Warm, 5 Hot
% Light:       1 Dark, 2 Dim, 3 Moderate, 4 Bright
% Occupancy:   1 Vacant, 2 Present
% Preference:  1 LikesCool, 2 Neutral, 3 LikesWarm
% Heater:      1 Zero, 2 Low, 3 Medium, 4 High
% Fan:         1 Zero, 2 Low, 3 Medium, 4 High
% Dimmer:      1 Off,  2 Low, 3 Medium, 4 High
% Rule row: [Temp Light Occup Pref  -> Heater Fan Dimmer  weight  connective]
% connective: 1=AND, 2=OR

rules = [ ...
 % --- HVAC: temperature vs preference/occupancy
 1  0  2  3   4  1  0   1 1;  % Cold & Present & LikesWarm -> Heater High, Fan Zero
 2  0  2  3   3  1  0   1 1;  % Cool & Present & LikesWarm -> Heater Med
 1  0  2  2   3  1  0   1 1;  % Cold & Present & Neutral   -> Heater Med
 2  0  2  2   2  1  0   1 1;  % Cool & Present & Neutral   -> Heater Low
 3  0  2  1   1  2  0   1 1;  % Comfortable & Present & LikesCool -> Heater Zero, Fan Low
 4  0  2  1   1  3  0   1 1;  % Warm & Present & LikesCool -> Fan Med
 5  0  2  1   1  4  0   1 1;  % Hot  & Present & LikesCool -> Fan High
 5  0  2  2   1  4  0   1 1;  % Hot  & Present & Neutral   -> Fan High
 4  0  2  2   1  3  0   1 1;  % Warm & Present & Neutral   -> Fan Med
 1  0  1  0   2  1  0   1 1;  % Cold & Vacant -> small anti-freeze Low heat
 0  0  1  0   1  1  0   1 1;  % Vacant -> HVAC mostly off

 % --- Lighting: ambient light vs occupancy
 0  1  2  0   0  0  4   1 1;  % Dark & Present   -> Dimmer High
 0  2  2  0   0  0  3   1 1;  % Dim  & Present   -> Dimmer Med
 0  3  2  0   0  0  2   1 1;  % Moderate & Present -> Dimmer Low
 0  4  2  0   0  0  1   1 1;  % Bright -> Dimmer Off
 0  0  1  0   0  0  1   1 1;  % Vacant -> Dimmer Off

 % --- Coupled comfort tweaks
 3  1  2  3   2  1  4   1 1;  % Comfortable & Dark & Present & LikesWarm -> a bit of heat, bright lights
 3  1  2  1   1  2  4   1 1;  % Comfortable & Dark & Present & LikesCool -> add a touch of fan, bright lights
];

fis = addRule(fis, rules);

%% -----------------------
% Inspect basic info
disp(fis)

%% -----------------------
% Plot membership functions (evidence for report)
figure('Name','Input MFs - Temperature'); plotmf(fis,'input',1);
figure('Name','Input MFs - Light');       plotmf(fis,'input',2);
figure('Name','Input MFs - Occupancy');   plotmf(fis,'input',3);
figure('Name','Input MFs - Preference');  plotmf(fis,'input',4);

figure('Name','Output MFs - Heater');     plotmf(fis,'output',1);
figure('Name','Output MFs - Fan');        plotmf(fis,'output',2);
figure('Name','Output MFs - Dimmer');     plotmf(fis,'output',3);

%% -----------------------
% Control surface plots (select pairs of inputs)
% Note: gensurf handles up to 2 inputs -> 1 output view at a time.
figure('Name','Surface: Temp & Pref -> Heater');
gensurf(fis,[1 4],1); % inputs [Temperature Preference] -> Heater

figure('Name','Surface: Temp & Pref -> Fan');
gensurf(fis,[1 4],2); % inputs [Temperature Preference] -> Fan

figure('Name','Surface: Light & Occupancy -> Dimmer');
gensurf(fis,[2 3],3); % inputs [Light Occupancy] -> Dimmer

%% -----------------------
% Example evaluations (for screenshots + table in report)
% [Temperature, Light, Occupancy, Preference]
scenarios = [ ...
    14   100   1   0.9;  % cold, dark, present, likes warm
    24   150   1   0.1;  % comfortable, dim, present, likes cool
    31   800   1   0.5;  % hot, bright, present, neutral
    18   50    0   0.5;  % cool, dark, vacant, neutral
    22   400   1   0.5]; % comfortable, moderate, present, neutral

out = evalfis(fis, scenarios);
T = array2table([scenarios out], ...
    "VariableNames",{'TempC','Lux','Occup','Pref','Heater','Fan','Dimmer'});
disp(T);
