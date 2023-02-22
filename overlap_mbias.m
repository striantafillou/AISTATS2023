% test the new algorithm
clear; clc;
comp ='sofia';

code_path = ['C:\Users\' comp '\Dropbox\MATLAB\causal_graphs'];
addpath(genpath(code_path));
code_path = ['C:\Users\' comp '\Dropbox\MATLAB\causal_effects'];
addpath(genpath(code_path));
code_path = ['C:\Users\' comp '\Dropbox\MATLAB\UAI2021'];
addpath(genpath(code_path));
code_path = ['C:\Users\' comp '\Dropbox\MATLAB\ICML2022'];
cd([code_path ]);
% control panel
N=5000;x=1; y=2;
doN =100;
load mbias_bn.mat  

%load confandemgraph.mat
nVars= sum(~isLatent);
smm = dag2smm(dag, isLatent);
domainCounts =[2 2 randi([2 2], 1, 3)];


nIters=50;
%scores = nan(nSets*2, nDoNs, nIters);

[nodes, dc] = dag2randBN2(dag, 'discrete', 'maxNumStates', domainCounts, 'minNumStates', domainCounts);

as=[0.8:0.02:1];
ia=0;
for a=as
    ia=ia+1;
    
    %Age
    nodes{4}.cpt = [.8 .2]';
    % 5: Gender F/M
    nodes{5}.cpt = [.3 .7]';
    % 3. noisy and
    nodes{3}.cpt(:, 1,1) =[a 1-a];
    nodes{3}.cpt(:, 2,1) =[0 1];
    nodes{3}.cpt(:, 1,2) =[0 1];
    nodes{3}.cpt(:, 2,2) =[0 1];
    % Outcome (no, yes) treatment (no/yes), gender(F/M)
    nodes{2}.cpt(:, 1,1) =[a 1-a];
    nodes{2}.cpt(:, 1,2) =[0 1];
    nodes{2}.cpt(:, 2,1) =[0 1];
    nodes{2}.cpt(:, 2,2) =[0 1];
    % treatment no/yes giv age
    nodes{1}.cpt(:, 1) =[a,1-a];
    nodes{1}.cpt(:, 2) =[0 1];
    isLatent  = [false false false true true];
    
    
    jtalg_do = tetradJtalgDo(dag,nodes,domainCounts, 1);
    [tIMdo] = tetradEIM(dag, nodes, domainCounts);
    jtalg =javaObject('edu.pitt.dbmi.custom.tetrad.lib.bayes.JunctionTree', tIMdo); % selection posterior 

    [pydoxz, ~, xzconfs] = estimateCondProbJT(2, [1,3], jtalg_do, nVars, domainCounts);
    pydox = estimateCondProbJT(2, 1, jtalg_do, nVars, domainCounts);
    pz = estimateCondProbJT(3, [], jtalg, nVars, domainCounts);
    
        
    bias(ia) =sum(abs(pydox(1, :) - pydoxz(1, 3:4)).*pz(1));
    
    %simulate observational, experimental data
    DoDs = simulatedata(nodes,N, 'discrete', 'domainCounts', domainCountsOr); % simulate data from original distribution
    DeDs = simulateDoData(nodes, 1,  0:domainCountsOr(1)-1, doN, 'discrete', 'domainCounts', domainCounts);
    DeDsTest = simulateDoData(nodes, 1,  0:domainCountsOr(1)-1, 500, 'discrete', 'domainCounts', domainCounts);

    cdjtalg =javaObject('edu.pitt.dbmi.custom.tetrad.lib.bayes.JunctionTree', tetradEIM(dag, nodes, domainCounts)); 
    y=2; x=1; ze=3; zo=[];
    [~, rnodespost, orderpost] = dag2BNPost(dag, DoDs.data, domainCountsOr); % Do posterior

    allze = allSubsets(nVars, ze);nZe = size(allze, 1);
    allzo = allSubsets(nVars, zo);nZo = size(allzo, 1);


    nZ = nZe*nZo;
    allz = allSubsets(nZ, nVars);
    probsz = zeros(nZ, 2);
    iZ =0;
    I=50;
    for iter=1:I
        samplepost = sampleBNpost(rnodespost, orderpost, domainCountsOr);
        [eIM(iter)] = tetradEIM(dag, samplepost, domainCountsOr);
        jtalgs(iter)  = javaObject('edu.pitt.dbmi.custom.tetrad.lib.bayes.JunctionTree', eIM(iter));
    end

    [cpha, cphabar, cpb, zezoconfs, zeconfs, pzezo, cptrue] = deal(cell(1, nZ)); 

    for iZe =1:nZe
        curZe =  find(allze(iZe, :));
        for iZo = 1:nZo
            iZ =iZ+1;
            curZo = find(allzo(iZo,:));
            curZ = any([allzo(iZo,:);allze(iZe, :)]);
            zs(iZ, :) = curZ;
    %         [pyxzezo] = overlapPyxzezo(y, x, curZe, curZo, jtalg, domainCounts);
    %         [pzoze] = overlapPzoze(curZo, curZe, jtalg, domainCounts);
    %         pydoxzehat = overlapAdjustment(pyxzezo, pzoze);
            [cptrue{iZ},zeconfigs] = overlapPyxz(y,x, curZe, jtalg_do, domainCounts);
            [cpbtrue{iZ}] = overlapPyxz(y,x, [curZe curZo], jtalg_do, domainCounts);

    %        bias{iZ} =abs(pydoxze - pydoxzehat);

            %list= tetradList(nVars, domainCounts);
            nyxze = overlapCondCounts(y, x, curZe, zeconfigs,DeDs);
            nyxze_ = overlapCondCounts(y, x, curZe, zeconfigs,DoDs);

            % score set 
            [logscoreha] = overlapScoreZeZo(y, x, curZe, curZo, jtalgs, domainCounts, nyxze, nyxze_);
            logscoresbarha = sum(dirichlet_bayesian_score(nyxze));
            probsz(iZ,:) = log2probs([logscoreha logscoresbarha]); 
            
          
            [cpha{iZ}, cphabar{iZ}, cpb{iZ}, zezoconfs{iZ}, zeconfs{iZ}] = overlap_cond_prob_hat(y, x, curZe, curZo, DeDs, DoDs, domainCounts, probsz(iZ,:));
            [pzezo{iZ}] = cond_prob_mult_inst([curZe curZo], [], DoDs);
            [eacc(iZ), maxp{iZ}] = estimateExpectedAccuracy(cpb{iZ}, pzezo{iZ}, zezoconfs{iZ}); 
            %[acc(iZ)] = estimateAccuracy(cpb{iZ}, zezoconfs{iZ}, DeDsTest.data(:, [1 2 find(curZ)])); 
            [ell(iZ), lls{iZ}] = estimateExpectedLogLoss(cpb{iZ}, pzezo{iZ}, zezoconfs{iZ}); 
           % [ll(iZ)] = estimateLogLoss(cpb{iZ}, zezoconfs{iZ}, DeDsTest.data(:, [1 2 find(curZ)])); 
        end
    end
% best Z based on eacc;
[overlap_ell(ia), bestzll(ia)] = min(ell);

test_ll(ia, 1)= estimateLogLoss(cpb{bestzll(ia)}, zezoconfs{bestzll(ia)}, DeDsTest.data(:, [1 2 find(zs(bestzll(ia), :))]));
test_ll(ia, 2) = estimateLogLoss(cpha{2}, zezoconfs{2}, DeDsTest.data(:, [1 2 3]));% observational data on all;
test_ll(ia, 3) = estimateLogLoss(cphabar{2}, zezoconfs{2}, DeDsTest.data(:, [1 2 3]));% experimental data  data on all;

test_ll_gt(ia) =estimateLogLoss(cpbtrue{2}, zezoconfs{2}, DeDsTest.data(:, [1 2 3]));
end
plot(bias, test_ll-test_ll_gt')
%%

numBins = 10;
llov = test_ll(:, 1)-test_ll_gt';
llo = test_ll(:, 2)-test_ll_gt';
lle = test_ll(:, 3)-test_ll_gt';
[mean_bin_ov, std_bin_ov, bin_size_ov, bin_centers_ov] = binvalues(llov, bias, 10);

[mean_bin_o, std_bin_o, bin_size_o, bin_centers_o] = binvalues(llo, bias, 10);
[mean_bin_e, std_bin_e, bin_size_e, bin_centers_e] = binvalues(lle, bias, 10);
close all;
figure;
scatter(bin_centers_ov, mean_bin_ov, (bin_size_ov+1)*3, 'filled', 'MarkerFaceAlpha',.8);

hold all;
scatter(bin_centers_o, mean_bin_o, (bin_size_o+1)*3, 'filled', 'MarkerFaceAlpha',.8);
hold on;
scatter(bin_centers_e, mean_bin_e, (bin_size_e+1)*3, 'filled', 'MarkerFaceAlpha',.8);

xlabel('average difference $P(Y|do(X),Z)-P(Y|do(X))$', 'Interpreter', 'latex')
ylabel('Difference from ground truth (log loss)', 'Interpreter', 'Latex')
legend({'$Overlap$', '$D_o$', '$D_e$'}, 'Interpreter', 'Latex','FontSize', 18, 'Location', 'NorthWest')

set(gca, 'XLim', [0.05,bin_centers_o(end)]);
line(get(gca, 'XLim'), [0 0], 'color','k', 'HandleVisibility','off')
fig = gcf;

fig.PaperPositionMode = 'auto';
fig_pos = fig.PaperPosition;
fig.PaperSize = [fig_pos(3) fig_pos(4)];

saveas(gcf, 'mbias', 'pdf');