clear; clc;
%comp ='sot16.PITT'; 
comp ='striant';
preamble
% simulate data from X->Y, X<-Z1->Z2->Y, Z3->Y: Ze should be before Zo
nVars =4;
dag = zeros(nVars);
dag(1, 2)=1; dag(4, [1 2])=1; dag(3, 2)=1; 

domainCounts =[2 2 2 3];
N=5000;doN=500;
j=0;
%%
J =500;
t =tic;
while j<J

    %  generate BN
    [nodes, dc] = dag2randBN2(dag, 'discrete', 'maxNumStates', domainCounts, 'minNumStates', domainCounts);
    nodes{4}.cpt = [0.2;0.1;0.7];

    jtalg_do = tetradJtalgDo(dag,nodes,domainCounts, 1);
    [tIMdo] = tetradEIM(dag, nodes, domainCounts);
    jtalg =javaObject('edu.pitt.dbmi.custom.tetrad.lib.bayes.JunctionTree', tIMdo); % selection posterior 

    [pyxz, ~, xzconfs] = estimateCondProbJT(2, [1, 3], jtalg, nVars, domainCounts);
    [pydoxz, ~, zconfs] = estimateCondProbJT(2, [1, 3], jtalg_do, nVars, domainCounts); 
    pw = estimateCondProbJT(4, [], jtalg, nVars, domainCounts);
    
   
    %if b>0.05
    j=j+1;
    bias(j)= mean(abs(pydoxz(1, :) - pyxz(1, :)));

    fprintf(' | Time elapsed %.3f, iteration %d\n',toc(t), j);

   
    %else
    %    continue;
    %end
    %simulate observational, experimental data
    DoDs = simulatedata(nodes,N, 'discrete', 'domainCounts', domainCounts); % simulate data from original distribution
    DeDs = simulateDoData(nodes, 1,  0:domainCounts(1)-1, doN, 'discrete', 'domainCounts', domainCounts);
    DeDsTest = simulateDoData(nodes, 1,  0:domainCounts(1)-1, 500, 'discrete', 'domainCounts', domainCounts);
    trueProbs(:, j) = estimateTruePs(DeDsTest.data, jtalg_do, [1 3]);

   
    jtalg_do = tetradJtalgDo(dag,nodes,domainCounts, 1);
    cdjtalg =javaObject('edu.pitt.dbmi.custom.tetrad.lib.bayes.JunctionTree', tetradEIM(dag, nodes, domainCounts)); 
    y=2; x=1; ze=3;

    % sampling BNs
    [~, rnodespost, orderpost] = dag2BNPost(dag, DoDs.data, domainCounts); % Do posterior

    I=50;
    for iter=1:I
        samplepost = sampleBNpost(rnodespost, orderpost, domainCounts);
        [eIM(iter)] = tetradEIM(dag, samplepost, domainCounts);
        jtalgs(iter)  = javaObject('edu.pitt.dbmi.custom.tetrad.lib.bayes.JunctionTree', eIM(iter));
    end

    to=tic;
    inExp = [1 1 1 0];
    [sets, nSets] = allSubsets(nVars,3:nVars-1);
    sets(:, x) = true;
    nSetsB = 2^(nVars-3); %sets with variables in both Do and De
    [logpDegivDoHw, logpDegivDoHn, logpDogivH] =deal(nan(nSetsB,1));    
    [curSet, curZos, curZes, Nexp, Nobs,cmbconfigs, nConfigs, pobs, probsExp] = deal(cell(1, nSets));


    for iSet=1:nSets
        %fprintf('iSet %d-----------\n', iSet)
        curSet{iSet} = find(sets(iSet,:));
        curZes{iSet} = intersect(curSet{iSet}, 2:nVars-1);
        curZos{iSet} = intersect(curSet{iSet}, nVars);
        [pobs{iSet}, ~, cmbconfigs{iSet}, Nobs{iSet}] = cond_prob_mult_inst(y, [curSet{iSet}], DoDs.data, domainCounts(1:nVars));
        nConfigs{iSet} = size(cmbconfigs{iSet}, 1);
    end

    logpDogivH(:) = dirichlet_bayesian_score(Nobs{iSet});
    [probsmbye, ~, mbyeconfigs] = cond_prob_mult_inst(y, [1 3], DeDs.data, domainCounts);
    bExp = 2;

    for iSet=1:nSetsB

        %findImb
        [probsExp{iSet}, ~, ~, Nexp{iSet}] = cond_prob_mult_inst(y, [curSet{iSet}], DeDs.data, domainCounts(1:nVars));            
        logpDegivDoHw(iSet) = dirichlet_bayesian_score(Nexp{iSet}, Nobs{iSet});
        logpDegivDoHn(iSet) = dirichlet_bayesian_score(Nexp{iSet});

        % overlap

        % split observational and experimental variables 
        curZ = curSet{iSet};curZ = curZ(2:end);
        curZe = curZes{iSet};
        curZo = curZos{iSet};

        % compute counts
        [~,zeconfigs] = overlapPyxz(y,x, curZe, jtalg_do, domainCounts);

        nyxze = overlapCondCounts(y, x, curZe, zeconfigs, DeDs);
        nyxze_ = overlapCondCounts(y, x, curZe, zeconfigs, DoDs);

        % score  
        [logscoreha] = overlapScoreZeZo(y, x, curZe, curZo, jtalgs, domainCounts, nyxze, nyxze_);
        logscoresbarha = sum(dirichlet_bayesian_score(nyxze));
        

        % score to prob
        overlapProbs(iSet,1:2) = log2probs([logscoreha logscoresbarha]); 
        % \hat p(y|do(x), z).
        [cpha{iSet}, cphabar{iSet}, cpb{iSet}, zezoconfs{iSet}, zeconfs{iSet}] = overlap_cond_prob_hat(y, x, curZe, curZo, DeDs, DoDs, domainCounts, overlapProbs(iSet,1:2));
        
        % compute E(acc), E(ll)
        [pzezo{iSet}] = cond_prob_mult_inst(curZ, [], DoDs);
        [eacc(iSet), maxp{iSet}] = estimateExpectedAccuracy(cpb{iSet}, pzezo{iSet}, zezoconfs{iSet}); 
        [ell(iSet), lls{iSet}] = estimateExpectedLogLoss(cpb{iSet}, pzezo{iSet}, zezoconfs{iSet}); 
    end



    for iSet=nSetsB+1:nSets

        % split observational and experimental variables 
        curZ = curSet{iSet};curZ = curZ(2:end);
        curZe = curZes{iSet};
        curZo = curZos{iSet};
    
        % compute counts
        [~,zeconfigs] = overlapPyxz(y,x, curZe, jtalg_do, domainCounts);
    
        nyxze = overlapCondCounts(y, x, curZe, zeconfigs, DeDs);
        nyxze_ = overlapCondCounts(y, x, curZe, zeconfigs, DoDs);
    
        % score  
        [logscoreha] = overlapScoreZeZo(y, x, curZe, curZo, jtalgs, domainCounts, nyxze, nyxze_);
        logscoresbarha = sum(dirichlet_bayesian_score(nyxze));
        
    
        % score to prob
        overlapProbs(iSet,1:2) = log2probs([logscoreha logscoresbarha]); 
        % \hat p(y|do(x), z).
        [cpha{iSet}, cphabar{iSet}, cpb{iSet}, zezoconfs{iSet}, zeconfs{iSet}] = overlap_cond_prob_hat(y, x, curZe, curZo, DeDs, DoDs, domainCounts, overlapProbs(iSet,1:2));
        
        % compute E(acc), E(ll)
        [pzezo{iSet}] = cond_prob_mult_inst(curZ, [], DoDs);
        [eacc(iSet), maxp{iSet}] = estimateExpectedAccuracy(cpb{iSet}, pzezo{iSet}, zezoconfs{iSet}); 
        [ell(iSet), lls{iSet}] = estimateExpectedLogLoss(cpb{iSet}, pzezo{iSet}, zezoconfs{iSet}); 
    end
    % overlap best sets
    [overlap_eacc, bestz] = max(eacc);bestzovacc = curSet{bestz};
    [overlap_ell, bestzll] = min(ell);bestzovll = curSet{bestzll};

    %fimb best sets
    numer = [logpDegivDoHw+logpDogivH;logpDegivDoHn+logpDogivH];
    denom =sumOfLogsV(numer);
    scores = numer-denom;
    [~, b] =max(scores);
    if b<nSetsB+1
        bestzfimb = curSet{b};
        configs = cmbconfigs{b};
        bestprobs = dirichlet_posterior_expectation(Nexp{b}, Nobs{b});
    else
        bestzfimb = curSet{b-nSetsB};
        configs = cmbconfigs{b-nSetsB};
        bestprobs = probsExp{b-nSetsB};
        b = b-nSetsB;
    end



    test_acc(j, 1) = estimateAccuracy(cpb{bestz}, zezoconfs{bestz}, DeDsTest.data(:, [1 2 bestzovacc(2:end)])); %overlap
    test_acc(j, 2) = estimateAccuracy(reshape(bestprobs, 2,2,[]), zezoconfs{b}, DeDsTest.data(:, [1 2 bestzfimb(2:end)]));% findimb
    test_acc(j, 3) = estimateAccuracy(cpha{end}, zezoconfs{end}, DeDsTest.data(:, [1 2 3])); % gt omb
    test_acc(j, 4) = estimateAccuracy(cphabar{bExp}, zezoconfs{bExp}, DeDsTest.data(:, [1 2 3])); % gt imb
  
    
    test_ll(j, 1) = estimateLogLoss(cpb{bestzll}, zezoconfs{bestzll}, DeDsTest.data(:, [1 2 bestzovll(2:end)])); %overlap
    test_ll(j, 2) = estimateLogLoss(reshape(bestprobs, 2,2,[]), zezoconfs{b}, DeDsTest.data(:, [1 2 bestzfimb(2:end)]));% findimb
    test_ll(j, 3) = estimateLogLoss(cpha{end}, zezoconfs{end}, DeDsTest.data(:, [1 2 3])); % gt omb
    test_ll(j, 4) = estimateLogLoss(cphabar{bExp}, zezoconfs{bExp}, DeDsTest.data(:, [1 2 3])); % gt omb
    ll_gt(j) = mean(logloss(1-trueProbs(:, j),  DeDsTest.data(:, 2)));


    fprintf(' | Time elapsed %.3f, nSets %d\n',toc(t), nSets);%%

end

%%
colors = colormap(lines);

numBins = 5;
llov = real(test_ll(:, 1)-ll_gt');
llo = real(test_ll(:, 3)-ll_gt');
lle = real(test_ll(:, 4)-ll_gt');
llfimb = real(test_ll(:, 2)-ll_gt');

[mean_bin_ov, std_bin_ov, bin_size_ov, bin_centers_ov, q_bin_ov] = binvalues(llov, bias, numBins);

[mean_bin_o, std_bin_o, bin_size_o, bin_centers_o, q_bin_o] = binvalues(llo, bias, numBins);
[mean_bin_e, std_bin_e, bin_size_e, bin_centers_e, q_bin_e] = binvalues(lle, bias, numBins);
[mean_bin_f, std_bin_f, bin_size_f, bin_centers_f, q_bin_f] = binvalues(llfimb, bias, numBins);

close all;
fig =figure;hold all;
%scatter(bin_centers_ov, mean_bin_ov, (bin_size_ov+1)*3, 'filled', 'MarkerFaceAlpha',.8);
p(1) =plot(bin_centers_ov, mean_bin_ov, 'LineWidth', 2, 'color', colors(1, :),'Marker', 'o');
%shadedErrorBar(bin_centers_ov, mean_bin_ov, std_bin_ov, 'LineProps', {'LineWidth', 2});
plot(bin_centers_ov, q_bin_ov, ':', 'LineWidth', 2, 'Color', p(1).Color);
hold all;

%scatter(bin_centers_o, mean_bin_o, (bin_size_o+1)*3, 'filled', 'MarkerFaceAlpha',.8);
p(2) =plot(bin_centers_o, mean_bin_o, 'LineWidth', 2, 'color', colors(2, :), 'Marker', 'x');
%shadedErrorBar(bin_centers_o, mean_bin_o, std_bin_o, 'LineProps', {'LineWidth', 2});
plot(bin_centers_o, q_bin_o, ':', 'LineWidth', 2, 'Color', p(2).Color);

hold on;
%scatter(bin_centers_e, mean_bin_e, (bin_size_e+1)*3, 'filled', 'MarkerFaceAlpha',.8);
%shadedErrorBar(bin_centers_e, mean_bin_e, std_bin_e, 'LineProps', {'LineWidth', 2});
p(3) =plot(bin_centers_e, mean_bin_e, 'LineWidth', 2, 'color', colors(3, :), 'Marker', '+');
plot(bin_centers_e, q_bin_e, ':', 'LineWidth', 2, 'Color', p(3).Color);

%plot(bin_centers_e, mean_bin_e, 'LineWidth', 2);
% hold on;
p(4) =plot(bin_centers_f, mean_bin_f, 'LineWidth', 2, 'color', colors(4, :), 'Marker', '*');
plot(bin_centers_f, q_bin_f, ':', 'LineWidth', 2, 'Color', p(4).Color);


xlabel('mean absolute difference $|P(Y|do(X),Z)-P(Y|do(X))|$', 'Interpreter', 'latex', 'FontSize', 20)
ylabel('$ll-ll_{GT}$', 'Interpreter', 'Latex','FontSize', 20)
legend(p,{'$Overlap$', '$OMB$', '$IMB$', "FindIMB"}, 'Interpreter', 'Latex','FontSize', 18, 'Location', 'NorthWest')
yline(0, 'HandleVisibility','off')

fig.PaperPositionMode = 'auto';
fig_pos = fig.PaperPosition;
fig.PaperSize = [fig_pos(3) fig_pos(4)];
%saveas(gcf, ['confounder_sim'], 'pdf');

%%
colors = colormap(lines); 

numBins = 4;
llov = real(test_ll(:, 1)-ll_gt');
llo = real(test_ll(:, 3)-ll_gt');
lle = real(test_ll(:, 4)-ll_gt');
llfimb = real(test_ll(:, 2)-ll_gt');


[mean_bin_ov, std_bin_ov, bin_size_ov, bin_centers_ov, q_bin_ov] = binvalues(llov, bias, numBins);
[mean_bin_o, std_bin_o, bin_size_o, bin_centers_o, q_bin_o] = binvalues(llo, bias, numBins);
[mean_bin_e, std_bin_e, bin_size_e, bin_centers_e, q_bin_e] = binvalues(lle, bias, numBins);
[mean_bin_f, std_bin_f, bin_size_f, bin_centers_f, q_bin_f] = binvalues(llfimb, bias, numBins);

close all;
fig =figure;hold all;box on;
p(1) =plot(bin_centers_ov, mean_bin_ov, 'LineWidth', 2, 'color', colors(1, :), 'Marker', 'o', 'MarkerSize', 15, 'MarkerFaceColor',colors(1, :));
%shadedBarQ(bin_centers_ov, q_bin_ov, colors(1, :), 0.1);
hold all;

%scatter(bin_centers_o, mean_bin_o, (bin_size_o+1)*3, 'filled', 'MarkerFaceAlpha',.8);
p(2) =plot(bin_centers_o, mean_bin_o, 'LineWidth', 2, 'color', colors(2, :), 'Marker', 's', 'MarkerSize', 20);
%shadedErrorBar(bin_centers_o, mean_bin_o, std_bin_o, 'LineProps', {'LineWidth', 2});
%shadedBarQ(bin_centers_o, q_bin_o, colors(2, :), 0.1);


hold on;
%scatter(bin_centers_e, mean_bin_e, (bin_size_e+1)*3, 'filled', 'MarkerFaceAlpha',.8);
%shadedErrorBar(bin_centers_e, mean_bin_e, std_bin_e, 'LineProps', {'LineWidth', 2});
p(3) =plot(bin_centers_e, mean_bin_e, 'LineWidth', 2, 'color', colors(3, :), 'Marker', 'x', 'MarkerSize', 20);
%shadedBarQ(bin_centers_e, q_bin_e, colors(3, :), 0.1);

%plot(bin_centers_e, mean_bin_e, 'LineWidth', 2);
% hold on;
p(4) =plot(bin_centers_f, mean_bin_f, 'LineWidth', 2, 'color', [0 .8 0.5], 'Marker', '*', 'MarkerSize', 20 );
%shadedBarQ(bin_centers_f, q_bin_f, [0 0.8 0.5], 0.1);

xlim([bin_centers_o(1) bin_centers_o(end)])
xlabel('mean absolute difference $|P(Y|do(X))-P(Y|X)|$', 'Interpreter', 'latex', 'FontSize', 18)
ylabel('$ll-ll_{GT}$', 'Interpreter', 'Latex','FontSize', 20)
legend(p,{'$Overlap$', 'OMB', 'IMB', "FindIMB"}, 'Interpreter', 'Latex','FontSize', 18, 'Location', 'NorthWest')
yline(0, 'HandleVisibility','off')

fig.PaperPositionMode = 'auto';
fig_pos = fig.PaperPosition;
fig.PaperSize = [fig_pos(3) fig_pos(4)];
saveas(gcf, ['scenario_2_doN_' num2str(doN)], 'pdf');

%%



close all;
fig =figure;hold all;box on
p(1) =plot(bin_centers_ov, mean_bin_ov, 'LineWidth', 2, 'color', colors(1, :), 'Marker', 'o', 'MarkerSize', 15, 'MarkerFaceColor',colors(1, :));
shadedBarQ(bin_centers_ov, q_bin_ov, colors(1, :), 0.1);
hold all;

%scatter(bin_centers_o, mean_bin_o, (bin_size_o+1)*3, 'filled', 'MarkerFaceAlpha',.8);
p(2) =plot(bin_centers_o, mean_bin_o, 'LineWidth', 2, 'color', colors(2, :), 'Marker', 's', 'MarkerSize', 20);
shadedBarQ(bin_centers_o, q_bin_o, colors(2, :), 0.1);


hold on;
%scatter(bin_centers_e, mean_bin_e, (bin_size_e+1)*3, 'filled', 'MarkerFaceAlpha',.8);
%shadedErrorBar(bin_centers_e, mean_bin_e, std_bin_e, 'LineProps', {'LineWidth', 2});
p(3) =plot(bin_centers_e, mean_bin_e, 'LineWidth', 2, 'color', colors(3, :), 'Marker', 'x', 'MarkerSize', 20);
shadedBarQ(bin_centers_e, q_bin_e, colors(3, :), 0.1);

%plot(bin_centers_e, mean_bin_e, 'LineWidth', 2);
% hold on;
p(4) =plot(bin_centers_f, mean_bin_f, 'LineWidth', 2, 'color', [0 .8 0.5], 'Marker', '*', 'MarkerSize', 20 );
shadedBarQ(bin_centers_f, q_bin_f, [0 0.8 0.5], 0.1);

xlim([bin_centers_o(1) bin_centers_o(end)])

xlabel('mean absolute difference $|P(Y|do(X),W)-P(Y|X, W)|$', 'Interpreter', 'latex', 'FontSize', 18)
ylabel('$ll-ll_{GT}$', 'Interpreter', 'Latex','FontSize', 20)
legend(p,{'$Overlap$', 'OMB', 'IMB', "FindIMB"}, 'Interpreter', 'Latex','FontSize', 18, 'Location', 'NorthWest')
yline(0, 'HandleVisibility','off')

fig.PaperPositionMode = 'auto';
fig_pos = fig.PaperPosition;
fig.PaperSize = [fig_pos(3) fig_pos(4)];
saveas(gcf, ['scenario_2_doN_' num2str(doN) '_quantiles'], 'pdf');

