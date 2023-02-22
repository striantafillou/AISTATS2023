% test the new algorithm
clear; clc;
comp = 'striant';
preamble;

% control panel
N=5000;x=1; y=2;
doNs =100;%[50 100 500];
maxDoN = doNs(end);nDoNs =length(doNs);iDoN =1;
doInds = getExperimentalInds(maxDoN, doNs, nDoNs, 2);

nVarsOr =5;
isLatent = false(1, nVarsOr); 
%load confandemgraph.mat
nVars= sum(~isLatent);
domainCountsOr =[2 2 randi([2 3], 1, nVarsOr-2)];


nIters=100;
% initialize
[auc, sens, spec] = deal(nan(nIters,6));
[cmbProbs, mbeProbs, mbProbs, imboProbs, overProbsll, overProbs] = deal(nan(2000, nIters, nDoNs));
[trueProbs, trueEvents] = deal(nan(2000, nIters));
diffs = nan(nIters,1);
timesObs = nan(nIters, 1);
timesExp = nan(nIters,1);
timesFCI = nan(nIters,1);
saveFolder = 'C:/Users/sot16.PITT/Dropbox/MATLAB/causal_effects/results/uai2021';
if(~isfolder(saveFolder));mkdir(saveFolder);end

imbys_true =false(nIters, nVars);
[imbys_fci, imbys_fimb, imbys_e] = deal(false(nIters, nVars, nDoNs));
[test_acc, test_ll] = deal(nan(nIters, 5, nDoNs));ll_gt = nan(nIters, 1);
fName = ['nVars_' num2str(nVars) '_N_' num2str(floor(N/1000)) 'K_res'] ;%

for iter=1:nIters
    totalt =tic;
    fprintf('Iter %d \t', iter);
    
    dag = randomDagWith12PreTreatmentConf(nVarsOr, 5);
    dag(5,2)=1; % V3->Y latent in D_e
    omby = find(dag(:,2));


    % nodesOr: with confounders.
    [nodesOr, domainCountsOr, orderOr, rnodes] = dag2randBN2(dag, 'discrete', 'maxNumStates', domainCountsOr, 'minNumStates', domainCountsOr);
    dagx = dag;dagx(:, x)=0;smmx = dag2smm(dagx, isLatent);
    imby_true =findMBsmm(smmx,y);imby = setdiff(imby_true, 1);
    imbys_true(iter, imby)=true;
    
    jtalg_do_true = tetradJtalgDo(dag,nodesOr,domainCountsOr, 1);
    jtalg_do = tetradJtalgDo(dag,nodesOr,domainCountsOr, 1);
    % simulate data with dummy nodes.
    obsDatasetOr = simulatedata(nodesOr,N, 'discrete', 'domainCounts', domainCountsOr, 'isLatent', isLatent); % simulate data from original distribution
 
    
    % observational data, learn dag, and BN
    obsDs = subdataset(obsDatasetOr);domainCounts=obsDs.domainCounts;   

    % simulate experimental data 
    expDsOr = simulateDoData(nodesOr, 1,  0:domainCountsOr(1)-1, maxDoN, 'discrete', 'domainCounts', domainCounts);
    expDs = subdataset(expDsOr,isLatent);
    expData = expDs.data;
    
    % simulate test data
    testDsOr = simulateDoData(nodesOr, 1,  0:domainCountsOr(1)-1, 1000, 'discrete', 'domainCounts', domainCounts);    
    testDs = subdataset(testDsOr,isLatent);
    testData = testDs.data;
    
    trueProbs(:, iter) = estimateTruePs(testData, jtalg_do_true, imby_true);
    
    to=tic;
    mby = [3:nVars]; % add all variables
    inExp = true(1,nVars); inExp(nVars)=false;
    timesObs(iter) = toc(to);
    fprintf(' | Time elapsed %.3f\t',toc(totalt));
    te=tic;
    [sets, nSets] = allSubsets(nVars,mby);
    sets(:, x) = true;
    nSetsB = 2^(nVars-3); %sets with variables in both Do and De
    [logpDegivDoHw, logpDegivDoHn] =deal(nan(nSets/2,nDoNs));    
    logpDogivH = nan(nSetsB,1);
    [curSet, curZos, curZes, Nexp, Nobs,cmbconfigs, nConfigs, pobs, probsExp] = deal(cell(1, nSets));


    for iSet=1:nSets
        %fprintf('iSet %d-----------\n', iSet)
        curSet{iSet} = find(sets(iSet,:));
        curZes{iSet} = intersect(curSet{iSet}, 2:nVars-1);
        curZos{iSet} = intersect(curSet{iSet}, nVars);
        [pobs{iSet}, ~, cmbconfigs{iSet}, Nobs{iSet}] = cond_prob_mult_inst(y, [curSet{iSet}], obsDs.data, domainCounts(1:nVars));
        nConfigs{iSet} = size(cmbconfigs{iSet}, 1);
    end

    logpDogivH(:) = dirichlet_bayesian_score(Nobs{iSet});
    timesExp(iter, 1) =toc(te);
    for iDoN =1:nDoNs
        ti =tic;
        
        [adage]= tetradFges12(expData(doInds(:, iDoN), 1:nVars-1), domainCounts, 'pretreat', true, 'onlyY', true); % only on experimental data
        mbye = find(adage(:, 2))';


        [probsmbye, ~, mbyeconfigs] = cond_prob_mult_inst(y, mbye, expData(doInds(:, iDoN), :), domainCounts(1:nVars));
        
        % overlap initialize matrices
        [eacc, ell] = deal(nan(nSets,1));
        overlapProbs = nan(nSets,domainCounts(y), nDoNs);
        [cpha, cphabar, cpb, zezoconfs, zeconfs, pzezo, maxp, lls] = deal(cell(nSets, 1));
        % overlap sample graphs
        I=50;
        [~, rnodespost, orderpost] = dag2BNPost(dag, obsDs.data, domainCounts); % Do posterior
        samplepost = sampleBNpost(rnodespost, orderpost, domainCounts);

        for i=1:I
            samplepost = sampleBNpost(rnodespost, orderpost, domainCounts);
            [eIM(i)] = tetradEIM(dag, samplepost, domainCounts);
            jtalgs(i)  = javaObject('edu.pitt.dbmi.custom.tetrad.lib.bayes.JunctionTree', eIM(i));
        end

        for iSet=1:nSetsB

            %findImb
            [probsExp{iSet}, ~, ~, Nexp{iSet}] = cond_prob_mult_inst(y, [curSet{iSet}], expData(doInds(:, iDoN), :), domainCounts(1:nVars));            
            logpDegivDoHw(iSet, iDoN) = dirichlet_bayesian_score(Nexp{iSet}, Nobs{iSet});
            logpDegivDoHn(iSet, iDoN) = dirichlet_bayesian_score(Nexp{iSet});

            % overlap

            % split observational and experimental variables 
            curZ = curSet{iSet};curZ = curZ(2:end);
            curZe = curZes{iSet};
            curZo = curZos{iSet};

            % compute counts
            [~,zeconfigs] = overlapPyxz(y,x, curZe, jtalg_do, domainCounts);

            nyxze = overlapCondCounts(y, x, curZe, zeconfigs, expDs);
            nyxze_ = overlapCondCounts(y, x, curZe, zeconfigs, obsDs);

            % score  
            [logscoreha] = overlapScoreZeZo(y, x, curZe, curZo, jtalgs, domainCounts, nyxze, nyxze_);
            logscoresbarha = sum(dirichlet_bayesian_score(nyxze));
            

            % score to prob
            overlapProbs(iSet,1:2, iDoN) = log2probs([logscoreha logscoresbarha]); 
            % \hat p(y|do(x), z).
            [cpha{iSet}, cphabar{iSet}, cpb{iSet}, zezoconfs{iSet}, zeconfs{iSet}] = overlap_cond_prob_hat(y, x, curZe, curZo, expDs, obsDs, domainCounts, overlapProbs(iSet,1:2, iDoN));
            
            % compute E(acc), E(ll)
            [pzezo{iSet}] = cond_prob_mult_inst(curZ, [], obsDs);
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

            nyxze = overlapCondCounts(y, x, curZe, zeconfigs, expDs);
            nyxze_ = overlapCondCounts(y, x, curZe, zeconfigs, obsDs);

            % score  
            [logscoreha] = overlapScoreZeZo(y, x, curZe, curZo, jtalgs, domainCounts, nyxze, nyxze_);
            logscoresbarha = sum(dirichlet_bayesian_score(nyxze));
            

            % score to prob
            overlapProbs(iSet,1:2, iDoN) = log2probs([logscoreha logscoresbarha]); 
            % \hat p(y|do(x), z).
            [cpha{iSet}, cphabar{iSet}, cpb{iSet}, zezoconfs{iSet}, zeconfs{iSet}] = overlap_cond_prob_hat(y, x, curZe, curZo, expDs, obsDs, domainCounts, overlapProbs(iSet,1:2, iDoN));
            
            % compute E(acc), E(ll)
            [pzezo{iSet}] = cond_prob_mult_inst(curZ, [], obsDs);
            [eacc(iSet), maxp{iSet}] = estimateExpectedAccuracy(cpb{iSet}, pzezo{iSet}, zezoconfs{iSet}); 
            [ell(iSet), lls{iSet}] = estimateExpectedLogLoss(cpb{iSet}, pzezo{iSet}, zezoconfs{iSet}); 
        end
        % overlap best sets
        [overlap_eacc, bestz] = max(eacc);bestzovacc = curSet{bestz};
        [overlap_ell, bestzll] = min(ell);bestzovll = curSet{bestzll};

        %fimb best sets
        numer = [logpDegivDoHw(:, iDoN)+logpDogivH;logpDegivDoHn(:, iDoN)+logpDogivH];
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
        timesExp(iter, 1+iDoN) =toc(ti);

        % fcitiers best sets
        [imbyfci, sameasCMB, pag]= contextFCI( subdataset(obsDs, ~inExp), expData(:, inExp));
        imbys_fci(iter, imbyfci)=true;
                    
        if sameasCMB
            [impbopobs, ~, imboconfigs] = cond_prob_mult_inst(y, [x imbyfci],[obsDs.data; expData(doInds(:, iDoN), :)], domainCounts(1:nVars));
        else
            [impbopobs, ~, imboconfigs] = cond_prob_mult_inst(y, [x imbyfci],[expData(doInds(:, iDoN), :)], domainCounts(1:nVars));
        end
        bFCI = find(cellfun(@(m)isequal(m,[x imbyfci]),curSet));

        % exp best set
        bExp =find(ismembercell(imby_true, curSet));
        

        timesFCI(iter)=toc;
        imbys_e(iter, mbye, iDoN)=true;
        imbys_fci(iter, imbyfci,iDoN) = true;
        imbys_fimb(iter, bestzfimb, iDoN)=true;
        

        test_acc(iter, 1, iDoN) = estimateAccuracy(cpb{bestz}, zezoconfs{bestz}, testData(:, [1 2 bestzovacc(2:end)])); %overlap
        test_acc(iter, 2, iDoN) = estimateAccuracy(reshape(bestprobs, 2,2,[]), zezoconfs{b}, testData(:, [1 2 bestzfimb(2:end)]));% findimb
        test_acc(iter, 3, iDoN) = estimateAccuracy(reshape(impbopobs,2,2,[]), zezoconfs{bFCI}, testData(:, [1 2 imbyfci]));% fcitiers
        test_acc(iter, 4, iDoN) = estimateAccuracy(cpha{end}, zezoconfs{end}, testData); % gt omb
        test_acc(iter, 5, iDoN) = estimateAccuracy(cphabar{bExp}, zezoconfs{bExp}, testData(:, [1 2 imby_true(2:end)])); % gt imb
      
        
        test_ll(iter, 1, iDoN) = estimateLogLoss(cpb{bestzll}, zezoconfs{bestzll}, testData(:, [1 2 bestzovll(2:end)])); %overlap
        test_ll(iter, 2, iDoN) = estimateLogLoss(reshape(bestprobs, 2,2,[]), zezoconfs{b}, testData(:, [1 2 bestzfimb(2:end)]));% findimb
        test_ll(iter, 3, iDoN) = estimateLogLoss(reshape(impbopobs,2,2,[]), zezoconfs{bFCI}, testData(:, [1 2 imbyfci]));% fcitiers
        test_ll(iter, 4, iDoN) = estimateLogLoss(cpha{end}, zezoconfs{end}, testData); % gt omb
        test_ll(iter, 5, iDoN) = estimateLogLoss(cphabar{bExp}, zezoconfs{bExp}, testData(:, [1 2 imby_true(2:end)])); % gt omb
        ll_gt(iter) = mean(logloss(1-trueProbs(:, iter), testData(:, 2)));
    end

    fprintf(' | Time elapsed %.3f, nSets %d, nSetsOr %d\n',toc(totalt), nSets, length(imby));
end

test_ll = real(test_ll);
%save([saveFolder filesep fName '.mat'])
%%
close all;
    figName = ['nVars_' num2str(nVars) '_N_' num2str(floor(N/1000)) 'K_doN' num2str(doNs(iDoN)) '_ll'] ;%
    
    figure; boxplot(test_ll-ll_gt');
    title(['$N_e$: ', num2str(2*doNs(iDoN))],'interpreter', 'latex')  
    xticklabels({'Overlap', 'FindIMB', 'FCIt-IMB', '$D_o$', '$D_e$'})
    set(gca,'TickLabelInterpreter','latex'); 
    ah = gca;
    colors = colormap(lines);
    yline(0)
    h = findobj(ah,'Tag','Box');lh = length(h);
    patch(get(h(1),'XData'),get(h(1),'YData'),colors(1,:),'FaceAlpha',.5);
    patch(get(h(2),'XData'),get(h(2),'YData'),colors(2,:),'FaceAlpha',.5);
    patch(get(h(3),'XData'),get(h(3),'YData'),colors(3,:),'FaceAlpha',.5);
    patch(get(h(4),'XData'),get(h(4),'YData'),colors(4,:),'FaceAlpha',.5);
    patch(get(h(5),'XData'),get(h(5),'YData'),colors(5,:),'FaceAlpha',.5);

    %ylabel('$|P(Y|do(X),\mathbf{V})-\hat P(Y|do(X), \mathbf{V})$', 'Interpreter', 'la
    % tex')
    ylabel('$ll-ll_{GT}$', 'Interpreter', 'latex')
    hAxes = gca;
    a = get(hAxes,'XTickLabel');  
    set(hAxes,'XTickLabel',a,'fontsize',14,'FontWeight','bold')
    set(hAxes.Title,'fontsize',18,'FontWeight','bold')
    set([hAxes.YAxis, hAxes.XAxis], ...
    'FontName'   , 'Helvetica', ...
    'FontSize'   , 16          );

    fig = gcf;

    fig.PaperPositionMode = 'auto';
    fig_pos = fig.PaperPosition;
    fig.PaperSize = [fig_pos(3) fig_pos(4)];
% 
%     [pvals(2) ]=signrank(biasOverlap, biasFGESIMB);
%     [pvals(3) ]=signrank(biasOverlap, biasFGESMB);
%     [pvals(4) ]=signrank(biasOverlap, biasFCIc);
% 
%     [pvals(1) ]=signrank(biasOverlap, biasFIMB);
% 
%     sigstar({[1 2], [1 3], [1 4], [1 5]}, pvals);

%    end
 


    saveas(gcf, ['Exp_1_Vo_' figName], 'pdf')


%%
figName = ['nVars_' num2str(nVars) '_N_' num2str(floor(N/1000)) 'K'] ;%
close all;nCols=nDoNs; nRows=1;
f=figure('Units', 'Normalized', 'Position',[0.1 .1 nCols/max(nRows,nCols)*0.9 nRows/max(nRows,nCols)*.8]);count=1; hold all;
for iDoN=1:nDoNs
    subplot(1, nDoNs,iDoN); hold all;
    ah = gca;boxplot(sens(:,2:4, iDoN)-sens(:,1, iDoN)); 
    plot(get(gca, 'xlim'), zeros(1, 2), 'r--');  
    title(['$N_e$: ', num2str(2*doNs(iDoN))],'interpreter', 'latex')  
    xticklabels({'FGES-IMB', 'FGES-MB', 'FCI-IMB'})
    set(gca,'TickLabelInterpreter','latex');

    colors = colormap(lines);

    h = findobj(ah,'Tag','Box');lh = length(h);
    patch(get(h(1),'XData'),get(h(1),'YData'),colors(1,:),'FaceAlpha',.5);
    patch(get(h(2),'XData'),get(h(2),'YData'),colors(2,:),'FaceAlpha',.5);
    patch(get(h(3),'XData'),get(h(3),'YData'),colors(3,:),'FaceAlpha',.5);
    patch(get(h(4),'XData'),get(h(4),'YData'),colors(4,:),'FaceAlpha',.5);
    patch(get(h(5),'XData'),get(h(4),'YData'),colors(5,:),'FaceAlpha',.5);

    
    hAxes = gca;
    a = get(hAxes,'XTickLabel');  
    set(hAxes,'XTickLabel',a,'fontsize',14,'FontWeight','bold')
    set(hAxes.Title,'fontsize',18,'FontWeight','bold')
    set([hAxes.YAxis, hAxes.XAxis], ...
    'FontName'   , 'Helvetica', ...
    'FontSize'   , 12          );
end

fig = gcf;

fig.PaperPositionMode = 'auto';
fig_pos = fig.PaperPosition;
fig.PaperSize = [fig_pos(3) fig_pos(4)];

saveas(gcf, [saveFolder filesep figName], 'pdf')
% fh=gcf;

