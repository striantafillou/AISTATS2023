clear; clc;
%comp ='sot16.PITT'; 
%comp ='sofia'
comp = 'striant';
preamble;
% simulate data from X->Y, X<-Z1->Z2->Y, Z3->Y: Ze should be before Zo
nVars =3;
dag = zeros(nVars);
dag(1, 2)=1; dag(3, 2)=1; 

domainCounts =[2 2 2];
N=5000;doN=100;
j=0;
J =500;
test_acc = nan(J, 3); % acc(j, 1): overlap, acc(j, 2)

while j<J
    [nodes, dc] = dag2randBN2(dag, 'discrete', 'maxNumStates', domainCounts, 'minNumStates', domainCounts);

    nodes{3}.cpt = [0.2;0.8];

    jtalg_do = tetradJtalgDo(dag,nodes,domainCounts, 1);
    [tIMdo] = tetradEIM(dag, nodes, domainCounts);
    jtalg =javaObject('edu.pitt.dbmi.custom.tetrad.lib.bayes.JunctionTree', tIMdo); % selection posterior 

    [pydoxz, ~, xzconfs] = estimateCondProbJT(2, [1,3], jtalg_do, nVars, domainCounts);
    pydox = estimateCondProbJT(2, 1, jtalg_do, nVars, domainCounts);
    pz = estimateCondProbJT(3, [], jtalg, nVars, domainCounts);
    
        
    b =sum(abs(pydox(1, :) - pydoxz(1, 1:2)).*pz(1));
    if b>0.05;
        j=j+1;
        bias(j)=b;
    else
        continue;
    end
    %simulate observational, experimental data
    DoDs = simulatedata(nodes,N, 'discrete', 'domainCounts', domainCounts); % simulate data from original distribution
    DeDs = simulateDoData(nodes, 1,  0:domainCounts(1)-1, doN, 'discrete', 'domainCounts', domainCounts);
    DeDsTest = simulateDoData(nodes, 1,  0:domainCounts(1)-1, 500, 'discrete', 'domainCounts', domainCounts);

    jtalg_do = tetradJtalgDo(dag,nodes,domainCounts, 1);
    cdjtalg =javaObject('edu.pitt.dbmi.custom.tetrad.lib.bayes.JunctionTree', tetradEIM(dag, nodes, domainCounts)); 
    y=2; x=1; ze=[]; zo=3;


    [~, rnodespost, orderpost] = dag2BNPost(dag, DoDs.data, domainCounts); % Do posterior


    allze = allSubsets(nVars, ze);nZe = size(allze, 1);
    allzo = allSubsets(nVars, zo);nZo = size(allzo, 1);


    nZ = nZe*nZo;
    allz = allSubsets(nZ, nVars);
    probsz = zeros(nZ, 2);
    iZ =0;

    I=50;
    for iter=1:I
        samplepost = sampleBNpost(rnodespost, orderpost, domainCounts);
        [eIM(iter)] = tetradEIM(dag, samplepost, domainCounts);
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
[overlap_eacc(j), bestz(j)] = max(eacc);
[overlap_ell(j), bestzll(j)] = min(ell);

test_acc(j, 1) = estimateAccuracy(cpb{bestz(j)}, zezoconfs{bestz(j)}, DeDsTest.data(:, [1 2 find(zs(bestz(j), :))]));
test_acc(j, 2) = estimateAccuracy(cpha{2}, zezoconfs{2}, DeDsTest.data(:, [1 2 3]));% observational data on all;
test_acc(j, 3) = estimateAccuracy(cphabar{1}, zezoconfs{1}, DeDsTest.data(:, [1 2]));% experimental data  data on all;

test_ll(j, 1) = estimateLogLoss(cpb{bestzll(j)}, zezoconfs{bestzll(j)}, DeDsTest.data(:, [1 2 find(zs(bestzll(j), :))]));
test_ll(j, 2) = estimateLogLoss(cpha{2}, zezoconfs{2}, DeDsTest.data(:, [1 2 3]));% observational data on all;
test_ll(j, 3) = estimateLogLoss(cphabar{1}, zezoconfs{1}, DeDsTest.data(:, [1 2]));% experimental data  data on all;

test_ll_gt(j) =estimateLogLoss(cpbtrue{2}, zezoconfs{2}, DeDsTest.data(:, [1 2 3]));
end

%%
numBins = 10;
llov = real(test_ll(:, 1)-test_ll_gt');
llo = real(test_ll(:, 2)-test_ll_gt');
lle = real(test_ll(:, 3)-test_ll_gt');
[mean_bin_ov, std_bin_ov, bin_size_ov, bin_centers_ov] = binvalues(llov, bias, 10);

[mean_bin_o, std_bin_o, bin_size_o, bin_centers_o] = binvalues(llo, bias, 10);
[mean_bin_e, std_bin_e, bin_size_e, bin_centers_e] = binvalues(lle, bias, 10);
% close all;
% figure;
% scatter(bin_centers_ov, mean_bin_ov, (bin_size_ov+1)*3, 'filled', 'MarkerFaceAlpha',.8);
% 
% hold all;
% scatter(bin_centers_o, mean_bin_o, (bin_size_o+1)*3, 'filled', 'MarkerFaceAlpha',.8);
% hold on;
% scatter(bin_centers_e, mean_bin_e, (bin_size_e+1)*3, 'filled', 'MarkerFaceAlpha',.8);

close all;
figure;
plot(bin_centers_ov, mean_bin_ov, 'LineWidth', 2);

hold all;
plot(bin_centers_o, mean_bin_o, 'LineWidth', 2);
hold on;
plot(bin_centers_e, mean_bin_e, 'LineWidth', 2);


xlabel('average absolute difference $|P(Y|do(X),Z)-P(Y|do(X))|$', 'Interpreter', 'latex')
ylabel('Difference from ground truth (log loss)', 'Interpreter', 'Latex')
legend({'$Overlap$', '$D_o$', '$D_e$'}, 'Interpreter', 'Latex','FontSize', 18, 'Location', 'NorthWest')

set(gca, 'XLim', [0.05,bin_centers_o(end)]);
line(get(gca, 'XLim'), [0 0], 'color','k', 'HandleVisibility','off')
fig = gcf;

fig.PaperPositionMode = 'auto';
fig_pos = fig.PaperPosition;
fig.PaperSize = [fig_pos(3) fig_pos(4)];

saveas(gcf, 'modifier_est2', 'pdf');

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


numBins = 10;
accov = real(test_acc(:, 1));
acco = real(test_acc(:, 2));
acce = real(test_acc(:, 3));
[mean_bin_ov, std_bin_ov, bin_size_ov, bin_centers_ov] = binvalues(accov, bias, 10);

[mean_bin_o, std_bin_o, bin_size_o, bin_centers_o] = binvalues(acco, bias, 10);
[mean_bin_e, std_bin_e, bin_size_e, bin_centers_e] = binvalues(acce, bias, 10);
% close aacc;
% figure;
% scatter(bin_centers_ov, mean_bin_ov, (bin_size_ov+1)*3, 'fiacced', 'MarkerFaceAlpha',.8);
% 
% hold aacc;
% scatter(bin_centers_o, mean_bin_o, (bin_size_o+1)*3, 'fiacced', 'MarkerFaceAlpha',.8);
% hold on;
% scatter(bin_centers_e, mean_bin_e, (bin_size_e+1)*3, 'fiacced', 'MarkerFaceAlpha',.8);

close all;
figure;

hold all ;
plot(bin_centers_o, mean_bin_o, 'LineWidth', 2);
hold on;
plot(bin_centers_e, mean_bin_e, 'LineWidth', 2);
hold all;
plot(bin_centers_ov, mean_bin_ov, 'LineWidth', 2);


xlabel('average absolute difference $|P(Y|do(X),Z)-P(Y|do(X))|$', 'Interpreter', 'latex')
ylabel('Difference from ground truth (log loss)', 'Interpreter', 'Latex')
legend({'$Overlap$', '$D_o$', '$D_e$'}, 'Interpreter', 'Latex','FontSize', 18, 'Location', 'NorthWest')

set(gca, 'XLim', [0.05,bin_centers_o(end)]);
line(get(gca, 'XLim'), [0 0], 'color','k', 'HandleVisibility','off')
fig = gcf;

fig.PaperPositionMode = 'auto';
fig_pos = fig.PaperPosition;
fig.PaperSize = [fig_pos(3) fig_pos(4)];

saveas(gcf, 'modifier_est2_acc', 'pdf');