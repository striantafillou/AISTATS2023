clear; clc;
%comp ='sot16.PITT'; 
comp ='striant';
preamble
% simulate data from X->Y, X<-Z1->Z2->Y, Z3->Y: Ze should be before Zo
nVars =3;
dag = zeros(nVars);
dag(1, 2)=1; dag(3, [1 2])=1; 

domainCounts =[2 2 2];
N=5000;doN=100;
j=0;nIts =20;
alphas = [0.2:0.05:0.8];
J =nIts*length(alphas);
test_acc = nan(J, 3); % acc(j, 1): overlap, acc(j, 2)
t =tic;
for a=alphas
    for iter=1:nIts

        j=j+1;

        [nodes, dc] = dag2randBN2(dag, 'discrete', 'maxNumStates', domainCounts, 'minNumStates', domainCounts);
    
        nodes{3}.cpt = [0.2;0.8];
        doNodes = nodes;
        doNodes{3}.cpt = [a;1-a];
        bias(j) =   abs(nodes{3}.cpt(1)-doNodes{3}.cpt(1));


        jtalg_do = tetradJtalgDo(dag, doNodes, domainCounts, 1);

        [tIMdo] = tetradEIM(dag, nodes, domainCounts);
        jtalg =javaObject('edu.pitt.dbmi.custom.tetrad.lib.bayes.JunctionTree', tIMdo); % selection posterior 
    
        [pydoxz, ~, xzconfs] = estimateCondProbJT(2, [1,3], jtalg_do, nVars, domainCounts);
        pydox = estimateCondProbJT(2, 1, jtalg_do, nVars, domainCounts);
        pz = estimateCondProbJT(3, [], jtalg, nVars, domainCounts);
        %estimateDoProbJTAdj(2, 1, 3, jtalg, nVars, domainCounts)
            
        DoDs = simulatedata(nodes,N, 'discrete', 'domainCounts', domainCounts); % simulate data from original distribution
        DeDs = simulateDoData(doNodes, 1,  0:domainCounts(1)-1, doN, 'discrete', 'domainCounts', domainCounts);
        DeDsTest = simulateDoData(doNodes, 1,  0:domainCounts(1)-1, 500, 'discrete', 'domainCounts', domainCounts);
    
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
                [ell(iZ), lls{iZ}] = estimateExpectedLogLoss(cpb{iZ}, pzezo{iZ}, zezoconfs{iZ}); 
    
                
            end
        end
        % best Z based on eacc;
        [overlap_eacc(j), bestz(j)] = max(eacc);
        [overlap_ell(j), bestzll(j)] = min(ell);
        
        test_acc(j, 1) = estimateAccuracy(cpb{bestz(j)}, zezoconfs{bestz(j)}, DeDsTest.data(:, [1 2 find(zs(bestz(j), :))]));
        test_acc(j, 2) = estimateAccuracy(cpha{2}, zezoconfs{2}, DeDsTest.data(:, [1 2 3]));% observational data on all;
        test_acc(j, 3) = estimateAccuracy(cphabar{1}, zezoconfs{1}, DeDsTest.data(:, [1 2]));% experimental data  data on all;
        test_acc(j, 4) = estimateAccuracy(cpb{bestz(1)}, zezoconfs{1}, DeDsTest.data(:, [1 2])); %fimb
        
        test_ll(j, 1) = estimateLogLoss(cpb{bestzll(j)}, zezoconfs{bestzll(j)}, DeDsTest.data(:, [1 2 find(zs(bestzll(j), :))]));
        test_ll(j, 2) = estimateLogLoss(cpha{2}, zezoconfs{2}, DeDsTest.data(:, [1 2 3]));% observational data on all;
        test_ll(j, 3) = estimateLogLoss(cphabar{1}, zezoconfs{1}, DeDsTest.data(:, [1 2]));% experimental data  data on all;
        test_ll(j, 4) = estimateLogLoss(cpb{1}, zezoconfs{1}, DeDsTest.data(:, [1 2]));% fimb
        
        test_ll_gt(j) =estimateLogLoss(cpbtrue{2}, zezoconfs{2}, DeDsTest.data(:, [1 2 3]));
    end
end
%%
colors = colormap(lines);

numBins = length(alphas);
llov = real(test_ll(:, 1)-test_ll_gt');
llo = real(test_ll(:, 2)-test_ll_gt');
lle = real(test_ll(:, 3)-test_ll_gt);

[mean_bin_ov, std_bin_ov, bin_size_ov, bin_centers_ov, q_bin_ov] = binvalues(llov, bias, numBins);

[mean_bin_o, std_bin_o, bin_size_o, bin_centers_o, q_bin_o] = binvalues(llo, bias, numBins);
[mean_bin_e, std_bin_e, bin_size_e, bin_centers_e, q_bin_e] = binvalues(lle, bias, numBins);

close all;
fig =figure;hold all;box on
p(1) =plot(alphas, mean_bin_ov, 'LineWidth', 2, 'color', colors(1, :), 'Marker', 'o', 'MarkerSize', 15, 'MarkerFaceColor',colors(1, :));
shadedBarQ(alphas, q_bin_ov, colors(1, :), 0.1);
hold all;

%scatter(bin_centers_o, mean_bin_o, (bin_size_o+1)*3, 'filled', 'MarkerFaceAlpha',.8);
p(2) =plot(alphas, mean_bin_o, 'LineWidth', 2, 'color', colors(2, :), 'Marker', 's', 'MarkerSize', 20);
shadedBarQ(alphas, q_bin_o, colors(2, :), 0.1);


hold on;
%scatter(bin_centers_e, mean_bin_e, (bin_size_e+1)*3, 'filled', 'MarkerFaceAlpha',.8);
%shadedErrorBar(bin_centers_e, mean_bin_e, std_bin_e, 'LineProps', {'LineWidth', 2});
p(3) =plot(alphas, mean_bin_e, 'LineWidth', 2, 'color', colors(3, :), 'Marker', 'x', 'MarkerSize', 20);
shadedBarQ(alphas, q_bin_e, colors(3, :), 0.1);


xlim([alphas(1),alphas(end)])
set(gca,'XTickLabel',get(gca,'XTickLabel'),'fontsize',16)

xlabel('$\alpha$', 'Interpreter', 'latex', 'FontSize', 18)
ylabel('$ll-ll_{GT}$', 'Interpreter', 'Latex','FontSize', 20)
legend(p,{'$Overlap$', 'OMB', 'IMB', "FindIMB"}, 'Interpreter', 'Latex','FontSize', 18, 'Location', 'NorthWest')
yline(0, 'HandleVisibility','off')

fig.PaperPositionMode = 'auto';
fig_pos = fig.PaperPosition;
fig.PaperSize = [fig_pos(3) fig_pos(4)];
saveas(gcf,'selection_bias.pdf')