clear; clc;
comp ='striant'
preamble;

% simulate data from X->Y, X<-Z1->Z2->Y, Z3->Y: Ze should be before Zo
nVars =3;
dag = zeros(nVars);
dag(1, 2)=1; dag(3, 1) =1; dag(3, 2)=1; 
domainCounts =[2 2 2];
N=5000;doN=50;

J =500;j=0;
[test_acc, test_ll] = deal(nan(J, 3)); % acc(j, 1): overlap, acc(j, 2)
[bias] = deal(nan(J, 1));
probha = nan(J, 2);
while j<J
    if mod(j, 10)==0; fprintf('J=%d\n', j);end
    [nodes, dc] = dag2randBN2(dag, 'discrete', 'maxNumStates', domainCounts, 'minNumStates', domainCounts);
    %[nodes, dc] = dag2randBN_NO(dag, 'discrete');

    nodes{3}.cpt = [0.2;0.8];
    %[nodes, dc] = dag2randBN2(dag, 'discrete', 'domainCounts', domainCounts);

    jtalg_do = tetradJtalgDo(dag,nodes,domainCounts, 1);
    [tIMdo] = tetradEIM(dag, nodes, domainCounts);
    jtalg =javaObject('edu.pitt.dbmi.custom.tetrad.lib.bayes.JunctionTree', tIMdo); % selection posterior 


    pydox = estimateCondProbJT(2, [1], jtalg_do, nVars, domainCounts);
    pyx = estimateCondProbJT(2, [1], jtalg, nVars, domainCounts);
    b = sum(abs(pydox(1, :)- pyx(1, :)));
    
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
 
    y=2; x=1; ze=[]; zo=[];

    [~, rnodespost, orderpost] = dag2BNPost(dag, DoDs.data, domainCounts); % Do posterior

    
    I=50;
    for iter=1:I
        samplepost = sampleBNpost(rnodespost, orderpost, domainCounts);
        [eIM(iter)] = tetradEIM(dag, samplepost, domainCounts);
        jtalgs(iter)  = javaObject('edu.pitt.dbmi.custom.tetrad.lib.bayes.JunctionTree', eIM(iter));
    end


    [cptrue,zeconfigs] = overlapPyxz(y,x, [], jtalg_do, domainCounts);
    [cpobs] = cond_prob_mult_inst(y, [x], [DoDs.data], domainCounts);
    cpobs = reshape(cpobs, domainCounts(y), domainCounts(x), []);
    
    nyxze = overlapCondCounts(y, x, [], zeconfigs,DeDs);
    nyxze_ = overlapCondCounts(y, x, [], zeconfigs,DoDs);

    % score set 
    [logscoreha] = overlapScoreZeZo(y, x, [], [], jtalgs, domainCounts, nyxze, nyxze_);
    logscoresbarha = sum(dirichlet_bayesian_score(nyxze));
    probsha(j,:) = log2probs([logscoreha logscoresbarha]); 

          
    [cpha, cphabar, cpb, zezoconfs, zeconfs] = overlap_cond_prob_hat(y, x, [], [], DeDs, DoDs, domainCounts, probsha(j, :));
    [eacc, maxp] = estimateExpectedAccuracy(cpb, 0, []); 
            %[acc(iZ)] = estimateAccuracy(cpb{iZ}, zezoconfs{iZ}, DeDsTest.data(:, [1 2 find(curZ)])); 
    [ell, lls] = estimateExpectedLogLoss(cpb, 0, zezoconfs); 
 
% best Z based on eacc;
[overlap_ell(j), bestzll(j)] = min(ell);


test_ll(j, 1) = estimateLogLoss(cpb, zezoconfs, DeDsTest.data(:, [1 2]));
test_ll(j, 2) = estimateLogLoss(cpobs, zezoconfs, DeDsTest.data(:, [1 2]));% observational data on all;
test_ll(j, 3) = estimateLogLoss(cphabar, zezoconfs, DeDsTest.data(:, [1 2]));% experimental data  data on all;
test_ll_gt(j) = estimateLogLoss(cptrue, zezoconfs, DeDsTest.data(:, [1 2]));% experimental data  data on all;

end

%%
colors = colormap(lines);

numBins = 5;
llov = real(test_ll(:, 1)-test_ll_gt');
llo = real(test_ll(:, 2)-test_ll_gt');
lle = real(test_ll(:, 3)-test_ll_gt');

[mean_bin_ov, std_bin_ov, bin_size_ov, bin_centers_ov, q_bin_ov] = binvalues(llov, bias, numBins);

[mean_bin_o, std_bin_o, bin_size_o, bin_centers_o, q_bin_o] = binvalues(llo, bias, numBins);
[mean_bin_e, std_bin_e, bin_size_e, bin_centers_e, q_bin_e] = binvalues(lle, bias, numBins);
%[mean_bin_f, std_bin_f, bin_size_f, bin_centers_f, q_bin_f] = binvalues(llfimb, bias, numBins);

close all;
figure;hold all;
%scatter(bin_centers_ov, mean_bin_ov, (bin_size_ov+1)*3, 'filled', 'MarkerFaceAlpha',.8);
p(1) =plot(bin_centers_ov, mean_bin_ov, 'LineWidth', 2, 'color', colors(1, :), 'Marker', 'o');
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


p(4) =plot(bin_centers_ov, mean_bin_ov, 'LineWidth', 2, 'color', colors(4, :), 'Marker', '*');
%shadedErrorBar(bin_centers_ov, mean_bin_ov, std_bin_ov, 'LineProps', {'LineWidth', 2});
plot(bin_centers_ov, q_bin_ov, ':', 'LineWidth', 2, 'Color', p(1).Color);
hold all;


xlabel('mean absolute bias $|P(Y|do(X))-P(Y|X)|$', 'Interpreter', 'latex','FontSize', 20)
ylabel('$ll-ll_{GT}$', 'Interpreter', 'Latex','FontSize',20)
legend(p, {'$Overlap$', '$D_o$', '$D_e$', '$FindIMB$'}, 'Interpreter', 'Latex','FontSize', 18, 'Location', 'NorthWest')
cxlim = get(gca, 'xlim');
set(gca, 'XLim', [0.05,bin_centers_o(end)]);
line(get(gca, 'XLim'), [0 0], 'color','k', 'HandleVisibility','off')
fig = gcf;

fig.PaperPositionMode = 'auto';
fig_pos = fig.PaperPosition;
fig.PaperSize = [fig_pos(3) fig_pos(4)];
saveas(gcf, 'hidden_confounder_sim', 'pdf');
%%
[mean_bin_pha,  std_bin_pha, bin_size_pha, bin_centers_pha,  q_bin_pha] = binvalues(probsha(:, 1), bias, 10);

figure;shadedErrorBar(bin_centers_pha, mean_bin_pha, std_bin_pha, 'lineprops',{'r-','linewidth', 2});% 'filled', 'MarkerFaceAlpha',.8);

xlabel('average bias $P(Y|do(X))-P(Y|X)$', 'Interpreter', 'latex')
ylabel('$P(\mathcal H_\mathbf Z ^a|D_e,D_o)$', 'Interpreter', 'Latex', 'Fontsize', 18)

saveas(gcf, 'single_confounder_prob', 'pdf');

