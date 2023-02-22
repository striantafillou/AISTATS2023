clear; clc;
comp ='sot16.PITT'; 
%comp ='sofia'
javaaddpath(['C:\Users\' comp '\Dropbox\MATLAB\causal_graphs\tetrad\custom-tetrad-lib-0.2.0.jar'])
import edu.cmu.tetrad.*
import java.util.*
import java.lang.*
import edu.cmu.tetrad.data.*
import edu.cmu.tetrad.search.*
import edu.cmu.tetrad.graph.*
import edu.cmu.tetrad.bayes.*

code_path = ['C:\Users\' comp '\Dropbox\MATLAB\causal_graphs'];
addpath(genpath(code_path));
code_path = ['C:\Users\' comp '\Dropbox\MATLAB\causal_effects'];
cd([code_path ]);
code_path = ['C:\Users\' comp '\Dropbox\MATLAB\ICML2022'];
addpath(genpath(code_path));


% simulate data from X->Y, X<-Z1->Z2->Y, Z3->Y: Ze should be before Zo
nVars =4;
dag = zeros(nVars);
dag(1, 2)=1; dag(3, 1) =1; dag(3, 2)=1; dag(4, 2)=1;

domainCounts =[2 2 2 2];
N=10000;doN=100;
[nodes, dc] = dag2randBN2(dag, 'discrete', 'maxNumStates', domainCounts, 'minNumStates', domainCounts);

%simulate observational, experimental data
DoDs = simulatedata(nodes,N, 'discrete', 'domainCounts', domainCounts); % simulate data from original distribution
DeDs = simulateDoData(nodes, 1,  0:domainCounts(1)-1, doN, 'discrete', 'domainCounts', domainCounts);

jtalg_do = tetradJtalgDo(dag,nodes,domainCounts, 1);
jtalg =javaObject('edu.pitt.dbmi.custom.tetrad.lib.bayes.JunctionTree', tetradEIM(dag, nodes, domainCounts)); 
y=2; x=1; ze = 3; zo=4;


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
    end
end

eacc = 
% estimate expected accuracy



