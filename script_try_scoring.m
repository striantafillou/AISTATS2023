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
addpath(genpath(code_path));
cd([code_path ]);
code_path = ['C:\Users\' comp '\Dropbox\MATLAB\UAI2021'];
addpath(genpath(code_path));


% simulate data from X->Y, X<-Z1->Z2->Y, Z3->Y
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
ze =[3];zo=[4];y=2; x=1;
[pyxzezo,zeconfs, zoconfs] = overlapPyxzezo(y, x, ze, zo, jtalg, domainCounts);
[pzoze,zoconfigs, zeconfsigs] = overlapPzoze(zo, ze, jtalg, domainCounts);
pydoxzehat = overlapAdjustment(pyxzezo, pzoze);
[pydoxze,zeconfigs] = overlapPyxz(y,x, ze, jtalg_do, domainCounts);
nnz(abs(pydoxze - pydoxzehat)>eps)
%
%list= tetradList(nVars, domainCounts);
nyxze = overlapCondCounts(y, x, ze, zeconfigs,DeDs);
nyxze_ = overlapCondCounts(y, x, ze, zeconfigs,DoDs);

[~, rnodespost, orderpost] = dag2BNPost(dag, DoDs.data, domainCounts); % Do posterior
%
I=50;
for iter=1:I
samplepost =sampleBNpost(rnodespost, orderpost, domainCounts);
[eIM(iter)] = tetradEIM(dag, samplepost, domainCounts);
jtalgs(iter)  = javaObject('edu.pitt.dbmi.custom.tetrad.lib.bayes.JunctionTree', eIM(iter));
end
[logscoreha, ~, thetas] = overlapScoreZeZo(y, x, ze, zo,jtalgs, domainCounts, nyxze, nyxze_);
logscoresbarha = sum(dirichlet_bayesian_score(nyxze));
log2probs([logscoreha logscoresbarha])


