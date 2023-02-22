clear; clc;
comp ='striant'; 
%comp ='sofia'
%comp ='sot16.PITT';
if isequal(comp, 'striant')
    javaaddpath(['/Users/' comp '/Dropbox/MATLAB/causal_graphs/tetrad/custom-tetrad-lib-0.2.0.jar'])
    code_path = ['/Users/' comp '/Dropbox/MATLAB/causal_graphs'];
    addpath(genpath(code_path));
    code_path = ['/Users/' comp '/Dropbox/MATLAB/causal_effects'];
    addpath(genpath(code_path));
    code_path = ['/Users/' comp '/Dropbox/MATLAB/UAI2021'];
    addpath(genpath(code_path));
else
    javaaddpath(['C:\Users\' comp '\Dropbox\MATLAB\causal_graphs\tetrad\custom-tetrad-lib-0.2.0.jar'])

    code_path = ['C:\Users\' comp '\Dropbox\MATLAB\causal_graphs'];
    addpath(genpath(code_path));
    code_path = ['C:\Users\' comp '\Dropbox\MATLAB\causal_effects'];
    addpath(genpath(code_path));
    code_path = ['C:\Users\' comp '\Dropbox\MATLAB\UAI2021'];
    addpath(genpath(code_path));
end
import edu.cmu.tetrad.*
import java.util.*
import java.lang.*
import edu.cmu.tetrad.data.*
import edu.cmu.tetrad.search.*
import edu.cmu.tetrad.graph.*
import edu.cmu.tetrad.bayes.*

% simulate data from X->Y, X<-Z1->Z2->Y, Z3->Y
nVarsOr =5;
dag = zeros(nVarsOr);
dag(1, 2)=1; dag(3, 1) =1; dag(3, 4)=1; dag(4, 2)=1;dag(5, 2)=1;

isLatent = false(1, nVarsOr);nVars= sum(~isLatent);
inDe = false(1, nVars); inDe([1 2 3 4 5]) = true;

domainCounts =[2 2 2 2 2];
N=10000;doN=1000;
[nodes, dc] = dag2randBN2(dag, 'discrete', 'maxNumStates', domainCounts, 'minNumStates', domainCounts);

%simulate observational, experimental data
DoDs = simulatedata(nodes,N, 'discrete', 'domainCounts', domainCounts, 'isLatent', isLatent); % simulate data from original distribution
% observational data, learn dag, and BN

% simulate experimental data 
DeDs = simulateDoData(nodes, 1,  0:domainCounts(1)-1, doN, 'discrete', 'domainCounts', domainCounts);

jtalg_do = tetradJtalgDo(dag,nodes,domainCounts, 1);
jtalg =javaObject('edu.pitt.dbmi.custom.tetrad.lib.bayes.JunctionTree', tetradEIM(dag, nodes, domainCounts)); 
ze =[3 4];zo=[5];y=2; x=1;
[pyxzezo,zeconfs, zoconfs] = overlapPyxzezo(2, 1, ze, zo, jtalg, domainCounts);
[pzoze,zoconfigs, zeconfsigs] = overlapPzoze(zo, ze, jtalg, domainCounts);
pydoxzehat = overlapAdjustment(pyxzezo, pzoze);
[pydoxze,zeconfigs] = overlapPyxz(2,1, ze, jtalg_do, domainCounts);
nnz(abs(pydoxze - pydoxzehat)>eps)
%%
list= tetradList(nVars, domainCounts);

dsj = javaObject('edu.cmu.tetrad.data.BoxDataSet',javaObject('edu.cmu.tetrad.data.VerticalIntDataBox',DoDs.data'), list);
aGraph=dag2tetrad(dag,list, nVars); % graph
tBM= javaObject('edu.cmu.tetrad.bayes.BayesPm', aGraph);
eIMprior = edu.cmu.tetrad.bayes.DirichletBayesIm.symmetricDirichletIm(tBM, 1);
eIMpost= edu.cmu.tetrad.bayes.DirichletEstimator.estimate(eIMprior, dsj);% 
I=100;
dirSamplers(I)=  edu.pitt.dbmi.custom.tetrad.lib.bayes.DirichletSampler.sampleFromPosterior(eIMpost); % sample from posterior
jtalgs(I)=javaObject('edu.cmu.tetrad.bayes.JunctionTreeAlgorithm', dirSamplers(I)); % junction tree
for iter=1:I-1
    dirSamplers(iter)=  edu.pitt.dbmi.custom.tetrad.lib.bayes.DirichletSampler.sampleFromPosterior(eIMpost); 
    jtalgs(iter)=javaObject('edu.cmu.tetrad.bayes.JunctionTreeAlgorithm', dirSamplers(iter));
end
[logscoresha,logscoreshbara, thetas] = overlapScoreZeZo(y, x, ze, zo,jtalgs, domainCounts, DeDs);

