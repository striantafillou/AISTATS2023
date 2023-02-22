if ismac
    javaaddpath(['/Users/' comp '/Dropbox/MATLAB/causal_graphs/tetrad/custom-tetrad-lib-0.2.0.jar'])
    code_path = ['/Users/' comp '/Dropbox/MATLAB/causal_graphs'];
    addpath(genpath(code_path));
    code_path = ['/Users/' comp '/Dropbox/MATLAB/causal_effects'];
    addpath(genpath(code_path));
    code_path = ['/Users/' comp '/Dropbox/MATLAB/UAI2021'];
    addpath(genpath(code_path));
elseif ispc
    javaaddpath(['C:\Users\' comp '\Dropbox\MATLAB\causal_graphs\tetrad\custom-tetrad-lib-0.2.0.jar'])

    code_path = ['C:\Users\' comp '\Dropbox\MATLAB\causal_graphs'];
    addpath(genpath(code_path));
    code_path = ['C:\Users\' comp '\Dropbox\MATLAB\causal_effects'];
    addpath(genpath(code_path));
    code_path = ['C:\Users\' comp '\Dropbox\MATLAB\UAI2021'];
    addpath(genpath(code_path));
end