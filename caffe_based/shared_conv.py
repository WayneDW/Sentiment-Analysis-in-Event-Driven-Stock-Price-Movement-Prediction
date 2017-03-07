import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import caffe


num_distributions = str(sys.argv[1])
sample_dimension = str(sys.argv[2])





os.system('python reProtocol_pars.py ' + num_distributions + ' ' + sample_dimension + ' joint')
caffe.set_mode_cpu() 

SOLVER = './protocol/dis_' + num_distributions + '_' + sample_dimension + '_dimension.solver_joint_'
PROTO = './protocol/dis_' + num_distributions + '_' + sample_dimension + '_dimension.protocol_'
MODEL = '/scratch/radon/d/deng106/CNNStatisticalModel/distributions/models/joint/'

max_steps = 200000
steps = 1
layers = ['conv1']

solver_par = caffe.get_solver(SOLVER + 'parameter') 
solver_dis = caffe.get_solver(SOLVER + 'distribution') 

if len(sys.argv) == 4: # if to continue from an existing model
    model_number = str(sys.argv[3])
    NUM = num_distributions + '_dis_' + sample_dimension + '_dim/_iter_' + model_number + '.caffemodel'
    model_par = caffe.Net(PROTO + 'parameter', MODEL + 'parameter/' + NUM, caffe.TRAIN)
    model_dis = caffe.Net(PROTO + 'distribution', MODEL + 'distribution/' + NUM, caffe.TRAIN)
    for name in model_par.params.keys():
        solver_par.net.params[name][0].data[...] = model_par.params[name][0].data[...]
        solver_par.net.params[name][1].data[...] = model_par.params[name][1].data[...]
    for name in model_dis.params.keys():
        solver_dis.net.params[name][0].data[...] = model_dis.params[name][0].data[...]
        solver_dis.net.params[name][1].data[...] = model_dis.params[name][1].data[...]
        

def has_nan(solver, layers):
    cnt = 0
    for name in layers:
        cnt += sum(np.isnan(solver_par.net.params[name][0].data[...].flatten()))
        cnt += sum(np.isnan(solver_par.net.params[name][1].data[...].flatten()))
    return cnt != 0

par_bak = {}
dis_bak = {}
for i in range(max_steps / steps):
    solver_par.step(steps)
    if not has_nan(solver_par, layers):
        for name in layers:
            dis_bak[name] = [solver_dis.net.params[name][0].data[...].copy(), solver_dis.net.params[name][1].data[...].copy()] # bak none-nan data
            solver_dis.net.params[name][0].data[...] = solver_par.net.params[name][0].data[...]
            solver_dis.net.params[name][1].data[...] = solver_par.net.params[name][1].data[...]
    else:
        print("Nan appears, go back to the previous step")
        for name in layers:
            solver_par.net.params[name][0].data[...] = par_bak[name][0]
            solver_par.net.params[name][1].data[...] = par_bak[name][1]
            

    solver_dis.step(steps)
    if not has_nan(solver_dis, layers):
        for name in layers:
            par_bak[name] = [solver_par.net.params[name][0].data[...].copy(), solver_par.net.params[name][1].data[...].copy()] # use copy to copy values
            solver_par.net.params[name][0].data[...] = solver_dis.net.params[name][0].data[...]
            solver_par.net.params[name][1].data[...] = solver_dis.net.params[name][1].data[...]
    else:
        print("Nan appears, go back to the previous step")
        for name in layers:
            solver_dis.net.params[name][0].data[...] = dis_bak[name][0]
            solver_dis.net.params[name][1].data[...] = dis_bak[name][1]



