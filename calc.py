""" test cpp embedding:
# pass variable from cpp to python and return
# uses pybind11 with MPI
"""
import os, sys, time, logging, numpy as np, scipy as sp
from mpi4py import MPI

import torch
from CDM_network import mini_U_Net

"""
allocate basic helpers defined in cpp
"""
def get_topology(crank,cd_glob,cn_glob,cm_glob,cl_glob):
    global rank, d_glob, n_glob, m_glob, l_glob
    rank = crank # cpp rank
    d_glob = cd_glob # cpp global dimensions
    n_glob = cn_glob # cpp topology information on n-axis
    m_glob = cm_glob # cpp topology information on m-axis
    l_glob = cl_glob # cpp topology information on l-axis

"""
fast interpolate input to any desired dimensions
ussing scipy's map_coordinates function
https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html
"""
def interpolate(field,prank,fac=1):
    # start a timer
    st = time.perf_counter()

    # map to any arbitrary dimensions
    ni,nj,nk=field.shape[:] # exemplray to a factor
    desired_x = ni*fac
    desired_y = nj*fac
    desired_z = nk*fac

    # create 3D list of coordonates
    des_dims = []
    for inp_l, out_l in zip(field.shape, (desired_x,desired_y,desired_z)):
        des_dims.append(np.linspace(0, inp_l-1, out_l))

    # create meshgrid from list
    coords = np.meshgrid(*des_dims, indexing='ij')

    # return mapped data to new coordinates, order=0 ensures that the process is fast (only copy, no interpolate)
    res = sp.ndimage.map_coordinates(field, coords, order=0, mode='nearest', cval=0, prefilter=False)

    # end timer
    if prank==0:
        logging.info('interpolate time: {:.2f}'.format(time.perf_counter()-st)+' s')

    return res

"""
split the data in MPI rank(s) to #GPU(s)
e.g., 5 MPI ranks (prank) in 2 GPUs (grank) will then
1. grank will've data from 0:2 pranks
2. grank will've data from 2:5 pranks
last grank will've leftover prank
note: psize is the MPI world size
"""
def split_to_gpu(field,grank,psize,gsize):
    # indice to start depending on grank
    n1 = grank * psize // gsize
    # indice to end depending on grank
    n2 = n1 + psize // gsize
    # leftover prank to last grank
    if grank==gsize-1:
        n2 += psize % gsize
    return field[n1:n2,:,:,:]

"""
restore model from a saved file
"""
def model_restore(model,grank):
    res_name='checkpoint.pth.tar'
    loc = {'cuda:%d' % 0: 'cuda:%d' % grank}
    checkpoint = torch.load('./'+res_name, map_location=loc)
    # only state_dict is needed
    model.load_state_dict(checkpoint['state_dict'])
    # fix parameters for evaluation
    model.eval()
    return model

# Get rank and size from mpi4py
def torch_couple(field):
    # start timer
    st = time.perf_counter()

    # debug
    logging.basicConfig(format='%(levelname)s: %(message)s', stream=sys.stdout, level=logging.INFO)

    # define MPI
    comm = MPI.COMM_WORLD
    prank = comm.Get_rank() # CPU rank
    psize = comm.Get_size() # CPU world size
    grank = prank%torch.cuda.device_count() # GPU rank
    gsize = torch.cuda.device_count() # GPU world size
    assert rank == prank # check MPI ranks with cpp
    if prank==0:
        print(f'Torch:')
        logging.info('CPU ranks:'+str(psize)+''+' / GPU ranks:'+str(gsize))

    # move tuple to torch array
    field = torch.asarray(field)

    # interpolate
    field_i = interpolate(field,prank,fac=2)

    # gather from all ranks
    field_global = comm.allgather(field_i) # leads to dim:ranks,n,m,l
    field_global = torch.asarray(np.array(field_global))

    # selective GPU distribution
    device='cuda:'+str(grank)

    # split data to GPU(s)
    inputs = split_to_gpu(field_global,grank,psize,gsize).to(device)

    # init model
    model = mini_U_Net().to(device)

    # restore model
    model = model_restore(model,grank)

    # apply input to model to get desired output
    with torch.no_grad():
        inputs = inputs.permute(0,3,1,2).reshape(inputs.size()[0]*inputs.size()[-1],1,*inputs.size()[1:3]).float()
        outputs = model(inputs).float()

    # compute maximum value
    test = torch.max(outputs)

    # timer end
    if prank==0:
        logging.info('final time: {:.2f}'.format(time.perf_counter()-st)+' s')

    # return an integer back to cpp
    return test

# eof