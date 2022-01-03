#!/usr/bin/env python
# coding: utf-8

# ### Libraries

# In[12]:


from gurobipy import *
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import distance
import pandas as pd
from copy import deepcopy as dc


# In[53]:


def SolverTotal(state):
#     num_cl_nodes,num_ncl_nodes,num_permanent_lanes,num_transient_lanes,num_clusters,num_carriers = 1000,500,[5,5],[1,1],5,2
    num_cl_nodes,num_ncl_nodes,num_permanent_lanes,num_transient_lanes,num_clusters,num_carriers = 10,5,[2,2],[1,1],5,2
    nodes = np.concatenate((make_blobs(n_samples=num_cl_nodes, centers=num_clusters, cluster_std=30, random_state=10,center_box=(0, 1000))[0], make_blobs(n_samples=num_ncl_nodes, centers=num_ncl_nodes, cluster_std=0.60, random_state=0,center_box=(0, 1000))[0]))
    num_lanes = sum(num_permanent_lanes) + sum(num_transient_lanes)
    total_lanes = [num_permanent_lanes[i] + num_transient_lanes[i] for i in range(num_carriers)]

    np.random.seed(state)
    permanent_lanes = np.random.randint(0,len(nodes),[sum(num_permanent_lanes),2])
    for j in range(sum(num_permanent_lanes)):
        if permanent_lanes[j][0] == permanent_lanes[j][1]:
            if j == 0:
                permanent_lanes[j] = [0,1]
            else:
                permanent_lanes[j] = permanent_lanes[0]

    np.random.seed(state+1)
    transient_lanes = np.random.randint(0,len(nodes),[sum(num_transient_lanes),2])
    for j in range(sum(num_transient_lanes)):
        if transient_lanes[j][0] == transient_lanes[j][1]:
            if j == 0:
                transient_lanes[j] = [0,1]
            else:
                transient_lanes[j] = transient_lanes[0]

    distance_matrix = distance.cdist(nodes, nodes, 'euclidean')
    carrier_idx_pr = np.ones((1,1))
    for k in range(num_carriers):
        temp = np.ones((num_permanent_lanes[k],1))*k
        carrier_idx_pr = np.vstack((carrier_idx_pr, temp))
    carrier_idx_pr = np.delete(carrier_idx_pr, (0), axis=0)
    carrier_idx_tr = np.ones((1,1))
    for k in range(num_carriers):
        temp = np.ones((num_transient_lanes[k],1))*k
        carrier_idx_tr = np.vstack((carrier_idx_tr, temp))
    carrier_idx_tr = np.delete(carrier_idx_tr, (0), axis=0)
    carrier_idx = np.vstack((carrier_idx_pr, carrier_idx_tr))
    permanent_lanes = np.hstack((permanent_lanes, carrier_idx_pr))
    transient_lanes = np.hstack((transient_lanes, carrier_idx_tr))
    lanes = np.concatenate((permanent_lanes,transient_lanes))

    mod_MCLCP = Model(name = 'MCLCP')
    mod_MCLCP.Params.LogToConsole = 0

    N = list(range(num_cl_nodes + num_ncl_nodes))
    K = list(range(num_carriers))
    I = permanent_lanes.astype(int)
    II = [x for x in I.tolist()]
    I_ijk = np.zeros((len(N),len(N))).astype(int)
    for ele in I:
        I_ijk[ele[0]][ele[1]] = I_ijk[ele[0]][ele[1]] + 1
    I_k = np.zeros((len(N),len(N),len(K))).astype(int)
    for ele in I:
        I_k[ele[0]][ele[1]][ele[2]] = I_k[ele[0]][ele[1]][ele[2]] + 1
    L = transient_lanes.astype(int)
    print(L)
    r = np.zeros((len(N),len(N))).astype(int)
    for ele in L:
        r[ele[0]][ele[1]] = r[ele[0]][ele[1]] + 1

    A = np.array([[i, j] for i in range((num_cl_nodes + num_ncl_nodes)) for j in range((num_cl_nodes + num_ncl_nodes)) if i != j])
    c = distance_matrix
    theta = np.array([0.5,0.5])

    x = mod_MCLCP.addVars(len(N),len(N),len(K), lb = 0, name = 'x', vtype = GRB.INTEGER)
    z = mod_MCLCP.addVars(len(N),len(N),len(K), lb = 0, name = 'z', vtype = GRB.INTEGER)

    mod_MCLCP.setObjective((quicksum(c[A[i][0]][A[i][1]]*x[A[i][0],A[i][1],k] for i in range(len(A)) for k in K)) + (quicksum(theta[k]*c[A[j][0]][A[j][1]]*z[A[j][0],A[j][1],k] for j in range(len(A)) for k in K)), GRB.MINIMIZE)
    mod_MCLCP.addConstrs(quicksum(x[i,j,k] for j in N) - quicksum(x[j,i,k] for j in N) + quicksum(z[i,j,k] for j in N) - quicksum(z[j,i,k] for j in N) == 0 for i in N for k in K)
    mod_MCLCP.addConstrs(quicksum(x[L[ele][0],L[ele][1],k] for k in K) - I_ijk[L[ele][0]][L[ele][1]] - r[L[ele][0]][L[ele][1]] == 0 for ele in range(len(L)))
    mod_MCLCP.addConstrs(x[I[ele][0],I[ele][1],k] - I_k[I[ele][0]][I[ele][1]][k] >= 0 for k in K for ele in range(len(I)))
    mod_MCLCP.addConstrs(x[i,i,k] == 0 for k in K for i in range(len(N)))
    mod_MCLCP.addConstrs(z[i,i,k] == 0 for k in K for i in range(len(N)))

    mod_MCLCP.optimize()
    Vars = []
    X={key:x[key].x for key in x.keys()}
    Z={key:z[key].X for key in z.keys()}
    for v in mod_MCLCP.getVars():
        if v.x>0:
            Vars.append(['%s %g' % (v.varName, v.x)])
    obj = mod_MCLCP.objVal
    sol = []
    for ele in X:
        if X[ele] > 0:
            sol.append([state,'x',ele[0],ele[1],ele[2],X[ele]])
    for ele in Z:
        if Z[ele] > 0:
            sol.append([state,'z',ele[0],ele[1],ele[2],Z[ele]])
    return  sol,c

def Cost_Cal (df,c,num_carriers):
    cost = []
    for i in range(num_carriers):
        cc = 0
        for index, row in df[(df.Var == 'x') & (df.Vehicle == i)].iterrows():
            cc = cc+ c[row['From']][int(row['To'])]
        for index, row in df[(df.Var == 'z') & (df.Vehicle == i)].iterrows():
            cc = cc+ 0.5*c[row['From']][int(row['To'])]
        cost.append(cc)
    return cost


# In[54]:


solution, costMat= dc(SolverTotal(0))

column_names = ["Instance", "Var", "From", "To", "Vehicle", 'traverse']
dfTotal = (pd.DataFrame(solution,columns=column_names))

Cost_Cal (dfTotal,costMat,2)
dfTotal


# In[47]:


def Solver(state):
#     num_cl_nodes,num_ncl_nodes,num_permanent_lanes,num_transient_lanes,num_clusters,num_carriers = 1000,500,[5,5],[1,1],5,2
    num_cl_nodes,num_ncl_nodes,num_permanent_lanes,num_transient_lanes,num_clusters,num_carriers = 10,5,[2,2],[1,1],5,2
    nodes = np.concatenate((make_blobs(n_samples=num_cl_nodes, centers=num_clusters, cluster_std=30, random_state=10,center_box=(0, 1000))[0], make_blobs(n_samples=num_ncl_nodes, centers=num_ncl_nodes, cluster_std=0.60, random_state=0,center_box=(0, 1000))[0]))
    num_lanes = sum(num_permanent_lanes) + sum(num_transient_lanes)
    total_lanes = [num_permanent_lanes[i] + num_transient_lanes[i] for i in range(num_carriers)]

    np.random.seed(state)
    permanent_lanes = np.random.randint(0,len(nodes),[sum(num_permanent_lanes),2])
    for j in range(sum(num_permanent_lanes)):
        if permanent_lanes[j][0] == permanent_lanes[j][1]:
            if j == 0:
                permanent_lanes[j] = [0,1]
            else:
                permanent_lanes[j] = permanent_lanes[0]

    np.random.seed(state+1)
    transient_lanes = np.random.randint(0,len(nodes),[sum(num_transient_lanes),2])
    for j in range(sum(num_transient_lanes)):
        if transient_lanes[j][0] == transient_lanes[j][1]:
            if j == 0:
                transient_lanes[j] = [0,1]
            else:
                transient_lanes[j] = transient_lanes[0]

    distance_matrix = distance.cdist(nodes, nodes, 'euclidean')
    carrier_idx_pr = np.ones((1,1))
    for k in range(num_carriers):
        temp = np.ones((num_permanent_lanes[k],1))*k
        carrier_idx_pr = np.vstack((carrier_idx_pr, temp))
    carrier_idx_pr = np.delete(carrier_idx_pr, (0), axis=0)
    carrier_idx_tr = np.ones((1,1))
    for k in range(num_carriers):
        temp = np.ones((num_transient_lanes[k],1))*k
        carrier_idx_tr = np.vstack((carrier_idx_tr, temp))
    carrier_idx_tr = np.delete(carrier_idx_tr, (0), axis=0)
    carrier_idx = np.vstack((carrier_idx_pr, carrier_idx_tr))
    permanent_lanes = np.hstack((permanent_lanes, carrier_idx_pr))
    transient_lanes = np.hstack((transient_lanes, carrier_idx_tr))
    lanes = np.concatenate((permanent_lanes,transient_lanes))
    mod_MCLCP = Model(name = 'MCLCP')
    mod_MCLCP.Params.LogToConsole = 0
    N = list(range(num_cl_nodes + num_ncl_nodes))
    K = list(range(num_carriers))
    I = permanent_lanes.astype(int)
    II = [x for x in I.tolist()]
    I_ijk = np.zeros((len(N),len(N))).astype(int)
    for ele in I:
        I_ijk[ele[0]][ele[1]] = I_ijk[ele[0]][ele[1]] + 1
    I_k = np.zeros((len(N),len(N),len(K))).astype(int)
    for ele in I:
        I_k[ele[0]][ele[1]][ele[2]] = I_k[ele[0]][ele[1]][ele[2]] + 1
    L = transient_lanes.astype(int)
    r = np.zeros((len(N),len(N))).astype(int)
    for ele in L:
        r[ele[0]][ele[1]] = r[ele[0]][ele[1]] + 1
    A = np.array([[i, j] for i in range((num_cl_nodes + num_ncl_nodes)) for j in range((num_cl_nodes + num_ncl_nodes)) if i != j])
    c = distance_matrix
    theta = np.array([0.5,0.5])
    
    

    
    x = mod_MCLCP.addVars(len(N),len(N),len(K), lb = 0, name = 'x', vtype = GRB.INTEGER)
    z = mod_MCLCP.addVars(len(N),len(N),len(K), lb = 0, name = 'z', vtype = GRB.INTEGER)
    mod_MCLCP.setObjective((quicksum(c[A[i][0]][A[i][1]]*x[A[i][0],A[i][1],k] for i in range(len(A)) for k in K)) + (quicksum(theta[k]*c[A[j][0]][A[j][1]]*z[A[j][0],A[j][1],k] for j in range(len(A)) for k in K)), GRB.MINIMIZE)
    mod_MCLCP.addConstrs(quicksum(x[i,j,k] for j in N) - quicksum(x[j,i,k] for j in N) + quicksum(z[i,j,k] for j in N) - quicksum(z[j,i,k] for j in N) == 0 for i in N for k in K)
    mod_MCLCP.addConstrs(quicksum(x[L[ele][0],L[ele][1],k] for k in K) - I_ijk[L[ele][0]][L[ele][1]] - r[L[ele][0]][L[ele][1]] == 0 for ele in range(len(L)))
    mod_MCLCP.addConstrs(x[I[ele][0],I[ele][1],k] - I_k[I[ele][0]][I[ele][1]][k] >= 0 for k in K for ele in range(len(I)))
    mod_MCLCP.addConstrs(x[i,i,k] == 0 for k in K for i in range(len(N)))
    mod_MCLCP.addConstrs(z[i,i,k] == 0 for k in K for i in range(len(N)))
    mod_MCLCP.optimize()
    Vars = []
    X={key:x[key].x for key in x.keys()}
    Z={key:z[key].X for key in z.keys()}
    for v in mod_MCLCP.getVars():
        if v.x>0:
            Vars.append(['%s %g' % (v.varName, v.x)])
    obj = mod_MCLCP.objVal
    
    

    Total_Cost = float(obj)
    print(Total_Cost)
    sol = []

    for q in range(len(transient_lanes)):
        print(q)
        transient_lanes_temp = dc(transient_lanes)
        transient_lanes_temp = np.delete(transient_lanes_temp, (q), axis=0)

        mod_MCLCP = Model(name = 'MCLCP')
        mod_MCLCP.Params.LogToConsole = 0

        N = list(range(num_cl_nodes + num_ncl_nodes))
        K = list(range(num_carriers))
        I = permanent_lanes.astype(int)
        II = [x for x in I.tolist()]
        I_ijk = np.zeros((len(N),len(N))).astype(int)
        for ele in I:
            I_ijk[ele[0]][ele[1]] = I_ijk[ele[0]][ele[1]] + 1
        I_k = np.zeros((len(N),len(N),len(K))).astype(int)
        for ele in I:
            I_k[ele[0]][ele[1]][ele[2]] = I_k[ele[0]][ele[1]][ele[2]] + 1
        L = transient_lanes_temp.astype(int)
        r = np.zeros((len(N),len(N))).astype(int)
        for ele in L:
            r[ele[0]][ele[1]] = r[ele[0]][ele[1]] + 1

        A = np.array([[i, j] for i in range((num_cl_nodes + num_ncl_nodes)) for j in range((num_cl_nodes + num_ncl_nodes)) if i != j])
        c = distance_matrix
        theta = np.array([0.5,0.5])

        x = mod_MCLCP.addVars(len(N),len(N),len(K), lb = 0, name = 'x', vtype = GRB.INTEGER)
        z = mod_MCLCP.addVars(len(N),len(N),len(K), lb = 0, name = 'z', vtype = GRB.INTEGER)

        mod_MCLCP.setObjective((quicksum(c[A[i][0]][A[i][1]]*x[A[i][0],A[i][1],k] for i in range(len(A)) for k in K)) + (quicksum(theta[k]*c[A[j][0]][A[j][1]]*z[A[j][0],A[j][1],k] for j in range(len(A)) for k in K)), GRB.MINIMIZE)
        mod_MCLCP.addConstrs(quicksum(x[i,j,k] for j in N) - quicksum(x[j,i,k] for j in N) + quicksum(z[i,j,k] for j in N) - quicksum(z[j,i,k] for j in N) == 0 for i in N for k in K)
        mod_MCLCP.addConstrs(quicksum(x[L[ele][0],L[ele][1],k] for k in K) - I_ijk[L[ele][0]][L[ele][1]] - r[L[ele][0]][L[ele][1]] == 0 for ele in range(len(L)))
        mod_MCLCP.addConstrs(x[I[ele][0],I[ele][1],k] - I_k[I[ele][0]][I[ele][1]][k] >= 0 for k in K for ele in range(len(I)))
        mod_MCLCP.addConstrs(x[i,i,k] == 0 for k in K for i in range(len(N)))
        mod_MCLCP.addConstrs(z[i,i,k] == 0 for k in K for i in range(len(N)))

        mod_MCLCP.optimize()
        Vars = []
        X={key:x[key].x for key in x.keys()}
        Z={key:z[key].X for key in z.keys()}
        for v in mod_MCLCP.getVars():
            if v.x>0:
                Vars.append(['%s %g' % (v.varName, v.x)])
        obj = float(mod_MCLCP.objVal)
        
#         temp = []
        for ele in X:
            if X[ele] > 0:
                sol.append([state,q,'x',ele[0],ele[1],ele[2],X[ele]])
        for ele in Z:
            if Z[ele] > 0:
                sol.append([state,q,'z',ele[0],ele[1],ele[2],Z[ele]])
#         print(obj)
#         sol.append([state,'y',int(transient_lanes[q][0]),int(transient_lanes[q][1]),int(transient_lanes[q][2]),Total_Cost-obj])
    return sol, transient_lanes






# In[48]:


column_names = ["Instance","RemovedLane","Var", "From", "To", "Vehicle", "MC"]
df = pd.DataFrame(columns = column_names)
for i in range(1):
    sol, transient_lanes = dc(Solver(i))
    df = df.append((pd.DataFrame(sol,columns=column_names)),ignore_index=True)
# df.to_csv (r'C:\Users\Student\Desktop\Research\LTL\export_dataframe.csv', index = None, header=True) 


# In[49]:


# df = df.append((pd.DataFrame(Solver(0),columns=column_names)),ignore_index=True)
df,transient_lanes


# In[51]:


num_carriers = 2
for i in range(len(transient_lanes)):
    temp = dc(df[(df.RemovedLane == i)])
    print(Cost_Cal (temp,costMat,num_carriers))


# In[39]:


a = [2,5,2,6,8,1]
a.pop(1)
a


# In[ ]:


Solver(0)


# In[ ]:


SolverTotal(0)


# In[ ]:




