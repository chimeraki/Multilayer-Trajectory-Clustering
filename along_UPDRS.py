##################code by Sanjukta Krishnagopal###############
##################           May 2020          ###############
###############       sanju33@gmail.com          #############

from numpy import *
#import cPickle, gzip
from matplotlib.pyplot import *
import scipy.linalg as linalg
import scipy
import csv
import igraph
import itertools
from scipy import signal
from copy import deepcopy
import operator
import pickle
import collections
import random
from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge
import matplotlib as plt
import math
import networkx as nx
from networkx.generators.classic import empty_graph, path_graph, complete_graph
from scipy.sparse import coo_matrix
import json
import louvain
import pandas as pd
import igraph as ig
from scipy.cluster.hierarchy import dendrogram,linkage
import community
from collections import Counter

matplotlib.rc('xtick', labelsize=18) 
matplotlib.rc('ytick', labelsize=18)
rand=42
random.seed(rand)

############# patients with complete data ###############
p1=[]
p2=[]
p3=[]
p4=[]
ver=4
config = json.loads(open('bl_v0'+str(ver)+'.json').read())
for i in range(len(config)):
   p1.append(config[i][[x for x in config[i].keys()][16]]) #patno
ver=6
config = json.loads(open('bl_v0'+str(ver)+'.json').read())
for i in range(len(config)):
   p2.append(config[i][[x for x in config[i].keys()][16]]) #patno
ver=8
config = json.loads(open('bl_v0'+str(ver)+'.json').read())
for i in range(len(config)):
   p3.append(config[i][[x for x in config[i].keys()][16]]) #patno
ver=10
config = json.loads(open('bl_v0'+str(ver)+'.json').read())
for i in range(len(config)):
   p4.append(config[i][[x for x in config[i].keys()][16]]) #patno

p=[p1,p2,p3,p4]
pat_set=set(p[0]).intersection(*p)
pat_num=np.sort(list(pat_set))
pat_num=[int(x) for x in pat_num]

ver_list=[4,4,6,8,10] #bl,4,6,8,10
shif=[0,2000,4000,6000,8000]

current =2020
v=[2,22,24,9,11,13,15,18,4,20,6,26,28,30,32,34] #organized by domains
bs=[1,21,23,8,10,12,14,17,3,19,5,25,27,29,31,33]

subnodelist = ['Age','JOLO','SDM','SFT','HVLT','LNS','MOCA','SEADL','RBDQ','ESS','SCOPA','GDS','STAI','PD1','PD2','PD3','T-PD']

dir=[1,-1,-1,-1,-1,-1,-1,-1,1,1,1,1,1,1,1,1,1]


### identify the four layers by splitting up outcome variable values in the final year
chosen=-2 #outcome variable UPDRS3
first=[]
pat1=[]
second=[]
pat2=[]
third=[]
pat3=[]
fourth=[]
pat4=[]

ver=10
config = json.loads(open('bl_v0'+str(ver)+'.json').read())
dat=[]
datbl=[]

for i in range(len(config)):
     dat1=config[i]
     dat1=[dat1[x] for x in dat1.keys()]
     dat11=list([dat1[index] for index in v])
     if dat1[16] in pat_set:
        #dat11.insert(0,dat1[16]) #patno
        dat11.insert(1,current-dat1[0]) #age
        #dat11.insert(1,dat1[7]) #gender value 2=male, 1=female
        dat11=[float(x) for x in dat11]
        dat.append(dat11)

for i in range(len(config)):
     dat1=config[i]
     dat1=[dat1[x] for x in dat1.keys()]
     dat11=list([dat1[index] for index in bs])
     if dat1[16] in pat_set:
        #dat11.insert(0,dat1[16]) #patno
        dat11.insert(1,current-dat1[0]) #age
        #dat11.insert(1,dat1[7]) #gender
        dat11=[float(x) for x in dat11]
        datbl.append(dat11)

          
dat=np.array(dat)
datbl=np.array(datbl)
n=shape(dat)[1]  #n variables
m=shape(dat)[0]  #m patients


mid=np.percentile(dat[:,chosen], 50, axis=0) #quartile based on UPDRS3
low=np.percentile(dat[:,chosen], 25, axis=0)
high=np.percentile(dat[:,chosen], 75, axis=0)

figure()
plt.pyplot.hist(dat[:,-1])
plt.pyplot.axvline(low)
plt.pyplot.axvline(mid)
plt.pyplot.axvline(high)
xlabel('MDS-UPDRS-3',fontsize=15)
ylabel('Frequency')
plt.pyplot.savefig('outcome_variable_distr.pdf')


#variables to be studied
l=np.arange(0,n,1)
var_cnt=list(l[:-4])  
dirs=np.array(dir)[var_cnt]
names=np.array(subnodelist)[var_cnt]

for yr in range(len(ver_list)):
   ver=ver_list[yr] #bl
   config = json.loads(open('bl_v0'+str(ver)+'.json').read())
   dat=[]
   dat_con=[]
   datbl=[]

   for i in range(len(config)):
        dat1=config[i]
        dat1=[dat1[x] for x in dat1.keys()]
        dat11=list([dat1[index] for index in v])
        if dat1[16] in pat_set:
           #dat11.insert(0,dat1[16]) #patno
           dat11.insert(1,current-dat1[0]) #age
           #dat11.insert(1,dat1[7]) #gender value 2=male, 1=female
           dat11=[float(x) for x in dat11]
           dat.append(dat11)



   for i in range(len(config)):
        dat1=config[i]
        dat1=[dat1[x] for x in dat1.keys()]
        dat11=list([dat1[index] for index in bs])
        if dat1[16] in pat_set:
           #dat11.insert(0,dat1[16]) #patno
           dat11.insert(1,current-dat1[0]) #age
           #dat11.insert(1,dat1[7]) #gender
           dat11=[float(x) for x in dat11]
           datbl.append(dat11)



   dat=np.array(dat)
   datbl=np.array(datbl)

   if yr==0:
      sol=datbl
   else:
      sol=dat
      
   mins=np.min(sol,axis=0)
   maxs=np.max(sol,axis=0)

      
   for k in range(m):
      z=(sol[k,var_cnt]-mins[var_cnt])/(maxs[var_cnt]-mins[var_cnt])
      for i in range(len(z)):
         if dirs[i]==-1:
            z[i]=1-z[i]
      if sol[k][chosen]<low:
         first.append(z)
         pat1.append(pat_num[k]+shif[yr])
      if sol[k][chosen]>low and sol[k][chosen]<mid:
         second.append(z)
         pat2.append(pat_num[k]+shif[yr])
      if sol[k][chosen]>mid and sol[k][chosen]<high:
         third.append(z)
         pat3.append(pat_num[k]+shif[yr])
      if sol[k][chosen]>high:
         fourth.append(z)
         pat4.append(pat_num[k]+shif[yr])
      #for borderline values
      if sol[k][chosen]==low:
         p=random.random()
         if p<0.5:
            first.append(z)
            pat1.append(pat_num[k]+shif[yr])
         else:
            second.append(z)
            pat2.append(pat_num[k]+shif[yr])
      if sol[k][chosen]==mid:
         p=random.random()
         if p<0.5:
            second.append(z)
            pat2.append(pat_num[k]+shif[yr])
         else:
            third.append(z)
            pat3.append(pat_num[k]+shif[yr])
      if sol[k][chosen]==high:
         p=random.random()
         if p<0.5:
            third.append(z)
            pat3.append(pat_num[k]+shif[yr])
         else:
            fourth.append(z)
            pat4.append(pat_num[k]+shif[yr])
   


#############


#identify variable communities in all four layers (quartiles of outcome variable)
first=np.array(first)
second=np.array(second)
third=np.array(third)
fourth=np.array(fourth)
pat1=np.array(pat1)
pat2=np.array(pat2)
pat3=np.array(pat3)
pat4=np.array(pat4)


var_list=[first,second,third,fourth]
total=0
noofcomm=[]
commsize=[]
x=[]
membership=[]
commcolor=[]
commnodes=[]
vall=[]


for yr in range(len(var_list)):  
   layer=var_list[yr]
   
   B=np.zeros((shape(layer)[0],shape(layer)[0]))
   C=np.zeros((shape(layer)[1],shape(layer)[1]))
   D=np.hstack((B,layer))
   E=np.hstack((layer.T,C))
   F=np.vstack((D,E))

   G_= nx.Graph(F)

   part = community.best_partition(G_, random_state=rand)
   val = [part.get(node) for node in G_.nodes()]
   mod = community.modularity(part,G_)
   vall.append(val)


   edges = G_.edges()
   weights = [G_[u][v]['weight'] for u,v in edges]
   cmap=cm.jet

   X, Y = nx.bipartite.sets(G_)   #X is patient indices, Y is variable indices

   x.append(np.unique(val[-len(Y):]))
   noofcomm.append(len(x[yr]))

   total+=len(x[yr])
   for i in range(len(x[yr])):
               place=np.where(val[-len(Y):]==x[yr][i])
               commnodes.append(list(names[j] for j in place[0]))

############################

commsize=np.zeros(total)

W=np.zeros((total,total))

pat_track = pat_num
pat_traj={}
pat_in_comm={} #community variable profiles
source={}
target={}

for i in range(m): #go through all patients
   patid=pat_num[i]
   pat_traj[patid]=[-1,-1,-1,-1]
   for j in range(4):  #identify which layer they are in at all times
      yoo=np.where(pat1==patid+j*2000)[0]
      if not len(yoo):
         yoo=np.where(pat2==patid+j*2000)[0]
         if not len(yoo):
            yoo=np.where(pat3==patid+j*2000)[0]
            if not len(yoo):
               yoo=np.where(pat4==patid+j*2000)[0]
               if not len(yoo):
                  continue
               k=vall[3][yoo[0]]   #which community is person in in layer 4
               if k>=noofcomm[3]:
                  continue
               commsize[k+noofcomm[0]+noofcomm[1]+noofcomm[2]]+=1
               pat_traj[patid][j]=noofcomm[0]+noofcomm[1]+noofcomm[2]+k
               if k+noofcomm[0]+noofcomm[1]+noofcomm[2] not in pat_in_comm:
                  pat_in_comm[k+noofcomm[0]+noofcomm[1]+noofcomm[2]]=[]
               pat_in_comm[k+noofcomm[0]+noofcomm[1]+noofcomm[2]].append(fourth[yoo])
            else:           #which community is person in in layer 3
               k=vall[2][yoo[0]]   
               if k>=noofcomm[2]:
                  continue
               commsize[k+noofcomm[0]+noofcomm[1]]+=1
               pat_traj[patid][j]=noofcomm[0]+noofcomm[1]+k
               if k+noofcomm[0]+noofcomm[1] not in pat_in_comm:
                  pat_in_comm[k+noofcomm[0]+noofcomm[1]]=[]
               pat_in_comm[k+noofcomm[0]+noofcomm[1]].append(third[yoo])
         else:                #which community is person in in layer 2
            k=vall[1][yoo[0]]
            if k>=noofcomm[1]:
                  continue
            commsize[k+noofcomm[0]]+=1
            pat_traj[patid][j]=noofcomm[0]+k
            if k+noofcomm[0] not in pat_in_comm:
               pat_in_comm[k+noofcomm[0]]=[]
            pat_in_comm[k+noofcomm[0]].append(second[yoo])
      else:
         k=vall[0][yoo[0]] #which community is person in in layer 1
         if k>=noofcomm[0]:
                  continue
         commsize[k]+=1
         pat_traj[patid][j]=k
         if k not in pat_in_comm:
            pat_in_comm[k]=[]
         pat_in_comm[k].append(first[yoo])


###############variable profiles for each community

comm_profile=[]
std_profile=[]
for i in range(total):
      comm_profile.append(np.mean(np.array(pat_in_comm[i]), axis=0)[0]) 
      std_profile.append(np.std(np.array(pat_in_comm[i]), axis=0)[0]) 

#Plotting the community profiles

commnames=deepcopy(commnodes)
#make variable titles fit
for i in range(len(commnodes)):  
   x=commnodes[i]
   s=" ".join(x)
   s = s.replace("'", '')
   commnames[i]=s

figure()
fig, axs = subplots(np.max(noofcomm),4, figsize=(25, 8), facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace = 1.2, wspace=0.5)

axs = axs.ravel()
tot=0
for i in range(4):
   for j in range(np.max(noofcomm)):
      if j in range(noofcomm[i]):
       axs[4*j+i].bar(np.arange(len(comm_profile[tot])), comm_profile[tot], yerr=std_profile[tot])
       nam=str(commnames[tot])
       nam=nam.replace("PD1 PD2 PD3 T-PD", "PD1,2,3,T")
       nam=nam.replace("PD1 PD2 PD3", "PD1,2,3")
       nam=nam.replace("PD2 PD3 T-PD", "PD2,3,T")
       nam=nam.replace("PD1 PD2", "PD1,2")
       nam=nam.replace("PD2 PD3", "PD2,3")
       nam=nam.replace("PD3 T-PD", "PD3,T")
       axs[4*j+i].set_title('$C_{}^{}$: '.format(i,j)+str(nam), fontsize=20)
       axs[4*j+i].set_yticks([0,1])
       if j== noofcomm[i]-1:
          axs[4*j+i].set_xticks(np.arange(len(names)))
          axs[4*j+i].set_xticklabels(names, rotation=90,fontsize=15)
       else:
          axs[4*j+i].set_xticks([])
       tot+=1
      else:
         fig.delaxes(axs.flatten()[4*j+i])

savefig(str(subnodelist[chosen])+'_variable_profile_in_comm.pdf', bbox_inches='tight')



#delete all incomplete trajectories
q=[]
for k,v in pat_traj.items():
        if type(v)==int:
           if k not in q:
              q.append(k)
        elif len(v)==0:
           if k not in q:
              q.append(k)
for i in range(len(q)):
   del pat_traj[q[i]] 

for k,v in pat_traj.items():
   source[k]=[v[0],v[1],v[2]]
   target[k]=[v[1],v[2],v[3]]
   W[v[0],v[1]]+=1
   W[v[1],v[0]]+=1
   W[v[2],v[1]]+=1
   W[v[1],v[2]]+=1
   W[v[2],v[3]]+=1
   W[v[3],v[2]]+=1


pat_num=list(pat_traj.keys())
pat_val=list(pat_traj.values())



#trajectory clustering

node_sim=np.zeros((total, total))
for i in range(total):
   for j in range(total):
      count=0
      s=list(pat_traj.values())
      x=[]
      y=[]
      for l in range(len(s)):
         if i in s[l]:
            x.append(l)
         if j in s[l]:
            y.append(l)
      #x=np.where(any(e==i for e in s)) #np.where(s==i)
      if not x or not y:
         continue
      #x=[s[l] for l in x]
      x=[pat_traj[list(pat_traj.keys())[b]] for b in x]# all trajectories going through this node
      y=[pat_traj[list(pat_traj.keys())[b]] for b in y]

      #method 2 using intersection set over union set with repetitions /multi-set
      yo1=[]
      yo2=[]
      for mm in range(len(x)):
         for k in range(3):# iterating through all trajectories related to node x
            yo1.append((x[mm][k],x[mm][k+1])) #all edges on any trajectory connected to x

         
      for nn in range(len(y)):
         for k in range(3): # iterating through all trajectories related to node y
            yo2.append((y[nn][k],y[nn][k+1])) #all edges on any trajectory connected to x


      tot=len(list(set(yo1).union(set(yo2))))
      com=len(list(set(yo1).intersection(set(yo2))))
      if tot!=0:
         node_sim[i,j]=node_sim[j,i]=com/tot
      v=np.diag(node_sim)
      
### get trajectory/(patient) similarity by summing node similarities
baseline=np.zeros((len(pat_num),len(pat_num)))
traj=np.zeros((len(pat_num),len(pat_num)))
for i in range(len(pat_num)):
   for j in range(len(pat_num)):
      for k in range(4):
         traj[i][j]+=node_sim[pat_val[i][k],pat_val[j][k]]
         baseline[i][j]+=(commsize[pat_val[i][k]]*commsize[pat_val[j][k]])/(m*m)

traj=traj/4 #normalizing wrt layers
baseline=baseline/4

H=nx.Graph(traj)              
part = community.best_partition(H, random_state=rand)
values = [part.get(node) for node in H.nodes()]
mod = community.modularity(part,H)

nocomm=len(np.unique(values))
pat_comm = dict(zip(pat_num, values))

count_=collections.Counter(values)

#print ('size of each community ' , pat_comm)
print (count_, 'trajectoryclustering_noofcomm')

###calculate trajectory separatedness

diff=traj-baseline
sep=0
cnt=0
for i in range(shape(traj)[0]):
   for j in range(i):
      sep+=diff[i,j]
      cnt+=1

seperatedness=sep/cnt

print ('separatedness: ', seperatedness)

#plot

figure()    
G=igraph.Graph.Weighted_Adjacency(W.tolist(),mode="undirected")
layout=[]
for i in range (4):
   noc=noofcomm[i]
   for j in range(noc):
      layout.append((i*3,j*4))
      #pos.append[(i*3,j*2)]

for i in range(len(commnodes)):  #make labels fit one after the other
   x=commnodes[i]
   s="\n".join(x)
   s = s.replace("'", '')
   commnodes[i]=s

G.vs['label']=commnodes
G.es['width']=0


#new plotting stuff--------------------------------------------------
#this is to plot only a few trajectories with thickness marking the flow of people in each trajectory communtiy



totcomm=np.sum(noofcomm)
mat=[]
for i in range(nocomm): #number of trajectory communities
   mat.append(np.zeros((totcomm,totcomm)))

for i in range(len(source)):
   patnid=list(source.keys())[i]
   v=pat_comm[list(source.keys())[i]]
   for k in range(3):
      a=source[patnid][k]
      b=target[patnid][k]
      mat[v][a][b]+=1         #add to thiccness

mat=np.array(mat)

#calculate the average profile of each trajectory community (by averaging the node profiles) in each layer

traj_prof=[]
thr=[0,noofcomm[0],noofcomm[0]+noofcomm[1],noofcomm[0]+noofcomm[1]+noofcomm[2],noofcomm[0]+noofcomm[1]+noofcomm[2]+noofcomm[3]]
for i in range(nocomm): #for each traj community
   traj_prof.append([[],[],[],[]])  #for all layers
for i in range(len(source)):
   patnid=list(source.keys())[i]
   v=pat_comm[list(source.keys())[i]]
   vals=[l for l in source[patnid]]
   vals.append(target[patnid][-1])
   for a in vals:
      for s in range(len(thr)-1):
         if a <thr[s+1] and a>=thr[s]:
               traj_prof[v][s].append(comm_profile[a])
               

traj_prof=np.array(traj_prof)

traj_mean=deepcopy(traj_prof)
traj_std=deepcopy(traj_prof)
traj_cnt=np.zeros((shape(traj_prof)[0],shape(traj_prof)[0]))
for i in range(shape(traj_prof)[0]):
   for j in range(shape(traj_prof)[1]):
      traj_mean[i,j]=np.mean(traj_prof[i,j],axis=0)
      traj_std[i,j]=np.std(traj_prof[i,j],axis=0)
      traj_cnt[i,j]=len(traj_prof[i,j])

figure()
fig, axs = subplots(4,nocomm, figsize=(25, 6), facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace = 0.6,wspace = 0.5)

axs = axs.ravel()
for s in range(nocomm):
   if s==0:
      i=2
   if s==1:
      i=3
   if s==2:
      i=0
   if s==3:
      i=1
   print (i)
   for j in range(4): #layers
       axs[s+j*nocomm].bar(np.arange(len(traj_mean[i,j])), traj_mean[i,j], yerr=traj_std[i,j])
       axs[nocomm*j+s].set_yticks([0,1])
       axs[nocomm*j+s].set_title('$S_{}^{} \quad \kappa_{}^{} = {}$'.format(s,j,s,j,int(traj_cnt[i,j])),fontsize=20)
       if j== 3:
          axs[nocomm*j+s].set_xticks(np.arange(len(names)))
          axs[nocomm*j+s].set_xticklabels(names, rotation=90,fontsize=15)
       else:
          axs[nocomm*j+s].set_xticks([])
savefig(str(subnodelist[chosen])+'_traj_variable_profile.pdf', bbox_inches='tight')



######find kruskall wallis pvalue for variable and difference:

'''kw_traj=[]
kw_traj_diff=[]
for i in range(len(names)):
   hello=[]
   hello_dif=[]
   for j in range(nocomm):
      hello.append(np.concatenate((np.array(traj_prof[j][0])[:,i],np.array(traj_prof[j][1])[:,i],np.array(traj_prof[j][2])[:,i],np.array(traj_prof[j][3])[:,i])))
      hello_dif.append(np.concatenate((np.array(traj_prof[j][0])[:,i],np.array(traj_prof[j][1])[:,i])
   kw_traj.append(scipy.stats.kruskal(hello[0],hello[1],hello[2],hello[3]).pvalue)
   kw_traj_diff.append(scipy.stats.kruskal(traj_mean[0][:,i]-traj_mean[1][:,i],traj_mean[1][:,i]-traj_mean[2][:,i],traj_mean[2][:,i]-traj_mean[3][:,i]).pvalue)
formatting_function = np.vectorize(lambda f: format(f, '6.3E'))
kw_traj=formatting_function(np.array(kw_traj))
kw_traj_diff=formatting_function(np.array(kw_traj_diff))
               
print ('kruskall_wallis_traj ', kw_traj)
print ('kruskall_wallis_diff_traj ', kw_traj_diff)'''

#don't show edges traversed by less than threshold people
thresh=3
mat[mat<thresh]=0

cols=["red","blue","green","orange", "cyan"]

for i in range(totcomm):
   for j in range(totcomm):
      for l in range (nocomm):
         G.add_edge(i,j,width=mat[l][i][j],color=cols[l],directed=True)
                 


igraph.plot(G,str(subnodelist[chosen])+'_flowlines.pdf',labels=True, layout=layout,vertex_size=commsize, vertex_color='grey',vertex_label_dist=0, vertex_label_size=24,mark_groups=True, bbox=(1624,1324), margin=250)

