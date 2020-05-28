from numpy import *
from matplotlib.pyplot import *
import scipy.linalg as linalg
import scipy
import scipy.stats
import igraph
from copy import deepcopy
import operator
import pickle
import collections
import matplotlib as plt
import math
import networkx as nx
import json
import louvain
import igraph as ig
from scipy.cluster.hierarchy import dendrogram,linkage
import community


matplotlib.rc('xtick', labelsize=18) 
matplotlib.rc('ytick', labelsize=18)
rand=44
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

current =2020
v=[2,22,24,9,11,13,15,18,4,20,6,26,28,30,32,34] #organized by domains
bs=[1,21,23,8,10,12,14,17,3,19,5,25,27,29,31,33]

subnodelist = ['Age','JOLO','SDM','SFT','HVLT','LNS','MOCA','SEADL','RBDQ','ESS','SCOPA','GDS','STAI','PD-1','PD-2','PD-3','PD-T']

dir=[1,-1,-1,-1,-1,-1,-1,-1,1,1,1,1,1,1,1,1,1]

total=0
noofcomm=[]
commsize=[]
x=[]
membership=[]
commcolor=[]
commnodes=[]
vall=[]
modu=[]

year={}
X={}
Y={}


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
   n=shape(dat)[1]  #n variables
   m=shape(dat)[0]  #m patients

   if yr==0:
      sol=datbl
   else:
      sol=dat

   mins=np.min(sol,axis=0)
   maxs=np.max(sol,axis=0)

   A=np.zeros((n,m))#random.randint(100, size=(n, n))
   for i in range(n): #runs over all symptoms except pat no.
             for k in range(m): #runs over all people
                  z=(sol[k,i]-mins[i])/(maxs[i]-mins[i])
                  if dir[i]==-1:
                     z=1-z
                  A[i][k]=z


   year[yr]=A

   B=np.zeros((n,n))
   C=np.zeros((m,m))
   D=np.hstack((B,A))
   E=np.hstack((A.T,C))
   F=np.vstack((D,E))

   G_= nx.Graph(F)

   part = community.best_partition(G_, random_state=50) #remove randomness
   val = [part.get(node) for node in G_.nodes()]
   mod = community.modularity(part,G_)
   modu.append(mod)
   vall.append(val)


   edges = G_.edges()
   weights = [G_[u][v]['weight'] for u,v in edges]
   cmap=cm.jet

   X, Y = nx.bipartite.sets(G_)   #X is variable indices, Y is patient indices

   x.append(np.unique(val[:len(X)]))
   noofcomm.append(len(x[yr]))

   total+=len(x[yr])
   for i in range(len(x[yr])):
               place=np.where(val[:len(X)]==x[yr][i])
               commnodes.append(list(subnodelist[j] for j in place[0]))

print ('layer_modularity: ', modu)

############################################

commsize=np.zeros(total)

W=np.zeros((total,total))

pat_track = pat_num   #track all patients
pat_traj={}
pat_in_comm={}
source={}
target={}


for i in Y: #patients in year0
      #year bl to year 1 
      k1=vall[0][i] #which comm is pat 1 in
      if k1>noofcomm[0]:
         continue
      coms=np.where(x[0]==k1)[0]
      patid=pat_num[i-list(Y)[0]]
      if k1 not in pat_in_comm:
         pat_in_comm[k1]=[]
      pat_in_comm[k1].append(patid)
      commsize[k1]+=1
      #where is the person in year 2
      k2=vall[1][i] 
      if k2>noofcomm[1]:
         continue
      comd=np.where(x[1]==k2)[0]      
      W[coms,comd+noofcomm[0]]+=1
      W[comd+noofcomm[0],coms]+=1
      source[patid]=[coms,-1,-1,-1]
      target[patid]=[comd+noofcomm[0],-1,-1,-1]
      pat_traj[patid]=[coms,-1,-1,-1,-1]
      ###year1 to 2
      coms=comd
      if k2+noofcomm[0] not in pat_in_comm:
         pat_in_comm[k2+noofcomm[0]]=[]
      pat_in_comm[k2+noofcomm[0]].append(patid)
      commsize[noofcomm[0]+k2]+=1
      k3=vall[2][i]
      if k3>noofcomm[2]:
         continue
      comd=np.where(x[2]==k3)[0]
      W[coms+noofcomm[0],comd+noofcomm[0]+noofcomm[1]]+=1
      W[comd+noofcomm[0]+noofcomm[1],coms+noofcomm[0]]+=1
      if patid in source:
         source[patid][1]=coms+noofcomm[0]
         target[patid][1]=comd+noofcomm[0]+noofcomm[1]
      if patid in pat_traj:
         pat_traj[patid][1]=coms+noofcomm[0]
      else:
         pat_traj[patid]=[-1,coms+noofcomm[0],0,0,0]
      ###year2 to 3
      coms=comd
      if k3+noofcomm[0]+noofcomm[1] not in pat_in_comm:
         pat_in_comm[k3+noofcomm[0]+noofcomm[1]]=[]
      pat_in_comm[k3+noofcomm[0]+noofcomm[1]].append(patid)
      commsize[noofcomm[0]+noofcomm[1]+k3]+=1
      k4=vall[3][i]
      if k4>noofcomm[3]:
         continue
      comd=np.where(x[3]==k4)[0]      
      W[comd+noofcomm[0]+noofcomm[1]+noofcomm[2],coms+noofcomm[0]+noofcomm[1]]+=1
      W[coms+noofcomm[0]+noofcomm[1],comd+noofcomm[0]+noofcomm[1]+noofcomm[2]]+=1
      if patid in source:
         source[patid][2]=coms+noofcomm[0]+noofcomm[1]
         target[patid][2]=comd+noofcomm[0]+noofcomm[1]+noofcomm[2]
      if patid in pat_traj:
         pat_traj[patid][2]=coms+noofcomm[0]+noofcomm[1]
      else:
         pat_traj[patid]=[-1,-1,coms+noofcomm[0]+noofcomm[1],0,0]
      ###year3 to 4
      coms=comd
      if k4+noofcomm[0]+noofcomm[1]+noofcomm[2] not in pat_in_comm:
         pat_in_comm[k4+noofcomm[0]+noofcomm[1]+noofcomm[2]]=[]
      pat_in_comm[k4+noofcomm[0]+noofcomm[1]+noofcomm[2]].append(patid)
      commsize[noofcomm[0]+noofcomm[1]+noofcomm[2]+k4]+=1
      k5=vall[4][i]
      if k5>noofcomm[4]:
         continue
      comd=np.where(x[4]==k5)[0]
      W[comd+noofcomm[0]+noofcomm[1]+noofcomm[2]+noofcomm[3],coms+noofcomm[0]+noofcomm[1]+noofcomm[2]]+=1
      W[coms+noofcomm[0]+noofcomm[1]+noofcomm[2],comd+noofcomm[0]+noofcomm[1]+noofcomm[2]+noofcomm[3]]+=1
      if patid in source:
         source[patid][3]=coms+noofcomm[0]+noofcomm[1]+noofcomm[2]
         target[patid][3]=comd+noofcomm[0]+noofcomm[1]+noofcomm[2]+noofcomm[3]
      if patid in pat_traj:
         pat_traj[patid][3]=coms+noofcomm[0]+noofcomm[1]+noofcomm[2]
      else:
         pat_traj[patid]=[-1,-1,-1,coms+noofcomm[0]+noofcomm[1]+noofcomm[2],0]     
      ###year4 to 5
      coms=comd
      if vall[4][i]==noofcomm[4]:
         vall[4][i]=noofcomm[4]-1
      if k5+noofcomm[0]+noofcomm[1]+noofcomm[2]+noofcomm[3] not in pat_in_comm:
            pat_in_comm[k5+noofcomm[0]+noofcomm[1]+noofcomm[2]+noofcomm[3]]=[]
      pat_in_comm[k5+noofcomm[0]+noofcomm[1]+noofcomm[2]+noofcomm[3]].append(patid)
      commsize[noofcomm[0]+noofcomm[1]+noofcomm[2]+noofcomm[3]+k5]+=1
      if patid in pat_traj:
         pat_traj[patid][4]=coms+noofcomm[0]+noofcomm[1]+noofcomm[2]+noofcomm[3]
   
   
#variable profiles for each community 

total_prof=[[],[],[],[],[]]
fin=[]
comm_profile=[]#np.zeros((total,m))
std_profile=[]
for i in range(total):
   if i < noofcomm[0]:
      b=[s for s, val in enumerate(pat_num) if val in set(pat_in_comm[i])]
      stor=year[0][:,b]
      comm_profile.append(np.mean(stor, axis=1))
      std_profile.append(np.std(stor, axis=1))
      total_prof[0].append(stor)
      fin.append(stor)
   elif i < noofcomm[0]+ noofcomm[1]:
      b=[s for s, val in enumerate(pat_num) if val in set(pat_in_comm[i])]
      stor=year[1][:,b]
      comm_profile.append(np.mean(stor, axis=1))
      std_profile.append(np.std(stor, axis=1))
      total_prof[1].append(stor)
      fin.append(stor)
   elif i < noofcomm[0]+ noofcomm[1]+noofcomm[2]:
      b=[s for s, val in enumerate(pat_num) if val in set(pat_in_comm[i])]
      stor=year[2][:,b]
      comm_profile.append(np.mean(stor, axis=1))
      std_profile.append(np.std(stor, axis=1))
      total_prof[2].append(stor)
      fin.append(stor)
   elif i < noofcomm[0]+ noofcomm[1]+noofcomm[2]+noofcomm[3]:
      b=[s for s, val in enumerate(pat_num) if val in set(pat_in_comm[i])]
      stor=year[3][:,b]
      comm_profile.append(np.mean(stor, axis=1))
      std_profile.append(np.std(stor, axis=1))
      total_prof[3].append(stor)
      fin.append(stor)
   elif i < noofcomm[0]+ noofcomm[1]+noofcomm[2]+noofcomm[3]+noofcomm[4]:
      b=[s for s, val in enumerate(pat_num) if val in set(pat_in_comm[i])]
      stor=year[4][:,b]
      comm_profile.append(np.mean(stor, axis=1))
      std_profile.append(np.std(stor, axis=1))
      total_prof[4].append(stor)
      fin.append(stor)

      
#Kruskal Wallis test between all groups in each layer

kw=[[],[],[],[],[]]
for j in range(len(subnodelist)):
      kw[0].append(scipy.stats.kruskal(total_prof[0][0][j],total_prof[0][1][j],total_prof[0][2][j],total_prof[0][3][j],total_prof[0][4][j]).pvalue)
      kw[1].append(scipy.stats.kruskal(total_prof[1][0][j],total_prof[1][1][j],total_prof[1][2][j],total_prof[1][3][j],total_prof[0][4][j]).pvalue)
      kw[2].append(scipy.stats.kruskal(total_prof[2][0][j],total_prof[2][1][j],total_prof[2][2][j],total_prof[2][3][j]).pvalue)
      kw[3].append(scipy.stats.kruskal(total_prof[3][0][j],total_prof[3][1][j],total_prof[3][2][j],total_prof[3][3][j]).pvalue)
      kw[4].append(scipy.stats.kruskal(total_prof[4][0][j],total_prof[4][1][j],total_prof[4][2][j],total_prof[4][3][j]).pvalue)

formatting_function = np.vectorize(lambda f: format(f, '6.3E'))
kw=formatting_function(np.array(kw))
print ('layer_kruskall', kw)   



#Plotting the community profiles

commnames=deepcopy(commnodes)
#make variable titles fit
for i in range(len(commnodes)):  
   x=commnodes[i]
   s=" ".join(x)
   s = s.replace("'", '')
   commnames[i]=s

fig, axs = subplots(np.max(noofcomm),5, figsize=(18, 6), facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace = 0.8, wspace=0.8)

   
axs = axs.ravel()
tot=0
for i in range(5):
   for j in range(np.max(noofcomm)):
      if j in range(noofcomm[i]):
       axs[5*j+i].bar(np.arange(len(comm_profile[tot])), comm_profile[tot], yerr=std_profile[tot])
       nam=str(commnames[tot])
       nam=nam.replace("PD-1 PD-2 PD-3 PD-T", "PD-1,2,3,T")
       nam=nam.replace("PD-1 PD-2 PD-3", "PD-1,2,3")
       nam=nam.replace("PD-2 PD-3 T-PD", "PD-2,3,T")
       nam=nam.replace("PD-1 PD-2", "PD-1,2")
       nam=nam.replace("PD-2 PD-3", "PD-2,3")
       nam=nam.replace("PD-3 PD-T", "PD-3,T")
       axs[5*j+i].set_title('$C_{}^{}$: '.format(i,j)+str(nam), fontsize=12)
       axs[5*j+i].set_yticks([0,1])
       if j== noofcomm[i]-1:
          axs[5*j+i].set_xticks(np.arange(len(subnodelist)))
          axs[5*j+i].set_xticklabels(subnodelist, rotation=90,fontsize=7.5)
       else:
          axs[5*j+i].set_xticks([])
       tot+=1
      else:
         fig.delaxes(axs.flatten()[5*j+i])
#for i in [17,21,22,23,24,25,26]:
#   fig.delaxes(axs.flatten()[i])

#show()
savefig('variable_profile_in_comm.pdf',bbox_inches='tight')



pat_num=list(pat_traj.keys())
pat_val=list(pat_traj.values())

#trajectory clustering

#total is the total number of communities over all layers
node_sim=np.zeros((total, total))

for i in range(total):
   for j in range(total):
      count=0
      s=list(pat_traj.values())
      x=[]
      y=[]
      for l in range(len(s)):
         s[l]=[v[0] for v in s[l]]
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
      #first order method
      
      
      #second order using intersection set over union set with repetitions /multi-set
      yo1=[]
      yo2=[]
      for mm in range(len(x)):
         for k in range(4):# iterating through all trajectories related to node x
            yo1.append((x[mm][k][0],x[mm][k+1][0])) #all edges on any trajectory connected to x

         
      for nn in range(len(y)):
         for k in range(4): # iterating through all trajectories related to node y
            yo2.append((y[nn][k][0],y[nn][k+1][0])) #all edges on any trajectory connected to x


      tot=len(list(set(yo1).union(set(yo2))))
      com=len(list(set(yo1).intersection(set(yo2))))
      if tot!=0:
         node_sim[i,j]=node_sim[j,i]=com/tot
      v=np.diag(node_sim)
      '''
      ##don't count repeated edges - it is advisable to leave this commented out
      rep = []
      for elem in deepcopy(yo1):
         if elem in yo2:
            yo1.pop(yo1.index(elem))
            rep.append(yo2.pop(yo2.index(elem)))'''
      


### get SECOND order trajectory/(patient) similarity by summing node similarities

baseline=np.zeros((len(pat_num),len(pat_num)))
traj=np.zeros((len(pat_num),len(pat_num)))
for i in range(len(pat_num)):
   for j in range(len(pat_num)):
      for k in range(5):
         traj[i][j]+=node_sim[pat_val[i][k],pat_val[j][k]]
         baseline[i][j]+=(commsize[pat_val[i][k]]*commsize[pat_val[j][k]])/(m*m)

traj=traj/5 #normalizing wrt layers
baseline=baseline/5

### get FIRST order trajectory/(patient) similarity as fraction of overlap in 2 trajectories
'''traj=np.zeros((len(pat_num),len(pat_num)))
for i in range(len(pat_num)):
   for j in range(len(pat_num)):
      tk=0
      for k in range(4):
         if pat_val[i][k]==pat_val[j][k] and pat_val[i][k+1]==pat_val[j][k+1]:
            traj[i][j]+=1
            tk+=1
         else:
            tk+=2
      print (tk)
      traj[i][j]=traj[i][j]/tk'''

##idenfity patient communities

H=nx.Graph(traj)              
part = community.best_partition(H, random_state=rand)
values = [part.get(node) for node in H.nodes()]
mod = community.modularity(part,H)

print ('traj_modularity: ', mod)

nocomm=len(np.unique(values))
pat_comm = dict(zip(pat_num, values))

count_=collections.Counter(values)

#print ('size of each community ' , pat_comm)
print (count_, 'trajectoryclustering_noofcomm')

#sort_pat = collections.OrderedDict(sorted(pat_comm.items())) #order pat_num acc to comm. Patients in same community are next to each other in array

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

###plot

figure()    
G=igraph.Graph.Weighted_Adjacency(W.tolist(),mode="undirected")
layout=[]
for i in range (5):
   noc=noofcomm[i]
   for j in range(noc):
      layout.append((i*3,j*4))
      #pos.append[(i*3,j*2)]

#make variable titles fit
for i in range(len(commnodes)):  
   x=commnodes[i]
   s="\n".join(x)
   s = s.replace("'", '')
   commnodes[i]=s


G.vs['label']=commnodes
G.es['width']=[w*0 for w in G.es['weight']]


#this edges with thickness marking the flow of people in each trajectory communtiy


totcomm=np.sum(noofcomm)
mat=[]
for i in range(nocomm): #number of trajectory communities
   mat.append(np.zeros((totcomm,totcomm)))

#find weight of trajectory edge
for i in range(len(source)):
   patnid=list(source.keys())[i]
   v=pat_comm[list(source.keys())[i]]  #which community does this trajectory/ person belong to
   for k in range(4):
      a=source[patnid][k][0]
      b=target[patnid][k][0]
      mat[v][a][b]+=1         #add to thiccness

mat=np.array(mat)


#calculate the average profile of each trajectory community (by averaging the node profiles) in each layer

traj_prof=[]
thr=[0,noofcomm[0],noofcomm[0]+noofcomm[1],noofcomm[0]+noofcomm[1]+noofcomm[2],noofcomm[0]+noofcomm[1]+noofcomm[2]+noofcomm[3],noofcomm[0]+noofcomm[1]+noofcomm[2]+noofcomm[3]+noofcomm[4]]
for i in range(nocomm): #for each traj community
   traj_prof.append([[],[],[],[],[]])  #for all layers
for i in range(len(source)):
   patnid=list(source.keys())[i]
   v=pat_comm[list(source.keys())[i]]
   vals=[l[0] for l in source[patnid]]
   vals.append(target[patnid][-1][0])
   for a in vals:
      for s in range(len(thr)-1):
         if a <thr[s+1] and a>=thr[s]:
               traj_prof[v][s].append(comm_profile[a])

traj_prof=np.array(traj_prof)
for i in range(shape(traj_prof)[0]):
   for j in range(shape(traj_prof)[1]):
      traj_prof[i,j]=np.array(traj_prof[i,j])


traj_mean=deepcopy(traj_prof)
traj_std=deepcopy(traj_prof)
traj_cnt=np.zeros((shape(traj_prof)[0],shape(traj_prof)[1]))
for i in range(shape(traj_prof)[0]):
   for j in range(shape(traj_prof)[1]):
      traj_mean[i,j]=np.mean(traj_prof[i,j],axis=0)
      traj_std[i,j]=np.std(traj_prof[i,j],axis=0)
      traj_cnt[i,j]=len(traj_prof[i,j])

#traj_prof=traj_prof.reshape(np.shape(traj_prof)[0],np.shape(traj_prof)[1],np.shape(traj_prof)[3])

#####plot the different trajectory profiles


#Plotting the community profiles
figure()
fig, axs = subplots(5,nocomm, figsize=(25, 8), facecolor='w', edgecolor='k')
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
   for j in range(5): #layers
       axs[s+j*nocomm].bar(np.arange(len(traj_mean[i,j])), traj_mean[i,j], yerr=traj_std[i,j])
       axs[nocomm*j+s].set_yticks([0,1])
       axs[nocomm*j+s].set_title('$S_{}^{} \quad \kappa_{}^{} = {}$'.format(s,j,s,j,int(traj_cnt[i,j])),fontsize=20)
       if j== 4:
          axs[nocomm*j+s].set_xticks(np.arange(len(subnodelist)))
          axs[nocomm*j+s].set_xticklabels(subnodelist, rotation=90,fontsize=15)
       else:
          axs[nocomm*j+s].set_xticks([])
savefig('traj_variable_profile.pdf', bbox_inches='tight')



######find kruskall wallis pvalue for variable and difference:
a=traj_prof[:,1]-traj_prof[:,0]
b=traj_prof[:,2]-traj_prof[:,1]
c=traj_prof[:,3]-traj_prof[:,2]
s=[a,b,c]
kw_diff=[]
for i in range(3): #for each inter-layer diff
      kw_diff.append([])
      for k in range(len(subnodelist)):
         a=scipy.stats.kruskal(s[i][0][:,k],s[i][1][:,k],s[i][2][:,k],s[i][3][:,k]).pvalue
         kw_diff[i].append(a)
kw_diff=formatting_function(np.array(kw_diff))
print ('kruskall_wallis_diff_traj ', kw_diff)

#######
'''for i in range(len(subnodelist)):
   hello=[]
   hello_diff=[]
   for s in range(nocomm):
      hello.append([traj_mean[s][0][i],traj_mean[s][1][i],traj_mean[s][2][i],traj_mean[s][3][i],traj_mean[s][4][i]])
      hello_diff.append([traj_mean[s][0][i]-traj_mean[s][1][i],traj_mean[s][1][i],traj_mean[s][2][i],traj_mean[s][2][i],traj_mean[s][3][i],traj_mean[s][3][i],traj_mean[s][4][i]])
      
   kw_traj.append(scipy.stats.kruskal(hello[0],hello[1],hello[2],hello[3]).pvalue)
   kw_traj_diff.append(scipy.stats.kruskal(hello_diff[0],hello_diff[1],hello_diff[2],hello_diff[3]).pvalue)

kw_traj=formatting_function(np.array(kw_traj))
kw_traj_diff=formatting_function(np.array(kw_traj_diff))
               
print ('kruskall_wallis_traj ', kw_traj)
print ('kruskall_wallis_diff_traj ', kw_traj_diff)'''


######find kruskall wallis pvalue for variable and difference:
kw_traj=[]
kw_traj_diff=[]
for i in range(len(subnodelist)):
   hello=[]
   hello_diff=[]
   for s in range(nocomm):
      hello.append([traj_mean[s][0][i],traj_mean[s][1][i],traj_mean[s][2][i],traj_mean[s][3][i],traj_mean[s][4][i]])
      hello_diff.append([traj_mean[s][0][i]-traj_mean[s][1][i],traj_mean[s][1][i],traj_mean[s][2][i],traj_mean[s][2][i],traj_mean[s][3][i],traj_mean[s][3][i],traj_mean[s][4][i]])
      
   kw_traj.append(scipy.stats.kruskal(hello[0],hello[1],hello[2],hello[3]).pvalue)
   kw_traj_diff.append(scipy.stats.kruskal(hello_diff[0],hello_diff[1],hello_diff[2],hello_diff[3]).pvalue)

kw_traj=formatting_function(np.array(kw_traj))
kw_traj_diff=formatting_function(np.array(kw_traj_diff))
               
print ('kruskall_wallis_traj ', kw_traj)
print ('kruskall_wallis_diff_traj ', kw_traj_diff)

#don't show edges traversed by less than threshold people
thresh=5
mat[mat<thresh]=0

cols=["red","blue","green","orange","cyan","pink","brown"]

for i in range(totcomm):
   for j in range(totcomm):
      for l in range (nocomm):
         G.add_edge(i,j,width=mat[l][i][j],color=cols[l],directed=True)
                 
figure()
igraph.plot(G,'biparite_flowlines_.pdf',labels=True, layout=layout,vertex_size=commsize, vertex_color='grey',vertex_label_dist=0, vertex_label_size=24,mark_groups=True, bbox=(1624,1324),margin=250)

close('all')
