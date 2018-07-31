## Set display and meta parameters
    #y1=1979
    #y2=2016
    #time=linspace(y1,y2 + 1,dot(12,(y2 - y1 + 1)) + 1)
    #time=(time(arange(1,end() - 1)) + time(arange(2,end()))) / 2
    #red_color=concat([1,0.3,0.4])
    #orange_color=concat([255,140,0]) / 256
    #blue_color=concat([0,0.447,0.741])
    #purple_color=concat([0.494,0.184,0.556])
    #green_color=concat([0.466,0.674,0.188])
    #lightblue_color=concat([0.301,0.745,0.933])
    #maroon_color=concat([0.635,0.078,0.184])
    ## Ticks
    #Ytick1=arange(- 60,60,1)
    #Ytick2=arange(- 60,60,2)
    #Ytick5=arange(- 60,60,5)
    #YtickLabels1=cellarray([])
    #YtickLabels2=cellarray([])
    #YtickLabels5=cellarray([])
    #S0N='S0N'
    #for y in arange(1,length(Ytick1),2).reshape(-1):
    #    if Ytick1(y) == 0:
    #        YtickLabels1[y]=cellarray(['0'])
    #    else:
    #        YtickLabels1[y]=cellarray([concat([int2str(abs(Ytick1(y))),S0N(sign(Ytick1(y)) + 2)])])
    #    if y < length(Ytick1):
    #        YtickLabels1[y + 1]=cellarray([''])
    #
    #for y in arange(1,length(Ytick2),2).reshape(-1):
    #    if Ytick2(y) == 0:
    #        YtickLabels2[y]=cellarray(['0'])
    #    else:
    #        YtickLabels2[y]=cellarray([concat([int2str(abs(Ytick2(y))),S0N(sign(Ytick2(y)) + 2)])])
    #    if y < length(Ytick2):
    #        YtickLabels2[y + 1]=cellarray([''])
    #
    #for y in arange(1,length(Ytick5),2).reshape(-1):
    #    if Ytick5(y) == 0:
    #        YtickLabels5[y]=cellarray(['0'])
    #    else:
    #        YtickLabels5[y]=cellarray([concat([int2str(abs(Ytick5(y))),S0N(sign(Ytick5(y)) + 2)])])
    #    if y < length(Ytick5):
    #        YtickLabels5[y + 1]=cellarray([''])

from __future__ import division
import numpy as np
import scipy as sp
from scipy.io import netcdf
from TropD_Metric_PSI import TropD_Metric_PSI 
from TropD_Metric_TPB import TropD_Metric_TPB 
from TropD_Metric_STJ import TropD_Metric_STJ
from TropD_Metric_EDJ import TropD_Metric_EDJ
from TropD_Metric_PE import TropD_Metric_PE
from TropD_Metric_UAS import TropD_Metric_UAS
from TropD_Metric_PSL import TropD_Metric_PSL
from TropD_Metric_OLR import TropD_Metric_OLR

from TropD_Calculate_Mon2Season import TropD_Calculate_Mon2Season


## 1) PSI -- Streamfunction zero crossing
f_V = netcdf.netcdf_file('../ValidationData/va.nc','r')
V = f_V.variables['va'][:]
#Change axes of V to be [time, lat, lev]
V = np.transpose(V, (2,1,0))
lat = f_V.variables['lat'][:]
lev = f_V.variables['lev'][:]

Phi_psi_nh = np.zeros((np.shape(V)[0],1))
Phi_psi_sh = np.zeros((np.shape(V)[0],1))

for j in range(np.shape(V)[0]):
  Phi_psi_sh[j], Phi_psi_nh[j] = TropD_Metric_PSI(V[j,:,:], lat, lev)


# Calculate metric from annual mean
V_ANN = TropD_Calculate_Mon2Season(V,np.arange(12))

Phi_psi_nh_ANN = np.zeros((np.shape(V_ANN)[0],1))
Phi_psi_sh_ANN = np.zeros((np.shape(V_ANN)[0],1))

for j in range(np.shape(V_ANN)[0]):
  Phi_psi_sh_ANN[j], Phi_psi_nh_ANN[j] = TropD_Metric_PSI(V_ANN[j,:,:], lat, lev)


#figure
#subplot('211')
#plot(time,Phi_psi_nh,'-','linewidth',1,'color',green_color)
#hold('on')
#plot(concat([arange(y1,y2)]) + 0.5,Phi_psi_nh_ANN,'-','linewidth',2,'color',blue_color)
#plot(concat([arange(y1,y2)]) + 0.5,TropD_Calculate_Mon2Season(Phi_psi_nh, np.arange(12)),'-k','linewidth',2)
#set(gca,'fontsize',12,'linewidth',2,'tickdir','out','box','off','xtick', np.arange(12)(1980,2020,5)]),'xticklabels',cellarray(['']),'ytick',Ytick5,'yticklabels',YtickLabels5)
#ylabel(cellarray([['NH \\Psi_{500}'],['latitude']]))
#xlim(concat([y1,y2 + 1]))
#l=legend('Latitude of \\Psi_{500} zero crossing from monthly mean V','Latitude of \\Psi_{500} zero crossing from annual mean V','Latitude of \\Psi_{500} zero crossing from annual means of monthly metric values')
#set(l,'box','off','location','north')
#subplot('212')
#plot(time,Phi_psi_sh,'-','linewidth',1,'color',green_color)
#hold('on')
#plot(concat([arange(y1,y2)]) + 0.5,Phi_psi_sh_ANN,'-','linewidth',2,'color',blue_color)
#plot(concat([arange(y1,y2)]) + 0.5,TropD_Calculate_Mon2Season(Phi_psi_sh, np.arange(12)),'-k','linewidth',2)
#set(gca,'fontsize',12,'linewidth',2,'tickdir','out','box','off','xtick', np.arange(12)(1980,2020,5)]),'ytick',Ytick5,'yticklabels',YtickLabels5)
#xlim(concat([y1,y2 + 1]))
#xlabel('Year','fontsize',14)
#ylabel(cellarray([['SH \\Psi zero crossing'],['latitude']]))
# Introduce latitude unertainty condition: no additional zero crossing is allowed within 10 degrees

Phi_psi_nh_L = np.zeros((np.shape(V)[0],1))
Phi_psi_sh_L = np.zeros((np.shape(V)[0],1))

for j in range(np.shape(V)[0]):
  Phi_psi_sh_L[j], Phi_psi_nh_L[j] = TropD_Metric_PSI(V[j,:,:], lat, lev, 'Psi_500', 10)

#figure
#plot(time,Phi_psi_nh,'-','linewidth',2,'color',green_color)
#hold('on')
#plot(time(isnan(Phi_psi_nh_L)),Phi_psi_nh(isnan(Phi_psi_nh_L)),'*','markersize',10,'color',red_color)
#hold('on')
#set(gca,'fontsize',12,'linewidth',2,'tickdir','out','box','off','xtick', np.arange(12)(1980,2020,5)]),'ytick',Ytick5,'yticklabels',YtickLabels5)
#ylabel(cellarray([['NH \\Psi_{500}'],['latitude']]))
#xlim(concat([y1,y2 + 1]))
#l=legend('\\Psi_{500} zero crossing from monthly mean V that qualify uncertainty criterion','\\Psi_{500} zero crossing from monthly mean V that fail uncertainty criterion')
#set(l,'box','off','location','north')
#xlabel('Year','fontsize',14)

## 2) TPB -- Tropopause break latitude
f_T = netcdf.netcdf_file('../ValidationData/ta.nc','r')
f_Z = netcdf.netcdf_file('../ValidationData/zg.nc','r')
T = f_T.variables['ta'][:]
Z = f_Z.variables['zg'][:]
#Change axes of T and Z to be [time, lat, lev]
T = np.transpose(T, (2,1,0))
Z = np.transpose(Z, (2,1,0))

lat = f_T.variables['lat'][:]
lev = f_T.variables['lev'][:]

Phi_tpb_nh = np.zeros((np.shape(T)[0],1))
Phi_tpb_sh = np.zeros((np.shape(T)[0],1))

for j in range(np.shape(V)[0]):
  Phi_tpb_sh[j], Phi_tpb_nh[j] = TropD_Metric_TPB(T[j,:,:], lat, lev)

# Calculate tropopause break from annual mean
T_ANN = TropD_Calculate_Mon2Season(T, np.arange(12))

Z_ANN = TropD_Calculate_Mon2Season(Z, np.arange(12))

Phi_tpb_nh_ANN = np.zeros((np.shape(T_ANN)[0],1))

Phi_tpb_sh_ANN = np.zeros((np.shape(T_ANN)[0],1))

Phi_tpbZ_nh_ANN = np.zeros((np.shape(T_ANN)[0],1))
                                                   
Phi_tpbZ_sh_ANN = np.zeros((np.shape(T_ANN)[0],1))

Phi_tpbT_nh_ANN = np.zeros((np.shape(T_ANN)[0],1))
                                                   
Phi_tpbT_sh_ANN = np.zeros((np.shape(T_ANN)[0],1))

for j in range(np.shape(T_ANN)[0]):
  Phi_tpb_sh_ANN[j], Phi_tpb_nh_ANN[j] = TropD_Metric_TPB(T_ANN[j,:,:], lat, lev, method='max_gradient')
  Phi_tpbT_sh_ANN[j], Phi_tpbT_nh_ANN[j] = TropD_Metric_TPB(T_ANN[j,:,:], lat, lev, method='max_potemp')
  Phi_tpbZ_sh_ANN[j], Phi_tpbZ_nh_ANN[j] = TropD_Metric_TPB(T_ANN[j,:,:], lat, lev, method='cutoff',\
                                                           Z=Z_ANN[j,:,:,], Cutoff=15*1000)


#figure
#subplot('211')
#plot(time,Phi_tpb_nh,'-','linewidth',1,'color',green_color)
#hold('on')
#plot(concat([arange(y1,y2)]) + 0.5,Phi_tpb_nh_ANN,'-','linewidth',2,'color',blue_color)
#plot(concat([arange(y1,y2)]) + 0.5,Phi_tpbZ_nh_ANN,'--','linewidth',1,'color',blue_color)
#plot(concat([arange(y1,y2)]) + 0.5,Phi_tpbT_nh_ANN,'--','linewidth',1,'color',red_color)
#plot(concat([arange(y1,y2)]) + 0.5,TropD_Calculate_Mon2Season(Phi_tpb_nh, np.arange(12)),'-k','linewidth',2)
#set(gca,'fontsize',12,'linewidth',2,'tickdir','out','box','off','xtick', np.arange(12)(1980,2020,5)]),'xticklabels',cellarray(['']),'ytick',Ytick5,'yticklabels',YtickLabels5)
#ylabel(cellarray([['NH tropopause break'],['latitude']]))
#xlim(concat([y1,y2 + 1]))
#l=legend('Latitude of tropopause break from monthly mean T -- potential temperature difference','Latitude of tropopause break from annual mean T -- maximal gradient','Latitude of tropopause break from annual mean T -- 15km cutoff height','Latitude of tropopause break from annual mean T -- potential temperature difference','Latitude of tropopause break from annual mean of monthly metric values -- potential temperature difference')
#set(l,'box','off','location','north')
#subplot('212')
#plot(time,Phi_tpb_sh,'-','linewidth',1,'color',green_color)
#hold('on')
#plot(concat([arange(y1,y2)]) + 0.5,Phi_tpb_sh_ANN,'-','linewidth',2,'color',blue_color)
#plot(concat([arange(y1,y2)]) + 0.5,Phi_tpbZ_sh_ANN,'--','linewidth',1,'color',blue_color)
#plot(concat([arange(y1,y2)]) + 0.5,Phi_tpbT_sh_ANN,'--','linewidth',1,'color',red_color)
#plot(concat([arange(y1,y2)]) + 0.5,TropD_Calculate_Mon2Season(Phi_tpb_sh, np.arange(12)),'-k','linewidth',2)
#set(gca,'fontsize',12,'linewidth',2,'tickdir','out','box','off','xtick', np.arange(12)(1980,2020,5)]),'ytick',Ytick5,'yticklabels',YtickLabels5)
#xlim(concat([y1,y2 + 1]))
#xlabel('Year','fontsize',14)
#ylabel(cellarray([['SH tropopause break'],['latitude']]))

##3) OLR -- OLR cutoff
#Note: OLR is assumed to be positive upwards and in units of W/m^2

f_olr = netcdf.netcdf_file('../ValidationData/rlnt.nc','r')
f_olrcs = netcdf.netcdf_file('../ValidationData/rlntcs.nc','r')
olr = -f_olr.variables['rlnt'][:]
olrcs = -f_olrcs.variables['rlntcs'][:]
lat = f_olr.variables['lat'][:]

#Change axes of olr and olrcs to be [time, lat]
olr = np.transpose(olr, (1,0))
olrcs = np.transpose(olrcs, (1,0))

olr_ANN = TropD_Calculate_Mon2Season(olr, np.arange(12))

olrcs_ANN = TropD_Calculate_Mon2Season(olrcs, np.arange(12))

Phi_olr_nh = np.zeros((np.shape(olr)[0],1))
Phi_olr_sh = np.zeros((np.shape(olr)[0],1))

Phi_olr_nh_ANN = np.zeros((np.shape(olr_ANN)[0],1))
Phi_olr_sh_ANN = np.zeros((np.shape(olr_ANN)[0],1))

Phi_olrcs_nh = np.zeros((np.shape(olr)[0],1))
Phi_olrcs_sh = np.zeros((np.shape(olr)[0],1))

Phi_olrcs_nh_ANN = np.zeros((np.shape(olr_ANN)[0],1))
Phi_olrcs_sh_ANN = np.zeros((np.shape(olr_ANN)[0],1))

Phi_olr20_nh_ANN = np.zeros((np.shape(olr_ANN)[0],1))
Phi_olr20_sh_ANN = np.zeros((np.shape(olr_ANN)[0],1))

Phi_olr240_nh_ANN = np.zeros((np.shape(olr_ANN)[0],1))
Phi_olr240_sh_ANN = np.zeros((np.shape(olr_ANN)[0],1))

for j in range(np.shape(olr)[0]):
  Phi_olr_sh[j], Phi_olr_nh[j] = TropD_Metric_OLR(olr[j,:], lat)
  Phi_olrcs_sh[j], Phi_olrcs_nh[j] = TropD_Metric_OLR(olrcs[j,:], lat)

for j in range(np.shape(olr_ANN)[0]):
  Phi_olr_sh_ANN[j], Phi_olr_nh_ANN[j] = TropD_Metric_OLR(olr_ANN[j,:], lat)
  Phi_olrcs_sh_ANN[j], Phi_olrcs_nh_ANN[j] = TropD_Metric_OLR(olrcs_ANN[j,:], lat)
  Phi_olr20_sh_ANN[j], Phi_olr20_nh_ANN[j] = TropD_Metric_OLR(olr_ANN[j,:], lat, method='20W')
  Phi_olr240_sh_ANN[j], Phi_olr240_nh_ANN[j] = TropD_Metric_OLR(olr_ANN[j,:],lat,method='cutoff',Cutoff=240)



#figure
#subplot('211')
#plot(time,Phi_olr_nh,'-','linewidth',3,'color',green_color)
#hold('on')
#plot(time,Phi_olrcs_nh,'-','linewidth',1,'color',multiply(green_color,0.5))
#hold('on')
#plot(concat([arange(y1,y2)]) + 0.5,Phi_olr_nh_ANN,'-','linewidth',3,'color',blue_color)
#plot(concat([arange(y1,y2)]) + 0.5,Phi_olrcs_nh_ANN,'-','linewidth',1,'color',multiply(blue_color,0.5))
#set(gca,'fontsize',12,'linewidth',2,'tickdir','out','box','off','xtick', np.arange(12)(1980,2020,5)]),'xticklabels',cellarray(['']),'ytick',Ytick2,'yticklabels',YtickLabels2)
#ylabel('NH OLR cutoff latitude')
#xlim(concat([y1,y2 + 1]))
#l=legend('Latitude of OLR 250W/m^2 cutoff latitude from monthly OLR','Latitude of OLR 250W/m^2 cutoff latitude from monthly clear-sky OLR','Latitude of OLR 250W/m^2 cutoff latitude from annual mean OLR','Latitude of OLR 250W/m^2 cutoff latitude from annual mean clear-sky OLR')
#set(l,'box','off','location','north')
#subplot('212')
#plot(time,Phi_olr_sh,'-','linewidth',3,'color',green_color)
#hold('on')
#plot(time,Phi_olrcs_sh,'-','linewidth',1,'color',dot(green_color,0.5))
#hold('on')
#plot(concat([arange(y1,y2)]) + 0.5,Phi_olr_sh_ANN,'-','linewidth',3,'color',blue_color)
#plot(concat([arange(y1,y2)]) + 0.5,Phi_olrcs_sh_ANN,'-','linewidth',1,'color',dot(blue_color,0.5))
#set(gca,'fontsize',12,'linewidth',2,'tickdir','out','box','off','xtick', np.arange(12)(1980,2020,5)]),'ytick',Ytick2,'yticklabels',YtickLabels2)
#xlim(concat([y1,y2 + 1]))
#xlabel('Year','fontsize',14)
#ylabel('SH OLR cutoff latitude')
#figure
#subplot('211')
#hold('on')
#plot(concat([arange(y1,y2)]) + 0.5,Phi_olr_nh_ANN,'-','linewidth',3,'color',multiply(blue_color,0.5))
#plot(concat([arange(y1,y2)]) + 0.5,Phi_olr240_nh_ANN,'-','linewidth',3,'color',blue_color)
#plot(concat([arange(y1,y2)]) + 0.5,Phi_olr20_nh_ANN,'-','linewidth',3,'color',green_color)
#set(gca,'fontsize',12,'linewidth',2,'tickdir','out','box','off','xtick', np.arange(12)(1980,2020,5)]),'xticklabels',cellarray(['']),'ytick',Ytick2,'yticklabels',YtickLabels2)
#ylabel('NH OLR cutoff latitude')
#xlim(concat([y1,y2 + 1]))
#l=legend('Latitude of OLR 250W/m^2 {default} cutoff latitude from annual-mean OLR','Latitude of OLR 240W/m^2 cutoff latitude from annual-mean OLR','Latitude of OLR -20W/m^2 cutoff latitude from annual-mean OLR')
#set(l,'box','off','location','north')
#subplot('212')
#hold('on')
#plot(concat([arange(y1,y2)]) + 0.5,Phi_olr_sh_ANN,'-','linewidth',3,'color',multiply(blue_color,0.5))
#plot(concat([arange(y1,y2)]) + 0.5,Phi_olr240_sh_ANN,'-','linewidth',3,'color',blue_color)
#plot(concat([arange(y1,y2)]) + 0.5,Phi_olr20_sh_ANN,'-','linewidth',3,'color',green_color)
#set(gca,'fontsize',12,'linewidth',2,'tickdir','out','box','off','xtick', np.arange(12)(1980,2020,5)]),'ytick',Ytick2,'yticklabels',YtickLabels2)
#xlim(concat([y1,y2 + 1]))
#xlabel('Year','fontsize',14)
#ylabel('SH OLR cutoff latitude')

## 4) STJ -- Subtropical Jet (STJ) latitude

f_u = netcdf.netcdf_file('../ValidationData/ua.nc','r')
U = f_u.variables['ua'][:]
lat = f_u.variables['lat'][:]
lev = f_u.variables['lev'][:]

#Change axes of u to be [time, lat]
U = np.transpose(U, (2,1,0))

# Calculate STJ latitude from annual mean
U_ANN = TropD_Calculate_Mon2Season(U, np.arange(12))

Phi_stj_nh_ANN_adj = np.zeros((np.shape(U_ANN)[0],1))
Phi_stj_sh_ANN_adj = np.zeros((np.shape(U_ANN)[0],1))
Phi_stj_nh_ANN_core = np.zeros((np.shape(U_ANN)[0],1))
Phi_stj_sh_ANN_core = np.zeros((np.shape(U_ANN)[0],1))

for j in range(np.shape(U_ANN)[0]):
  Phi_stj_sh_ANN_adj [j], Phi_stj_nh_ANN_adj[j] = TropD_Metric_STJ(U_ANN[j,:,:], lat, lev)
  Phi_stj_sh_ANN_core[j], Phi_stj_nh_ANN_core[j] = TropD_Metric_STJ(U_ANN[j,:,:], lat, lev, method='core')


#figure
#subplot('211')
#plot(concat([arange(y1,y2)]) + 0.5,Phi_stj_nh_ANN_adj,'-','linewidth',2,'color',green_color)
#hold('on')
#plot(concat([arange(y1,y2)]) + 0.5,Phi_stj_nh_ANN_core,'-','linewidth',2,'color',blue_color)
#set(gca,'fontsize',12,'linewidth',2,'tickdir','out','box','off','xtick', np.arange(12)(1980,2020,5)]),'xticklabels',cellarray(['']),'ytick',Ytick2,'yticklabels',YtickLabels2)
#ylabel('NH STJ latitude')
#xlim(concat([y1,y2 + 1]))
#l=legend('Latitude of STJ from anual mean U, using \'adjusted\' method','Latitude of STJ from anual mean U, using \'core\' method')
#set(l,'box','off','location','north')
#subplot('212')
#plot(concat([arange(y1,y2)]) + 0.5,Phi_stj_sh_ANN_adj,'-','linewidth',2,'color',green_color)
#hold('on')
#plot(concat([arange(y1,y2)]) + 0.5,Phi_stj_sh_ANN_core,'-','linewidth',2,'color',blue_color)
#set(gca,'fontsize',12,'linewidth',2,'tickdir','out','box','off','xtick', np.arange(12)(1980,2020,5)]),'ytick',Ytick2,'yticklabels',YtickLabels2)
#xlim(concat([y1,y2 + 1]))
#xlabel('Year','fontsize',14)
#ylabel('SH STJ latitude')

## 5) EDJ -- Eddy Driven Jet (EDJ) latitude
f_u = netcdf.netcdf_file('../ValidationData/ua.nc','r')
U = f_u.variables['ua'][:]
lat = f_u.variables['lat'][:]
lev = f_u.variables['lev'][:]

#Change axes of u to be [time, lat]
U = np.transpose(U, (2,1,0))

Phi_edj_nh = np.zeros((np.shape(U)[0],1))
Phi_edj_sh = np.zeros((np.shape(U)[0],1))

for j in range(np.shape(U)[0]):
  Phi_edj_sh[j], Phi_edj_nh[j] = TropD_Metric_EDJ(U[j,:,:,] ,lat, lev, method='max')

# Calculate EDJ latitude from annual mean
U_ANN = TropD_Calculate_Mon2Season(U, np.arange(12))

Phi_edj_nh_ANN = np.zeros((np.shape(U_ANN)[0],1))
Phi_edj_sh_ANN = np.zeros((np.shape(U_ANN)[0],1))

for j in range(np.shape(U_ANN)[0]):
  Phi_edj_sh_ANN[j], Phi_edj_nh_ANN[j] = TropD_Metric_EDJ(U_ANN[j,:,:], lat, lev)

#figure
#subplot('211')
#plot(time,Phi_edj_nh,'-','linewidth',1,'color',green_color)
#hold('on')
#plot(concat([arange(y1,y2)]) + 0.5,Phi_edj_nh_ANN,'-','linewidth',2,'color',blue_color)
#plot(concat([arange(y1,y2)]) + 0.5,TropD_Calculate_Mon2Season(Phi_edj_nh, np.arange(12)),'-k','linewidth',2)
#set(gca,'fontsize',12,'linewidth',2,'tickdir','out','box','off','xtick', np.arange(12)(1980,2020,5)]),'xticklabels',cellarray(['']),'ytick',Ytick2,'yticklabels',YtickLabels2)
#ylabel('NH EDJ latitude')
#xlim(concat([y1,y2 + 1]))
#l=legend('Latitude of EDJ from monthly mean U','Latitude of EDJ from annual mean U','Latitude of EDJ from annual mean of monthly metric values')
#set(l,'box','off','location','north')
#subplot('212')
#plot(time,Phi_edj_sh,'-','linewidth',1,'color',green_color)
#hold('on')
#plot(concat([arange(y1,y2)]) + 0.5,Phi_edj_sh_ANN,'-','linewidth',2,'color',blue_color)
#plot(concat([arange(y1,y2)]) + 0.5,TropD_Calculate_Mon2Season(Phi_edj_sh, np.arange(12)),'-k','linewidth',2)
#set(gca,'fontsize',12,'linewidth',2,'tickdir','out','box','off','xtick', np.arange(12)(1980,2020,5)]),'ytick',Ytick2,'yticklabels',YtickLabels2)
#xlim(concat([y1,y2 + 1]))
#xlabel('Year','fontsize',14)
#ylabel('SH EDJ latitude')


## 6) PE -- Precipitation minus evaporation subtropical zero crossing latitude
f_pr = netcdf.netcdf_file('../ValidationData/pr.nc','r')
f_er = netcdf.netcdf_file('../ValidationData/hfls.nc','r')

L=2510400.0

pr = f_pr.variables['pr'][:]
er = -f_er.variables['hfls'][:]/L
PE = pr - er 

lat = f_pr.variables['lat'][:]

#Change axes of pr and er to be [time, lat]
PE = np.transpose(PE, (1,0))

PE_ANN = TropD_Calculate_Mon2Season(PE, np.arange(12))

Phi_pe_nh = np.zeros((np.shape(PE)[0],1))
Phi_pe_sh = np.zeros((np.shape(PE)[0],1))
Phi_pe_nh_ANN = np.zeros((np.shape(PE_ANN)[0],1))
Phi_pe_sh_ANN = np.zeros((np.shape(PE_ANN)[0],1))

for j in range(np.shape(PE)[0]):
  Phi_pe_sh[j], Phi_pe_nh[j] = TropD_Metric_PE(PE[j,:], lat)

for j in range(np.shape(PE_ANN)[0]):
  Phi_pe_sh_ANN[j], Phi_pe_nh_ANN[j] = TropD_Metric_PE(PE_ANN[j,:], lat)

#figure
#subplot('211')
#plot(time,Phi_pe_nh,'-','linewidth',2,'color',green_color)
#hold('on')
#plot(concat([arange(y1,y2)]) + 0.5,Phi_pe_nh_ANN,'-','linewidth',2,'color',blue_color)
#plot(concat([arange(y1,y2)]) + 0.5,TropD_Calculate_Mon2Season(Phi_pe_nh, np.arange(12)),'-k','linewidth',2)
#set(gca,'fontsize',12,'linewidth',2,'tickdir','out','box','off','xtick', np.arange(12)(1980,2020,5)]),'xticklabels',cellarray(['']),'ytick',Ytick2,'yticklabels',YtickLabels2)
#ylabel('NH P - E zero-crossing')
#xlim(concat([y1,y2 + 1]))
#l=legend('Latitude of P minus E zero-crossing','Latitude of P minus E zero-crossing from annual mean field','Latitude of P minus E zero-crossing from annual mean of monthly metric')
#set(l,'box','off','location','north')
#subplot('212')
#plot(time,Phi_pe_sh,'-','linewidth',2,'color',green_color)
#hold('on')
#plot(concat([arange(y1,y2)]) + 0.5,Phi_pe_sh_ANN,'-','linewidth',2,'color',blue_color)
#plot(concat([arange(y1,y2)]) + 0.5,TropD_Calculate_Mon2Season(Phi_pe_sh, np.arange(12)),'-k','linewidth',2)
#set(gca,'fontsize',12,'linewidth',2,'tickdir','out','box','off','xtick', np.arange(12)(1980,2020,5)]),'ytick',Ytick2,'yticklabels',YtickLabels2)
#xlim(concat([y1,y2 + 1]))
#xlabel('Year','fontsize',14)
#ylabel('SH P - E zero-crossing')

## 7) UAS -- Zonal surface wind subtropical zero crossing latitude
f_u = netcdf.netcdf_file('../ValidationData/ua.nc','r')
f_uas = netcdf.netcdf_file('../ValidationData/uas.nc','r')
U = f_u.variables['ua'][:]
uas = f_uas.variables['uas'][:]
lat = f_u.variables['lat'][:]
lev = f_u.variables['lev'][:]

#Change axes of u to be [time, lat]
U = np.transpose(U, (2,1,0))
uas = np.transpose(uas, (1,0))

uas_ANN = TropD_Calculate_Mon2Season(uas, np.arange(12))
u_ANN = TropD_Calculate_Mon2Season(U, np.arange(12))

Phi_uas_nh = np.zeros((np.shape(uas)[0],1))
Phi_uas_sh = np.zeros((np.shape(uas)[0],1))

Phi_uas_nh_ANN = np.zeros((np.shape(uas_ANN)[0],1))
Phi_uas_sh_ANN = np.zeros((np.shape(uas_ANN)[0],1))
Phi_Uas_nh_ANN = np.zeros((np.shape(uas_ANN)[0],1))
Phi_Uas_sh_ANN = np.zeros((np.shape(uas_ANN)[0],1))

for j in range(np.shape(uas)[0]):
  Phi_uas_sh[j], Phi_uas_nh[j] = TropD_Metric_UAS(uas[j,:], lat)

for j in range(np.shape(uas_ANN)[0]):
  Phi_uas_sh_ANN[j], Phi_uas_nh_ANN[j] = TropD_Metric_UAS(uas_ANN[j,:], lat)
  Phi_Uas_sh_ANN[j], Phi_Uas_nh_ANN[j] = TropD_Metric_UAS(U_ANN[j,:], lat, lev)


#figure
#subplot('211')
#plot(time,Phi_uas_nh,'-','linewidth',2,'color',green_color)
#hold('on')
#plot(concat([arange(y1,y2)]) + 0.5,Phi_uas_nh_ANN,'-','linewidth',2,'color',blue_color)
#plot(concat([arange(y2,y2)]) + 0.5,Phi_Uas_nh_ANN,'-','linewidth',2,'color',red_color)
#plot(concat([arange(y1,y2)]) + 0.5,TropD_Calculate_Mon2Season(Phi_uas_nh, np.arange(12)),'-k','linewidth',2)
#set(gca,'fontsize',12,'linewidth',2,'tickdir','out','box','off','xtick', np.arange(12)(1980,2020,5)]),'xticklabels',cellarray(['']),'ytick',Ytick2,'yticklabels',YtickLabels2)
#ylabel('NH uas zero-crossing')
#xlim(concat([y1,y2 + 1]))
#l=legend('Latitude of surface zonal wind zero crossing','Latitude of surface zonal wind zero crossing from annual mean field','Latitude of 850 hPa zonal wind zero crossing from annual mean field','Latitude of surface zonal wind zero crossing from annual mean of monthly metric')
#set(l,'box','off','location','north')
#subplot('212')
#plot(time,Phi_uas_sh,'-','linewidth',2,'color',green_color)
#hold('on')
#plot(concat([arange(y1,y2)]) + 0.5,Phi_uas_sh_ANN,'-','linewidth',2,'color',blue_color)
#plot(concat([arange(y1,y2)]) + 0.5,Phi_Uas_sh_ANN,'-','linewidth',2,'color',red_color)
#plot(concat([arange(y1,y2)]) + 0.5,TropD_Calculate_Mon2Season(Phi_uas_sh, np.arange(12)),'-k','linewidth',2)
#set(gca,'fontsize',12,'linewidth',2,'tickdir','out','box','off','xtick', np.arange(12)(1980,2020,5)]),'ytick',Ytick2,'yticklabels',YtickLabels2)
#xlim(concat([y1,y2 + 1]))
#xlabel('Year','fontsize',14)
#ylabel('SH uas zero-crossing')

## 8) PSL -- Sea-level Pressure Maximum
f_ps = netcdf.netcdf_file('../ValidationData/psl.nc','r')
ps = f_ps.variables['psl'][:]
lat = f_ps.variables['lat'][:]

#Change axes of u to be [time, lat]
ps = np.transpose(ps, (1,0))

ps_DJF = TropD_Calculate_Mon2Season(ps, np.array([0,1,11]))
ps_JJA = TropD_Calculate_Mon2Season(ps, np.array([5,6,7]))

Phi_ps_DJF_nh = np.zeros((np.shape(ps_DJF)[0],1))
Phi_ps_JJA_nh = np.zeros((np.shape(ps_JJA)[0],1))
Phi_ps_DJF_sh = np.zeros((np.shape(ps_DJF)[0],1))
Phi_ps_JJA_sh = np.zeros((np.shape(ps_JJA)[0],1))

for j in range(np.shape(ps_DJF)[0]):
  Phi_ps_DJF_sh[j], Phi_ps_DJF_nh[j] = TropD_Metric_PSL(ps_DJF[j,:], lat)

for j in range(np.shape(ps_JJA)[0]):
  Phi_ps_JJA_sh[j], Phi_ps_JJA_nh[j] = TropD_Metric_PSL(ps_JJA[j,:], lat)

#figure
#subplot('211')
#plot(concat([arange(y1,y2)]) + 0.5,Phi_ps_DJF_nh,'-','linewidth',2,'color',green_color)
#hold('on')
#plot(concat([arange(y1,y2)]) + 0.5,Phi_ps_JJA_nh,'-','linewidth',2,'color',blue_color)
#set(gca,'fontsize',12,'linewidth',2,'tickdir','out','box','off','xtick', np.arange(12)(1980,2020,5)]),'ytick',Ytick1,'yticklabels',YtickLabels1)
#ylabel('NH max psl latitude')
#xlim(concat([y1,y2 + 1]))
#l=legend('Latitude of max sea-level pressure during DJF','Latitude of max sea-level pressure during JJA')
#set(l,'box','off','location','south')
#subplot('212')
#plot(concat([arange(y1,y2)]) + 0.5,Phi_ps_DJF_sh,'-','linewidth',2,'color',green_color)
#hold('on')
#plot(concat([arange(y1,y2)]) + 0.5,Phi_ps_JJA_sh,'-','linewidth',2,'color',blue_color)
#set(gca,'fontsize',12,'linewidth',2,'tickdir','out','box','off','xtick', np.arange(12)(1980,2020,5)]),'ytick',Ytick1,'yticklabels',YtickLabels1)
#ylabel('SH max psl latitude')
#xlim(concat([y1,y2 + 1]))


## 9) Compare annual mean metrics
#Psi500
f_V = netcdf.netcdf_file('../ValidationData/va.nc','r')
V = f_V.variables['va'][:]
lat = f_V.variables['lat'][:]
lev = f_V.variables['lev'][:]

#Change axes of V to be [time, lat]
V = np.transpose(V, (1,0))

V_ANN = TropD_Calculate_Mon2Season(V, np.arange(12))

Phi_ps_nh_ANN = np.zeros((np.shape(V_ANN)[0],1))
Phi_ps_sh_ANN = np.zeros((np.shape(V_ANN)[0],1))

for j in range(np.shape(V_ANN)[0]):
  Phi_psi_sh_ANN[j], Phi_psi_nh_ANN[j] = TropD_Metric_PSI(V_ANN[j,:,:], lat, lev)

# Tropopause break
f_T = netcdf.netcdf_file('../ValidationData/ta.nc','r')
T = f_T.variables['ta'][:]

#Change axes of T to be [time, lat]
T = np.transpose(T, (1,0))

T_ANN = TropD_Calculate_Mon2Season(T, np.arange(12))

Phi_tpb_nh_ANN = np.zeros((np.shape(T_ANN)[0],1))
Phi_tpb_sh_ANN = np.zeros((np.shape(T_ANN)[0],1))

for j in range(np.shape(T_ANN)[0]):
  Phi_tpb_sh_ANN[j], Phi_tpb_nh_ANN[j] = TropD_Metric_TPB(T_ANN[j,:,:], lat, lev)

# Surface pressure max
f_ps = netcdf.netcdf_file('../ValidationData/psl.nc','r')
ps = f_ps.variables['psl'][:]

#Change axes of ps to be [time, lat]
ps = np.transpose(ps, (1,0))

ps_ANN = TropD_Calculate_Mon2Season(ps, np.arange(12))

Phi_ps_nh_ANN = np.zeros((np.shape(ps_ANN)[0],1))

Phi_ps_sh_ANN = np.zeros((np.shape(ps_ANN)[0],1))

for j in range(np.shape(ps_ANN)[0]):
  Phi_ps_sh_ANN[j], Phi_ps_nh_ANN[j] = TropD_Metric_PSL(ps_ANN[j,:,:], lat)

# Eddy driven jet
f_U = netcdf.netcdf_file('../ValidationData/ua.nc','r')
U = f_ps.variables['ua'][:]

#Change axes of U to be [time, lat]
U = np.transpose(U, (1,0))

U_ANN = TropD_Calculate_Mon2Season(U, np.arange(12))

Phi_edj_nh_ANN = np.zeros((np.shape(U_ANN)[0],1))

Phi_edj_sh_ANN = np.zeros((np.shape(U_ANN)[0],1))

for j in range(np.shape(U_ANN)[0]):
  Phi_edj_sh_ANN[j], Phi_edj_nh_ANN[j] = TropD_Metric_EDJ(U_ANN[j,:,:], lat, lev)

# Subtropical jet
Phi_stj_nh_ANN = np.zeros((np.shape(U_ANN)[0],1))

Phi_stj_sh_ANN = np.zeros((np.shape(U_ANN)[0],1))

for j in range(np.shape(U_ANN)[0]):
  Phi_stj_sh_ANN[j], Phi_stj_nh_ANN[j] = TropD_Metric_STJ(U_ANN[j,:,:], lat, lev)

# OLR
f_olr = - ncread('../ValidationData/rlnt.nc','r')
olr = f_olr.variables['rlnt'][:]

#Change axes of olr to be [time, lat]
olr = np.transpose(olr, (1,0))

olr_ANN = TropD_Calculate_Mon2Season(olr, np.arange(12))

Phi_olr_nh_ANN = np.zeros((np.shape(olr_ANN)[0],1))

Phi_olr_sh_ANN = np.zeros((np.shape(olr_ANN)[0],1))

for j in range(np.shape(olr_ANN)[0]):
  Phi_olr_sh_ANN[j], Phi_olr_nh_ANN[j] = TropD_Metric_OLR(olr_ANN[j,:,:], lat)

# P minus E
f_pr = netcdf.netcdf_file('../ValidationData/pr.nc','r')
pr = f_pr.variables['pr'][:]

L = 2510400.0

f_er = - ncread('../ValidationData/hfls.nc','r')
er = f_er.variables['hfls'][:] / L

#Change axes of ps and er to be [time, lat]
pr = np.transpose(pr, (1,0))
er = np.transpose(er, (1,0))

PE = pr - er
PE_ANN = TropD_Calculate_Mon2Season(PE, np.arange(12))

Phi_pe_nh_ANN = np.zeros((np.shape(PE_ANN)[0],1))

Phi_pe_sh_ANN = np.zeros((np.shape(PE_ANN)[0],1))

for j in range(np.shape(PE_ANN)[0]):
  Phi_pe_sh_ANN[j], Phi_pe_nh_ANN[j] = TropD_Metric_PE(PE_ANN[j,:,:], lat)

# Surface winds
f_uas = netcdf.netcdf_file('../ValidationData/uas.nc','r')
uas = f_ps.variables['uas'][:]

#Change axes of uas to be [time, lat]
uas = np.transpose(uas, (1,0))

uas_ANN = TropD_Calculate_Mon2Season(uas, np.arange(12))

Phi_uas_nh_ANN = np.zeros((np.shape(uas_ANN)[0],1))

Phi_uas_sh_ANN = np.zeros((np.shape(uas_ANN)[0],1))

for j in range(np.shape(uas_ANN)[0]):
  Phi_uas_sh_ANN[j], Phi_uas_nh_ANN[j] = TropD_Metric_UAS(uas_ANN[j,:,:], lat)


#figure
#subplot('211')
#plot(concat([arange(y1,y2)]),Phi_psi_nh_ANN,'-','linewidth',2,'color',concat([0,0,0]))
#hold('on')
#plot(concat([arange(y1,y2)]),Phi_tpb_nh_ANN,'-','linewidth',2,'color',green_color)
#plot(concat([arange(y1,y2)]),Phi_edj_nh_ANN,'-','linewidth',2,'color',blue_color)
#plot(concat([arange(y1,y2)]),Phi_stj_nh_ANN,'-','linewidth',2,'color',red_color)
#plot(concat([arange(y1,y2)]),Phi_olr_nh_ANN,'-','linewidth',2,'color',lightblue_color)
#plot(concat([arange(y1,y2)]),Phi_pe_nh_ANN,'-','linewidth',2,'color',orange_color)
#plot(concat([arange(y1,y2)]),Phi_uas_nh_ANN,'-','linewidth',2,'color',purple_color)
#plot(concat([arange(y1,y2)]),Phi_ps_nh_ANN,'--','linewidth',2,'color',maroon_color)
#set(gca,'fontsize',12,'linewidth',2,'tickdir','out','box','off','xtick', np.arange(12)(1980,2020,5)]),'xticklabels',cellarray(['']),'ytick',Ytick5,'yticklabels',YtickLabels5)
#l=legend('PSI','TPB','EDJ','STJ','OLR','P-E','UAS','PSL')
#set(l,'box','off','location','eastoutside')
#ylabel('NH HC edge')
#xlim(concat([y1,y2 + 1]))
#ylim(concat([25,50]))
#subplot('212')
#plot(concat([arange(y1,y2)]),Phi_psi_sh_ANN,'-','linewidth',2,'color',concat([0,0,0]))
#hold('on')
#plot(concat([arange(y1,y2)]),Phi_tpb_sh_ANN,'-','linewidth',2,'color',green_color)
#plot(concat([arange(y1,y2)]),Phi_edj_sh_ANN,'-','linewidth',2,'color',blue_color)
#plot(concat([arange(y1,y2)]),Phi_stj_sh_ANN,'-','linewidth',2,'color',red_color)
#plot(concat([arange(y1,y2)]),Phi_olr_sh_ANN,'-','linewidth',2,'color',lightblue_color)
#plot(concat([arange(y1,y2)]),Phi_pe_sh_ANN,'-','linewidth',2,'color',orange_color)
#plot(concat([arange(y1,y2)]),Phi_uas_sh_ANN,'-','linewidth',2,'color',purple_color)
#plot(concat([arange(y1,y2)]),Phi_ps_sh_ANN,'--','linewidth',2,'color',maroon_color)
#set(gca,'fontsize',12,'linewidth',2,'tickdir','out','box','off','xtick', np.arange(12)(1980,2020,5)]),'ytick',Ytick5,'yticklabels',YtickLabels5)
#l=legend('PSI','TPB','EDJ','STJ','OLR','P-E','UAS','PSL')
#set(l,'box','off','location','eastoutside')
#xlim(concat([y1,y2 + 1]))
#xlabel('Year','fontsize',14)
#ylabel('SH HC edge')

## 10) Validate metrics
#Psi500
V = netcdf.netcdf_file('../ValidationData/va.nc','va')

lat = netcdf.netcdf_file('../ValidationData/va.nc','lat')

lev = netcdf.netcdf_file('../ValidationData/va.nc','lev')

V_ANN = TropD_Calculate_Mon2Season(V, np.arange(12))

Phi_psi_nh = np.zeros((np.shape(V)[0],1))

Phi_psi_sh = np.zeros((np.shape(V)[0],1))

Phi_psi_nh_ANN = np.zeros((np.shape(V_ANN)[0],1))

Phi_psi_sh_ANN = np.zeros((np.shape(V_ANN)[0],1))

for j in range(np.shape(V)[0]):
  Phi_psi_sh[j], Phi_psi_nh[j] = TropD_Metric_PSI(V[j,:,:], lat, lev)

for j in range(np.shape(V_ANN)[0]):
  Phi_psi_sh_ANN[j], Phi_psi_nh_ANN[j] = TropD_Metric_PSI(V_ANN[j,:,:], lat, lev)

Phi_nh = netcdf.netcdf_file('../ValidationMetrics/PSI_ANN.nc','PSI_NH')
Phi_sh = netcdf.netcdf_file('../ValidationMetrics/PSI_ANN.nc','PSI_SH')
if logical_not((std(Phi_nh - Phi_psi_nh_ANN) < 1e-10 and std(Phi_sh - Phi_psi_sh_ANN) < 1e-10)):
  disp('Warning: annual-mean Validation and calculated PSI metrics are NOT equal!')
else:
  disp('OK. Annual-mean Validation and calculated PSI metrics are the same!')

Phi_nh = netcdf.netcdf_file('../ValidationMetrics/PSI.nc','PSI_NH')
Phi_sh = netcdf.netcdf_file('../ValidationMetrics/PSI.nc','PSI_SH')
if logical_not((std(Phi_nh - Phi_psi_nh) < 1e-10 and std(Phi_sh - Phi_psi_sh) < 1e-10)):
  disp('Warning: monthly Validation and calculated PSI metrics are NOT equal!')
else:
  disp('OK. Monthly Validation and calculated PSI metrics are the same!')

# Tropopause break
T = netcdf.netcdf_file('../ValidationData/ta.nc','ta')

T_ANN = TropD_Calculate_Mon2Season(T, np.arange(12))

Phi_tpb_nh=np.zeros((np.shape(T)[0],1))

Phi_tpb_sh=np.zeros((np.shape(T)[0],1))

Phi_tpb_nh_ANN = np.zeros((np.shape(T_ANN)[0],1))

Phi_tpb_sh_ANN = np.zeros((np.shape(T_ANN)[0],1))

for j in range(np.shape(T)[0]):
  Phi_tpb_sh[j], Phi_tpb_nh[j] = TropD_Metric_TPB(T[j,:,:], lat, lev)

for j in range(np.shape(T_ANN)[0]):
  Phi_tpb_sh_ANN[j], Phi_tpb_nh_ANN[j] = TropD_Metric_TPB(T_ANN[j,:,:], lat, lev)

Phi_nh = netcdf.netcdf_file('../ValidationMetrics/TPB_ANN.nc','TPB_NH')
Phi_sh = netcdf.netcdf_file('../ValidationMetrics/TPB_ANN.nc','TPB_SH')
if logical_not((std(Phi_nh - Phi_tpb_nh_ANN) < 1e-10 and std(Phi_sh - Phi_tpb_sh_ANN) < 1e-10)):
  disp('Warning: annual-mean Validation and calculated TPB metrics are NOT equal!')
else:
  disp('OK. Annual-mean Validation and calculated TPB metrics are the same!')

Phi_nh = netcdf.netcdf_file('../ValidationMetrics/TPB.nc','TPB_NH')
Phi_sh = netcdf.netcdf_file('../ValidationMetrics/TPB.nc','TPB_SH')
if logical_not((std(Phi_nh - Phi_tpb_nh) < 1e-10 and std(Phi_sh - Phi_tpb_sh) < 1e-10)):
  disp('Warning: monthly Validation and calculated TPB metrics are NOT equal!')
else:
  disp('OK. Monthly Validation and calculated TPB metrics are the same!')

# Surface pressure max (Invalid in NH)
ps = netcdf.netcdf_file('../ValidationData/psl.nc','psl')

ps_DJF=TropD_Calculate_Mon2Season(ps,concat([1,2,12]))

ps_JJA=TropD_Calculate_Mon2Season(ps, np.arange(12)(6,8)]))

ps_MAM=TropD_Calculate_Mon2Season(ps, np.arange(12)(3,5)]))

ps_SON = TropD_Calculate_Mon2Season(ps, np.arange(12)(9,11)]))

Phi_ps_sh_DJF=np.zeros((np.shape(ps_DJF)[0],1))
Phi_ps_sh_JJA=np.zeros((np.shape(ps_JJA)[0],1))
Phi_ps_sh_MAM=np.zeros((np.shape(ps_MAM)[0],1))
Phi_ps_sh_SON = np.zeros((np.shape(ps_SON)[0],1))
Phi_ps_nh_DJF=np.zeros((np.shape(ps_DJF)[0],1))
Phi_ps_nh_JJA=np.zeros((np.shape(ps_JJA)[0],1))
Phi_ps_nh_MAM=np.zeros((np.shape(ps_MAM)[0],1))
Phi_ps_nh_SON = np.zeros((np.shape(ps_SON)[0],1))
for j in range(np.shape(ps_DJF)[0]):
  Phi_ps_sh_DJF[j], Phi_ps_nh_DJF[j] = TropD_Metric_PSL(ps_DJF[j,:,:], lat)

for j in range(np.shape(ps_JJA)[0]):
  Phi_ps_sh_JJA[j], Phi_ps_nh_JJA[j] = TropD_Metric_PSL(ps_JJA[j,:,:], lat)

for j in range(np.shape(ps_MAM)[0]):
  Phi_ps_sh_MAM[j], Phi_ps_nh_MAM[j] = TropD_Metric_PSL(ps_MAM[j,:,:], lat)

for j in range(np.shape(ps_SON)[0]):
  Phi_ps_sh_SON[j], Phi_ps_nh_SON[j] = TropD_Metric_PSL(ps_SON[j,:,:], lat)

Phi_sh = netcdf.netcdf_file('../ValidationMetrics/PSL_DJF.nc','PSL_SH')
Phi_nh = netcdf.netcdf_file('../ValidationMetrics/PSL_DJF.nc','PSL_NH')
if logical_not((std(Phi_sh - Phi_ps_sh_DJF) < 1e-10)) or logical_not((std(Phi_nh - Phi_ps_nh_DJF) < 1e-10)):
  disp('Warning: DJF Validation and calculated PSL metrics are NOT equal!')
else:
  disp('OK. DJF Validation and calculated PSL metrics are the same!')

Phi_sh = netcdf.netcdf_file('../ValidationMetrics/PSL_JJA.nc','PSL_SH')
Phi_nh = netcdf.netcdf_file('../ValidationMetrics/PSL_JJA.nc','PSL_NH')
if logical_not((std(Phi_sh - Phi_ps_sh_JJA) < 1e-10)) or logical_not((std(Phi_nh - Phi_ps_nh_JJA) < 1e-10)):
  disp('Warning: JJA Validation and calculated PSL metrics are NOT equal!')
else:
  disp('OK. JJA Validation and calculated PSL metrics are the same!')

Phi_sh = netcdf.netcdf_file('../ValidationMetrics/PSL_MAM.nc','PSL_SH')
Phi_nh = netcdf.netcdf_file('../ValidationMetrics/PSL_MAM.nc','PSL_NH')
if logical_not((std(Phi_sh - Phi_ps_sh_MAM) < 1e-10)) or logical_not((std(Phi_nh - Phi_ps_nh_MAM) < 1e-10)):
  disp('Warning: MAM Validation and calculated PSL metrics are NOT equal!')
else:
  disp('OK. MAM Validation and calculated PSL metrics are the same!')

Phi_sh = netcdf.netcdf_file('../ValidationMetrics/PSL_SON.nc','PSL_SH')
Phi_nh = netcdf.netcdf_file('../ValidationMetrics/PSL_SON.nc','PSL_NH')
if logical_not((std(Phi_sh - Phi_ps_sh_SON) < 1e-10)) or logical_not((std(Phi_nh - Phi_ps_nh_SON) < 1e-10)):
  disp('Warning: SON Validation and calculated PSL metrics are NOT equal!')
else:
  disp('OK. SON Validation and calculated PSL metrics are the same!')

# Eddy driven jet
U = netcdf.netcdf_file('../ValidationData/ua.nc','ua')

U_ANN = TropD_Calculate_Mon2Season(U, np.arange(12))

Phi_edj_nh=np.zeros((np.shape(U)[0],1))

Phi_edj_sh=np.zeros((np.shape(U)[0],1))

Phi_edj_nh_ANN = np.zeros((np.shape(U_ANN)[0],1))

Phi_edj_sh_ANN = np.zeros((np.shape(U_ANN)[0],1))

for j in range(np.shape(U)[0]):
  Phi_edj_sh[j], Phi_edj_nh[j] = TropD_Metric_EDJ(U[j,:,:], lat, lev)

for j in range(np.shape(U_ANN)[0]):
  Phi_edj_sh_ANN[j], Phi_edj_nh_ANN[j] = TropD_Metric_EDJ(U_ANN[j,:,:], lat, lev)

Phi_nh = netcdf.netcdf_file('../ValidationMetrics/EDJ_ANN.nc','EDJ_NH')
Phi_sh = netcdf.netcdf_file('../ValidationMetrics/EDJ_ANN.nc','EDJ_SH')
if logical_not((std(Phi_nh - Phi_edj_nh_ANN) < 1e-10 and std(Phi_sh - Phi_edj_sh_ANN) < 1e-10)):
  disp('Warning: annual-mean Validation and calculated EDJ metrics are NOT equal!')
else:
  disp('OK. Annual-mean Validation and calculated EDJ metrics are the same!')

Phi_nh = netcdf.netcdf_file('../ValidationMetrics/EDJ.nc','EDJ_NH')
Phi_sh = netcdf.netcdf_file('../ValidationMetrics/EDJ.nc','EDJ_SH')
if logical_not((std(Phi_nh - Phi_edj_nh) < 1e-10 and std(Phi_sh - Phi_edj_sh) < 1e-10)):
  disp('Warning: monthly Validation and calculated EDJ metrics are NOT equal!')
else:
  disp('OK. Monthly Validation and calculated EDJ metrics are the same!')

# Subtropical jet
Phi_stj_nh=np.zeros((np.shape(U)[0],1))

Phi_stj_sh=np.zeros((np.shape(U)[0],1))

Phi_stj_nh_ANN = np.zeros((np.shape(U_ANN)[0],1))

Phi_stj_sh_ANN = np.zeros((np.shape(U_ANN)[0],1))

for j in range(np.shape(U)[0]):
  Phi_stj_sh[j], Phi_stj_nh[j] = TropD_Metric_STJ(U[j,:,:], lat, lev)

for j in range(np.shape(U_ANN)[0]):
  Phi_stj_sh_ANN[j], Phi_stj_nh_ANN[j] = TropD_Metric_STJ(U_ANN[j,:,:], lat, lev)

Phi_nh = netcdf.netcdf_file('../ValidationMetrics/STJ_ANN.nc','STJ_NH')
Phi_sh = netcdf.netcdf_file('../ValidationMetrics/STJ_ANN.nc','STJ_SH')
if logical_not((std(Phi_nh - Phi_stj_nh_ANN) < 1e-10 and std(Phi_sh - Phi_stj_sh_ANN) < 1e-10)):
  disp('Warning: annual-mean Validation and calculated STJ metrics are NOT equal!')
else:
  disp('OK. Annual-mean Validation and calculated STJ metrics are the same!')

Phi_nh = netcdf.netcdf_file('../ValidationMetrics/STJ.nc','STJ_NH')
Phi_sh = netcdf.netcdf_file('../ValidationMetrics/STJ.nc','STJ_SH')
if logical_not((std(Phi_nh - Phi_stj_nh) < 1e-10 and std(Phi_sh - Phi_stj_sh) < 1e-10)):
  disp('Warning: monthly Validation and calculated STJ metrics are NOT equal!')
else:
  disp('OK. Monthly Validation and calculated STJ metrics are the same!')

# OLR
olr=- ncread('../ValidationData/rlnt.nc','rlnt')

olr_ANN = TropD_Calculate_Mon2Season(olr, np.arange(12))

Phi_olr_nh=np.zeros((np.shape(olr)[0],1))

Phi_olr_sh=np.zeros((np.shape(olr)[0],1))

Phi_olr_nh_ANN = np.zeros((np.shape(olr_ANN)[0],1))

Phi_olr_sh_ANN = np.zeros((np.shape(olr_ANN)[0],1))

for j in range(np.shape(olr)[0]):
  Phi_olr_sh[j], Phi_olr_nh[j] = TropD_Metric_OLR(olr[j,:,:], lat)

for j in range(np.shape(olr_ANN)[0]):
  Phi_olr_sh_ANN[j], Phi_olr_nh_ANN[j] = TropD_Metric_OLR(olr_ANN[j,:,:], lat)

Phi_nh = netcdf.netcdf_file('../ValidationMetrics/OLR_ANN.nc','OLR_NH')
Phi_sh = netcdf.netcdf_file('../ValidationMetrics/OLR_ANN.nc','OLR_SH')
if logical_not((std(Phi_nh - Phi_olr_nh_ANN) < 1e-10 and std(Phi_sh - Phi_olr_sh_ANN) < 1e-10)):
  disp('Warning: annual-mean Validation and calculated OLR metrics are NOT equal!')
else:
  disp('OK. Annual-mean Validation and calculated OLR metrics are the same!')

Phi_nh = netcdf.netcdf_file('../ValidationMetrics/OLR.nc','OLR_NH')
Phi_sh = netcdf.netcdf_file('../ValidationMetrics/OLR.nc','OLR_SH')
if logical_not((std(Phi_nh - Phi_olr_nh) < 1e-10 and std(Phi_sh - Phi_olr_sh) < 1e-10)):
  disp('Warning: monthly Validation and calculated OLR metrics are NOT equal!')
else:
  disp('OK. Monthly Validation and calculated OLR metrics are the same!')

# P minus E
pr = netcdf.netcdf_file('../ValidationData/pr.nc','pr')

L=2510400.0

er=- ncread('../ValidationData/hfls.nc','hfls') / L

PE=pr - er
PE_ANN = TropD_Calculate_Mon2Season(PE, np.arange(12))

Phi_pe_nh=np.zeros((np.shape(PE)[0],1))

Phi_pe_sh=np.zeros((np.shape(PE)[0],1))

Phi_pe_nh_ANN = np.zeros((np.shape(PE_ANN)[0],1))

Phi_pe_sh_ANN = np.zeros((np.shape(PE_ANN)[0],1))

for j in range(np.shape(PE)[0]):
  Phi_pe_sh[j], Phi_pe_nh[j] = TropD_Metric_PE(PE[j,:,:], lat)

for j in range(np.shape(PE_ANN)[0]):
  Phi_pe_sh_ANN[j], Phi_pe_nh_ANN[j] = TropD_Metric_PE(PE_ANN[j,:,:], lat)

Phi_nh = netcdf.netcdf_file('../ValidationMetrics/PE_ANN.nc','PE_NH')
Phi_sh = netcdf.netcdf_file('../ValidationMetrics/PE_ANN.nc','PE_SH')
if logical_not((std(Phi_nh - Phi_pe_nh_ANN) < 1e-10 and std(Phi_sh - Phi_pe_sh_ANN) < 1e-10)):
  disp('Warning: annual-mean Validation and calculated P-E metrics are NOT equal!')
else:
  disp('OK. Annual-mean Validation and calculated P-E metrics are the same!')

Phi_nh = netcdf.netcdf_file('../ValidationMetrics/PE.nc','PE_NH')
Phi_sh = netcdf.netcdf_file('../ValidationMetrics/PE.nc','PE_SH')
if logical_not((std(Phi_nh - Phi_pe_nh) < 1e-10 and std(Phi_sh - Phi_pe_sh) < 1e-10)):
  disp('Warning: monthly Validation and calculated P-E metrics are NOT equal!')
else:
  disp('OK. Monthly Validation and calculated P-E metrics are the same!')

# Surface winds
uas = netcdf.netcdf_file('../ValidationData/uas.nc','uas')

uas_ANN = TropD_Calculate_Mon2Season(uas, np.arange(12))

Phi_uas_nh=np.zeros((np.shape(uas)[0],1))

Phi_uas_sh=np.zeros((np.shape(uas)[0],1))

Phi_uas_nh_ANN = np.zeros((np.shape(uas_ANN)[0],1))

Phi_uas_sh_ANN = np.zeros((np.shape(uas_ANN)[0],1))

for j in range(np.shape(uas)[0]):
  Phi_uas_sh[j], Phi_uas_nh[j] = TropD_Metric_UAS(uas[j,:,:], lat)

for j in range(np.shape(uas_ANN)[0]):
  Phi_uas_sh_ANN[j], Phi_uas_nh_ANN[j] = TropD_Metric_UAS(uas_ANN[j,:,:], lat)

Phi_nh = netcdf.netcdf_file('../ValidationMetrics/UAS_ANN.nc','UAS_NH')
Phi_sh = netcdf.netcdf_file('../ValidationMetrics/UAS_ANN.nc','UAS_SH')
if logical_not((std(Phi_nh - Phi_uas_nh_ANN) < 1e-10 and std(Phi_sh - Phi_uas_sh_ANN) < 1e-10)):
  disp('Warning: annual-mean Validation and calculated UAS metrics are NOT equal!')
else:
  disp('OK. Annual-mean Validation and calculated UAS metrics are the same!')

Phi_nh = netcdf.netcdf_file('../ValidationMetrics/UAS.nc','UAS_NH')
Phi_sh = netcdf.netcdf_file('../ValidationMetrics/UAS.nc','UAS_SH')
if logical_not((std(Phi_nh - Phi_uas_nh) < 1e-10 and std(Phi_sh - Phi_uas_sh) < 1e-10)):
  disp('Warning: monthly Validation and calculated UAS metrics are NOT equal!')
else:
  disp('OK. Monthly Validation and calculated UAS metrics are the same!')

