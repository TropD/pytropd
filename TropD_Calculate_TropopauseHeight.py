    
def TropD_Calculate_TropopauseHeight(T ,P, Z=None,*args,**kwargs):
  ''' Calculate the Tropopause Height in isobaric coordinates 

  Written by Ori Adam Mar.17.2017 as part of TropD package
  Converted to python by Alison Ming Jul.4.2017

  Based on the method described in Birner (2010), according to the WMO definition: first level at 
  which the lapse rate <= 2K/km and for which the lapse rate <= 2K/km in all levels at least 2km 
  above the found level 

  Positional arguments:
  T -- Temperature array of dimensions (latitude, levels) on (longitude, latitude, levels)
  P -- pressure levels in hPa

  Keyword arguments:
  Z (optional) -- geopotential height [m] or any field with the same dimensions as T

  Output:
  Pt(lat) or Pt(lon,lat) = tropopause level in hPa 
  Ht(lat) or Ht(lon,lat) = the field Z evaluated at the tropopause. For Z=geopotential heigt, Ht is the tropopause altitude in m '''


  Rd=287.04
  Cpd=1005.7
    
    Pk=repmat(reshape((dot(P,100)) ** k,1,1,length(P)),concat([size(T,1),size(T,2),1]))
    Pk2=(Pk(arange(),arange(),arange(1,end() - 1)) + Pk(arange(),arange(),arange(2,end()))) / 2
    
    T2=(T(arange(),arange(),arange(1,end() - 1)) + T(arange(),arange(),arange(2,end()))) / 2
    Pk1=squeeze(Pk2(1,1,arange()))
    Gamma=reshape(multiply(multiply((T(arange(),arange(),arange(2,end())) - T(arange(),arange(),arange(1,end() - 1))) / (Pk(arange(),arange(),arange(2,end())) - Pk(arange(),arange(),arange(1,end() - 1))),Pk2) / T2,Factor),dot(size(T,1),size(T,2)),size(T,3) - 1)
    
    T2=reshape(T2,dot(size(T,1),size(T,2)),size(T,3) - 1)
    
    Pt=zeros(dot(size(T,1),size(T,2)),1)
    for j in arange(1,size(Gamma,1)).reshape(-1):
        G1=interp1(Pk1,double(Gamma(j,arange()).T),PI,'linear','extrap')
        T1=interp1(Pk1,double(T2(j,arange()).T),PI,'linear','extrap')
        idx=find(G1 <= logical_and(2,PI) < logical_and((dot(550,100)) ** k,PI) > (dot(75,100)) ** k)
        Pidx=PI(idx)
        if isempty(Pidx):
            Pt[j]=nan
        else:
            for c in arange(1,length(Pidx)).reshape(-1):
                dpk_2km=multiply(dot(dot(- 2000,k),g) / Rd / T1(c),Pidx(c))
                idx2=dsearchn(Pidx(arange(c,end())).T,Pidx(c) + dpk_2km)
                if sum(G1(arange(idx(c),idx(c) + idx2 - 1)) <= 2) == idx2:
                    Pt[j]=Pidx(c)
                    break
                else:
                    continue
    
    Pt=Pt ** (1 / k) / 100
    if nargin > 2:
        if isequal(size(T),size(Z)):
            Zt=reshape(Z,dot(size(Z,1),size(Z,2)),size(Z,3))
            Ht=zeros(dot(size(T,1),size(T,2)),1)
            for j in arange(1,size(Ht,1)).reshape(-1):
                Ht[j]=interp1(P,Zt(j,arange()).T,Pt(j))
            Ht=reshape(Ht,size(T,1),size(T,2))
        else:
            disp('TropD_Calculate_TropopauseHeight: ERROR :  T and Z must have the same dimensions')
            Ht=[]
    
    Pt=reshape(Pt,size(T,1),size(T,2))
    
