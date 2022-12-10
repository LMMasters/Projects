"""## Quark and Gluon Nsubs

A dataset consisting of 45 $N$-subjettiness observables for 100k quark and 
gluon jets generated with Pythia 8.230. Following [1704.08249](https:
//arxiv.org/abs/1704.08249), the observables are in the following order:

\[\{\tau_1^{(\beta=0.5)},\tau_1^{(\beta=1.0)},\tau_1^{(\beta=2.0)},
\tau_2^{(\beta=0.5)},\tau_2^{(\beta=1.0)},\tau_2^{(\beta=2.0)},
\ldots,
\tau_{15}^{(\beta=0.5)},\tau_{15}^{(\beta=1.0)},\tau_{15}^{(\beta=2.0)}\}.\]

The dataset contains two members: `'X'` which is a numpy array of the nsubs
that has shape `(100000,45)` and `'y'` which is a numpy array of quark/gluon 
labels (quark=`1` and gluon=`0`).
"""

#   ____   _____          _   _  _____ _    _ ____   _____
#  / __ \ / ____|        | \ | |/ ____| |  | |  _ \ / ____|
# | |  | | |  __         |  \| | (___ | |  | | |_) | (___
# | |  | | | |_ |        | . ` |\___ \| |  | |  _ < \___ \
# | |__| | |__| | ______ | |\  |____) | |__| | |_) |____) |
#  \___\_\\_____||______||_| \_|_____/ \____/|____/|_____/

# EnergyFlow - Python package for high-energy particle physics.
# Copyright (C) 2017-2020 Patrick T. Komiske III and Eric Metodiev

from __future__ import absolute_import, division, print_function

import sys, os
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle
from matplotlib import image
import h5py 
from tqdm import tqdm

__all__ = ['load']


def load(fname, num_data=50000, cache_dir='~/.energyflow'):

    print ('Unpacking file', fname) 
    
    hf = h5py.File(fname, 'r')

    y = hf.get('type')
    nobj = hf.get('nobj')
    reg = hf.get('reg')

    y = np.array(y, dtype=np.float32)
    nobj = np.array(nobj)
    reg = np.array(reg)


    #Filter 
    """
    ttbarWW : 1
    ttbarZ_lep : 2
    ttbarW_lep : 3
    ttbarHiggs_lep : 4
    4top_1jet : 5
    """

    indx=(y==5)

    ftop_nobj = nobj[indx]
    ftop_reg = reg[indx]

    indx=(y==3)

    ttbarW_nobj = nobj[indx]
    ttbarW_reg = reg[indx]

    indx=(y==2)

    ttbarZ_nobj = nobj[indx]
    ttbarZ_reg = reg[indx]

    indx=(y==1)

    ttbarWW_nobj = nobj[indx]
    ttbarWW_reg = reg[indx]

    indx=(y==4)

    ttbarH_nobj = nobj[indx]
    ttbarH_reg = reg[indx]

    ftop_type = np.ones(ftop_nobj.shape[0])
    ttbarW_type = np.zeros(ttbarW_nobj.shape[0])
    ttbarZ_type = np.zeros(ttbarZ_nobj.shape[0])
    ttbarH_type = np.zeros(ttbarH_nobj.shape[0])
    ttbarWW_type = np.zeros(ttbarWW_nobj.shape[0])

    #Choose max. 50k events per process excepting 4tops which 
    #is then 200k to get a balanced dataset
    max_num_events = num_data #50000
    if ftop_nobj.shape[0] > 4.*max_num_events:
     ftop_nobj = ftop_nobj[:4*max_num_events]
     ftop_reg = ftop_reg[:4*max_num_events]
     ftop_type = np.ones(4*max_num_events)

    if ttbarW_nobj.shape[0] > max_num_events:
     ttbarW_nobj = ttbarW_nobj[:max_num_events]
     ttbarW_reg = ttbarW_reg[:max_num_events]
     ttbarW_type = np.zeros(max_num_events)

    if ttbarZ_nobj.shape[0] > max_num_events:
     ttbarZ_nobj = ttbarZ_nobj[:max_num_events]
     ttbarZ_reg = ttbarZ_reg[:max_num_events]
     ttbarZ_type = np.zeros(max_num_events)

    if ttbarH_nobj.shape[0] > max_num_events:
     ttbarH_nobj = ttbarH_nobj[:max_num_events]
     ttbarH_reg = ttbarH_reg[:max_num_events]
     ttbarH_type = np.zeros(max_num_events)
     print(ttbarH_nobj.shape)

    if ttbarWW_nobj.shape[0] > max_num_events:
     ttbarWW_nobj = ttbarWW_nobj[:max_num_events]
     ttbarWW_reg = ttbarWW_reg[:max_num_events]
     ttbarWW_type = np.zeros(max_num_events)

    #Now I have to concatenate and suffle(perm)
    _nobj = np.vstack((ftop_nobj, ttbarW_nobj, ttbarZ_nobj, ttbarH_nobj, ttbarWW_nobj)) 
    _reg = np.vstack((ftop_reg, ttbarW_reg, ttbarZ_reg, ttbarH_reg, ttbarWW_reg))
    _y = np.concatenate((ftop_type, ttbarW_type, ttbarZ_type, ttbarH_type, ttbarWW_type)) 

    #randomize
    perm = np.random.permutation(_nobj.shape[0])

    nobj = _nobj[perm] 
    reg = _reg[perm] 
    y = _y[perm] 

    #print(type(nobj), type(reg), nobj[0], reg[0])

    X = np.concatenate((nobj, reg), axis = 1) 

    #if num_data > -1:
    #    X, y = X[:num_data], y[:num_data]

    return X, y

def load_pfm(fname, num_data=50000, cache_dir='~/.energyflow'):

    print ('Unpacking file', fname) 
    
    hf = h5py.File(fname, 'r')

    y = hf.get('type')
    reg = hf.get('reg')

    y = np.array(y, dtype=np.float32)
    reg = np.array(reg)

    
    #Filter 
    """
    ttbarWW : 1
    ttbarZ_lep : 2
    ttbarW_lep : 3
    ttbarHiggs_lep : 4
    4top_1jet : 5
    """

    indx=(y==5)

    ftop_reg = reg[indx]

    indx=(y==3)

    ttbarW_reg = reg[indx]

    indx=(y==2)

    ttbarZ_reg = reg[indx]

    indx=(y==1)

    ttbarWW_reg = reg[indx]

    indx=(y==4)

    ttbarH_reg = reg[indx]

    ftop_type = np.ones(ftop_reg.shape[0])
    ttbarW_type = np.zeros(ttbarW_reg.shape[0])
    ttbarZ_type = np.zeros(ttbarZ_reg.shape[0])
    ttbarH_type = np.zeros(ttbarH_reg.shape[0])
    ttbarWW_type = np.zeros(ttbarWW_reg.shape[0])

    #Choose max. 50k events per process excepting 4tops which 
    #is then 200k to get a balanced dataset
    max_num_events = num_data #50000
    if ftop_reg.shape[0] > 4.*max_num_events:
     ftop_reg = ftop_reg[:4*max_num_events]
     ftop_type = np.ones(4*max_num_events)

    if ttbarW_reg.shape[0] > max_num_events:
     ttbarW_reg = ttbarW_reg[:max_num_events]
     ttbarW_type = np.zeros(max_num_events)

    if ttbarZ_reg.shape[0] > max_num_events:
     ttbarZ_reg = ttbarZ_reg[:max_num_events]
     ttbarZ_type = np.zeros(max_num_events)

    if ttbarH_reg.shape[0] > max_num_events:
     ttbarH_reg = ttbarH_reg[:max_num_events]
     ttbarH_type = np.zeros(max_num_events)

    if ttbarWW_reg.shape[0] > max_num_events:

     ttbarWW_reg = ttbarWW_reg[:max_num_events]
     ttbarWW_type = np.zeros(max_num_events)

    #Now I have to concatenate and suffle(perm)

    _reg = np.vstack((ftop_reg, ttbarW_reg, ttbarZ_reg, ttbarH_reg, ttbarWW_reg))
    _y = np.concatenate((ftop_type, ttbarW_type, ttbarZ_type, ttbarH_type, ttbarWW_type)) 

    #randomize
    perm = np.random.permutation(_reg.shape[0])

    reg = _reg[perm] 
    y = _y[perm] 

    #print(type(nobj), type(reg), nobj[0], reg[0])

    X = reg

    #if num_data > -1:
    #    X, y = X[:num_data], y[:num_data]

    return X, y


def load_old(fname, num_data=50000, cache_dir='~/.energyflow'):
    """Loads the dataset. The first time this is called, it will automatically
    download the dataset. Future calls will attempt to use the cached dataset 
    prior to redownloading.

    **Arguments**

    - **num_data** : _int_
        - The number of events to return. A value of `-1` means read in all events.
    - **cache_dir** : _str_
        - The directory where to store/look for the file.

    **Returns**

    - _3-d numpy.ndarray_, _1-d numpy.ndarray_
        - The `X` and `y` components of the dataset as specified above.
    """

    print ('Unpacking file', fname) 
    
    hf = h5py.File(fname, 'r')

    y = hf.get('type')
    nobj = hf.get('nobj')
    reg = hf.get('reg')

    y = np.array(y, dtype=np.float32)
    nobj = np.array(nobj)
    reg = np.array(reg)

    #Filter 
    """
    ttbarWW : 1
    ttbarZ_lep : 2
    ttbarW_lep : 3
    4top_1jet : 4
    ttbarHiggs_lep : 5
    """

    indx=(y==4)

    ftop_nobj = nobj[indx]
    ftop_reg = reg[indx]

    indx=(y==3)

    ttbarW_nobj = nobj[indx]
    ttbarW_reg = reg[indx]

    indx=(y==2)

    ttbarZ_nobj = nobj[indx]
    ttbarZ_reg = reg[indx]

    indx=(y==1)

    ttbarWW_nobj = nobj[indx]
    ttbarWW_reg = reg[indx]

    indx=(y==5)

    ttbarH_nobj = nobj[indx]
    ttbarH_reg = reg[indx]

    ftop_type = np.ones(ftop_nobj.shape[0])
    ttbarW_type = np.zeros(ttbarW_nobj.shape[0])
    ttbarZ_type = np.zeros(ttbarZ_nobj.shape[0])
    ttbarH_type = np.zeros(ttbarH_nobj.shape[0])
    ttbarWW_type = np.zeros(ttbarWW_nobj.shape[0])

    #Choose max. 50k events per process
    max_num_events = num_data #50000
    if ftop_nobj.shape[0] > max_num_events:
     ftop_nobj = ftop_nobj[:max_num_events]
     ftop_reg = ftop_reg[:max_num_events]
     ftop_type = np.ones(max_num_events)

    if ttbarW_nobj.shape[0] > max_num_events:
     ttbarW_nobj = ttbarW_nobj[:max_num_events]
     ttbarW_reg = ttbarW_reg[:max_num_events]
     ttbarW_type = np.zeros(max_num_events)

    if ttbarZ_nobj.shape[0] > max_num_events:
     ttbarZ_nobj = ttbarZ_nobj[:max_num_events]
     ttbarZ_reg = ttbarZ_reg[:max_num_events]
     ttbarZ_type = np.zeros(max_num_events)

    if ttbarH_nobj.shape[0] > max_num_events:
     ttbarH_nobj = ttbarH_nobj[:max_num_events]
     ttbarH_reg = ttbarH_reg[:max_num_events]
     ttbarH_type = np.zeros(max_num_events)

    if ttbarWW_nobj.shape[0] > max_num_events:
     ttbarWW_nobj = ttbarWW_nobj[:max_num_events]
     ttbarWW_reg = ttbarWW_reg[:max_num_events]
     ttbarWW_type = np.zeros(max_num_events)

    #Now I have to concatenate and suffle(perm)
    _nobj = np.vstack((ftop_nobj, ttbarW_nobj, ttbarZ_nobj, ttbarH_nobj, ttbarWW_nobj)) 
    _reg = np.vstack((ftop_reg, ttbarW_reg, ttbarZ_reg, ttbarH_reg, ttbarWW_reg))
    _y = np.concatenate((ftop_type, ttbarW_type, ttbarZ_type, ttbarH_type, ttbarWW_type)) 

    #randomize
    perm = np.random.permutation(_nobj.shape[0])

    nobj = _nobj[perm] 
    reg = _reg[perm] 
    y = _y[perm] 

    #print(type(nobj), type(reg), nobj[0], reg[0])

    X = np.concatenate((nobj, reg), axis = 1) 

    #if num_data > -1:
    #    X, y = X[:num_data], y[:num_data]

    return X, y
  
def drawEvent(Event, imageName, Class):
    
    # Create the empty figure
    plt.ioff()
    fig = plt.figure(frameon=False)
    fig.set_size_inches(4,4)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.axis([-3.2,3.2,-5,5])
    ax.set_axis_off()
    fig.add_axes(ax)
    
    # Draw the electrons
    if 'electron' in Event:
        # Loop over the electrons
        electronSet = list(Event['electron'])
        for electronNumber in range(len(electronSet)):
            electron = Event['electron'][electronNumber] # Get the electron
            pt = electron['pt']
            eta = electron['eta']
            phi = electron['phi']

            scalePt = math.log(pt,11/10)
            phiAxis = np.array([scalePt*2*math.pi/224]) # Ellypse axis
            etaAxis = np.array([scalePt*6/224])
            
            center = np.array([eta,phi])
            Object = Ellipse(xy = center, width=etaAxis, height=phiAxis, angle=0.0, facecolor = 'none', edgecolor= 'b', lw = 4)
            ax.add_artist(Object)
            
    # Draw the muons
    if 'muon' in Event:
        # Loop over the electrons
        muonSet = list(Event['muon'])
        for muonNumber in range(len(muonSet)):
            muon = Event['muon'][muonNumber] # Get the electron
            pt = muon['pt']
            eta = muon['eta']
            phi = muon['phi']

            scalePt = math.log(pt,11/10)
            phiAxis = np.array([scalePt*2*math.pi/224]) # Ellypse axis
            etaAxis = np.array([scalePt*6/224])
            
            center = np.array([eta,phi])
            Object = Ellipse(xy = center, width=etaAxis, height=phiAxis, angle=0.0, facecolor = 'none', edgecolor= 'g', lw = 4)
            ax.add_artist(Object)
            
    # Draw the muons
    if 'photon' in Event:
        # Loop over the electrons
        photonSet = list(Event['photon'])
        for photonNumber in range(len(photonSet)):
            muon = Event['photon'][photonNumber] # Get the electron
            pt = muon['pt']
            eta = muon['eta']
            phi = muon['phi']

            scalePt = math.log(pt,11/10)
            phiAxis = np.array([scalePt*2*math.pi/224]) # Ellypse axis
            etaAxis = np.array([scalePt*6/224])
            
            center = np.array([eta,phi])
            Object = Ellipse(xy = center, width=etaAxis, height=phiAxis, angle=0.0, facecolor = 'none', edgecolor= 'y', lw = 4)
            ax.add_artist(Object)

    # Draw the jets
    if 'jet' in Event:
        # Loop over the electrons
        jetSet = list(Event['jet'])
        for jetNumber in range(len(jetSet)):
            jet = Event['jet'][jetNumber] # Get the electron
            pt = jet['pt']
            eta = jet['eta']
            phi = jet['phi']
            #CSV = jet['CSV']

            scalePt = math.log(pt,11/10)
            phiAxis = np.array([scalePt*2*math.pi/224]) # Ellypse axis
            etaAxis = np.array([scalePt*6/224])
            
            # Define the color of the jet object
#            if CSV > 2 or CSV == 2:
#                colorScale = 0
#            elif CSV<0:
#                colorScale = 225
#            else:
#                colorScale = 200-100*CSV
            #if CSV > 0.679: # medium working point
            #    colorScale = 0 # b-tag
            #else: 
            #    colorScale = 125 # no b-tag
            colorScale = 125
            #if obj == 'bjet':
            #   colorScale = 0             
   
            colorRGB = colorScale/255
            
            center = np.array([eta,phi])
            #Object = Ellipse(xy = center, width=etaAxis, height=phiAxis, angle=0.0, facecolor = 'none', edgecolor= (1, colorRGB, colorRGB), lw = 4)
            Object = Ellipse(xy = center, width=etaAxis, height=phiAxis, angle=0.0, facecolor = 'none', edgecolor= 'r', lw = 4)

            ax.add_artist(Object)
            
    # Draw the jets
    if 'bjet' in Event:
        # Loop over the electrons
        jetSet = list(Event['bjet'])
        for jetNumber in range(len(jetSet)):
            jet = Event['bjet'][jetNumber] # Get the electron
            pt = jet['pt']
            eta = jet['eta']
            phi = jet['phi']
            #CSV = jet['CSV']

            scalePt = math.log(pt,11/10)
            phiAxis = np.array([scalePt*2*math.pi/224]) # Ellypse axis
            etaAxis = np.array([scalePt*6/224])
            
            # Define the color of the jet object
#            if CSV > 2 or CSV == 2:
#                colorScale = 0
#            elif CSV<0:
#                colorScale = 225
#            else:
#                colorScale = 200-100*CSV
            #if CSV > 0.679: # medium working point
            #    colorScale = 0 # b-tag
            #else: 
            #    colorScale = 125 # no b-tag
            colorScale = 0
            #if obj == 'bjet':
            #   colorScale = 0             
   
            colorRGB = colorScale/255
            
            center = np.array([eta,phi])
            #Object = Ellipse(xy = center, width=etaAxis, height=phiAxis, angle=0.0, facecolor = 'none', edgecolor= (1, colorRGB, colorRGB), lw = 4)
            Object = Ellipse(xy = center, width=etaAxis, height=phiAxis, angle=0.0, facecolor = 'none', edgecolor= 'pink', lw = 4)
            ax.add_artist(Object)

    # Draw the met
    if 'met' in Event:
        met = Event['met']['pt']
        scaleMet = math.log(met,11/10)
        met_phi = Event['met']['phi'] 
        phiAxis = np.array([scaleMet*2*math.pi/224]) # Ellypse axis
        etaAxis = np.array([scaleMet*6/224])
        center = np.array([0,met_phi])
        #Object = Ellipse(xy = center, width=etaAxis, height=phiAxis, angle=0.0, facecolor = 'none', edgecolor= 'k', lw = 4)
        Object = Ellipse(xy = center, width=etaAxis, height=phiAxis, angle=0.0, facecolor = 'none', edgecolor= 'k', lw = 4)
        #Object = Rectangle(xy = np.array([3*9/10, 9/10*math.pi]), width = 0.08*3, height = 0.08*math.pi, angle = 0.0, facecolor = color, edgecolor = color)
        ax.add_artist(Object)
    
    # Image name
    if not os.path.exists('CreatedImages/'+Class): os.makedirs('CreatedImages/'+Class)
    image_dir = 'CreatedImages/'+Class + '/' #+ ' Images/'
    #fig.savefig(image_dir+imageName+'.jpg', dpi=56)
    #check how it looks like
    #plt.close(fig) 
    #plt.show()
    #plt.close(fig) 
    #sys.exit()
    Image = image.imread(image_dir+imageName+'.jpg')
    # convert image to numpy array
    #data = asarray(Image)

    return Image
  
def pixelate_org(Events):

   njets_max = 4
   nbjets_max = 4
   nel_max = 1
   nep_max = 1
   nmu_max = 1
   nmup_max = 1

   matrix = []

   #Go trhough events
   for idx, event in enumerate(tqdm(Events)):

        num_jets = int(event[0] if event[0] <= njets_max else njets_max)
        num_bjets = int(event[1] if event[1] <= nbjets_max else nbjets_max)
        num_el = int(event[2] if event[2] <= nel_max else nel_max)
        num_ep = int(event[3] if event[3] <= nep_max else nep_max)
        num_mu = int(event[4] if event[4] <= nmu_max else nmu_max)
        num_mup = int(event[5] if event[5] <= nmup_max else nmup_max)

        #print(event)

        #print(num_jets, num_bjets, num_el, num_ep, num_mu, num_mup)

        Event = {}

        pt = event[6]
        eta = 0.  
        phi = event[7]

        Event['met'] = {'pt':pt, 'eta':eta, 'phi':phi} 
    
        offset = 9
        for i in range(num_jets):
            typ = 'jet'
            pt = event[4*i + offset]
            eta = event[4*i + (offset +1)]
            phi = event[4*i + (offset +2)]
            Event.setdefault(typ, []).append({'pt':pt, 'eta':eta, 'phi':phi})

        offset += 4*njets_max
        for i in range(num_bjets):
            typ = 'bjet'
            pt = event[4*i + offset]
            eta = event[4*i + (offset +1)]
            phi = event[4*i + (offset +2)]
            if pt == 0. and eta == 0. and phi == 0.:
               continue
            Event.setdefault(typ, []).append({'pt':pt, 'eta':eta, 'phi':phi})

        offset += 4*nbjets_max
        for i in range(num_el):
            typ = 'electron'
            pt = event[4*i + offset]
            eta = event[4*i + (offset +1)]
            phi = event[4*i + (offset +2)]
            if pt == 0. and eta == 0. and phi == 0.:
               continue
            Event.setdefault(typ, []).append({'pt':pt, 'eta':eta, 'phi':phi})

        offset += 4*nel_max
        for i in range(num_ep):
            typ = 'electron'
            pt = event[4*i + offset]
            eta = event[4*i + (offset +1)]
            phi = event[4*i + (offset +2)]
            if pt == 0. and eta == 0. and phi == 0.:
               continue
            Event.setdefault(typ, []).append({'pt':pt, 'eta':eta, 'phi':phi})

        offset += 4*nep_max
        for i in range(num_mu):
            typ = 'muon'
            pt = event[4*i + offset]
            eta = event[4*i + (offset +1)]
            phi = event[4*i + (offset +2)]
            if pt == 0. and eta == 0. and phi == 0.:
               continue
            Event.setdefault(typ, []).append({'pt':pt, 'eta':eta, 'phi':phi})

        offset += 4*nmu_max
        for i in range(num_mup):
            typ = 'muon'
            pt = event[4*i + offset]
            eta = event[4*i + (offset +1)]
            phi = event[4*i + (offset +2)]
            if pt == 0. and eta == 0. and phi == 0.:
               continue
            Event.setdefault(typ, []).append({'pt':pt, 'eta':eta, 'phi':phi})

#        print(Event)
#        sys.exit()

        mat = drawEvent(Event, 'test', 'train')
         
        #print(mat.shape)
        #sys.exit()
        #data_event_type.append(event_t)
        matrix.append(mat)

   return np.array(matrix)

def pixelate(Events):

   njets_max = 4
   nbjets_max = 4
   nel_max = 1
   nep_max = 1
   nmu_max = 1
   nmup_max = 1

   matrix = []

   #Go trhough events
   for idx, event in enumerate(tqdm(Events)):

        num_jets = int(event[0]) 
        num_bjets = int(event[1])
        num_el = int(event[2]) 
        num_ep = int(event[3])
        num_mu = int(event[4])
        num_mup = int(event[5]) 
        num_ga = int(event[6]) 

        #print(event)

        #print(num_jets, num_bjets, num_el, num_ep, num_mu, num_mup)

        Event = {}

        pt = event[7]
        eta = 0.  
        phi = event[8]

        Event['met'] = {'pt':pt, 'eta':eta, 'phi':phi} 
    
        offset = 10
        for i in range(num_jets):
            typ = 'jet'
            pt = event[4*i + offset]
            eta = event[4*i + (offset +1)]
            phi = event[4*i + (offset +2)]
            Event.setdefault(typ, []).append({'pt':pt, 'eta':eta, 'phi':phi})

        offset += 4*num_jets
        for i in range(num_bjets):
            typ = 'bjet'
            pt = event[4*i + offset]
            eta = event[4*i + (offset +1)]
            phi = event[4*i + (offset +2)]
            if pt == 0. and eta == 0. and phi == 0.:
               continue
            Event.setdefault(typ, []).append({'pt':pt, 'eta':eta, 'phi':phi})

        offset += 4*num_bjets
        for i in range(num_el):
            typ = 'electron'
            pt = event[4*i + offset]
            eta = event[4*i + (offset +1)]
            phi = event[4*i + (offset +2)]
            if pt == 0. and eta == 0. and phi == 0.:
               continue
            Event.setdefault(typ, []).append({'pt':pt, 'eta':eta, 'phi':phi})

        offset += 4*num_el
        for i in range(num_ep):
            typ = 'electron'
            pt = event[4*i + offset]
            eta = event[4*i + (offset +1)]
            phi = event[4*i + (offset +2)]
            if pt == 0. and eta == 0. and phi == 0.:
               continue
            Event.setdefault(typ, []).append({'pt':pt, 'eta':eta, 'phi':phi})

        offset += 4*num_ep
        for i in range(num_mu):
            typ = 'muon'
            pt = event[4*i + offset]
            eta = event[4*i + (offset +1)]
            phi = event[4*i + (offset +2)]
            if pt == 0. and eta == 0. and phi == 0.:
               continue
            Event.setdefault(typ, []).append({'pt':pt, 'eta':eta, 'phi':phi})

        offset += 4*num_mu
        for i in range(num_mup):
            typ = 'muon'
            pt = event[4*i + offset]
            eta = event[4*i + (offset +1)]
            phi = event[4*i + (offset +2)]
            if pt == 0. and eta == 0. and phi == 0.:
               continue
            Event.setdefault(typ, []).append({'pt':pt, 'eta':eta, 'phi':phi})

        offset += 4*num_mup
        for i in range(num_ga):
            typ = 'photon'
            pt = event[4*i + offset]
            eta = event[4*i + (offset +1)]
            phi = event[4*i + (offset +2)]
            if pt == 0. and eta == 0. and phi == 0.:
               continue
            Event.setdefault(typ, []).append({'pt':pt, 'eta':eta, 'phi':phi})

#        print(Event)
#        sys.exit()

        mat = drawEvent(Event, 'test', 'train')
         
        print(mat.shape)
        sys.exit()
        #data_event_type.append(event_t)
        matrix.append(mat)

   return np.array(matrix)
