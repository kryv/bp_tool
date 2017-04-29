import pyqtgraph as pg
import os
from pyqtgraph.Qt import QtCore, QtGui

import time
import numpy as np

app = QtGui.QApplication([])
import pyqtgraph.parametertree.parameterTypes as pTypes
from pyqtgraph.parametertree import Parameter, ParameterTree, ParameterItem, registerParameterType

from phantasy import MachinePortal
from cothread.catools import caget, caput

from flame import Machine, setLogLevel, FLAME_ERROR
setLogLevel(FLAME_ERROR)

#machine = "FRIB_FLAME"
machine = "FRIB_FE"
mp = MachinePortal(machine=machine)
io = mp.work_lattice_conf

mm = 1e3

### Plottting ###
model_flag = True

pw1 = pg.PlotWidget()
pw2 = pg.PlotWidget()
pw3 = pg.PlotWidget()

p1 = pw1.plotItem
p2 = pw2.plotItem
p3 = pw3.plotItem

p1.setLabel('left', 'Horizontal [mm]')
p1.setLabel('bottom', 'z [m]')

p2.setLabel('left', 'Vertical [mm]')
p2.setLabel('bottom', 'z [m]')

p3.showAxis('left', False)
#p3.setXRange(-1.5,1.5,padding=0)
p3.setRange(xRange=[-1.5,1.5])
p3.setMouseEnabled(y=False)
p3.setLabel('bottom', 'z [m]')

bpms = mp.get_elements(type='BPM')
pms = mp.get_elements(type='PM')

bpm_pos = [getattr(i,'sb') for i in bpms]
pm_pos = [getattr(i,'sb') for i in pms]

bpm_col = 'r'
pm_col = 'c'

dt_bpmx = p1.plot(pen=bpm_col, symbol='s', symbolSize=5, symbolBrush=bpm_col)
dt_pmx = p1.plot(pen=pm_col, symbol='s', symbolSize=5, symbolBrush=pm_col)
dt_pmxp = p1.plot(pen=None, symbol='s', symbolSize=5, symbolBrush=pm_col)
dt_pmxm = p1.plot(pen=None, symbol='s', symbolSize=5, symbolBrush=pm_col)


dt_bpmy = p2.plot(pen=bpm_col, symbol='s', symbolSize=5, symbolBrush=bpm_col)
dt_pmy = p2.plot(pen=pm_col, symbol='s', symbolSize=5, symbolBrush=pm_col)
dt_pmyp = p2.plot(pen=None, symbol='s', symbolSize=5, symbolBrush=pm_col)
dt_pmym = p2.plot(pen=None, symbol='s', symbolSize=5, symbolBrush=pm_col)

def monitor_res():
    if len(bpms) != 0:
        bpmx = np.array(mp.get_pv_values(bpms,'X')['X'])*mm
        bpmy = np.array(mp.get_pv_values(bpms,'Y')['Y'])*mm

        dt_bpmx.setData(bpm_pos, bpmx)
        dt_bpmy.setData(bpm_pos, bpmy)

    if len(pms) != 0:
        pmx = np.array(mp.get_pv_values(pms,'X')['X'])*mm
        pmy = np.array(mp.get_pv_values(pms,'Y')['Y'])*mm
        pmxr = np.array(mp.get_pv_values(pms,'XRMS')['XRMS'])*mm
        pmyr = np.array(mp.get_pv_values(pms,'YRMS')['YRMS'])*mm

        dt_pmx.setData(pm_pos, pmx)
        dt_pmxp.setData(pm_pos, pmx+pmxr)
        dt_pmxm.setData(pm_pos, pmx-pmxr)

        dt_pmy.setData(pm_pos, pmy)
        dt_pmyp.setData(pm_pos, pmy+pmyr)
        dt_pmym.setData(pm_pos, pmy-pmyr)

monitor_res()

mcol = (255, 255, 0, 100)
scol = (255, 255, 0, 50)
mcol0 = (255, 255, 0, 0)
scol0 = (255, 255, 0, 0)

cv_x  = p1.plot(pen=mcol)
cv_xp = p1.plot(pen=scol)
cv_xm = p1.plot(pen=scol)
fill_x = pg.FillBetweenItem(curve1=cv_xp, curve2=cv_xm, brush=scol)
p1.addItem(fill_x)

cv_y  = p2.plot(pen=mcol)
cv_yp = p2.plot(pen=scol)
cv_ym = p2.plot(pen=scol)
fill_y = pg.FillBetweenItem(curve1=cv_yp, curve2=cv_ym, brush=scol)
p2.addItem(fill_y)


etypes = mp.get_all_types()
elem_col = {'HCOR':'w', 'VCOR':'w', 'BPM':'r', 'PM':'c', 'BEND':(0,80,0), 'SOL':'b', 'CAV':'y',
            'QUAD':(255,165,0),'SEXT':'m', 'EBEND':(0, 150, 0), 'EQUAD':(128,0,128)}

drift = p3.plot([0, io.length], [0.0, 0.0], pen=pg.mkPen((70,70,70),width=10))

for etype in etypes:
    elems = mp.get_elements(type=etype)
    tpen = pg.mkPen(elem_col[etype])
    tbrs = pg.mkBrush(elem_col[etype])

    for elem in elems:
        wid = np.abs(elem.sb-elem.se)
        tbox = QtGui.QGraphicsRectItem(elem.sb, -1.0, wid, 2.0)
        tbox.setPen(tpen)
        tbox.setBrush(tbrs)
        tbox.setToolTip(elem.name)
        p3.addItem(tbox)

for etype in etypes:
    elems = mp.get_elements(type=etype)
    for elem in elems:
        text = pg.TextItem('- '+elem.name)
        offset = 0.5 if elem.family == 'VCOR' else 0.0
        text.setPos(elem.sb, -0.7+offset)
        text.setAngle(50)
        p3.addItem(text)


p2.setXLink(p1)
p3.setXLink(p1)

def line_renew():
    lat = mp.work_lattice_conf.model_factory.build()
    M = Machine(lat.conf())
    ids = range(len(M))
    S = M.allocState({})
    res = M.propagate(S,observe=ids)
    pos = np.asarray([res[i][1].pos for i in ids])
    x,y = np.asarray([[res[i][1].moment0_env[j] for i in ids] for j in [0,2]])
    xr,yr = np.asarray([[res[i][1].moment0_rms[j] for i in ids] for j in [0,2]])
    cv_x.setData(pos,x)
    cv_xp.setData(pos,x+xr)
    cv_xm.setData(pos,x-xr)
    cv_y.setData(pos,y)
    cv_yp.setData(pos,y+yr)
    cv_ym.setData(pos,y-yr)

line_renew()

p1.disableAutoRange(True)
p2.disableAutoRange(True)

#################


### ParameterTree ###
auto_flag = False
steps = {'CAV':0.01,'HCOR':0.0001, 'VCOR':0.0001, 'SOL':0.1, 'QUAD':0.1, 'BEND':0.0001, 'EBEND':0.0001, 'EQUAD':50}

class MainParameter(pTypes.GroupParameter):
    def __init__(self, **opts):
        opts['type'] = 'bool'
        opts['value'] = True
        opts['expanded'] = False
        pTypes.GroupParameter.__init__(self, **opts)
        elems=opts['info']
        for d in elems:
            data = io.get(d) if d.family != 'EQUAD' else io.get(d,'V')
            for key in data.keys():
                val = io.get(d,key)[key]
                linename = d.name+'('+key+')   '
                self.addChild({'name':linename, 'elem':d.name, 'field':key, 'type':'float', 'value':val, 'step':steps[d.family], 'decimals':5})
                a = self.param(linename)
                a.setOpts(tmp=val, flag=True)
                a.sigValueChanging.connect(self.changing)
                a.sigValueChanged.connect(self.changed)
                a.sigStateChanged.connect(self.update)

    def changing(self,a):
        a.setOpts(flag=False)

    def changed(self,a):
        cur = a.value()
        tmp = a.opts['tmp']
        if cur != tmp:
            a.setOpts(tmp=cur)
            io.set(a.opts['elem'], cur, a.opts['field'])
        a.setOpts(flag=True)

    def update(self,a):
        if a.opts['flag']:
            a.setValue(a.opts['tmp'])

class Boolparameter(pTypes.GroupParameter):
    def __init__(self, **opts):
        opts['type'] = 'bool'
        opts['value'] = True
        opts['expanded'] = True
        pTypes.GroupParameter.__init__(self, **opts)
        self.addChild({'name':'Auto-update', 'type':'bool', 'value': False})
        a = self.param('Auto-update')
        a.sigValueChanged.connect(self.achanged)

        self.addChild({'name':'Plot model', 'type':'bool', 'value': True})
        b = self.param('Plot model')
        b.sigValueChanged.connect(self.bchanged)

    def achanged(self,a):
        global auto_flag
        auto_flag = a.value()

    def bchanged(self,b):
        global model_flag
        model_flag = b.value()
        if b.value():
            cv_x.setPen(mcol)
            cv_xp.setPen(scol)
            cv_xm.setPen(scol)
            fill_x.setBrush(scol)
            cv_y.setPen(mcol)
            cv_yp.setPen(scol)
            cv_ym.setPen(scol)
            fill_y.setBrush(scol)
        else:
            cv_x.setPen(mcol0)
            cv_xp.setPen(scol0)
            cv_xm.setPen(scol0)
            fill_x.setBrush(scol0)
            cv_y.setPen(mcol0)
            cv_yp.setPen(scol0)
            cv_ym.setPen(scol0)
            fill_y.setBrush(scol0)

etypes = mp.get_all_types()
for ntype in ['BPM','PM','SEXT']:
    if ntype in etypes:
        etypes.remove(ntype)

etypes = sorted(etypes, key=lambda x: x[-1])
etypes.reverse()

params = [Boolparameter(name='Operation parameters')]

for etype in etypes:
    elems = mp.get_elements(type=etype)
    params.append(MainParameter(name=etype, info=elems))

p = Parameter.create(name='params',type='group', children=params)

t = ParameterTree()
t.setParameters(p,showTop=False)
font = QtGui.QFont()
font.setPointSize(12)
font.setFixedPitch(True)
font.setFamily('Monospace')
t.setFont(font)

#####################

def update():
    if auto_flag:
        for child in p.children()[1:]:
            for gc in child:
                field = gc.opts['field']
                new = io.get(gc.opts['elem'],field)[field]
                cur = gc.value()
                tmp = gc.opts['tmp']
                if cur == tmp:
                    gc.setOpts(value=new, tmp=new)

    if model_flag:
        line_renew()

    monitor_res()

timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(500)

win = QtGui.QWidget()

layout = QtGui.QVBoxLayout()

layout_v  = QtGui.QSplitter(QtCore.Qt.Vertical)
layout_v.addWidget(pw1)
layout_v.addWidget(pw2)
layout_v.addWidget(pw3)

layout_h = QtGui.QSplitter(QtCore.Qt.Horizontal)
layout_h.addWidget(t)
layout_h.addWidget(layout_v)

layout.addWidget(layout_h)

win.setLayout(layout)

win.setWindowTitle('VA console')
win.show()
win.resize(1200,800)

## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()