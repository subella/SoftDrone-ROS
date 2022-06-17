import Sofa
from math import sin,cos,pi

import os
path = os.path.dirname(os.path.abspath(__file__))+'/../details/data/mesh/'
path3D = os.path.dirname(os.path.abspath(__file__))+'/../details/data/mesh2/'
from rigidControl import *

def AddConstantAndtransformTableInString(Table, add):
	sizeT =  len(Table);
	strOut= ' ';
	for p in range(sizeT):
		strOut = strOut+ str(Table[p]+add)+' '

	return strOut



def transformTableInString(Table):
	sizeT =  len(Table);
	strOut= ' ';
	for p in range(sizeT):
		strOut = strOut+ str(Table[p])+' '

	return strOut

#Takes a n dimensional vector of vector and transform it into a simple vector
def transformDoubleTableInSimpleTable(Table):
    size0 =  len(Table);

    # count the size
    size=0;
    for i in range(size0):
        size = size+len(Table[i]);

    TableOut=[0]*size;
    s=0;
    for i in range(size0):
        for j in range(len(Table[i])):
            TableOut[s] = Table[i][j];
            s=s+1;

    return TableOut



#Units: cm and kg

def createScene(rootNode):

    rootNode.createObject('VisualStyle', displayFlags='showBehaviorModels showForceFields');

    rootNode.findData('dt').value= 0.05;
    rootNode.findData('gravity').value= '0. 0. -9810';

    rootNode.createObject('BackgroundSetting', color='0 0.168627 0.211765');
    rootNode.createObject('OglSceneFrame', style="Arrows", alignment="TopRight");
    #rootNode.createObject('PythonScriptController', classname="controller", filename="ControlKeyboard2.py");


    ###############################
    ## MECHANICAL MODEL
    ###############################

    robot = rootNode.createChild('robot')
    robot.createObject('EulerImplicit');
    #robot.createObject('CGLinearSolver', iterations='1000', tolerance='1e-5', threshold='1e-10')
    robot.createObject('SparseLDLSolver');

    nodeFEM = robot.createChild('nodeFEM')

    loader=nodeFEM.createObject('GIDMeshLoader', name='loader', filename=path3D+'tripod_mid.gidmsh', scale='1');
    loader.init()
    loader2=nodeFEM.createObject('MeshVTKLoader', name='loader2', filename=path3D+'tripodDeformed.vtu')
    topo=nodeFEM.createObject('TetrahedronSetTopologyContainer', position='@loader2.position', tetrahedra='@loader2.tetrahedra' , name='container', createTriangleArray='1', checkConnexity='1');
    topo.init()
    #print test.position
    
    ### indices found thanks to displayBoxROI_indices = 1 in actuatedarm.py
    
    BoxROI0= [[711], [712], [727], [728], [729], [736], [738], [741], [743], [745], [748], [749], [750], [751], [752], [754], [755], [758], [759], [760], [763], [764], [765], [766], [768], [769], [770], [771], [772], [773], [774], [775], [776], [777], [778], [779],[780]]
    BoxROI1= [[508], [510], [547], [561], [564], [565], [567], [590], [592], [593], [595], [601], [621], [622], [630], [632], [650], [652], [654], [657], [660], [673], [674], [676], [687], [689], [690], [697], [700], [702], [710], [720], [721], [723], [725], [740], [742]]
    BoxROI2= [[0], [1], [2], [3], [4], [5], [6], [8], [9], [11], [12], [14], [19], [21], [24], [27], [33], [37], [42], [44], [46], [57], [60], [64], [66], [80], [84], [91], [93], [116], [118], [119], [123]]
    branch0= transformDoubleTableInSimpleTable(BoxROI0) 
    branch1= transformDoubleTableInSimpleTable(BoxROI1) 
    branch2= transformDoubleTableInSimpleTable(BoxROI2) 



    center = nodeFEM.createObject('SphereROI', template='Vec3d', centers='0 0 0', radii='5', position='@loader.position', name='ROI', computeTriangles='0', computeEdges='0')
    center.init()
    centerList = transformDoubleTableInSimpleTable(center.indices)


    nodeFEM.createObject('TetrahedronSetTopologyModifier');
    nodeFEM.createObject('TetrahedronSetTopologyAlgorithms');
    nodeFEM.createObject('TetrahedronSetGeometryAlgorithms');

    tetras= nodeFEM.createObject('MechanicalObject', name='tetras',  position='@container.position', rest_position='@loader.position');
    mass = nodeFEM.createObject('UniformMass', totalmass='0.4');
    tetraFF= nodeFEM.createObject('TetrahedronFEMForceField', poissonRatio='0.45',  youngModulus='600');

    #fix = nodeFEM.createObject('RestShapeSpringsForceField', name='fix', stiffness='1e10', points=AddConstantAndtransformTableInString(list_branch0,-1) ) #external_rest_shape='@../actuators/attach0/mm0'
    #robot.createObject('RestShapeSpringsForceField', external_rest_shape='@../actuators/attach1/mm1', stiffness='1e10', points=AddConstantAndtransformTableInString(list_branch1,-1), external_points=transformTableInString(range(len(list_branch1))) )
    #robot.createObject('RestShapeSpringsForceField', external_rest_shape='@../actuators/attach2/mm2', stiffness='1e10', points=AddConstantAndtransformTableInString(list_branch2,-1), external_points=transformTableInString(range(len(list_branch2))) )




    #robot.createObject('BoxROI', name='boxROI', box="-30 80 70 30 140 130   -130 -100 70  -60  -20 130  130 -100 70  60  -20 130", drawBoxes='true');
    #robot.createObject('FixedConstraint', indices="@boxROI.indices");


    y0=0; y1=0; y2=0
    x0=0;
    z0=-23.5
    z1=cos(2*pi/3)*z0;
    x1=sin(2*pi/3)*z0;
    z2=cos(4*pi/3)*z0;
    x2=sin(4*pi/3)*z0;
    
    transform= [ [x0,y0,z0,0,0,0,1], [x1,y1,z1,0,-sin(4.*pi/6.),0,cos(4.*pi/6.)],  [x2,y2,z2,0,-sin(2.*pi/6.),0,cos(2*pi/6.)], [0,0,0,0,0,0,1] ]


    controllerNode = robot.createChild('controllerNode')
    myController = rigidControl(controllerNode, "superCaMarche", [branch0, branch1, branch2, centerList],[tetraFF, mass], tetras, nodeFEM, transform,0)

    visuServo = rootNode.createChild('VisuServo')
    visuServo.createObject('MeshSTLLoader', name='loader', filename=path3D+'SG90_servomotor.stl', rotation="0 0 0")
    visuServo.createObject('OglModel', name='servo1', position='@loader.position', triangles='@loader.triangles', scale="1", translation=[x0,y0,z0], rotation=[0,0,0], color='green')
    visuServo.createObject('OglModel', name='servo2', position='@loader.position', triangles='@loader.triangles', scale="1", translation=[x1,y1,z1], rotation=[0,120,0], color='white')
    visuServo.createObject('OglModel', name='servo3', position='@loader.position', triangles='@loader.triangles', scale="1", translation=[x2,y2,z2], rotation=[0,240,0], color='red')


    return rootNode
