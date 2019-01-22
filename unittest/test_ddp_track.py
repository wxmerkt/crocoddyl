import pinocchio
from pinocchio.utils import *
from numpy.linalg import inv,pinv,norm,svd,eig
from numpy import dot,asarray
import warnings
from crocoddyl import CostModelSum,CostModelPosition,CostModelState,CostModelControl,DifferentialActionModelFloatingInContact,IntegratedActionModelEuler,ActuationModelFreeFloating,StatePinocchio,ContactModel6D,ContactModelMultiple,ActivationModelWeightedQuad,m2a,a2m,CostModelPlacementVelocity,CostModelPosition6D
from robots import loadTalosLegs

robot = loadTalosLegs()
rmodel = robot.model

opPointName = 'right_sole_link'
contactName = 'left_sole_link'

opPointName,contactName = contactName,opPointName
CONTACTFRAME = rmodel.getFrameId(contactName)
OPPOINTFRAME = rmodel.getFrameId(opPointName)

def createModel():
    State = StatePinocchio(rmodel)
    actModel = ActuationModelFreeFloating(rmodel)
    contactModel = ContactModelMultiple(rmodel)
    contact6 = ContactModel6D(rmodel,rmodel.getFrameId(contactName),ref=None)
    contactModel.addContact(name='contact',contact=contact6)
    costModel = CostModelSum(rmodel,nu=actModel.nu)
    cost1 = CostModelPosition6D(rmodel,nu=actModel.nu,
                                frame=rmodel.getFrameId(opPointName),
                                ref=pinocchio.SE3(eye(3),np.matrix([.2,.0848,0.]).T))
    cost2 = CostModelState(rmodel,State,ref=State.zero(),nu=actModel.nu)
    cost3 = CostModelControl(rmodel,nu=actModel.nu)
    costModel.addCost( name="pos", weight = 10, cost = cost1)
    costModel.addCost( name="regx", weight = 0.1, cost = cost2) 
    costModel.addCost( name="regu", weight = 0.01, cost = cost3)
    
    dmodel = DifferentialActionModelFloatingInContact(rmodel,actModel,contactModel,costModel)
    model  = IntegratedActionModelEuler(dmodel)
    return model

q = robot.q0.copy()
v = zero(rmodel.nv)
x = m2a(np.concatenate([q,v]))


# --- DDP 
# --- DDP 
# --- DDP 
from refact import ShootingProblem, SolverDDP,SolverKKT
from logger import *
disp = lambda xs: disptraj(robot,xs)

DT = 1.0 
T = 20
timeStep = DT/T

models = [ createModel() for _ in range(T+1) ]

for k,model in enumerate(models[:-1]):
    t = k*timeStep
    model.timeStep = timeStep
    model.differential.costs['pos' ].weight =    100
    model.differential.costs['regx'].weight = .1
    model.differential.costs['regx'].cost.weights = np.array([0]*6+[0.01]*(rmodel.nv-6)+[10]*rmodel.nv)
    model.differential.costs['regu'].weight = 0.001
    model.differential.costs['pos' ].cost.ref.translation = np.matrix([ .2*t/DT, .0848, 0.0 ]).T

termmodel = models[-1]
termmodel.differential.costs.addCost(name='veleff',
                                     cost=CostModelPlacementVelocity(rmodel,OPPOINTFRAME),
                                     weight=10000)

termmodel.differential.costs['veleff' ].weight = 10000
termmodel.differential.costs['pos' ]   .weight = 300000
termmodel.differential.costs['regx']   .weight = 1
termmodel.differential.costs['regu']   .weight = 0.01
termmodel.differential.costs['regx'].cost.weights = np.array([0]*6+[0.01]*(rmodel.nv-6)+[10]*rmodel.nv)
termmodel.differential.costs['pos' ].cost.ref.translation = np.matrix([ .2, .0848, 0.0 ]).T
    
# --- SOLVER

problem = ShootingProblem(x, models[:-1], models[-1] )

ddp = SolverDDP(problem)
ddp.callback = SolverLogger(robot)
ddp.th_stop = 1e-9
ddp.solve(verbose=True,maxiter=1000,regInit=.1)


# --- Contact velocity
# cost = || v ||

