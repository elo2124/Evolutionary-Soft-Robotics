import random
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import csv
import time
import datetime
from multiprocessing.pool import Pool
import os, psutil

T = 0
dt = 0.0005
unit_mass = 1.0
k_wall = 10000 # Spring constant of floor / ceiling
k_robot = 10000 # spring constant of robot
u_frict_s = 1 # Coefficient of friction (static)
u_frict_k = 0.8 # Coefficient of friction (kinetic)
max_robot_size = 50
max_tree_depth = 3


fig = plt.figure()
ax1 = fig.add_subplot(111, projection='3d')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('z')

# Change these parameters to alter size of grid
Xfloor = -10
Xroof = 10
Yfloor = -10
Yroof = 10
Zfloor = 0
Zroof = 10

X1 = np.arange(Xfloor,Xroof+1,1)
Y1 = np.arange(Yfloor,Yroof+1,1)
X1, Y1 = np.meshgrid(X1, Y1)
Z1 = (X1+Y1)*0

X2 = np.arange(Xfloor,Xroof+1,1)
Z2 = np.arange(Zfloor,Zroof+1,1)
X2, Z2 = np.meshgrid(X2, Z2)
Y2 = (X2+Z2)*0

Y3 = np.arange(Yfloor,Yroof+1,1)
Z3 = np.arange(Zfloor,Zroof+1,1)
Y3, Z3 = np.meshgrid(Y3, Z3)
X3 = (Y3+Z3)*0

ax1.plot_surface(X1, Y1, Z1, color='gray', alpha=0.3)
#ax1.plot_surface(X2, Y2, Z2, color='gray', alpha=0.3)
#ax1.plot_surface(X3, Y3, Z3, color='gray', alpha=0.3)

ax1.set_xlim(Xfloor, Xroof)
ax1.set_ylim(Yfloor, Yroof)
ax1.set_zlim(Zfloor, Zroof)

line0, = ax1.plot3D([],[],[], 'b-')

#shadow, = ax1.plot3D([],[],[], color = '0.3')
#ax1.scatter3D([],[],[])


class Mass:
    def __init__(self, mass, posX, posY, posZ, velX, velY, velZ, accX, accY, accZ):
        self.mass = mass
        self.posX = posX
        self.posY = posY
        self.posZ = posZ
        self.velX = velX
        self.velY = velY
        self.velZ = velZ
        self.accX = accX
        self.accY = accY
        self.accZ = accZ

    def __str__(self):
        return f"mass({self.mass}), " \
               f"pos({self.posX},{self.posY},{self.posZ}), " \
               f"vel({self.velX},{self.velY},{self.velZ}), " \
               f"acc({self.accX},{self.accY},{self.accZ})  "

class Spring:
    def __init__(self, k, rLength, m1, m2):
        self.k = k
        self.rLength = rLength
        self.rLength_a = rLength
        self.m1 = m1
        self.m2 = m2

    def __str__(self):
        return f"k({self.k}), Rest Length({self.rLength}), m1({self.m1}), m2({self.m2})"

class MusclePoint:
    def __init__(self, m, b, w, c):
        self.m = m
        self.b = b
        self.w = w
        self.c = c

    def __str__(self):
        return f"massPoint({self.m}), Amplitude({self.b}), rate({self.w}), phase({self.c})"

class Cube:
    def __init__(self, origin, length, k):
        self.origin = origin
        self.length = length
        self.k = k
        self.masses, self.springs, self.faces = createStaticCube(self)

    def __str__(self):
        return f"origin({self.origin}), Side Length({self.length}), k({self.k}, masses({len(self.masses)}), " \
               f"springs({len(self.springs)})"

class Tetra:
    def __init__(self, origin, length, k):
        self.origin = origin
        self.length = length
        self.k = k
        self.masses, self.springs = createStaticTetra(self)

    def __str__(self):
        return f"origin({self.origin}), Side Length({self.length}), k({self.k}, masses({len(self.masses)}), " \
               f"springs({len(self.springs)})"

class OriginTetra:
    def __init__(self):
        self.origin = [0, 0, 0]
        self.length = 1
        self.k = k_robot
        self.masses, self.springs, self.faces = createOriginTetra(self)

    def __str__(self):
        return f"origin({self.origin}), Side Length({self.length}), k({self.k}, masses({len(self.masses)}), " \
               f"springs({len(self.springs)})"

class OriginCube:
    def __init__(self):
        self.origin = [0,0,0]
        self.length = 1
        self.k = k_robot
        self.masses, self.springs, self.faces = createOriginCube(self)

    def __str__(self):
        return f"origin({self.origin}), Side Length({self.length}), k({self.k}, masses({len(self.masses)}), " \
               f"springs({len(self.springs)})"

class CubeStructure:
    def __init__(self, origin, length, k, form):
        self.origin = origin
        self.length = length
        self.k = k
        self.form = form
        self.masses, self.springs, self.faces = createStaticCubeStructure(self)

    def __str__(self):
        return f"origin({self.origin}), Side Length({self.length}), k({self.k}, masses({len(self.masses)}), " \
               f"springs({len(self.springs)}), form({self.form})"

class Robot:
    def __init__(self):
        self.heap = generateRandomTernaryHeap()
        self.masses, self.springs, self.musclePoints, self.heap = generateNewTetraRobot()

        def __str__(self):
            return f"masses({self.masses}), springs({springs}), musclePoints({self.musclePoints}), heap({heap})"


def generateRobotFromHeapNumpy(heap):
    if idx == 1:
        m_np = np.full()

def generateNewTetraRobot(heap, masses = [], springs = [], faces = [], idx = 1, musclePoints = []):
    # Generate a robot structure from ternary heap (pre Muscle addition)

    if idx == 1:
        masses = []
        springs = []
        originTetra = OriginTetra()
        for mass in originTetra.masses:
            masses.append(mass)
        for spring in originTetra.springs:
            springs.append(spring)
        faces = originTetra.faces

    # Build a new Tetra structure off of face 0
    if int(heap[idx-1][0]) == 1:
        tri_center = [(faces[0][0].posX+faces[0][1].posX+faces[0][2].posX)/3,
                      (faces[0][0].posY+faces[0][1].posY+faces[0][2].posY)/3,
                      (faces[0][0].posZ+faces[0][1].posZ+faces[0][2].posZ)/3]
        h = math.sqrt(2/3) * 2 # Assumes height of 1
        a = (faces[0][0].posX - faces[0][1].posX, faces[0][0].posY - faces[0][1].posY,
             faces[0][0].posZ - faces[0][1].posZ)
        b = (faces[0][0].posX - faces[0][2].posX, faces[0][0].posY - faces[0][2].posY,
             faces[0][0].posZ - faces[0][2].posZ)
        c = (1/np.linalg.norm(np.cross(a, b)))*np.cross(a, b)
        newX = tri_center[0] - h*c[0]
        newY = tri_center[1] - h*c[1]
        newZ = tri_center[2] - h * c[2]
        overlap_flag = False # Check if mass is overlapping
        err = 0.5
        for mass in masses:
            if newX + err >= mass.posX >= newX - err \
                    and newY + err >= mass.posY >= newY - err \
                    and newZ + err >= mass.posZ >= newZ - err:
                overlap_flag = True
                repeatMass = mass
        if not overlap_flag:
            m = Mass(unit_mass, tri_center[0] - h*c[0], tri_center[1] - h*c[1], tri_center[2] - h*c[2], 0, 0, 0, 0, 0, 0)
            masses.append(m)
            springs.append(Spring(k=k_robot, rLength=calcMassDist(m, faces[0][0]), m1=faces[0][0], m2=m))
            springs.append(Spring(k=k_robot, rLength=calcMassDist(m, faces[0][1]), m1=faces[0][1], m2=m))
            springs.append(Spring(k=k_robot, rLength=calcMassDist(m, faces[0][2]), m1=faces[0][2], m2=m))
        else:
            m = repeatMass
            springFlag1 = False
            springFlag2 = False
            springFlag3 = False
            for spring in springs:
                if {spring.m1, spring.m2} == {faces[0][0], m}:
                    springFlag1 = True
                if {spring.m1, spring.m2} == {faces[0][1], m}:
                    springFlag2 = True
                if {spring.m1, spring.m2} == {faces[0][2], m}:
                    springFlag3 = True
            if not springFlag1:
                springs.append(Spring(k=k_robot, rLength=calcMassDist(m, faces[0][0]), m1=faces[0][0], m2=m))
            if not springFlag2:
                springs.append(Spring(k=k_robot, rLength=calcMassDist(m, faces[0][1]), m1=faces[0][1], m2=m))
            if not springFlag3:
                springs.append(Spring(k=k_robot, rLength=calcMassDist(m, faces[0][2]), m1=faces[0][2], m2=m))
        faces_new = [[faces[0][1], faces[0][2], m], [faces[0][0], faces[0][2], m], [faces[0][0], faces[0][1], m]]
        if getChild1Idx(idx) <= len(heap):
            results = generateNewTetraRobot(heap=heap, masses=masses, springs=springs, faces=faces_new, idx=getChild1Idx(idx), musclePoints=musclePoints)

    # Build a new Tetra structure off of face 1
    if int(heap[idx-1][1]) == 1:
        tri_center = [(faces[1][0].posX+faces[1][1].posX+faces[1][2].posX)/3,
                      (faces[1][0].posY+faces[1][1].posY+faces[1][2].posY)/3,
                      (faces[1][0].posZ+faces[1][1].posZ+faces[1][2].posZ)/3]
        h = math.sqrt(2/3) * 2 # Assumes height of 1
        a = (faces[1][0].posX - faces[1][1].posX, faces[1][0].posY - faces[1][1].posY,
             faces[1][0].posZ - faces[1][1].posZ)
        b = (faces[1][0].posX - faces[1][2].posX, faces[1][0].posY - faces[1][2].posY,
             faces[1][0].posZ - faces[1][2].posZ)
        c = (1/np.linalg.norm(np.cross(a, b)))*np.cross(a, b)
        newX = tri_center[0] + h * c[0]
        newY = tri_center[1] + h * c[1]
        newZ = tri_center[2] + h * c[2]
        overlap_flag = False  # Check if mass is overlapping
        err = 0.5
        for mass in masses:
            if newX + err >= mass.posX >= newX - err \
                    and newY + err >= mass.posY >= newY - err \
                    and newZ + err >= mass.posZ >= newZ - err:
                overlap_flag = True
                repeatMass = mass
        if not overlap_flag:
            m = Mass(unit_mass, tri_center[0] + h*c[0], tri_center[1] + h*c[1], tri_center[2] + h*c[2], 0, 0, 0, 0, 0, 0)
            masses.append(m)
            springs.append(Spring(k=k_robot, rLength=calcMassDist(m, faces[1][0]), m1=faces[1][0], m2=m))
            springs.append(Spring(k=k_robot, rLength=calcMassDist(m, faces[1][1]), m1=faces[1][1], m2=m))
            springs.append(Spring(k=k_robot, rLength=calcMassDist(m, faces[1][2]), m1=faces[1][2], m2=m))
        else:
            m = repeatMass
            springFlag1 = False
            springFlag2 = False
            springFlag3 = False
            for spring in springs:
                if {spring.m1, spring.m2} == {faces[1][0], m}:
                    springFlag1 = True
                if {spring.m1, spring.m2} == {faces[1][1], m}:
                    springFlag2 = True
                if {spring.m1, spring.m2} == {faces[1][2], m}:
                    springFlag3 = True
            if not springFlag1:
                springs.append(Spring(k=k_robot, rLength=calcMassDist(m, faces[1][0]), m1=faces[1][0], m2=m))
            if not springFlag2:
                springs.append(Spring(k=k_robot, rLength=calcMassDist(m, faces[1][1]), m1=faces[1][1], m2=m))
            if not springFlag3:
                springs.append(Spring(k=k_robot, rLength=calcMassDist(m, faces[1][2]), m1=faces[1][2], m2=m))
        faces_new = [[faces[1][1], faces[1][2], m], [faces[1][0], faces[1][2], m], [faces[1][0], faces[1][1], m]]
        if getChild2Idx(idx) <= len(heap):
            results = generateNewTetraRobot(heap=heap, masses=masses, springs=springs,
                                                                        faces=faces_new, idx=getChild2Idx(idx), musclePoints=musclePoints)

    # Build a new Tetra structure off of face 0
    if int(heap[idx-1][2]) == 1:
        tri_center = [(faces[2][0].posX+faces[2][1].posX+faces[2][2].posX)/3,
                      (faces[2][0].posY+faces[2][1].posY+faces[2][2].posY)/3,
                      (faces[2][0].posZ+faces[2][1].posZ+faces[2][2].posZ)/3]
        h = math.sqrt(2/3) * 2 # Assumes height of 1
        a = (faces[2][0].posX - faces[2][1].posX, faces[2][0].posY - faces[2][1].posY,
             faces[2][0].posZ - faces[2][1].posZ)
        b = (faces[2][0].posX - faces[2][2].posX, faces[2][0].posY - faces[2][2].posY,
             faces[2][0].posZ - faces[2][2].posZ)
        c = (1/np.linalg.norm(np.cross(a, b)))*np.cross(a, b)
        newX = tri_center[0] - h * c[0]
        newY = tri_center[1] - h * c[1]
        newZ = tri_center[2] - h * c[2]
        overlap_flag = False  # Check if mass is overlapping
        err = 0.5
        for mass in masses:
            if newX + err >= mass.posX >= newX - err \
                    and newY + err >= mass.posY >= newY - err \
                    and newZ + err >= mass.posZ >= newZ - err:
                overlap_flag = True
                repeatMass = mass
        if not overlap_flag:
            m = Mass(unit_mass, tri_center[0] - h*c[0], tri_center[1] - h*c[1], tri_center[2] - h*c[2], 0, 0, 0, 0, 0, 0)
            masses.append(m)
            springs.append(Spring(k=k_robot, rLength=calcMassDist(m, faces[2][0]), m1=faces[2][0], m2=m))
            springs.append(Spring(k=k_robot, rLength=calcMassDist(m, faces[2][1]), m1=faces[2][1], m2=m))
            springs.append(Spring(k=k_robot, rLength=calcMassDist(m, faces[2][2]), m1=faces[2][2], m2=m))
        else:
            m = repeatMass
            springFlag1 = False
            springFlag2 = False
            springFlag3 = False
            for spring in springs:
                if {spring.m1, spring.m2} == {faces[2][0], m}:
                    springFlag1 = True
                if {spring.m1, spring.m2} == {faces[2][1], m}:
                    springFlag2 = True
                if {spring.m1, spring.m2} == {faces[2][2], m}:
                    springFlag3 = True
            if not springFlag1:
                springs.append(Spring(k=k_robot, rLength=calcMassDist(m, faces[2][0]), m1=faces[2][0], m2=m))
            if not springFlag2:
                springs.append(Spring(k=k_robot, rLength=calcMassDist(m, faces[2][1]), m1=faces[2][1], m2=m))
            if not springFlag3:
                springs.append(Spring(k=k_robot, rLength=calcMassDist(m, faces[2][2]), m1=faces[2][2], m2=m))
        faces_new = [[faces[2][1], faces[2][2], m], [faces[2][0], faces[2][2], m], [faces[2][0], faces[2][1], m]]
        if getChild3Idx(idx) <= len(heap):
            results = generateNewTetraRobot(heap=heap, masses=masses, springs=springs, faces=faces_new, idx=getChild3Idx(idx), musclePoints=musclePoints)

        while int(heap[-1][0]) == -1:
            heap = np.delete(heap, -1, 0)

    # Assign Muscle Points
    if float(heap[idx - 1][3]) != 0.0:
        musclePoints.append(MusclePoint(m=faces[0][2], b=heap[idx - 1][3], w=heap[idx - 1][4], c=heap[idx - 1][5]))

    results = [masses, springs, musclePoints]
    return results

def generateRandomTernaryHeap():
    # Build robot from genetic string heap
    # Fill an array with None types
    #heap = [None] * ((pow(3, max_tree_depth) - 1))
    heap = np.full((int((pow(3, max_tree_depth+1) - 1)/2), 6), -1.0)
    #print(heap)

    # Data structure: [[faces to branch off], [muscle point params], [tissue type]]
    # faces to branch: 0 (no) or 1 (yes)
    # Muscle Points: [0]: 0 (no point) or 1 (yes point), [1-3]: b, w, c
    heap[0] = np.array([random.randint(0, 1), random.randint(0, 1), random.randint(0, 1),
               0, 0, 0])
    while heap[0][0] == 0.0 and heap[0][1] == 0.0 and heap[0][2] == 0.0:
        heap[0] = np.array([random.randint(0, 1), random.randint(0, 1), random.randint(0, 1),
                            0, 0, 0])

    j = 1
    while j <= int((pow(3, max_tree_depth) - 1)/2):
        if heap[j-1][0] != -1 and j <= int((pow(3, max_tree_depth-1) - 1)/2):
            if heap[j-1][0]:
                heap[getChild1Idx(j)-1] = np.array([random.randint(0, 1), random.randint(0, 1), random.randint(0, 1),
                   0, 0, 0])
            if heap[j-1][1]:
                heap[getChild2Idx(j)-1] = np.array([random.randint(0, 1), random.randint(0, 1), random.randint(0, 1),
                   0, 0, 0])
            if heap[j-1][2]:
                heap[getChild3Idx(j)-1] = np.array([random.randint(0, 1), random.randint(0, 1), random.randint(0, 1),
                   0, 0, 0])
        elif heap[j-1][0] != -1 and j > int((pow(3, max_tree_depth-1) - 1)/2):
            if heap[j-1][0]:
                heap[getChild1Idx(j)-1] = np.array([0, 0, 0, 0, 0, 0])
            if heap[j-1][1]:
                heap[getChild2Idx(j)-1] = np.array([0, 0, 0, 0, 0, 0])
            if heap[j-1][2]:
                heap[getChild3Idx(j)-1] = np.array([0, 0, 0, 0, 0, 0])
        j += 1

    while int(heap[-1][0]) == -1:
        heap = np.delete(heap, -1, 0)

    # Assign Muscle Points
    depth = math.floor((math.log(2*len(heap)+1)/math.log(3))-1)
    #numMusclePoints = random.randint(1, depth)
    numMusclePoints = depth
    mp_idx = []
    for i in range(numMusclePoints):
        if i==0:
            val = random.randint(0,len(heap)-1)
            while heap[val][0] == -1:
                val = random.randint(0, len(heap) - 1)
            mp_idx.append(val)
        else:
            val = random.randint(0, len(heap)-1)
            while val in mp_idx or heap[val][0] == -1:
                val = random.randint(0, len(heap)-1)
            mp_idx.append(val)
    for i in range(len(mp_idx)):
        b = random.uniform(0.05, 0.15)
        w = random.uniform(8, 15)
        c = random.uniform(-1 * math.pi, math.pi)

        heap[mp_idx[i]][3] = b
        heap[mp_idx[i]][4] = w
        heap[mp_idx[i]][5] = c


    return heap


def record_state(masses, springs):
    X = []
    Y = []
    Z = []
    L = []
    for mass in masses:
        X.append(mass.posX)
        Y.append(mass.posY)
        Z.append(mass.posZ)
    for spring in springs:
        L.append(spring.rLength)

    return [X, Y, Z, L]

def reset_state(masses, springs, start_state):
    i = 0
    for mass in masses:
        mass.posX = start_state[0][i]
        mass.posY = start_state[1][i]
        mass.posZ = start_state[2][i]
        mass.velX = 0
        mass.velY = 0
        mass.velZ = 0
        mass.accX = 0
        mass.accY = 0
        mass.accZ = 0
        i+=1
    i = 0
    for spring in springs:
        spring.rLength = start_state[3][i]
        i+=1
    return 0


def getParentIdx(index):
    # return index of parent from child index
    # This is assuming a ternary tree
    result = math.floor((index+1)/3)
    return result

def getChild1Idx(index):
    # return index of left child from parent index
    result = (3*index)-1
    return result


def getChild2Idx(index):
    # return index of left child from parent index
    result = (3*index)
    return result

def getChild3Idx(index):
    # return index of left child from parent index
    result = (3*index)+1
    return result


def createOriginTetra(self):
    # Create origin Tetra of length 1
    masses = []
    springs = []
    faces = []

    masses = [
        Mass(unit_mass, 1, 0, (-1 / math.sqrt(2)), 0, 0, 0, 0, 0, 0),
        Mass(unit_mass, -1, 0, (-1 / math.sqrt(2)), 0, 0, 0, 0, 0, 0),
        Mass(unit_mass, 0, 1, (1 / math.sqrt(2)), 0, 0, 0, 0, 0, 0),
        Mass(unit_mass, 0, -1, (1 / math.sqrt(2)), 0, 0, 0, 0, 0, 0)]

    for i in range(0, len(masses)):
        j = i + 1
        while j < len(masses):
            springs.append(Spring(k=self.k, rLength=calcMassDist(masses[i], masses[j]),
                                     m1=masses[i], m2=masses[j]))
            j += 1

    faces = [[masses[1], masses[2], masses[3]],
             [masses[0], masses[2], masses[3]],
             [masses[0], masses[1], masses[3]]]

    return [masses, springs, faces]

def createStaticTetra(self):
    massList = []
    springList = []

    massList = [
        Mass(unit_mass, self.origin[0] + self.length, self.origin[1],
             self.origin[2]+ (self.length*(-1/math.sqrt(2))), 0, 0, 0, 0, 0, 0),
        Mass(unit_mass, self.origin[0] + (self.length*-1), self.origin[1],
             self.origin[2] + (self.length * (-1 / math.sqrt(2))), 0, 0, 0, 0, 0, 0),
        Mass(unit_mass, self.origin[0], self.origin[1] + self.length,
             self.origin[2] + (self.length*(1/math.sqrt(2))), 0, 0, 0, 0, 0, 0),
        Mass(unit_mass, self.origin[0], self.origin[1] + (self.length*-1),
             self.origin[2] + (self.length * (1 / math.sqrt(2))), 0, 0, 0, 0, 0, 0)
    ]

    for i in range(0, len(massList)):
        j = i + 1
        while j < len(massList):
            springList.append(Spring(k=self.k, rLength=calcMassDist(massList[i], massList[j]),
                                     m1=massList[i], m2=massList[j]))
            j += 1

    return [massList, springList]


def calcMassDist(m1,m2):
    # Returns total distance (abs value) between two masses
    # Parameters:
    # m1: First Mass object
    # m2: Second Mass object

    return math.sqrt(pow(m1.posX-m2.posX,2)+pow(m1.posY-m2.posY,2)+pow(m1.posZ-m2.posZ,2))


def init():
    line0.set_data([], [])
    line0.set_3d_properties([])
    return line0,

def animate(i, line, massList, springList, musclePoints=[]):
    # Animate simulation for given masses
    X = []
    Y = []
    Z = []

    for i in range(0,100):
        applyForces(massList, springList, musclePoints)
        updateVelocities(massList)
        updatePositions(massList)



    # Create X,Y,Z for each spring/mass
    for spring in springList:
        X.append(spring.m1.posX)
        X.append(spring.m2.posX)
        Y.append(spring.m1.posY)
        Y.append(spring.m2.posY)
        Z.append(spring.m1.posZ)
        Z.append(spring.m2.posZ)


    line0.set_data(X, Y)
    line0.set_3d_properties(Z)

    return line0,


def execute_sim(args):
    iterations = args[0]
    heap = args[1]
    masses, springs, musclePoints = generateNewTetraRobot(heap)
    gravity_a = [0, 0, -9.81]
    global T
    T=0

    # Bring robot to floor
    min_z = float('inf')
    for mass in masses:
        if mass.posZ < min_z:
            min_z = mass.posZ
    for mass in masses:
        mass.posZ -= min_z
    x_sum = 0
    y_sum = 0
    for mass in masses:
        x_sum += mass.posX
        y_sum += mass.posY
    starting_x = x_sum/len(masses)
    starting_y = y_sum/len(masses)
    try:
        # Execute simulation
        for i in range(0, iterations):
            applyForces(masses, springs, musclePoints)
            updateVelocities(masses)
            updatePositions(masses)

            # Detect if model is unstable
            threshold = 15000
            if i%100 == 0:
                for mass in masses:
                    if abs(mass.accX) > threshold or abs(mass.accY) > threshold or abs(mass.accZ) > threshold:
                        print('Model is unstable!')
                        return 0
    except:
        return 0

    x_sum_final = 0
    y_sum_final = 0
    for mass in masses:
        x_sum_final += mass.posX
        y_sum_final += mass.posY
    ending_x = x_sum_final / len(masses)
    ending_y = y_sum_final / len(masses)
    speed_final = math.sqrt(pow(ending_x-starting_x, 2) + pow(ending_y-starting_y, 2))/(dt*iterations)
    del masses
    del springs
    del musclePoints

    return speed_final

def physics_cupy(args):
    # Physics sim using cp/np arrays
    iterations = args[0]
    heap = args[1]
    massList, springList, musclePoints = generateNewTetraRobot(heap)
    gravity_a = [0, 0, -9.81]
    global T
    T = 0
    time_st = time.time()

    # Initialize mass np array
    m_np = np.array([[massList[0].posX, massList[0].posY, massList[0].posZ, massList[0].velX,
                     massList[0].velY, massList[0].velZ, massList[0].accX, massList[0].accY, massList[0].accZ]])

    for j in range(1, len(massList)):
        m_np = np.concatenate((m_np, [[massList[j].posX, massList[j].posY, massList[j].posZ, massList[j].velX,
                                       massList[j].velY, massList[j].velZ, massList[j].accX, massList[j].accY,
                                       massList[j].accZ]]), axis=0)

    # Initialize spring np array
    m1_idx = -1
    m2_idx = -1
    for m in range(0, len(massList)):
        if springList[0].m1 == massList[m]:
            m1_idx = m
        elif springList[0].m2 == massList[m]:
            m2_idx = m
    if m1_idx == -1 or m2_idx == -1:
        print('Error finding mass index')
        exit()
    s_np = np.array([[springList[0].rLength, springList[0].rLength_a, m1_idx, m2_idx,
                      massList[0].posX-massList[1].posX,
                      massList[0].posY-massList[1].posY,
                      massList[0].posZ-massList[1].posZ]])
    for j in range(1, len(springList)):
        m1_idx = -1
        m2_idx = -1
        for m in range(0, len(massList)):
            if springList[j].m1 == massList[m]:
                m1_idx = m
            elif springList[j].m2 == massList[m]:
                m2_idx = m
        if m1_idx == -1 or m2_idx == -1:
            print('Error finding mass index')
            exit()
        # [rLength, rLength_a, m1 index, m2 index, Lx, Ly, Lz]
        s_np = np.concatenate((s_np, [[springList[j].rLength, springList[j].rLength_a, m1_idx, m2_idx,
                                       massList[m1_idx].posX-massList[m2_idx].posX,
                                       massList[m1_idx].posY-massList[m2_idx].posY,
                                       massList[m1_idx].posZ-massList[m2_idx].posZ]]), axis=0)

    # Create numpy list of musclePoints
    if musclePoints != []:
        m_idx = -1
        for m in range(0, len(massList)):
            if musclePoints[0].m == massList[m]:
                m_idx = m
        if m_idx == -1:
            print('Error finding mass index')
            exit()
        mp_np = np.array([[musclePoints[0].b, musclePoints[0].w, musclePoints[0].c, m_idx]])
        for j in range(1, len(musclePoints)):
            m_idx = -1
            for m in range(0, len(massList)):
                if musclePoints[j].m == massList[m]:
                    m_idx = m
            if m_idx == -1:
                print('Error finding mass index')
                exit()
            mp_np = np.concatenate((mp_np, [[musclePoints[j].b, musclePoints[j].w, musclePoints[j].c, m_idx]]))

    # Define mapping matrix for springs to masses
    s2m = np.zeros(shape=(len(massList), len(springList))) # Force mapping function
    s2m_ap = np.zeros(shape=(len(springList), len(massList))) # average spring position
    for i in range(0, len(springList)):
        s2m[int(s_np[i, 2])][i] = 1
        s2m[int(s_np[i, 3])][i] = -1
        s2m_ap[i][int(s_np[i, 2])] = 0.5
        s2m_ap[i][int(s_np[i, 3])] = 0.5

    # Define mapping matrix for masses to springs (transpose of s2m)
    m2s = s2m.transpose()

    # Define mapping matrix for musclepoints to mass positions
    m2mp = np.zeros(shape=(len(musclePoints), len(massList)))
    for i in range(0, len(musclePoints)):
        m2mp[i][int(mp_np[i, 3])] = 1

    y_sum = np.sum(m_np[:, 1])
    x_sum = np.sum(m_np[:, 0])
    starting_x = x_sum / len(massList)
    starting_y = y_sum / len(massList)

    np.seterr(invalid='ignore')
    for i in range(iterations):
        # Zero out acceleration
        m_np[:, 6:9] *= 0

        # Add gravity constant
        m_np[:, 6] += gravity_a[0]
        m_np[:, 7] += gravity_a[1]
        m_np[:, 8] += gravity_a[2]

        # Update spring Lengths
        s_np[:, 4:7] = np.matmul(m2s, m_np[:, 0:3])

        # Apply muscle Actuation
        mp_mpos = np.matmul(m2mp, m_np[:, 0:3])
        s_pos_avg = np.matmul(s2m_ap, m_np[:, 0:3])
        muscleSum = np.zeros(len(springList))
        if len(musclePoints) != 0:
            for k in range(0, len(musclePoints)):
                mp_mpos_rs = np.resize(mp_mpos[k], (len(springList), 3))
                dist_r = np.reciprocal(
                    np.sqrt(np.square(s_pos_avg[:, 0] - mp_mpos_rs[:, 0]) + np.square(s_pos_avg[:, 1] - mp_mpos_rs[:, 1]) +
                            np.square(s_pos_avg[:, 2] - mp_mpos_rs[:, 2])))

                sin_np = np.multiply(mp_np[k, 0], np.sin(T*mp_np[k, 1] + mp_np[k, 2]))
                mp_fact = np.multiply(s_np[:, 1], sin_np*dist_r)
                muscleSum += mp_fact
            s_np[:, 0] = s_np[:, 1] + muscleSum


        # Add spring forces
        # F = k * (rLength - Length)
        L = np.sqrt(np.add(np.add(np.square(s_np[:, 4]), np.square(s_np[:, 5])), np.square(s_np[:, 6])))
        Fs = k_robot*np.subtract(s_np[:, 0], L)
        Fs_3 = np.array([Fs, Fs, Fs]).transpose()
        L_3 = np.array([np.reciprocal(L),np.reciprocal(L),np.reciprocal(L)]).transpose()
        Fs_n = np.multiply(np.multiply(L_3, s_np[:, 4:7]), Fs_3)
        m_np[:, 6:9] = np.add(m_np[:, 6:9], np.matmul(s2m, Fs_n))

        # Apply normal forces and friction
        Fnorm = np.where(m_np[:, 2] < 0, abs(k_wall*m_np[:, 2]), 0)
        m_np[:, 8] = m_np[:, 8] + Fnorm
        F_h = np.sqrt(np.square(m_np[:, 6]) + np.square(m_np[:, 7]))
        F_hn = np.subtract(F_h, u_frict_k * Fnorm)
        x_r = np.where(F_h != 0.0, np.divide(m_np[:, 6], F_h), 0)
        y_r = np.where(F_h != 0.0, np.divide(m_np[:, 7], F_h), 0)
        m_np[:, 6] = np.where(m_np[:, 2] < 0, np.where(F_h < u_frict_s * Fnorm, 0, np.multiply(F_hn, x_r)), m_np[:, 6])
        m_np[:, 7] = np.where(m_np[:, 2] < 0, np.where(F_h < u_frict_s * Fnorm, 0, np.multiply(F_hn, y_r)), m_np[:, 7])


        # Update velocities of masses (with dampening)
        m_np[:, 3:6] = 0.999*np.add(m_np[:, 3:6], dt * m_np[:, 6:9])

        # Update positions of masses
        m_np[:, 0:3] = np.add(m_np[:, 0:3], dt * m_np[:, 3:6])

    y_sum_final = np.sum(m_np[:, 1])
    x_sum_final = np.sum(m_np[:, 0])

    ending_x = x_sum_final / len(massList)
    ending_y = y_sum_final / len(massList)
    del m_np
    del s_np
    if len(musclePoints) != 0:
        del mp_np

    speed_final = math.sqrt(pow(ending_x - starting_x, 2) + pow(ending_y - starting_y, 2)) / (dt * iterations)

    return speed_final



def applyForces(massList, springList, musclePoints=[]):
    # Apply all net forces acting on each mass in massList
    global T
    global spring_evals
    global spring_eval_plot
    gravity_a = [0, 0, -9.81]
    T = T + dt

    # Zero all accelerations
    for mass in massList:
        mass.accX = 0.0
        mass.accY = 0.0
        mass.accZ = 0.0


    # Apply Gravity
    for mass in massList:
        mass.accX = mass.accX + gravity_a[0]
        mass.accY = mass.accY + gravity_a[1]
        mass.accZ = mass.accZ + gravity_a[2]

    # Apply muscle actuations
    for i in range(0, len(springList)):
        for musclePoint in musclePoints:
            avgX = (springList[i].m1.posX + springList[i].m2.posX) / 2
            avgY = (springList[i].m1.posY + springList[i].m2.posY) / 2
            avgZ = (springList[i].m1.posZ + springList[i].m2.posZ) / 2
            dist = math.sqrt(
                pow(avgX - musclePoint.m.posX, 2) + pow(avgY - musclePoint.m.posY, 2) + pow(avgZ - musclePoint.m.posZ,
                                                                                            2))
            # springList[i].rLength = springList[i].rLength * (
            # 1 + (1 / dist) * musclePoint.b * (math.sin(musclePoint.w * T + musclePoint.c)))
            springList[i].rLength = springList[i].rLength_a * (
                        1 + (1 / dist) * musclePoint.b * (math.sin(musclePoint.w * T + musclePoint.c)))
    
    # Apply spring forces
    for spring in springList:
        distVect = np.array([spring.m1.posX - spring.m2.posX, spring.m1.posY - spring.m2.posY, spring.m1.posZ - spring.m2.posZ])
        springLength = calcMassDist(spring.m1, spring.m2)
        distVect = distVect*(1/springLength)
        springForce = spring.k * (spring.rLength - springLength)
        spring.m1.accX = spring.m1.accX + (springForce * distVect[0])
        spring.m1.accY = spring.m1.accY + (springForce * distVect[1])
        spring.m1.accZ = spring.m1.accZ + (springForce * distVect[2])
        spring.m2.accX = spring.m2.accX - (springForce * distVect[0])
        spring.m2.accY = spring.m2.accY - (springForce * distVect[1])
        spring.m2.accZ = spring.m2.accZ - (springForce * distVect[2])



    # Apply normal forces + friction
    for mass in massList:
        # apply normal force from the floor
        if mass.posZ < Zfloor:
            F_n = k_wall * (Zfloor - mass.posZ)
            mass.accZ = mass.accZ + F_n
            F_h = math.sqrt(pow(mass.accX*mass.mass,2) + pow(mass.accY*mass.mass,2))
            if F_h < abs(F_n*u_frict_s):
                mass.accX = 0
                mass.accY = 0
            else:
                F_hn = F_h - (u_frict_k*F_n)
                X_R = (mass.accX * unit_mass) / F_h
                Y_R = (mass.accY * unit_mass) / F_h
                mass.accX = (F_hn/unit_mass)*X_R
                mass.accY = (F_hn/unit_mass)*Y_R


    dampening_const = 0.999
    # Apply dampening
    for mass in massList:
        mass.velX = mass.velX * dampening_const
        mass.velY = mass.velY * dampening_const
        mass.velZ = mass.velZ * dampening_const


def updateVelocities(massList):
    # Update mass velocities based on latest acceleration

    for mass in massList:
        mass.velX = mass.velX + (dt * mass.accX)
        mass.velY = mass.velY + (dt * mass.accY)
        mass.velZ = mass.velZ + (dt * mass.accZ)


def updatePositions(massList):
    # Update mass positions based on latest velocities

    for mass in massList:
        mass.posX = mass.posX + (dt * mass.velX)
        mass.posY = mass.posY + (dt * mass.velY)
        mass.posZ = mass.posZ + (dt * mass.velZ)

def findMassIdx(robot, mass):
    for n in range(len(robot.struct.masses)):
        if mass == robot.struct.masses[n]:
            massIdx = n
    return massIdx


def mutateMuscle(heap):
    # Mutate a random muscle in robot heap
    muscleIdxList = []
    for i in range(0, len(heap)):
        if heap[i][3] != 0.0 and heap[i][3] != -1.0:
            muscleIdxList.append(i)
    if muscleIdxList == []:
        return heap
    randPoint = random.choice(muscleIdxList)
    randElement = random.choice(['b', 'w', 'c'])
    randFactor = random.uniform(0.9, 1.1)

    if randElement == 'b':
        heap[randPoint][3] = heap[randPoint][3]*randFactor
    elif randElement == 'w':
        heap[randPoint][4] = heap[randPoint][4]*randFactor
    elif randElement == 'c':
        heap[randPoint][5] = heap[randPoint][5]*randFactor

    return heap

def crossover_linked(heap1, heap2):
    # Performs GA crossover between two heaps and produces a child heap
    if len(heap1) < len(heap2):
        randIdx = random.randint(1, len(heap1))
        while heap1[randIdx-1][0] == -1.0 or heap2[randIdx-1][0] == -1.0:
            randIdx = random.randint(1, len(heap1))
    else:
        randIdx = random.randint(1, len(heap2))
        while heap1[randIdx-1][0] == -1.0 or heap2[randIdx-1][0] == -1.0:
            randIdx = random.randint(1, len(heap2))

    heap2_subtree = build_subtree(heap=heap2, idx=randIdx)

    heap_new = replace_branch(heap=heap1, heap_subtree=heap2_subtree, idx=randIdx)

    return heap_new

def crossover_unlinked(heap1, heap2):
    # Performs GA crossover between two heaps and produces a child heap
    randIdx1 = random.randint(1, len(heap1))
    randIdx2 = random.randint(1, len(heap2))
    while heap1[randIdx1-1][0] == -1.0:
        randIdx1 = random.randint(1, len(heap1))
    while heap2[randIdx2-1][0] == -1.0:
        randIdx2 = random.randint(1, len(heap2))

    heap2_subtree = build_subtree(heap=heap2, idx=randIdx2)

    heap_new = replace_branch(heap=heap1, heap_subtree=heap2_subtree, idx=randIdx1)

    return heap_new

def breed_generation(population_list, best_heap, linked=False, selection_pressure=2):
    # Breeds a population to create the next generation

    # Sort population based on simulation speed
    population_list.sort(key = lambda x: x[1], reverse=True)
    pop_size = len(population_list)
    appended_population_list = population_list[:math.floor(pop_size/selection_pressure)]
    del population_list

    new_random_heap = generateRandomTernaryHeap()
    appended_population_list.append([new_random_heap, 0.0])
    appended_population_list.append([best_heap, 0.0])

    new_population = []
    for i in range(0, pop_size):
        parent1Idx = random.randint(0, len(appended_population_list)-1)
        parent2Idx = random.randint(0, len(appended_population_list)-1)
        while parent1Idx == parent2Idx:
            parent2Idx = random.randint(0, len(appended_population_list)-1)
        if linked:
            new_population.append([crossover_linked(appended_population_list[parent1Idx][0], appended_population_list[parent2Idx][0]), 0.0])
        else:
            new_population.append([crossover_unlinked(appended_population_list[parent1Idx][0], appended_population_list[parent2Idx][0]), 0.0])

    return new_population


def prune_heap(heap, max_depth):
    max_length = int((pow(3, max_depth+1)-1)/2)
    while len(heap) >= max_length:

        if heap[-1][0] == -1.0:
            heap = np.delete(heap, -1, axis=0)
        else:
            parentIdx = getParentIdx(len(heap))
            while parentIdx >= max_length:
                parentIdx = getParentIdx(parentIdx)
            heap = snip_tree(heap=heap, idx=parentIdx)

    return heap

def snip_tree(heap, idx, start_idx=0, flag=False, muscleSum=[0.0, 0.0, 0.0, 0]):
    # Snips tree for prune heap to include muscle points
    if not flag:
        start_idx = idx
        flag=True
    # Snip tree starting at idx and sum muscle points
    if heap[idx-1][0] == 1:
        childIdx = getChild1Idx(idx)
        if childIdx <= len(heap):
            heap = snip_tree(heap=heap, idx=childIdx, start_idx=start_idx, flag=flag, muscleSum=muscleSum)
    if heap[idx-1][1] == 1:
        childIdx = getChild2Idx(idx)
        if childIdx <= len(heap):
            heap = snip_tree(heap=heap, idx=childIdx, start_idx=start_idx, flag=flag, muscleSum=muscleSum)
    if heap[idx-1][2] == 1:
        childIdx = getChild3Idx(idx)
        if childIdx <= len(heap):
            heap = snip_tree(heap=heap, idx=childIdx, start_idx=start_idx, flag=flag, muscleSum=muscleSum)

    if heap[idx-1][3] != 0.0:
        muscleSum[0] += heap[idx-1][3]
        muscleSum[1] += heap[idx-1][4]
        muscleSum[2] += heap[idx-1][5]
        muscleSum[3] += 1
    if idx == start_idx:
        heap[idx-1][0] = 0.0
        heap[idx-1][1] = 0.0
        heap[idx-1][2] = 0.0
        if muscleSum[3] != 0:
            heap[idx-1][3] = float(muscleSum[0] / muscleSum[3])
            heap[idx-1][4] = float(muscleSum[1] / muscleSum[3])
            heap[idx-1][5] = float(muscleSum[2] / muscleSum[3])
    else:
        # Replace idx with empty list (-1)
        heap[idx-1] = np.full(6, -1.0)

    return heap

def replace_branch(heap, heap_subtree, idx, idx_sub=1, heap_cp = [], cleared=False, start_idx = 1):

    if not cleared:
        heap_cp = np.copy(heap)
        for i in range(0,len(heap_subtree)):
            if float(heap_subtree[i][0]) != -1.0:
                idx_sub=i+1
                break
        heap_cp = clear_subtree(heap_cp, idx)
        cleared = True
        start_idx = idx
    timeout = 0
    if len(heap_cp) < idx:
        # Pad heap with empty values
        heap_cp = np.concatenate((heap_cp, np.full((idx-len(heap_cp), 6), -1.0)), axis=0)
    if heap_subtree[idx_sub-1][0] == 1:
        childIdx = getChild1Idx(idx)
        childIdxSub = getChild1Idx(idx_sub)
        if childIdxSub < len(heap_subtree)+1:
            heap_cp = replace_branch(heap, heap_subtree, childIdx, childIdxSub, heap_cp, cleared, start_idx)
    if heap_subtree[idx_sub-1][1] == 1:
        childIdx = getChild2Idx(idx)
        childIdxSub = getChild2Idx(idx_sub)
        if childIdxSub < len(heap_subtree)+1:
            heap_cp = replace_branch(heap, heap_subtree, childIdx, childIdxSub, heap_cp, cleared, start_idx)
    if heap_subtree[idx_sub-1][2] == 1:
        childIdx = getChild3Idx(idx)
        childIdxSub = getChild3Idx(idx_sub)
        if childIdxSub < len(heap_subtree)+1:
            heap_cp = replace_branch(heap, heap_subtree, childIdx, childIdxSub, heap_cp, cleared, start_idx)
    heap_cp[idx-1] = heap_subtree[idx_sub-1]

    if idx == start_idx:
        while heap_cp[-1][0] == -1:
            heap_cp = np.delete(heap_cp, -1, axis=0)

    return heap_cp

def build_subtree(heap, idx, heap_subtree = []):
    # Creates a subtree of length = len(heap)
    # Containing values of subtree off of
    # index idx-1 and empty (-1) for all other nodes
    # Recursive function

    # Define heap_subtree of length of heap
    if len(heap_subtree) == 0:
        heap_subtree = np.full((len(heap), 6), -1.0)

    if heap[idx-1][0] == 1:
        childIdx = getChild1Idx(idx)
        if childIdx < len(heap)+1:
            heap_subtree = build_subtree(heap, childIdx, heap_subtree)
    if heap[idx-1][1] == 1:
        childIdx = getChild2Idx(idx)
        if childIdx < len(heap)+1:
            heap_subtree = build_subtree(heap, childIdx, heap_subtree)
    if heap[idx-1][2] == 1:
        childIdx = getChild3Idx(idx)
        if childIdx < len(heap)+1:
            heap_subtree = build_subtree(heap, childIdx, heap_subtree)

    heap_subtree[idx-1] = heap[idx-1]

    return heap_subtree

def clear_subtree(heap, idx):
    # Clear out subtree starting at index idx
    if heap[idx-1][0] == 1:
        childIdx = getChild1Idx(idx)
        if childIdx < len(heap)+1:
            heap = clear_subtree(heap, childIdx)
    if heap[idx-1][1] == 1:
        childIdx = getChild2Idx(idx)
        if childIdx < len(heap) + 1:
            heap = clear_subtree(heap, childIdx)
    if heap[idx-1][2] == 1:
        childIdx = getChild3Idx(idx)
        if childIdx < len(heap) + 1:
            heap = clear_subtree(heap, childIdx)

    # Replace idx with empty list (-1)
    heap[idx-1] = np.full(6, -1.0)

    return heap


def GA_Optimize(generations=100, population_size=5, simulation_frames=50000):
    # Cross-breeding optimization using GA for fastest robot (+x direction)
    gravity_a = [0, 0, -9.81]
    prune_depth = 7
    best_speed = float('-inf')
    best_heap = []
    best_generation = 0
    p_mutation = 0.2 # Probability of mutation
    p_crossover_linked = 0.5 # Probability of crossover linked
    p_crossover_unlinked = 0.3 # Probability of crossover unlinked
    operators_list = [['mutation', p_mutation], ['crossover_linked', p_crossover_linked],
                      ['crossover_unlinked', p_crossover_unlinked]]

    population_list = []
    for i in range(0, population_size):
        population_list.append([generateRandomTernaryHeap(), 0.0])
    try:
        for i in range(0, generations):
            print(f'Generation: {i}')

            # Prune heaps that are too large
            for p in population_list:
                p[0] = prune_heap(p[0], prune_depth)

            # Append arguments list for pool simulator
            args_list = []
            for p in population_list:
                #[masses, springs, musclePoints] = generateNewTetraRobot(p[0])
                args_list.append([simulation_frames, p[0]])

            print('Executing Simulations...')
            st_time = time.time()
            with Pool() as p:
                results = p.map(physics_cupy, args_list)

            print('Simulation Done')
            print(f'Simulation Time: {math.floor(time.time()-st_time)} seconds')
            process = psutil.Process(os.getpid())
            print(process.memory_info().rss)  # in bytes
            lengths = []
            for j in range(0, len(results)):
                population_list[j][1] = results[j]
                lengths.append(len(population_list[j][0]))

            population_list.sort(key=lambda x: x[1], reverse=True)
            if population_list[0][1] > best_speed:
                best_speed = population_list[0][1]
                best_heap = population_list[0][0]
                best_generation = i
                if best_speed > 0:
                    print(f'New Best Speed: {best_speed}')
            else:
                if best_speed > 0:
                    print(f'Previous Best Speed: {best_speed}')
                    print(f'Last Generation of improvement: {best_generation}')

            operator_val = random.uniform(0,1)
            sum = 0
            operation = ''
            for operator in operators_list:
                sum += operator[1]
                if sum >= operator_val:
                    operation = operator[0]
                    break

            print(f'Operation: {operation}')
            if operation == 'crossover_linked':
                population_list = breed_generation(population_list, best_heap=best_heap, linked=True)
            elif operation == 'crossover_unlinked':
                population_list = breed_generation(population_list, best_heap=best_heap, linked=False)
            elif operation == 'mutation':
                for k in range(0,len(population_list)):
                    population_list[k][0] = mutateMuscle(population_list[k][0])
            else:
                print('INVALID OPERATOR... EXITING...')
                exit()
    except Exception as e:
        print(e)

    print(f'Best Speed: {best_speed}')
    now = datetime.datetime.now()

    file = "/home/elo2124/saved_heaps/robot_" + now.strftime("%Y%m%d%H%M%S") + ".txt"
    print(f'Writing to file: {file}')
    with open(file, 'w') as f:
        f.write('[')
        for h in best_heap:
            f.write(f'[{h[0]}')
            for v in range(1,len(h)):
                f.write(f', {h[v]}')
            f.write('], \n')
        f.write(']')


    return best_speed, best_heap

if __name__ == '__main__':
    # Main function
    test = 'numpy evolve' # Change this parameter for different simulations
    global gravity_a
    start_time = time.time()

    if test == 'simple bounce':
        cube1 = Cube([3,3,6], 2, 10000)
        objList = [cube1]
        breathing = False
        gravity_a = [0,0,-9.81]
    elif test == 'Optimize Robot':
        gravity_a = [0, 0, -9.81]
        best_speed, best_heap = GA_Optimize()
        [masses, springs, musclePoints] = generateNewTetraRobot(best_heap)
        # Bring robot to floor
        min_z = float('inf')
        for mass in masses:
            if mass.posZ < min_z:
                min_z = mass.posZ
        for mass in masses:
            mass.posZ -= min_z
    elif test == 'numpy evolve':
        gravity_a = [0, 0, -9.81]
        best_speed, best_heap = GA_Optimize()
        [masses, springs, musclePoints] = generateNewTetraRobot(best_heap)
        # Bring robot to floor
        min_z = float('inf')
        for mass in masses:
            if mass.posZ < min_z:
                min_z = mass.posZ
        for mass in masses:
            mass.posZ -= min_z




    # Start simulation with input cubes
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   fargs=(line0, masses, springs, musclePoints),
                                   frames=1000, interval=1, repeat=False)


    show_anim = False
    if show_anim:
        plt.show() # Show simulation

    record=True # Change to True to record Gif of simulation
    if record:
        now = datetime.datetime.now()
        f = "/home/elo2124/saved_videos/robot_" + now.strftime("%Y%m%d%H%M%S") +  ".gif"
        print(f'Saving file to {f}')
        writergif = animation.PillowWriter(fps=30)
        anim.save(f, writer=writergif)

    end_time = time.time()
    print('Run time: {} seconds'.format(end_time-start_time))
