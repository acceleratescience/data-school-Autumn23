
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 18:05:09 2020
@author: RKDaniels
The simulation of electrical current through the NWN.

TODO There is probably time to be gained. I have experimented with using Cholesky decomposition to solve the linear
system but it was slower than just using spsolve. 
TODO Switching! The junctions don't switch, but some of the commented out code exists from a previous attempt
"""
import numpy as np
import networkx as nx
import scipy as sp


class MemristorSim:
    def __init__(self, network, simParams):
        self.network = network
        self.simParams = simParams
        self.edges = np.array(np.where(np.triu(network) == 1)).T
        self.junctionCount = len(self.edges)
        self.inputElectrodes = np.array([0])
        self.outputElectrodes = np.array([0])
        self.networkCurrent = np.zeros(0)
        self.wireVoltage = np.zeros(self.network.shape[0])
        self.junctionVoltage = np.zeros(self.junctionCount)
        self.junctionFilament = np.zeros(self.junctionCount)
        self.junctionG = np.zeros(self.junctionCount) + 1/simParams['R_OFF']
        # self.isSwitched = False
        self.THETA = self.simParams['THETA']
        z_0 = self.simParams['z_0']
        var = self.simParams['var']
        if self.simParams['dist'] == 'uniform':
            self.z_0 = np.abs(np.random.uniform(var[0],var[1], self.junctionCount))
        if self.simParams['dist'] == 'normal':
            self.z_0 = np.random.normal(z_0, var*z_0, self.junctionCount)


    def applyVoltage(self, voltageVector, outputs):
        # elif electrodeType == 'single':
        #     self.inputElectrodes, self.outputElectrodes = self._get_electrodes_single()
        dt = 1
        time = np.arange(0, len(voltageVector), 1)
        inputs = voltageVector.shape[1]

        self.inputElectrodes = np.arange(0, inputs, 1, dtype=int)
        self.outputElectrodes = np.arange(inputs, outputs + inputs, 1, dtype=int)
        # wireVoltages = np.zeros((len(time), self.network.nWires))
        # junctionVoltages = np.zeros((len(time), self.junctionCount))
        junctionConductances = np.zeros((len(time), self.junctionCount))
        self.networkCurrent = np.zeros((len(time), len(self.outputElectrodes)))
        self.networkZs = np.zeros((len(time), self.junctionCount))
        edges = np.array(np.where(np.triu(self.network) == 1)).T
        times = np.zeros(len(time))
        
        for i in range(len(time)):
            appliedVoltage = np.array([voltageVector[i], 0], dtype=object)
            # junctionVoltages[i, :] = self.junctionVoltage
            junctionConductances[i, :] = self.junctionG
            wire_voltage = self.updateJunction(edges, appliedVoltage, self.inputElectrodes, self.outputElectrodes, dt, i)
            # wireVoltages[i,:] = wire_voltage
            # self.networkZs[i,:] = self.junctionFilament
        return self.networkCurrent, junctionConductances#, wireVoltages, junctionVoltages, junctionConductances, self.networkZs


    def updateJunction(self, edges, appliedVoltage, inputElectrodes, outputElectrodes, dt, j):
        # Kirchhoff's Laws
        nWires = self.network.shape[0]
        z = np.zeros(nWires + len(inputElectrodes) + len(outputElectrodes))
        A = np.zeros((nWires + len(inputElectrodes) + len(outputElectrodes),
                      nWires + len(inputElectrodes) + len(outputElectrodes)))
        G = np.zeros((nWires, nWires))
        G[edges[:, 0], edges[:, 1]] = self.junctionG
        G[edges[:, 1], edges[:, 0]] = self.junctionG
        G = np.diag(np.sum(G, axis=0)) - G
        A[:nWires, :nWires] = G
        for i, e in enumerate(inputElectrodes):
            A[nWires  + i, e] = 1
            A[e, nWires + i] = 1
            z[nWires + i] = appliedVoltage[0][i]
        for i, e in enumerate(outputElectrodes):
            A[nWires + len(inputElectrodes) + i, e] = 1
            A[e, nWires + len(inputElectrodes) + i] = 1
            z[nWires + len(inputElectrodes) + i] = appliedVoltage[1]
        A = sp.sparse.csc_matrix(A)
        x = sp.sparse.linalg.spsolve(A, z)
        wire_voltage = x[0:nWires]
        self.networkCurrent[j, :] = x[nWires+len(inputElectrodes):]
        self.junctionVoltage = wire_voltage[edges[:, 0]] - wire_voltage[edges[:, 1]]
    
        # Junction dynamics
        MU = self.simParams['MU'] 
        KAPPA = self.simParams['KAPPA']
        # THETA = 10
        T = 10
        THETA_0 = 26
        dz = (THETA_0/(T*self.THETA))*( ( ( MU * (np.abs(self.junctionVoltage)) / (self.z_0 - self.junctionFilament))) - ( KAPPA * (self.junctionFilament) ) )
        self.junctionFilament = self.junctionFilament + dz
        # mask = np.where(self.junctionFilament >= self.z_0)
        # if len(mask[0]) > 0:
            # self.isSwitched = True
        # self.junctionFilament[mask] = 0#self.z_0[mask]
        # self.junctionFilament[self.junctionFilament < 0.0] = 0.0
        alpha = 1
        beta = 200
        Conductance = np.exp(-beta*(self.z_0 - self.junctionFilament))
        self.junctionG = Conductance
        
        return wire_voltage


    def _get_wire_currents(self):
        edge_list = np.array(np.where(np.triu(self.network) == 1)).T
        diMat = np.zeros(self.network.shape)
        diMat[edge_list[:,0], edge_list[:,1]] = np.sign(self.junctionVoltage)
        diMat = diMat-diMat.T
        diMat[diMat<0] = 0
        DiGraph = nx.from_numpy_array(diMat, create_using=nx.DiGraph())
        diMat = np.array(nx.adjacency_matrix(DiGraph).todense())
        wireCurrents = np.zeros(self.network.shape[0])
        junctionCurrent = self.junctionVoltage*self.junctionG
        for i in range(self.network.shape[0]):
            outGoing = np.where(diMat[i,:])[0]
            junctionIdx = [self.findJunctionIndex(edge_list, i, j) for j in outGoing]
            wireCurrents[i] = np.sum(abs(junctionCurrent[junctionIdx]))

        return wireCurrents


    @staticmethod
    def findJunctionIndex(edge_list, wire1, wire2):
        index = np.where((edge_list[:,0] == wire1) & (edge_list[:,1] == wire2))[0]
        if len(index) == 0:
            index = np.where((edge_list[:,0] == wire2) & (edge_list[:,1] == wire1))[0]
        if len(index) == 0:
            print(f'Unfortunately, no junction between nanowire {wire1} and {wire2}')
            return None
        else:
            return index[0]