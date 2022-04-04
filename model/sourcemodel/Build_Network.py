# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 09:56:09 2019

"""

import torch.nn as nn
import torch.nn.functional as F
class first_layer_cascade_Net(nn.Module):
    def _int_(self,layer_index,hyper_parameters,layer_output_size):
        super(first_layer_cascade_Net, self).__init__()
        self.layer_index=layer_index
        self.hyper_parameters=hyper_parameters
        # =====================================================
        self.conv1 = nn.Conv2d(self.hyper_parameters['Cascade layer %d shape'%self.layer_index][0],self.hyper_parameters['Cascade layer %d shape'%self.layer_index][1], kernel_size=self.hyper_parameters['Cascade kernal size'],padding=self.hyper_parameters['Cascade kernal size'])
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1_bn = nn.BatchNorm2d(self.hyper_parameters['Cascade layer %d shape'%self.layer_index][1])
        # =================================================
        if self.hyper_parameters['Cascade add dropout']:
           self.conv1_drop = nn.Dropout2d(0.2)
        # ======================================CNN Auxiliary network part=============================================  
        self.aux_network_size=self.hyper_parameters['Auxiliary network nodes number']
        # =====================================================
        self.conv2=nn.Conv2d(self.hyper_parameters['Cascade layer %d shape'%self.layer_index][1],self.aux_network_size,kernel_size=self.hyper_parameters['Cascade kernal size'],padding=self.hyper_parameters['Cascade kernal size'])
        self.pool_aux = nn.MaxPool2d(2, 2)
        self.conv2_bn = nn.BatchNorm2d(self.aux_network_size)
        # =====================================================
        # self.conv3=nn.Conv2d(self.aux_network_size,self.aux_network_size,kernel_size=self.hyper_parameters['Cascade kernal size'],padding=self.hyper_parameters['Cascade kernal size'])
        # self.pool_aux1 = nn.MaxPool2d(2, 2)
        # self.conv3_bn = nn.BatchNorm2d(self.aux_network_size)
        # =====================================================
        min_s=self.hyper_parameters['Cascade kernal size']-2*self.hyper_parameters['Cascade kernal size']
        self.size_calculated=224
        for au_index in range(self.hyper_parameters['Auxiliary size']+1):
                    self.size_calculated=int(int((((self.size_calculated-min_s))/1+1)-2)/2+1)#int(((self.size_calculated-min_s)/1+1)/2)#int(((32-min_s)/1+1)/2)
        # print('the calculating size is',self.size_calculated)
        self.size_calculated_prelayer=int(((224-min_s)/1+1)/2)
        layer_output_size.append(self.size_calculated_prelayer)
        self.fc1 = nn.Linear(self.aux_network_size*self.size_calculated*self.size_calculated,self.hyper_parameters['Auxiliary linear network nodes number'][0])#(W-F+2P)/S+1   W:input volume size, F: converlutional layer neuros, P: padding, S: stride
        # ================================================MLP auxiliary network part==========================================================
        # self.size_calculated=int(((self.hyper_parameters['Cascade layer %d shape'%self.layer_index][1]-self.hyper_parameters['Cascade kernal size']+2*self.hyper_parameters['Cascade kernal size'])/1+1)/2)
        # print('the size is',self.size_calculated)
        # layer_output_size.append(self.size_calculated)
        # self.fc1 = nn.Linear(self.hyper_parameters['Cascade layer %d shape'%self.layer_index][1]*self.size_calculated*self.size_calculated, 84)#(W-F+2P)/S+1   W:input volume size, F: converlutional layer neuros, P: padding, S: stride
        # ================================================================================================================
        self.drop = nn.Dropout2d(hyper_parameters['Dropout percent'])
        self.fc2=nn.Linear(self.hyper_parameters['Auxiliary linear network nodes number'][0],self.hyper_parameters['Auxiliary linear network nodes number'][1])
        self.fc3 = nn.Linear(self.hyper_parameters['Auxiliary linear network nodes number'][1],hyper_parameters['Number of classess'])
        
    def forward(self, x): #The forward() pass defines the way we compute our output using the given layers and functions
        last_layer_out = self.pool(F.relu(self.conv1(x)))
        last_layer_out=self.conv1_bn(last_layer_out)
        # ======================================CNN Auxiliary network part=============================================
        last_layer_out = self.pool_aux(F.relu(self.conv2(last_layer_out)))
        last_layer_out=self.conv2_bn(last_layer_out)
        # =====================================================
        # last_layer_out = self.pool_aux1(F.relu(self.conv3(last_layer_out)))
        # last_layer_out=self.conv3_bn(last_layer_out)
        # =====================================================
        # print('The output of relu is:',last_layer_out.shape)
        x= last_layer_out.view(-1, self.aux_network_size*self.size_calculated*self.size_calculated)
        # ============================================================================================        
        # x= last_layer_out.view(-1, self.hyper_parameters['Cascade layer %d shape'%self.layer_index][1]*self.size_calculated*self.size_calculated)
        # =========================================================================================
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x=F.relu(self.fc2(x))
        x = self.fc3(x)
        # print('The output shape of layer %d is'%self.layer_index,x.shape)
        return x                         #F.log_softmax(x)
class non_first_layer_cascade_Net(nn.Module):
    def _int_(self,layer_index,hyper_parameters,layer_output_size):
        super(non_first_layer_cascade_Net,self).__init__()
        self.layer_index=layer_index
        self.hyper_parameters=hyper_parameters
        # =====================================================
        self.new_conv1 = nn.Conv2d(self.hyper_parameters['Cascade layer %d shape'%self.layer_index][0],self.hyper_parameters['Cascade layer %d shape'%self.layer_index][1], kernel_size=self.hyper_parameters['Cascade kernal size'],padding=self.hyper_parameters['Cascade kernal size'])
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1_bn = nn.BatchNorm2d(self.hyper_parameters['Cascade layer %d shape'%self.layer_index][1])
        # ======================================CNN Auxiliary network part=============================================
        if self.hyper_parameters['Cascade add dropout']:
           self.new_conv1_drop = nn.Dropout2d(0.2)
        # =====================================================   
        self.aux_network_size=self.hyper_parameters['Auxiliary network nodes number']
        # =====================================================
        self.conv2=nn.Conv2d(self.hyper_parameters['Cascade layer %d shape'%self.layer_index][1],self.aux_network_size,kernel_size=self.hyper_parameters['Cascade kernal size'],padding=self.hyper_parameters['Cascade kernal size'])
        self.pool_aux = nn.MaxPool2d(2, 2)
        self.conv2_bn = nn.BatchNorm2d(self.aux_network_size)
        # =====================================================
        # self.conv3=nn.Conv2d(self.aux_network_size,self.aux_network_size,kernel_size=self.hyper_parameters['Cascade kernal size'],padding=self.hyper_parameters['Cascade kernal size'])
        # self.pool_aux1 = nn.MaxPool2d(2, 2)
        # self.conv3_bn = nn.BatchNorm2d(self.aux_network_size)
        # =====================================================
        min_s=self.hyper_parameters['Cascade kernal size']-2*self.hyper_parameters['Cascade kernal size']
        self.size_calculated=layer_output_size[self.layer_index-1]
        # print('before calcutation is:',self.size_calculated)
        for au_index in range(self.hyper_parameters['Auxiliary size']+1):
                    self.size_calculated=int(int((((self.size_calculated-min_s))/1+1)-2)/2+1)
        # print('calcutation is:',self.size_calculated)
        self.size_calculated_prelayer=int(int(((layer_output_size[self.layer_index-1]-min_s))/1+1)/2)
        layer_output_size.append(self.size_calculated_prelayer)
        self.fc1 = nn.Linear(self.aux_network_size*self.size_calculated*self.size_calculated,self.hyper_parameters['Auxiliary linear network nodes number'][0])
        print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++for layer %d the layer output size is:'%layer_index,layer_output_size,self.size_calculated_prelayer,'++++++++++++++++++++++++++++++++++++++++++++')
        # ================================================MLP auxiliary network part==========================================================
        # self.size_calculated=int(int((((layer_output_size[self.layer_index-1]-self.hyper_parameters['Cascade kernal size']+2*self.hyper_parameters['Cascade kernal size']))/1+1)-2)/2+1)
        # layer_output_size.append(self.size_calculated)
        # self.fc1 = nn.Linear(self.hyper_parameters['Cascade layer %d shape'%self.layer_index][1]*self.size_calculated*self.size_calculated, 84)#(W-F+2P)/S+1   W:input volume size, F: converlutional layer neuros, P: padding, S: stride
        # ==================================================================================
        self.drop = nn.Dropout2d(hyper_parameters['Dropout percent'])
        self.fc2=nn.Linear(self.hyper_parameters['Auxiliary linear network nodes number'][0],self.hyper_parameters['Auxiliary linear network nodes number'][1])
        self.fc3 = nn.Linear(self.hyper_parameters['Auxiliary linear network nodes number'][1],hyper_parameters['Number of classess'])
        # print('The current classesifacations is',self.hyper_parameters['Number of classess'])
    def forward(self,x):
        last_layer_out = self.pool(F.relu(self.new_conv1(x)))
        last_layer_out=self.conv1_bn(last_layer_out)
        # ======================================CNN Auxiliary network part=============================================
        last_layer_out = self.pool_aux(F.relu(self.conv2(last_layer_out)))
        last_layer_out=self.conv2_bn(last_layer_out)
        # =====================================================
        # last_layer_out = self.pool_aux1(F.relu(self.conv3(last_layer_out)))
        # last_layer_out=self.conv3_bn(last_layer_out)
        # print('The shape is',last_layer_out.shape,self.size_calculated)
        x= last_layer_out.view(-1, self.aux_network_size*self.size_calculated*self.size_calculated)
        # ==========================================MLP AUX==================================================
        # print('The shape is',last_layer_out.shape)
        # x= last_layer_out.view(-1, self.hyper_parameters['Cascade layer %d shape'%self.layer_index][1]*self.size_calculated*self.size_calculated)
        # ===========================================================================================================
        x = F.relu(self.fc1(x))
        # if self.hyper_parameters['Cascade add dropout']:
        x = self.drop(x)
        x=F.relu(self.fc2(x))
        x = self.fc3(x)
        # print('The output shape of layer %d is'%self.layer_index,x.shape)
        return x#F.log_softmax(x)
