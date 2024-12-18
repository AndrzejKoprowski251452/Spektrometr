
#from __future__ import division
import matplotlib
#matplotlib.use('Agg')
from matplotlib.patches import Patch
from pylab import *
from scipy.optimize import curve_fit

import re
import sys
import matplotlib.font_manager

import glob
import numpy
import subprocess
import serial
#import usb
import pyvisa
import logging
import time

#import LDC4005_24_1




#1. Add SUBSYSTEM=="usb", MODE="0666", GROUP="usbusers" to /etc/udev/rules.d/99-com.rules as root for pyvisa to see USB devices (otherwise have to use sudo to run the script)

#2. Run sudo usermod -a -G dialout $USER (add user to dialout group) to gain an access to the serial ports (or USB-RS232 adapters)



def search_for_devs():
    rm = pyvisa.ResourceManager()
    #rm = pyvisa.ResourceManager('/usr/lib/x86_64-linux-gnu/libiovisa.so')
    print('Found following devices:', rm.list_resources())


def connect_device(ID):
    print('Connecting to', ID, '...')
    rm = pyvisa.ResourceManager('/usr/lib/x86_64-linux-gnu/libiovisa.so')
    #print('Wykryte urządzenia:', rm.list_resources())
    #DEV = rm.open_resource(ID, write_termination = '\n')
    DEV = rm.open_resource(ID)
    #DEV.read_termination = '\n'
    print('Connected to:', DEV.query('*IDN?'))
    return DEV


def multimeter_display_text(MM, text, interval=1):
    print("Multimeter:", text)
    if len(text) < 17:
        #print('DISPlay:TEXT "%s"' % text, MM.write('DISPlay:TEXT "%s"'  % text))
        MM.write('DISPlay:TEXT "%s"'  % text)
    else:
        for i in range((len(text)-15)):
            #print(text[i:i+16])
            #print('DISPlay:TEXT "%s"' % text[i:i+16], MM.write('DISPlay:TEXT "%s"'  % text[i:i+16]))
            MM.write('DISPlay:TEXT "%s"'  % text[i:i+16])
            time.sleep(0.5)
    time.sleep(3)
    print('DISPlay:TEXT:CLEar"', MM.write('DISPlay:TEXT:CLEar'))
    print('SYSTem:LOCal', MM.write('SYSTem:LOCal'))
    #time.sleep(interval)


def dump_to_txt(foo_I, foo_U, foo_P, file_name, mode='w'):
    txt_file = open(file_name, mode)
    txt_file.write("Voltage [V} Current [A] Power [W]\n")
    if len(foo_I) == len(foo_U) and len(foo_U) == len(foo_P):
        for foo_i in range(len(foo_I)):
            if foo_I[foo_i] != None and foo_U[foo_i] != None and foo_P[foo_i] != None:
                txt_file.write("%g\t%g\t%g\n" % (foo_U[foo_i], foo_I[foo_i], foo_P[foo_i]))
            else:
                txt_file.write("None")
            #txt_file.write('\t')
    #txt_file.write('\n')
    txt_file.close()


def fetch_current_array(I_start, I_stop, res, offset=0.0):
    currents = []
    i = I_start
    while i < I_stop:
        currents.append(i+offset)
        i += res
    return currents


def mask_info(mask):
    if mask == 'VG' or mask == 'U' or mask == 'HP':
        mask_rows_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G']
        mask_rows_coordinates = linspace(8000.0, 0.0, 17)
        mask_columns_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
        mask_columns_coordinates = linspace(0.0, 9000.0, 19)
    elif mask == 'HF':
        mask_rows_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'R', 'S', 'T', 'U']
        mask_rows_coordinates = linspace(11400.0, 0.0, 20)
        mask_columns_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'R', 'S', 'T', 'U', 'V', 'W', 'Z']
        mask_columns_coordinates = linspace(0.0, 13200.0, 23)
    elif mask == 'TUB':
        mask_rows_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F']
        mask_rows_coordinates = linspace(9000.0, 0.0, 16)
        mask_columns_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E']
        mask_columns_coordinates = linspace(0.0, 8400.0, 15)
    else:
        print("mask_info: Wrong maskset name!")

    return mask_rows_names, mask_rows_coordinates, mask_columns_names, mask_columns_coordinates


def mask_mapping(mask, dev, offset=[0.0, 0.0]):
    if len(dev) == 5:
        dev_UC_row = dev[0]
        dev_UC_column = dev[1]
        dev_row = dev[3]
        dev_column = dev[4]
        dev_x = None
        dev_y = None

        foo_rows_names, foo_rows_coordinates, foo_columns_names, foo_columns_coordinates = mask_info(mask)

        for foo_i in range(len(foo_rows_names)):
            if foo_rows_names[foo_i] == dev_row:
                dev_y = foo_rows_coordinates[foo_i] + offset[1]
        for foo_j in range(len(foo_columns_names)):
            if foo_columns_names[foo_j] == dev_column:
                dev_x = foo_columns_coordinates[foo_j] + offset[0]
    else:
        print("mask_mapping: Wrong device name!")

    return dev_UC_row, dev_UC_column, dev_row, dev_column, dev_x, dev_y


#def first_differential(m_X, m_Y, average_points=3):
    #diff_X = None
    #diff_Y = None
    #if len(m_X) >= 2*average_points+1 and len(m_Y) >= 2*average_points+1:
        #diff_X = average(m_X[len(m_X)-(2*average_points+1):])
        #diff_Y = (m_Y[-1]-m_Y[len(m_Y)-(2*average_points+1)])/(m_X[-1]-m_X[len(m_X)-(2*average_points+1)])

    #return diff_X, diff_Y


def first_differential_v2(m_X, m_Y, average_points=3):
    diff_X = None
    diff_Y = None
    if len(m_X) >= 2*average_points+1 and len(m_Y) >= 2*average_points+1:
        diff_X = average(m_X[len(m_X)-(2*average_points+1):])
        diff_Y = polyfit(m_X[len(m_X)-(2*average_points+1):], m_Y[len(m_Y)-(2*average_points+1):], 1)[0]

    return diff_X, diff_Y



def single_point_measurement(val, LD, MM=None, PM=None, mode='current', wait='0.0'):
    liv = [None, None, None]
    LD.write('SOURce:CURRent %s' % val)
    time.sleep(wait)

    liv[1] = float(LD.query('MEASure:VOLTage?'))
    if MM != None:
        liv[0] = float(MM.query('MEASure:CURRent:DC?'))
    else:
        liv[0] = float(LD.query('MEASure:CURRent?'))
    if PM != None:
        liv[2] = float(PM.query('MEASure?'))
    #print('SYSTem:ERRor?', LD.query('SYSTem:ERRor?'))
    return liv[0], liv[1], liv[2]


def measure_LIV(I_start, I_stop, res, PS, MM=None, PM=None, average_points=3, use_RO_fuse=True, RO_fuse_max=1, draw_figs=True):
    if draw_figs == True:
        fig1, ax1IV = subplots(figsize=(12, 9), dpi=80)
        ax1LI = ax1IV.twinx()

        fig2, ax2LI = subplots(figsize=(12, 9), dpi=80)
        #ax2LI = ax2IV.twinx()
    else:
        fig1 = None
        fig2 = None

    I = fetch_current_array(I_start, I_stop, res, offset=0.0008)
    m_P = []
    m_I = []
    m_U = []

    diff_I_v2 = []
    diff_P_v2 = []

    RO_fuse = 0

    wait_time = 0.05
    print("Measuring LIV for currents from %sA to %sA with resolution %sA..." % (I_start, I_stop, res))
    print('PS -> SOURce:CURRent %s' % I[0], PS.write('SOURce:CURRent %s' % I[0]))
    print('PS -> OUTPut:DELay?', PS.query('OUTPut:DELay?'))
    print('PS -> OUTPut ON', PS.write('OUTPut ON'))
    if_PS_on = int(PS.query('OUTPut?'))
    print('PS -> OUTPut?', PS.query('OUTPut?'))
    print('PS -> SYSTem:ERRor?', PS.query('SYSTem:ERRor?'))
    time.sleep(float(PS.query('OUTPut:DELay?'))+1.5)
    initial_voltage = float(PS.query('MEASure:VOLTage?'))

    if initial_voltage >= 0.1:
        i = 0
        current_i = I[0]
        while current_i < I[-1] and RO_fuse < RO_fuse_max and if_PS_on == 1:
            if_PS_on = int(PS.query('OUTPut?'))
            LIV = single_point_measurement(current_i, PS, MM, PM, wait=wait_time)
            if MM != None:
                MM.write('SYSTem:LOCal')
            m_I.append(LIV[0])
            m_U.append(LIV[1])
            m_P.append(LIV[2])
            if draw_figs == True:
                ax1IV.plot(m_I, m_U, label="VI", linewidth=2, color='red')
                ax1LI.plot(m_I, m_P, label="LI", linewidth=2, color='blue')

            if len(m_I) >= 2*average_points+1 and len(m_P) >= 2*average_points+1:
                current_diff_I_v2, current_diff_P_v2 = first_differential_v2(m_I, m_P, average_points)
                if current_diff_P_v2 < 0:
                    RO_fuse += 1
                if RO_fuse != 0 and current_diff_P_v2 > 0 or use_RO_fuse == False:
                    RO_fuse = 0
                #print("RO_fuse =", RO_fuse)
                diff_I_v2.append(current_diff_I_v2)
                diff_P_v2.append(current_diff_P_v2)
                if draw_figs == True:
                    ax2LI.plot(diff_I_v2, diff_P_v2, label="dL/dI", linewidth=2, color='green')
                    ax2LI.set_xlabel("$I$ [A]", size=16)
                    ax2LI.set_ylabel("$dP/dI$ [W/A]", size=16)
                    ax2LI.set_xlim(left=diff_I_v2[0], right=diff_I_v2[-1])
                    #ax2LI.set_ylim(bottom=diff_P_v2[0], top=max(diff_P_v2)*1.2)
                    #ax2IV.set_ylim(bottom=0.0, top=m_U[-1])
                    ax2LI.tick_params(axis='both', which='major', labelsize=16)
                    pause(0.05)
            if draw_figs == True:
                ax1IV.set_xlabel("$I$ [A]", size=16)
                ax1IV.set_ylabel("$U$ [V]", size=16)
                ax1LI.set_ylabel("$P$ [W]", size=16)
                if i>1:
                    ax1IV.set_xlim(left=m_I[0], right=m_I[-1])
                    #xlim([m_I[0], m_I[-1]])
                ax1IV.set_ylim(bottom=0.0, top=m_U[-1])
                ax1LI.set_ylim(bottom=m_P[0], top=max(m_P)*1.2)
                ax1IV.tick_params(axis='both', which='major', labelsize=16)
                ax1LI.tick_params(axis='both', which='major', labelsize=16)
                pause(0.05)
            i+=1
            current_i = I[i]
        print('PS -> OUTPut OFF', PS.write('OUTPut OFF'))
        #show()
    else:
        print('PS -> OUTPut OFF', PS.write('OUTPut OFF'))
    print('PS -> SOURce:CURRent 0', PS.write('SOURce:CURRent 0'))

    #return LIV
    return m_I, m_U, m_P, fig1, fig2


def measure_LIV_B2901BL(I_start, I_stop, V_stop, res, PS, PM=None, average_points=3, use_RO_fuse=True, RO_fuse_max=1, draw_figs=True):


    if draw_figs == True:
        fig1, ax1IV = subplots(figsize=(12, 9), dpi=80)
        ax1LI = ax1IV.twinx()

        fig2, ax2LI = subplots(figsize=(12, 9), dpi=80)
        #ax2LI = ax2IV.twinx()
    else:
        fig1 = None
        fig2 = None

    I = fetch_current_array(I_start, I_stop, res)
    m_P = []
    m_I = []
    m_U = []

    diff_I_v2 = []
    diff_P_v2 = []

    RO_fuse = 0

    #wait_time = 0.05
    print("Measuring LIV for currents from %sA to %sA with resolution %sA..." % (I_start, I_stop, res))

    #meas_voltage = []
    #meas_current = []
    #meas_power = []

    if_voltage_mode = True
    max_voltage_reached = False
    switch_to_current_mode = False
    voltage_mode_max_current = 0.001    #[A]
    voltage_mode_resolution = 0.01  #[V]

    PS.write(':FUNC:MODE CURR') #current source mode
    print("110:PS.query(':FUNC:MODE?')", PS.query(':FUNC:MODE?'))

    PS.write('CURR ' + str(I[1]))   #set current
    PS.write(':OUTP 1')
    if_PS_on = int(PS.query(':OUTP?'))
    print("PS.query(':OUTP?')", if_PS_on)


    if float(PS.query(':SENS:VOLT:PROT:TRIP?')) == 0:
        print("PS.query(':SENS:VOLT:PROT:TRIP?') 0")
        if if_voltage_mode == True:
            current_u = 0.0
            PS.write(':OUTP 0')
            PS.write(':FUNC:MODE VOLT') #voltage source mode
            print("124:PS.query(':FUNC:MODE?')", PS.query(':FUNC:MODE?'))
            PS.write(':VOLT ' + str(current_u))   #set voltage
            PS.write(':OUTP 1')
        i = 0
        current_i = I[0]
        while current_i < I[-1] and max_voltage_reached == False and RO_fuse < RO_fuse_max and if_PS_on == 1:
            if if_voltage_mode == True and switch_to_current_mode == True:
                if_voltage_mode = False
                PS.write(':OUTP 0')
                PS.write(':FUNC:MODE CURR') #current source mode
                PS.write('CURR ' + str(current_i))   #set current
                PS.write(':OUTP 1')
                print("136:PS.query(':FUNC:MODE?')", PS.query(':FUNC:MODE?'))
                I = fetch_current_array(voltage_mode_max_current, I_stop, res)
                current_i = I[0]
                #input()
            #print("PS.query(':SENS:FUNC:ON?')", PS.query(':SENS:FUNC:ON?'))  #what is being measured

            if if_voltage_mode == True:
                if_PS_on = int(PS.query(':OUTP?'))
                PS.write('VOLT ' + str(current_u))   #set voltage
                #time.sleep(0.1)
                ret_vals = PS.query(':MEAS? (@1)')
                current_i = float(ret_vals.split(',')[1])
                print("147:current_i = ", current_i)
                if current_i < voltage_mode_max_current:
                    m_U.append(float(ret_vals.split(',')[0]))
                    m_I.append(current_i)
                    m_P.append(float(PM100USB.query('MEASure?')))
                    print(m_U[-1], m_I[-1], m_P[-1])
                    current_u += voltage_mode_resolution
                else:
                    switch_to_current_mode = True
                if m_U[-1] > V_stop:
                    max_voltage_reached = True

            else:
                print("165:current_i =", current_i)
                print("if_voltage_mode =", if_voltage_mode)
                #input()
                if_PS_on = int(PS.query(':OUTP?'))
                PS.write('CURR ' + str(current_i))   #set current
                #time.sleep(0.1)
                ret_vals = PS.query(':MEAS? (@1)')
                m_U.append(float(ret_vals.split(',')[0]))
                m_I.append(float(ret_vals.split(',')[1]))
                m_P.append(float(PM100USB.query('MEASure?')))
                i+=1
                current_i = I[i]
                if m_U[-1] > V_stop:
                    max_voltage_reached = True


            if draw_figs == True:
                ax1IV.plot(m_I, m_U, label="VI", linewidth=2, color='red')
                ax1LI.plot(m_I, m_P, label="LI", linewidth=2, color='blue')

            if len(m_I) >= 2*average_points+1 and len(m_P) >= 2*average_points+1 and if_voltage_mode == False:
                current_diff_I_v2, current_diff_P_v2 = first_differential_v2(m_I, m_P, average_points)
                if current_diff_P_v2 < 0:
                    RO_fuse += 1
                if RO_fuse != 0 and current_diff_P_v2 > 0 or use_RO_fuse == False:
                    RO_fuse = 0
                #print("RO_fuse =", RO_fuse)
                diff_I_v2.append(current_diff_I_v2)
                diff_P_v2.append(current_diff_P_v2)
                if draw_figs == True:
                    ax2LI.plot(diff_I_v2, diff_P_v2, label="dL/dI", linewidth=2, color='green')
                    ax2LI.set_xlabel("$I$ [A]", size=16)
                    ax2LI.set_ylabel("$dP/dI$ [W/A]", size=16)
                    ax2LI.set_xlim(left=diff_I_v2[0], right=diff_I_v2[-1])
                    #ax2LI.set_ylim(bottom=diff_P_v2[0], top=max(diff_P_v2)*1.2)
                    #ax2IV.set_ylim(bottom=0.0, top=m_U[-1])
                    ax2LI.tick_params(axis='both', which='major', labelsize=16)
                    pause(0.05)
            if draw_figs == True:
                ax1IV.set_xlabel("$I$ [A]", size=16)
                ax1IV.set_ylabel("$U$ [V]", size=16)
                ax1LI.set_ylabel("$P$ [W]", size=16)
                if i>2:
                    ax1IV.set_xlim(left=m_I[0], right=m_I[-1])
                    #xlim([m_I[0], m_I[-1]])
                ax1IV.set_ylim(bottom=0.0, top=m_U[-1])
                ax1LI.set_ylim(bottom=m_P[0], top=max(m_P)*1.2)
                ax1IV.tick_params(axis='both', which='major', labelsize=16)
                ax1LI.tick_params(axis='both', which='major', labelsize=16)
                pause(0.05)
            #print(current_i, current_u, voltage_mode_max_current, ifS_on, RO_fuse, RO_fuse_max)
            #input()
        PS.write(':OUTP 0')
        print("PS.query(':OUTP?')", PS.query(':OUTP?'))

        #for i in range(len(meas_voltage)):
            #print(meas_voltage[i], meas_current[i], meas_power[i])
        #print('PS -> OUTPut OFF', PS.write('OUTPut OFF'))
        #show()
    else:
        print("Protection voltage tripped!")
        PS.write(':OUTP 0')
        print("PS.query(':OUTP?')", PS.query(':OUTP?'))
    #print('PS -> SOURce:CURRent 0', PS.write('SOURce:CURRent 0'))

    #return LIV
    return m_I, m_U, m_P, fig1, fig2


def znajdzUSB():
    listaUSB = glob.glob("/dev/ttyUSB*")
    print(listaUSB)
    USCBzpl = []
    for USB in listaUSB:
        komenda = "udevadm info -a -n " + USB + "|grep DRIVERS "
        data = subprocess.check_output(komenda, shell=True).decode()
        if 'pl2303' in data or 'PL2303' in data:
           USCBzpl.append(USB) 
    return USCBzpl                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 


class stage:
    def __init__(self, port_number):
        print(port_number)
        self.port = serial.Serial(port_number, timeout=1) # open port under name 'port_number' using serial pySerial API
        self.steptime = 1.5/10000.0 # estimated single step time
        print("Sucessfully opened serial port", port_number)
        self.update_status()
    
    
    def read_port(self):
        #self.line = self.port.readline().decode() # read the answer
        self.line = self.port.readline().decode() # read the answer
        return self.line.strip('\r\n')
        
    def check_origin_offset(self):
        self.port.write(str.encode("V:N\r\n")) # ask the dev for origin offset
        #while self.check_status() != 'R':
                #time.sleep(0.001)
        self.line = self.port.readline().decode() # read the answer
        self.line = ''.join(self.line.split()) # remove spaces
        print("Origin offset: ", self.line.strip('\r\n'))


    def set_cs_origin(self):
        self.port.write(str.encode("R:1\r\n")) # ask the dev for origin offset
        #while self.check_status() != 'R':
                #time.sleep(0.001)
        self.line = self.port.readline().decode() # read the answer
        self.line = ''.join(self.line.split()) # remove spaces
        print("Current position set to origin of the coordinate system: ", self.line.strip('\r\n'))


    def update_status(self):
        self.port.write(str.encode("Q:\r\n")) # ask the dev for status
        self.line = self.port.readline().decode() # read the answer
        self.line = ''.join(self.line.split()) # remove spaces
        #print(self.line)
        while len(self.line.split(',')) != 4:
            #time.sleep(0.1)
            self.port.write(str.encode("Q:\r\n")) # ask the dev for status
            self.line = self.port.readline().decode() # read the answer
            self.line = ''.join(self.line.split()) # remove spaces
        self.position = int(self.line.split(',')[0])
        self.command_status = self.line.split(',')[1]
        self.stop_status = self.line.split(',')[2]
        self.readiness = self.line.split(',')[3].strip('\r\n')
        
        
    def check_status(self):
        self.port.write(str.encode("!:\r\n")) # ask the dev for status
        self.line = self.port.readline().decode() # read the answer
        return self.line.strip('\r\n')


    def check_internal(self):
        self.port.write(str.encode("?:V\r\n")) # ask the dev for status
        self.line = self.port.readline().decode() # read the answer
        print("ROM version:", self.line.strip('\r\n'))
        self.port.write(str.encode("?:-\r\n")) # ask the dev for status
        self.line = self.port.readline().decode() # read the answer
        print("Revision:", self.line.strip('\r\n'))
        #return self.line.strip('\r\n')


    def port_settings(self):
        print("Settings of port", self.port.name)
        print("cts =", self.port.cts)
        print("dsr =", self.port.dsr)
        print("ri =", self.port.ri)
        print("cd =", self.port.cd)
        print("is_open =", self.port.is_open)
        print("bytesize =", self.port.bytesize)
        print("parity =", self.port.parity)
        print("stopbits =", self.port.stopbits)
        print("timeout =", self.port.timeout)
        print("write_timeout =", self.port.write_timeout)
        print("inter_byte_timeout =", self.port.inter_byte_timeout)
        print("xonxoff =", self.port.xonxoff)
        print("rtscts =", self.port.rtscts)
        print("dsrdtr =", self.port.dsrdtr)
        print("rs485_mode =", self.port.rs485_mode)
        print("get_settings() =", self.port.get_settings())


    def stage_speed(self, min_speed, max_speed, t_ad): # set minimal speed, maximal speed, nad accelearation/decelaration time
        self.parameters_good = True
        if min_speed < 100 or min_speed > 20000:
            print("Minimum speed s has to be 100pps < s < 20000pps! Setting: ", min_speed)
            self.parameters_good = False
        if min_speed%100 != 0:
            print("Minimum speed s has to be set in multiples of 100pps! Setting: ", min_speed)
            self.parameters_good = False
        if max_speed < 100 or max_speed > 20000:
            print("Maximum speed S has to be 100pps < S < 20000pps! Setting: ", max_speed)
            self.parameters_good = False
        if max_speed%100 != 0:
            print("Maximum speed S has to be set in multiples of 100pps! Setting: ", max_speed)
            self.parameters_good = False
        if max_speed < min_speed:
            print("Maximum speed S has to be larger than minimum speed s! Settings: S =", max_speed, "s =", min_speed)
            self.parameters_good = False
        if t_ad < 0 or t_ad > 1000:
            print("Acceleration/decelaration time t has to be 0ms < t < 1000ms! Setting: ", t_ad)
            self.parameters_good = False
        if t_ad%100 != 0:
            print("Acceleration/decelaration time t has to be set in multiples of 100ms! Setting: ", t_ad)
            self.parameters_good = False
        print("Parameters good?", self.parameters_good)
        if self.parameters_good:
            #print(("D:1S"+str(int(min_speed))+"F"+str(int(max_speed))+"R"+str(int(t_ad))+"\r\n").encode(encoding="ascii"))
            self.port.write(("D:1S"+str(int(min_speed))+"F"+str(int(max_speed))+"R"+str(int(t_ad))+"\r\n").encode(encoding="ascii")) # send the command with the new speed parameters to the stage
            print("Setting new stage speed parameters:", self.read_port())
    
    
    def report_status(self):
        print("Current position/command status/stop status/readiness: ", self.position, self.command_status, self.stop_status, self.readiness, "\n")
        #print("\n")


    def move(self, journey, unit='pulses'): # move 'journey' pulses or um
        self.safe_move = False
        if unit == 'pulses':
            self.steps = journey
            self.safe_move = True
        elif unit == 'um':
            if journey%2 == 0:  # the resolution is 2um/step
                self.steps = int(journey/2)
                self.safe_move = True
            else:
                print("Wrong journey length! The resolution of the move is 2um/pulse.")
                self.safe_move = False
        print("Safe move:", self.safe_move)
        #self.update_status()
        if self.readiness == 'R' and self.safe_move == True: # check if the stage is ready (should always be)
            #print("Now moving", steps, "steps")
            if(self.steps>=0):
                self.strM = ("M:1+P"+str(int(self.steps))+"\r\n").encode(encoding="ascii") # moving positive direction of the axis
            else:
                self.strM = ("M:1-P"+str(-int(self.steps))+"\r\n").encode(encoding="ascii") # moving negative direction of the axis
            self.port.write(self.strM) # send the command with the number of steps to the stage controller
            if self.read_port() == 'OK':
                if unit == 'pulses':
                    print("Successfuly set move by", self.steps, "steps")
                elif unit == 'um':
                    print("Successfuly set move by", 2*self.steps, "um")
            self.port.write(str.encode("G:\r\n")) # send the execute command
            if self.read_port() == 'OK':
                if unit == 'pulses':
                    print("Now moving by", self.steps, "steps")
                elif unit == 'um':
                    print("Successfuly set move by", 2*self.steps, "um")
            while self.check_status() != 'R':
                time.sleep(0.001)
                #print(self.check_status())
            self.update_status()
            if unit == 'pulses':
                print("Current position:", self.position)
            elif unit == 'um':
                print("Current position:", 2*self.position, 'um')
            #self.report_status()


    def move_to(self, _position, unit='pulses'): # move to 'position' in absolute coordinates (pulses or um)
        self.safe_move = False
        if unit == 'pulses':
            self.coordinate = _position
            self.safe_move = True
        elif unit == 'um':
            if _position%2 == 0:  # the resolution is 2um/step
                self.coordinate = int(_position/2)
                self.safe_move = True
            else:
                print("Wrong journey length! The resolution of the move is 2um/pulse.")
                self.safe_move = False
        print("Safe move:", self.safe_move)
        #self.update_status()
        if self.readiness == 'R' and self.safe_move == True: # check if the stage is ready (should always be)
            if(self.coordinate>=0):
                self.strA = ("A:1+P"+str(int(self.coordinate))+"\r\n").encode(encoding="ascii") # moving positive direction of the axis
            else:
                self.strA = ("A:1-P"+str(-int(self.coordinate))+"\r\n").encode(encoding="ascii") # moving negative direction of the axis
            self.port.write(self.strA) # send the command with the number of steps to the stage controller
            if self.read_port() == 'OK':
                if unit == 'pulses':
                    print("Successfuly set move to", self.coordinate)
                elif unit == 'um':
                    print("Successfuly set move to", 2*self.coordinate, "um")
            self.port.write(str.encode("G:\r\n")) # send the execute command
            if self.read_port() == 'OK':
                if unit == 'pulses':
                    print("Now moving to", self.coordinate)
                elif unit == 'um':
                    print("Now moving to", 2*self.coordinate, "um")
            self.read_port()
            while self.check_status() != 'R':
                time.sleep(0.001)
                #print(self.check_status())
            self.update_status()
            if unit == 'pulses':
                print("Current position:", self.position)
            elif unit == 'um':
                print("Current position:", 2*self.position, 'um')
            #self.report_status()    


def go_to_device(foo_X_stage, foo_Y_stage, foo_Z_stage, R, C, r, c):
    foo_Z_stage.move_to(1000, unit='pulses')
    foo_X_stage.move_to(c, unit='um')
    foo_Y_stage.move_to(r, unit='um')
    foo_Z_stage.move_to(0, unit='pulses')


def user_prompt():
    answer = input('What would you like to do? a[bort] / b[ind] / M[ap] / m[ove] / j[ump move]:\n')
    #print("answer:", answer)
    check_if_abort = 0
    check_next_step = 0
    if answer == 'a' or answer == 'A':
        check_if_abort = 1
    else:
        check_next_step = answer

    return check_if_abort, check_next_step


def read_config(config_filename):
    foo_cs_offset = [0.0, 0.0]
    foo_cs_theta = 0.0
    in_txt = open(config_filename, 'r')

    for i, line in enumerate(in_txt):
        print("line:", i, ":", line)
        if line[0:11] == "cs_offset_x":
            foo_cs_offset[0] = float(line[12:])
        if line[0:11] == "cs_offset_y":
            foo_cs_offset[1] = float(line[12:])
        if line[0:8] == "cs_theta":
            foo_cs_theta = float(line[9:])
    in_txt.close()
    return foo_cs_offset, foo_cs_theta


def whereami(foo_X_axis, foo_Y_axis, foo_Z_axis, verbose=0):
    if verbose == 1:
        print("whereami:foo_X_axis.position =", foo_X_axis.position*2, "um")
        print("whereami:foo_Y_axis.position =", -foo_Y_axis.position*2, "um")
        print("whereami:foo_Z_axis.position =", foo_Z_axis.position*2, "um")
    return foo_X_axis.position*2, -foo_Y_axis.position*2, foo_Z_axis.position*2


def user_move(foo_X_axis, foo_Y_axis, foo_Z_axis):
    check_if_abort = 0
    check_if_done = 0
    answer_axis = input('Which axis and how much? x/y/z(+/-move in um) / d[one] / a[bort]:\n')
    if answer_axis[0] == 'x' or answer_axis[0] == 'y' or answer_axis[0] == 'z':
        foo_move = float(answer_axis[1:])
        really_move = input("Move %s um in %s direction!?\n" % (foo_move, answer_axis[0]))
        if really_move == '':
            if answer_axis[0] == 'x':
                foo_X_axis.move(foo_move, unit='um')
            elif answer_axis[0] == 'y':
                foo_Y_axis.move(-foo_move, unit='um')
            elif answer_axis[0] == 'z':
                foo_Z_axis.move(foo_move, unit='um')
    elif answer_axis[0] == 'd':
        check_if_done = 1
    else:
        check_if_abort = 1

    return check_if_abort, check_if_done


def jump_move(mask, foo_X_axis, foo_Y_axis, foo_Z_axis, offset=[0.0, 0.0], rotation=0.0):
    check_if_abort = 0
    check_if_done = 0
    answer = input('Which device would you like to go to? (dev) / / d[one] / a[bort]:\n')
    if len(answer) == 5:
        foo_mask_map = mask_mapping(mask, answer, offset)
        really_move = input("Move to device %s%s_%s%s!?\n" % (foo_mask_map[0], foo_mask_map[1], foo_mask_map[2], foo_mask_map[3]))
        if really_move == '':
            if foo_mask_map[4] != None and foo_mask_map[5] != None:
                foo_Z_axis.move_to(1000.0, unit='pulses')
                foo_theta = (rotation*numpy.pi)/180.0
                foo_x = foo_mask_map[4]*cos(foo_theta) + foo_mask_map[5]*sin(foo_theta)
                foo_y = -foo_mask_map[4]*sin(foo_theta) + foo_mask_map[5]*cos(foo_theta)
                if int(round(foo_x))%2 == 0:
                    foo_x = int(round(foo_x))
                else:
                    foo_x = int(round(foo_x))-1
                    #foo_x = int(round(numpy.ceil(foo_x)))
                if int(round(foo_y))%2 == 0:
                    foo_y = int(round(foo_y))
                else:
                    foo_y = int(round(foo_y))-1
                    #foo_y = int(round(numpy.ceil(foo_y)))

                print("foo_x =", foo_x, int(round(foo_x)))
                print("foo_y =", foo_y, int(round(foo_y)))
                foo_X_axis.move_to(foo_x, unit='um')
                foo_Y_axis.move_to(-foo_y, unit='um')
                print("Done!")
            #if answer_axis[0] == 'x':
            #elif answer_axis[0] == 'y':
                #foo_Y_axis.move(-foo_move, unit='um')
            #elif answer_axis[0] == 'z':
                #foo_Z_axis.move(foo_move, unit='um')
        elif really_move == 'a':
            check_if_abort = 1
    elif answer == 'd':
        check_if_done = 1
    elif answer == 'a':
        check_if_abort = 1
    else:
        print("Wrong device name. Try again!")

    return check_if_abort, check_if_done


def bind_coordinate_systems(mask, foo_X_axis, foo_Y_axis, foo_Z_axis):
    check_if_abort = 0
    check_if_done = 0
    foo_cs_offset = [0.0, 0.0]
    theta = 0.0
    foo_center_device = None
    foo_second_device = None
    print("Binding together coordinate systems of probe station and the sample.\n")
    print("Initial status of the stages:\n")
    whereami(foo_X_axis, foo_Y_axis, foo_Z_axis, verbose=1)
    #foo_X_axis.report_status()
    #foo_Y_axis.report_status()
    #foo_Z_axis.report_status()
    answer = input("Move the probe to any device. Confirm with ENTER or a[bort]...\n" )
    if answer == 'a':
        check_if_abort = 1
    elif answer == '':
        while check_if_done == 0 and check_if_abort == 0:
            check_if_abort, check_if_done = user_move(Xaxis, Yaxis, Zaxis)
    else:
        print("bind_coordinate_systems(): Wrong answer. Try again.\n")


    if check_if_done == 1 and check_if_abort == 0:
        foo_X_axis.set_cs_origin()
        foo_Y_axis.set_cs_origin()
        foo_Z_axis.set_cs_origin()

        foo_center_device = input("Which device is it?\n" )
        foo_center_device_coordinates = mask_mapping(mask, foo_center_device)
        #print("foo_center_device_coordinates:", foo_center_device_coordinates)
        print("Center of the coordinate system of the stage set to device %s%s_%s%s (x = %s, y = %s in coordinate system of the sample)." % (foo_center_device_coordinates[0], foo_center_device_coordinates[1], foo_center_device_coordinates[2], foo_center_device_coordinates[3], foo_center_device_coordinates[4], foo_center_device_coordinates[5]))

        foo_cs_offset = [-foo_center_device_coordinates[4], -foo_center_device_coordinates[5]]

        #foo_X_axis.report_status()
        #foo_Y_axis.report_status()
        #foo_Z_axis.report_status()

        answer = input("Move the probe to another device in the same unit cell. Confirm with ENTER or a[bort]...\n" )
        check_if_done = 0
        if answer == 'a':
            check_if_abort = 1
        elif answer == '':
            while check_if_done == 0 and check_if_abort == 0:
                check_if_abort, check_if_done = user_move(Xaxis, Yaxis, Zaxis)
        else:
            print("bind_coordinate_systems(): Wrong answer. Try again.\n")

        if check_if_done == 1 and check_if_abort == 0:
            foo_second_device = input("Which device is it?\n" )
            foo_second_device_coordinates = mask_mapping(mask, foo_second_device, offset=foo_cs_offset)
            print("The second device is %s%s_%s%s (x = %s, y = %s in coordinate system of the sample)." % (foo_second_device_coordinates[0], foo_second_device_coordinates[1], foo_second_device_coordinates[2], foo_second_device_coordinates[3], foo_second_device_coordinates[4], foo_second_device_coordinates[5]))

            foo_x_prim, foo_y_prim = whereami(foo_X_axis, foo_Y_axis, foo_Z_axis, verbose=0)[0:2]
            foo_x = foo_second_device_coordinates[4]
            foo_y = foo_second_device_coordinates[5]

            theta = (180.0/numpy.pi)*arccos((foo_x*foo_x_prim + foo_y*foo_y_prim)/(numpy.sqrt(foo_x*foo_x + foo_y*foo_y)*numpy.sqrt(foo_x_prim*foo_x_prim + foo_y_prim*foo_y_prim)))
            print("x, y, x_prim, y_prim, theta =", foo_x, foo_y, foo_x_prim, foo_y_prim, theta)
        elif answer == 'a':
            check_if_abort = 1
        else:
            print("bind_coordinate_systems(): Wrong answer. Try again.\n")
    #elif answer == 'a':
        #check_if_abort = 1

    return check_if_abort, check_if_done, foo_cs_offset, theta


def initialize_stage3D():
    listaUSB = glob.glob("/dev/ttyUSB*")
    for USB in listaUSB:
        komenda = "udevadm info -a -n " + USB + "|grep DRIVERS "
        data = subprocess.check_output(komenda, shell=True).decode()
        #print(data)
    print(listaUSB)

    foo_Xaxis = stage(listaUSB[2])
    print("X axis:")
    foo_Xaxis.report_status()
    #Xaxis.set_cs_origin()
    #Xaxis.stage_speed(500,2000,200)
    #Xaxis.check_origin_offset()
    #Xaxis.move(10.0, unit='um')
    #Xaxis.move_to(0.0, unit='um')
    #Xaxis.report_status()

    foo_Yaxis = stage(listaUSB[1])
    print("Y axis:")
    foo_Yaxis.report_status()
    #Yaxis.set_cs_origin()
    #Yaxis.stage_speed(500,2000,200)
    #Yaxis.check_origin_offset()
    #Yaxis.move_to(0.0, unit='um')
    #Xaxis.move(1000, unit='um')
    #Xaxis.report_status()

    foo_Zaxis = stage(listaUSB[0])
    print("Z axis:")
    foo_Zaxis.report_status()
    #Zaxis.set_cs_origin()

    #Zaxis.move_to(1000.0, unit='pulses')
    #Xaxis.move_to(0.0, unit='um')
    #Yaxis.move_to(0.0, unit='um')
    #Zaxis.move(00, unit='pulses')
    return foo_Xaxis, foo_Yaxis, foo_Zaxis


#print("arccos(0.5) =", arccos(0.5))
#print("in deg. =", (arccos(0.5)*180.0)/numpy.pi)
cs_offset = [0.0, 0.0]
cs_theta = 0.0
cs_offset, cs_theta = read_config("probe_station_config.txt")
print("cs_offset:", cs_offset)
print("cs_theta:", cs_theta)
#next_step = 0
#if_abort = 0
#which_power_supply = "LDC4005"
which_power_supply = "B2901BL"
print("Power supply:", which_power_supply)


#rm = pyvisa.ResourceManager('@py')
#print("rm.get_visa_attribute('usb_protocol')", rm.get_visa_attribute('usb_protocol'))
#print('Wykryte urządzenia:')
#for dev in rm.list_resources():
    #print(dev)
#ID = 'GPIB0::23::INSTR'
#ID = 'USB0::10893::37121::MY63320362::0::INSTR'
#ID = "USB0::0x2A8D::0x9101::MY63320362::0::INSTR"
#print('Connecting to', ID, '...')
#DEV = rm.open_resource(ID)
#DEV.set_visa_attribute('usb_protocol', "normal")
#print('DEV.__dict__:', DEV.__dict__)
#print('dir(DEV):', dir(DEV))
#print('vars(DEV):', vars(DEV))
#print('DEV.encoding:', DEV.encoding)
#print('DEV.usb_protocol:', DEV.get_visa_attribute('usb_protocol'))
#print('DEV.io_protocol:', DEV.io_protocol)
#DEV = rm.open_resource(ID, write_termination = '\n')
#print('DEV.session:', DEV.session)
#DEV.timeout = 10000
#DEV.write_termination = '\n'
#DEV.read_termination = '\n'
#DEV.write_termination = '\n'
#DEV.read_termination = '\n'
#print('DEV.timeout:', DEV.timeout)
#print('DEV.write_termination:', DEV.write_termination)
#print('DEV.read_termination:', DEV.read_termination)
#DEV.write(':FORM:ASCII')
#DEV.read_termination = '\n'
#print('Connected to:', DEV.query('*IDN?'))



if_abort, next_step = user_prompt()
if_done = 0
#print(if_abort, next_step)

while if_abort == 0:
    Xaxis, Yaxis, Zaxis = initialize_stage3D()

    if next_step == 'M' or next_step == 'b' or next_step == 'j':
        sample_name = input('What is the name of the sample?:\n')
        if 'VG' in sample_name:
            really_know_mask = input("Is it VG maskset?\n")
            if really_know_mask == '':
                mask = 'VG'
            else:
                mask = input("What is the name of the maskset?\n")
        elif 'U' in sample_name:
            really_know_mask = input("Is it U maskset?\n")
            if really_know_mask == '':
                mask = 'U'
            else:
                mask = input("What is the name of the maskset?\n")
        elif 'HP' in sample_name:
            really_know_mask = input("Is it HP maskset?\n")
            if really_know_mask == '':
                mask = 'HP'
            else:
                mask = input("What is the name of the maskset?\n")
        elif 'HF' in sample_name:
            really_know_mask = input("Is it HF maskset?\n")
            if really_know_mask == '':
                mask = 'HF'
            else:
                mask = input("What is the name of the maskset?\n")
        else:
                mask = input("What is the name of the maskset?\n")

    while next_step == 'm' and if_abort == 0:
        if_abort, if_done = user_move(Xaxis, Yaxis, Zaxis)
        if if_abort == 0 and if_done == 1:
            if_abort, next_step = user_prompt()

    while next_step == 'b' and if_abort == 0:
        if_abort, if_done, cs_offset, cs_theta = bind_coordinate_systems(mask, Xaxis, Yaxis, Zaxis)

        if if_abort == 0 and if_done == 1:
            print("Binding of the coordinate systems finished! Theta = %s, coordinate system offset = %s" % (cs_theta, cs_offset))
            config_file = open("probe_station_config.txt", 'w')
            config_file.write("cs_offset_x: %s\n" % cs_offset[0])
            config_file.write("cs_offset_y: %s\n" % cs_offset[1])
            config_file.write("cs_theta: %s" % cs_theta)
            config_file.close()

            if_abort, next_step = user_prompt()

    while next_step == 'j' and if_abort == 0:
        if_abort, if_done = jump_move(mask, Xaxis, Yaxis, Zaxis, offset=cs_offset, rotation=cs_theta)
        if if_abort == 0 and if_done == 1:
            if_abort, next_step = user_prompt()

    if next_step == 'M' and if_abort == 0:
        #search_for_devs()
        if which_power_supply == "LDC4005":
            K2100 = connect_device('USB0::1510::8448::8000977::0::INSTR')
            print('K2100 -> *CLS', K2100.write('*CLS'))

            #multimeter_display_text(K2100, "Ive got bad feelings about this...")
            #multimeter_display_text(K2100, "Goooood!")
            PS = connect_device('USB0::4883::32833::M00298547::0::INSTR')
            print('LDC4005 -> SYSTem:BEEPer:STATe?', PS.query('SYSTem:BEEPer:STATe?'))
            print('LDC4005 -> SYSTem:BEEPer', PS.write('SYSTem:BEEPer'))
            print('LDC4005 -> SYSTem:BEEPer:STATe 0', PS.write('SYSTem:BEEPer:STATe 0'))
            print('LDC4005 -> SYSTem:BEEPer:STATe?', PS.query('SYSTem:BEEPer:STATe?'))
        
        if which_power_supply == "B2901BL":
            PS = connect_device('USB0::10893::37121::MY63320362::0::INSTR')
            PS.write('SYST:BEEP:STAT ON')
            PS.write('SENS:REM 0')  #2-port measurement
            PS.write(':SENS:VOLT:RANG:AUTO ON')  #voltage measurement range
            #PS.write(':SENS:VOLT:NPLC:AUTO ON')    #number of nplc frames to average
            PS.write(':SENS:VOLT:NPLC: 16')    #number of nplc frames to average
            
        PM100USB = connect_device('USB0::4883::32882::P2001205::0::INSTR')
        PM_lambda = input('Whats the emission wavelength in nm?\n')
        PM100USB.write('SENSe:CORRection:WAVelength %s' % PM_lambda)
        PM100USB.write ('SENSe:POWer:RANGe:AUTO ON')
        PM100USB.write ('SENSe:AVERage:COUNt 8')
        max_current = float(input('Whats the maximal current in mA?\n'))/1000.0
        max_voltage = 5.0
        #PM100USB.write('SENSe:CORRection:WAVelength 940')
        #resolution = 8e-5
        
        no_of_measurements = int(input('How many times shall I measure LIVs for each device?\n'))
        
        resolution = 1e-4

        #devices_coordinates_set_X = [0.0, 500.0, 1000.0]
        #devices_coordinates_set_Y = [0.0, 500.0, 1000.0]
        devices_rows_names, devices_rows_coordinates, devices_columns_names, devices_columns_coordinates = mask_info(mask)

        print("devices_rows_names:", devices_rows_names)
        print("devices_rows_coordinates:", devices_rows_coordinates)
        print("devices_columns_names:", devices_columns_names)
        print("devices_columns_coordinates:", devices_columns_coordinates)

        unit_cell = input('Which unit cell would you like to map?\n')

        start_row = None
        stop_row = None
        start_column = None
        stop_column = None

        while start_row == None:
            start_row = input('Which row would you like to start at?\n')
            if start_row not in devices_rows_names:
                print('Row %s not in the predefined rows of the %s mask! Try again.' %(start_row, mask))
                start_row = None

        while stop_row == None:
            stop_row = input('Which row would you like to stop at?\n')
            if stop_row not in devices_rows_names:
                print('Row %s not in the predefined rows of the %s mask! Try again.' %(stop_row, mask))
                stop_row = None

        while start_column == None:
            start_column = input('Which column would you like to start at?\n')
            if start_column not in devices_columns_names:
                print('Column %s not in the predefined columns of the %s mask! Try again.' %(start_column, mask))
                start_column= None

        while stop_column == None:
            stop_column = input('Which column would you like to stop at?\n')
            if stop_column not in devices_columns_names:
                print('Column %s not in the predefined columns of the %s mask! Try again.' %(stop_column, mask))
                stop_column = None

        print("start_row:", start_row)
        print("stop_row:", stop_row)
        print("start_column:", start_column)
        print("stop_column:", stop_column)

        if start_row != None and stop_row != None and start_column != None and stop_column != None:
            i_start_row = None
            i_stop_row = None
            i_start_column = None
            i_stop_column = None

            for foo_i in range(len(devices_rows_names)):
                if devices_rows_names[foo_i] == start_row:
                    i_start_row = foo_i
                if devices_rows_names[foo_i] == stop_row:
                    i_stop_row = foo_i

            for foo_i in range(len(devices_columns_names)):
                if devices_columns_names[foo_i] == start_column:
                    i_start_column = foo_i
                if devices_columns_names[foo_i] == stop_column:
                    i_stop_column = foo_i
            print("sliced devices_rows_names:", devices_rows_names[i_start_row:i_stop_row+1])
            print("sliced devices_columnss_names:", devices_columns_names[i_start_column:i_stop_column+1])
            print("Now mapping!")
            if_draw_figs = False

            for current_row in devices_rows_names[i_start_row:i_stop_row+1]:
                for current_column in devices_columns_names[i_start_column:i_stop_column+1]:
                    current_dev = "%s_%s%s" % (unit_cell, current_row, current_column)
                    print("Current_device: %s" % current_dev)
                    foo_mask_map = mask_mapping(mask, current_dev, cs_offset)
                    print(foo_mask_map)
                    if foo_mask_map[4] != None and foo_mask_map[5] != None:
                        Zaxis.move_to(1000.0, unit='pulses')
                        foo_theta = (cs_theta*numpy.pi)/180.0
                        foo_x = foo_mask_map[4]*cos(foo_theta) + foo_mask_map[5]*sin(foo_theta)
                        foo_y = -foo_mask_map[4]*sin(foo_theta) + foo_mask_map[5]*cos(foo_theta)
                        if int(round(foo_x))%2 == 0:
                            foo_x = int(round(foo_x))
                        else:
                            foo_x = int(round(foo_x))-1
                            #foo_x = int(round(numpy.ceil(foo_x)))
                        if int(round(foo_y))%2 == 0:
                            foo_y = int(round(foo_y))
                        else:
                            foo_y = int(round(foo_y))-1
                            #foo_y = int(round(numpy.ceil(foo_y)))

                        print("foo_x =", foo_x, int(round(foo_x)))
                        print("foo_y =", foo_y, int(round(foo_y)))
                        Xaxis.move_to(foo_x, unit='um')
                        Yaxis.move_to(-foo_y, unit='um')
                        Zaxis.move_to(0.0, unit='pulses')

                        for foo_n in range(no_of_measurements):
                            if which_power_supply == "LDC4005":
                                PS.write('SOURce:CURRent 0')
                                measured_I, measured_U, measured_P, fig1, fig2 = measure_LIV(4e-4, max_current, resolution, PS, MM=K2100, PM=PM100USB, use_RO_fuse=True, draw_figs=if_draw_figs)
                            if which_power_supply == "B2901BL":
                                measured_I, measured_U, measured_P, fig1, fig2 = measure_LIV_B2901BL(0.0, max_current, max_voltage, resolution, PS, PM=PM100USB, use_RO_fuse=True, draw_figs=if_draw_figs)
                            if len(measured_I) > 0:
                                dump_to_txt(measured_I, measured_U, measured_P, '%s_%s_RTC_%s.dat' % (sample_name, current_dev, foo_n))
                            #fig1.savefig('%s_%s_RTC_LIV_1.png' % (sample_name, current_dev), bbox_inches='tight')
                            #fig2.savefig('%s_%s_RTC_dLdI_1.png' % (sample_name, current_dev), bbox_inches='tight')
                            if if_draw_figs == True:
                                close(fig1)
                                close(fig2)
                        Zaxis.move_to(1000.0, unit='pulses')
                        #time.sleep(1)


                        print("Done!")
                Zaxis.move_to(1000.0, unit='pulses')

        if_abort = 1
