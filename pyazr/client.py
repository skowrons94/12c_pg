import socket
import struct

import numpy as np

class client:

    CMD = { 'INITIALIZE': 0, 
            'UPDATE_SEGMENTS': 1, 
            'GET_PARAMS': 2, 
            'GET_PARAMS_NAMES': 3,
            'GET_PARAMS_ALL': 4, 
            'CALCULATE_EXTERNAL_CAPTURE': 5, 
            'GET_DATA_ENERGIES': 6, 
            'GET_DATA_SEGMENTS': 7, 
            'GET_DATA_SEGMENTS_ERRORS': 8, 
            'UPDATE_DATA': 9, 
            'SET_DATA_MODE': 10, 
            'SET_EXTRAP_MODE': 11, 
            'GET_CALCULATED_SEGMENT': 12, 
            'GET_CALCULATED_ENERGIES': 13, 
            'SET_CHANNEL_RADIUS': 14, 
            'GET_NORMS': 15, 
            'GET_NORMS_ERRORS': 16, 
            'GET_CALCULATED_SEGMENT_E1': 17, 
            'GET_CALCULATED_SEGMENT_E2': 18, 
            'GET_DATA_CONV': 19,
            'GET_CALCULATED_CONV': 20 }
    
    BUFFER_SIZE = 10000 * 8

    def __init__(self, server='localhost', port=20000):
        # Server information
        self.server_address = server
        self.server_port = port
        self.connect( )

    def __del__(self):
        self.disconnect( )

    # Connect to the server
    def connect(self):
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket.connect((self.server_address, self.server_port))

    # Disconnect from the server
    def disconnect(self):
        self.client_socket.close( )

    # Function to send data
    def send(self, cmd, data, type = 'd'):
        buffer = bytearray(self.BUFFER_SIZE)
        
        buffer[:8], buffer[8:16] = struct.pack('d', cmd), struct.pack('d', len(data))
        for i in range( len( data ) ): buffer[16+i*8:16+(i+1)*8] = struct.pack( type, data[i] )

        self.client_socket.sendall(buffer)

    # Function to receive data
    def receive(self, type):

        buffer = self.client_socket.recv(self.BUFFER_SIZE, socket.MSG_WAITALL)
        try: buffer_size = int(struct.unpack(type, buffer[:8])[0])
        except: buffer_size = 0

        data = [ struct.unpack( type, buffer[8+i*8:8+(i+1)*8] )[0] for i in range( buffer_size )]

        return np.asarray( data )
    
    def communicate(self, cmd, data, type = 'd'):
        self.send( self.CMD[cmd], data, type )
        data = self.receive( type )
        return data