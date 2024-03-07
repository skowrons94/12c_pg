import os
import time

import numpy as np

from .client import client
from .server import server

class azure2:

    PORT = 20000
    servers = []

    def __init__( self, file, nprocs = 1 ):
        self.file= file
        self.clients = []
        self.spawn( nprocs )
        self.configure( )

    def __del__( self ):
        for instance in self.instances: instance.stop( )

    def spawn( self, nprocs ):
        for i in range( nprocs ): self.servers.append( server( self.PORT + i, self.file ) )
        time.sleep( 1 )
        for i in range( nprocs ): self.clients.append( client( port=self.PORT + i ) )
        time.sleep( 1 )
        for i in range( nprocs ): self.clients[i].communicate( "INITIALIZE", [0] )
        time.sleep( 1 )

    def configure( self ):
        self.nsegments = int(self.clients[0].communicate( "UPDATE_DATA", [0] )[0])
        self.energies = [ self.clients[0].communicate( "GET_DATA_ENERGIES", [i] ) for i in range( self.nsegments ) ]
        self.cross = [ self.clients[0].communicate( "GET_DATA_SEGMENTS", [i] ) for i in range( self.nsegments ) ]
        self.cross_err = [ self.clients[0].communicate( "GET_DATA_SEGMENTS_ERRORS", [i] ) for i in range( self.nsegments ) ]
        self.conv = [ self.clients[0].communicate( "GET_DATA_CONV", [i] ) for i in range( self.nsegments ) ]
        self.sfactor = [ self.cross[i] * self.conv[i] for i in range( self.nsegments ) ]
        self.sfactor_err = [ self.cross_err[i] * self.conv[i] for i in range( self.nsegments ) ]
        self.params = self.clients[0].communicate( "GET_PARAMS", [0] )

    def calculate( self, params, proc = 0 ):
        nsegments = int(self.clients[proc].communicate( "UPDATE_SEGMENTS", params ))
        segments = [ self.clients[proc].communicate( "GET_CALCULATED_SEGMENT", [i] ) for i in range( nsegments ) ]
        return segments
    
    def calculate_energies( self, params, proc = 0 ):
        nsegments = int(self.clients[proc].communicate( "UPDATE_SEGMENTS", params ))
        segments = [ self.clients[proc].communicate( "GET_CALCULATED_ENERGIES", [i] ) for i in range( nsegments ) ]
        return segments
    
    def calculate_sfactor( self, params, proc = 0 ):
        nsegments = int(self.clients[proc].communicate( "UPDATE_SEGMENTS", params ))
        segments = [ self.clients[proc].communicate( "GET_CALCULATED_SEGMENT", [i] ) for i in range( nsegments ) ]
        conv = [ self.clients[proc].communicate( "GET_CALCULATED_CONV", [i] ) for i in range( nsegments ) ]
        for i in range( nsegments ): segments[i] *= conv[i]
        return segments
    
    def extrap_mode( self ):
        for i in range( len( self.clients ) ): 
            self.clients[i].communicate( "SET_EXTRAP_MODE", [0] )
            self.clients[i].communicate( "INITIALIZE", [0] )

    def data_mode( self ):
        for i in range( len( self.clients ) ): 
            self.clients[i].communicate( "SET_DATA_MODE", [0] )
            self.clients[i].communicate( "INITIALIZE", [0] )

