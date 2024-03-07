import subprocess 
import threading

class server:

    def __init__(self,port,file):
        self.port, self.file = port, file
        self.thread = threading.Thread( target=self.func )
        self.start( )

    def start(self):
        self.thread.start( )

    def stop(self):
        self.thread.join( )

    def func(self):
        try: 
            subprocess.call("AZURE2 --no-gui --gsl-coul --use-api {} {}".format(self.port,self.file), shell=True)
        except:
            print("Error: Failed to run AZURE2")
            quit( )
    
