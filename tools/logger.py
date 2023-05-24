import sys


class Logger(object):
    '''
    Class to save the terminal output into a file
    '''
    def __init__(self, outputsPath, logfile_name):
        self.terminal = sys.stdout
        self.log = open(outputsPath + "{}.log".format(logfile_name), "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass 
    
    
if __name__ == "__main__":
    
    sys.stdout = Logger(outputsPath="", logfile_name="logfile_test")
    print("Testing log file")
    