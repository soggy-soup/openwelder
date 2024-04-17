import serial
import time


def generate_path_gcode(transformed_path, feed_rate):
    #print(len(transformed_path[0][0][0]))
    with open('gcode.txt', 'w') as file:
        file.write('$X\nG92 X0 Y0 Z0\nG17 G21 G90 G94 G54\nG01 X0 Y0 Z0 F5000\n' ) # X{} Y{} Z0 F{}\n'.format(transformed_path[0][0][0][0],transformed_path[0][0][0][1],feed_rate))
        
        for i in range(len(transformed_path[0])):
            xcoord =transformed_path[0][i][0][0]+5
            ycoord = transformed_path[0][i][0][1]
            if i == 0:
                file.write('X{} Y{} Z0\n'.format(xcoord,ycoord))
                file.write('X{} Y{} Z0 F{}\n'.format(xcoord,ycoord,feed_rate))
                file.write('M8\nG4 P0.5\n')
            else:
                file.write('X{} Y{} Z0 F{}\n'.format(xcoord,ycoord,feed_rate))
            
        file.write('M9\nG4 P01\nG01 Z5 F5000\nX0 Y0\nZ0\n') #CHANGE G4 DWELL
        
        
def stream_gcode(gcode_file):
    #Following function is based off of GRBL source code simple_stream.py file 
    #https://github.com/gnea/grbl/blob/master/doc/script/simple_stream.py
    
    # Open grbl serial port
    s = serial.Serial('COM3',115200)

    # Open g-code file
    f = open(gcode_file,'r')

    # Wake up grbl
    s.write("\r\n\r\n".encode())
    time.sleep(2)   # Wait for grbl to initialize 
    s.flushInput()  # Flush startup text in serial input

    # Stream g-code to grbl
    for line in f:
        li = line.strip() # Strip all EOL characters for consistency
        print('Sending: ', li)
        s.write(li.encode()+ '\n'.encode()) # Send g-code block to grbl
        grbl_out = s.readline() # Wait for grbl response with carriage return
        print(' : ', grbl_out.strip())

    # Wait here until grbl is finished to close serial port and file.
    #raw_input("  Press <Enter> to exit and disable grbl.") 

    # Close file and serial port
    f.close()
    s.close()   