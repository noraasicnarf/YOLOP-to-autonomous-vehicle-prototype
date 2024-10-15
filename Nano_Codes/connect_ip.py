import socket
import serial

def setup_jetson_socket(port):
    jetson_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    jetson_socket.bind(("0.0.0.0", port))
    return jetson_socket

def setup_arduino_serial(port, baudrate):
    return serial.Serial(port, baudrate)

def main():
    jetson_port = 8889
    arduino_port = '/dev/ttyUSB0'
    arduino_baudrate = 9600

    
    jetson_socket = setup_jetson_socket(jetson_port)
    arduino_serial = setup_arduino_serial(arduino_port, arduino_baudrate)

    while True:
        data, addr = jetson_socket.recvfrom(1024)
        received_message = data.decode()
        print(f"Received message: {received_message}")

        arduino_serial.write(received_message.encode())


if __name__ == "__main__":
    main()
