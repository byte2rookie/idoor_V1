import time
import serial #引入外设
import 核心_人脸识别门禁

if __name__ == '__main__':
    #1.使用串口通信发送启动指令
    serial = serial.Serial('COM4', 115200, timeout=2)  # 连接COM14,波特率位115200
    if serial.isOpen():
        print('串口已打开')
    else:
        print('串口未打开')

    sign=serial.read(1)#记录从单片机收到的信息
    print(sign)
    data=b'\x01'
    if sign == b'\x01':
        serial.write(data)#向外设中写入1的信号试试看
        print('输入的数据为:',data) 

while True:
        time.sleep(1)
        sign1=核心_人脸识别门禁.recognization()
        if sign1==1:#识别成功
            #order=进制转换.conv_message(1)
            order=b'\x02'
            serial.write(order)
            #发送命令回去
        elif sign1==0:
            #order=进制转换.conv_message(0)
            order=b'\x03'
            serial.write(order)

        #databack = serial.read(1)
        #print(f"返回的值识别指令为为：{databack}")

        #reset=b'\x09'
        #serial.write(reset)
        time.sleep(6)
        if ord(' ') == True:
            turnoff=b'\x00'
            serial.write(turnoff)
            break


