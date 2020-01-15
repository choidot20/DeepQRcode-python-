## 클라이언트(자바로 만들어야됨)

import socket

s = socket.socket()         # 동일
host = '1.245.48.126' # 동일
port = 12345                  # 동일
print('host : ', host)
s.connect((host, port))     # 동일
f = open('3.png','rb')      # 보낼 파일명
l = f.read(1024)            # 파일을 읽어들임
while (l):                  # 파일을 다 읽을 때까지
    s.send(l)               # 파일을 보냄
    l = f.read(1024)        # 파일이 끝나지않았으면 더 읽음
print("finish")
f.close()
print('fclose')
##data = s.recv(1024)
##print('받은 데이터 : ', data.decode('utf-8'))
s.close()                     # disconnection