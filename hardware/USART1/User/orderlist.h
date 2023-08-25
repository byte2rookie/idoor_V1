#ifndef __ORDERLIST_H
#define __ORDERLIST_H



#define COMMUNICATION_REQUEST 0x01
#define SUCCESS_COMMUNICATION 0x01    // 用python提前把命令对应的号码提取出来了
#define SUCCESS_recognization 0x02
#define FALSE_recognization 0x03
#define RESET_RECOGNIZATION 0x09


#define GET_NEWINFO 0x10
#define CONFIRM_NEWINFO 0x11
#define PAUSE 0x12


#define TURNOFF 0x00



#define MAX_WAITTIME 200
#endif