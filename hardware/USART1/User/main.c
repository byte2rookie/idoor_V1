/**
  ******************************************************************************
  * @file    main.c
  * 
  * 
  *     
  *   
  ******************************************************************************
  * @attention
  *
  * 实验平台:野火 F103-指南者 STM32 开发板 
  * 
  * 
  *
  ******************************************************************************
  */ 
 
 
#include "stm32f10x.h"
//#include "bsp_usart.h"
//#include "bsp_led.h"
//#include "orderlist.h"
//#include "bsp_pwm_led.h"
#include "bsp_ili9341_lcd.h"
#include "fonts.h"
#include "my_lcd.h"
//#include "delay.h"




/**
  * @brief  主函数
  * @param  无
  * @retval 无
  */
#if 1
int main(void)
{	
  /*初始化USART 配置模式为 115200 8-N-1，中断接收*/
  USART_Config();
	led_config();
  TIM_COM_config();
  TIM_COM2_config();
  start_LCD();//打开液晶
  LED_B(ON);//红灯表示已经开始工作(可以拓展成呼吸灯之类的？)
  
  //上机后先通过串口一直向上位机发送信息，直到上位机回应
  //如果超过等待时间就自动关闭
 extern uint32_t respon;
  uint8_t data_start;
  
  data_start=COMMUNICATION_REQUEST; //发送一个请求信号
  
  while(1)
  {
    Usart_SendByte(DEBUG_USARTx,data_start);
    respon=USART_ReceiveData(DEBUG_USARTx);   
  if(respon==SUCCESS_COMMUNICATION)
  {   
    LED_B(OFF)
    //与上位机通信成功，亮起黄灯
    LED_R(ON);
    delay_ms(100);
    LED_G(ON);
    break;
  }
  delay_ms(1000);
}      
   
while(1)
{
  	
    recongization_LCD();
    //respon = USART_ReceiveData(DEBUG_USARTx);  //或许是这里重复了中断？
   
  
    //LCD展示结果
    result_LCD(respon); 
                
  if(respon==SUCCESS_recognization)
    {  
      led_reset();
        openbox(GREENBOX);
        LED_G(ON);
      delay_ms(1000);      
      closebox(GREENBOX);
    }
  else if (respon==FALSE_recognization)
    { 
      led_reset();
      openbox(REDBOX);      
      LED_R(ON);
      delay_ms(1000);     
      closebox(REDBOX);
      }
  else if(respon==RESET_RECOGNIZATION)
    {                
      led_reset();      
      LED_G(ON);      
      LED_R(ON);
      LED_B(ON);
      delay_ms(100);      
    }	
  else if(respon==TURNOFF)
    {
      led_reset();
      openbox(GREENBOX);
      openbox(REDBOX);
      delay_ms(1000);
      closebox(REDBOX);
      closebox(GREENBOX);
      break;
    }
    respon=PAUSE;
}  
      
}
/*********************************************END OF FILE**********************/


#elif 0
int main(void)
{
led_config();
breathLED_B_config();
int i;
while(1)
{
for(;i<100;i++)
{
  pwm_set_compare(i);
    delay_ms(10);
}
for(i=100;i>0;i--)
{
  pwm_set_compare(i);
    delay_ms(10);
}}
}

#elif 0//舵机程序，启动！
int main(void)
{
TIM_COM_config();
TIM_COM2_config();
while(1)
  {
// SetRedbox(500);
//  delay_ms(1000);
// SetRedbox(2500);
//  delay_ms(1000);
// SetGreenbox(500);
//  delay_ms(1000);
// SetGreenbox(2500);
//  delay_ms(1000);   
    delay_ms(1000);   
    openbox(REDBOX);
    openbox(GREENBOX);
}
  //SetDJ3(1500);//停止了
 // SetDJ3(2500);
}

#elif 0
int main(void)//测试一下LCD显示
{
  //1.LCD 初始化
  ILI9341_Init ();
  //2.设置液晶单元指针的扫描模式
  ILI9341_GramScan(LCD_SCAN_MODE);//默认设置也是6号模式，这一步可以省略
  //3.进行显示测试
  LCD_SetBackColor(GBLUE);
  ILI9341_Clear(0,0,LCD_X_LENGTH,LCD_Y_LENGTH);	/* 清屏，显示全黑 */
  //先从文字显示来试试看吧
  LCD_SetColors(BLACK,YELLOW);//调一下颜色
  LCD_SetFont(&Font8x16);//调一下字号  
  ILI9341_DispStringLine_EN(LINE(0),"hello!my deer friend!");
  delay_ms(500);
  ILI9341_DispStringLine_EN(LINE(1),"HERE IS .......");
  LCD_SetColors(WHITE,GREY);
  LCD_SetFont(&Font24x32);
  delay_ms(500);
  ILI9341_DispStringLine_EN(LINE(1),"   ZZY   ");
  LCD_SetFont(&Font8x16);
  LCD_SetColors(RED,GREEN);
  ILI9341_DispStringLine_EN(LINE(4),"   from UESTC   ");
  delay_ms(500);
  int flag=0;
  while(1)
  {
  ILI9341_DispStringLine_EN(LINE(5),"let's start the program!");
  delay_ms(1000);
    ILI9341_DispStringLine_EN(LINE(6),"the result of recognization:");
  delay_ms(1000);  
  if(flag)
  {
    LCD_SetColors(RED,GREEN);
     LCD_SetFont(&Font24x32);
    ILI9341_Clear(0,0,LCD_X_LENGTH,LCD_Y_LENGTH);	/* 清屏，显示全黑 */
    delay_ms(2000);
  ILI9341_DispStringLine_EN(LINE(4),"  PASS!   ");
  }
  else
  {
    LCD_SetColors(RED,GREEN);
     LCD_SetFont(&Font24x32);
    ILI9341_Clear(0,0,LCD_X_LENGTH,LCD_Y_LENGTH);	/* 清屏，显示全黑 */
    delay_ms(2000);
  ILI9341_DispStringLine_EN(LINE(4),"  DENIED!   ");
  }
   delay_ms(2000);
  LCD_SetFont(&Font8x16);
  LCD_SetColors(RED,GREEN);
  ILI9341_Clear(0,0,LCD_X_LENGTH,LCD_Y_LENGTH);	/* 清屏，显示全黑 */
  
}
}


#endif
#if 0
void start_LCD(void)
{
  //1.LCD 初始化
  ILI9341_Init ();
  //2.设置液晶单元指针的扫描模式
  ILI9341_GramScan(LCD_SCAN_MODE);//默认设置也是6号模式，这一步可以省略
  //3.进行显示测试
  LCD_SetBackColor(GBLUE);
  ILI9341_Clear(0,0,LCD_X_LENGTH,LCD_Y_LENGTH);	/* 清屏，显示全黑 */
  //先从文字显示来试试看吧
  LCD_SetColors(BLACK,YELLOW);//调一下颜色
  LCD_SetFont(&Font8x16);//调一下字号  
  ILI9341_DispStringLine_EN(LINE(0),"hello!my deer friend!");
  delay_ms(500);
  ILI9341_DispStringLine_EN(LINE(1),"HERE IS .......");
  LCD_SetColors(WHITE,GREY);
  LCD_SetFont(&Font24x32);
  delay_ms(500);
  ILI9341_DispStringLine_EN(LINE(1),"   ZZY   ");
  LCD_SetFont(&Font8x16);
  LCD_SetColors(RED,GREEN);
  ILI9341_DispStringLine_EN(LINE(4),"   from UESTC   ");
  delay_ms(500);
  int flag=0;
  
}

void recongization_LCD(void)
{
ILI9341_Clear(0,0,LCD_X_LENGTH,LCD_Y_LENGTH);
  ILI9341_DispStringLine_EN(LINE(5),"let's start the program!");
  delay_ms(1000);
  ILI9341_DispStringLine_EN(LINE(6),"the result of recognization:");
  delay_ms(1000);  
}


void result_LCD(int flag)
{

    
  if(flag==SUCCESS_recognization)
  {
    LCD_SetColors(RED,GREEN);
     LCD_SetFont(&Font24x32);
    ILI9341_Clear(0,0,LCD_X_LENGTH,LCD_Y_LENGTH);	/* 清屏*/
    delay_ms(2000);
  ILI9341_DispStringLine_EN(LINE(4),"  PASS!   ");
  }
  else if(flag==FALSE_recognization)
  {
    LCD_SetColors(RED,GREEN);
     LCD_SetFont(&Font24x32);
    ILI9341_Clear(0,0,LCD_X_LENGTH,LCD_Y_LENGTH);	/* 清屏 */
    delay_ms(2000);
  ILI9341_DispStringLine_EN(LINE(4),"  DENIED!   ");
  }
  else if(flag==TURNOFF)
  {
    
    LCD_SetColors(WHITE,BLACK);
    ILI9341_Clear(0,0,LCD_X_LENGTH,LCD_Y_LENGTH);
    LCD_SetFont(&Font24x32);
    ILI9341_DispStringLine_EN(LINE(4),"  BYE!   ");
    return;
  }
   delay_ms(2000);
  LCD_SetFont(&Font8x16);
  LCD_SetColors(RED,GREEN);
  ILI9341_Clear(0,0,LCD_X_LENGTH,LCD_Y_LENGTH);	/* 清屏，显示绿色背景*/
}

#endif