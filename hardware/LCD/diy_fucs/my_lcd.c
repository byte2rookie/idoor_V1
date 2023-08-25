#include "my_lcd.h"

void start_LCD(void)
{
    // 1.LCD 初始化
    ILI9341_Init();
    // 2.设置液晶单元指针的扫描模式
    ILI9341_GramScan(LCD_SCAN_MODE); // 默认设置也是6号模式，这一步可以省略
    // 3.进行显示测试
    LCD_SetBackColor(GBLUE);
    ILI9341_Clear(0, 0, LCD_X_LENGTH, LCD_Y_LENGTH); /* 清屏，显示全黑 */
    // 先从文字显示来试试看吧
    LCD_SetColors(BLACK, YELLOW); // 调一下颜色
    LCD_SetFont(&Font8x16);       // 调一下字号
    ILI9341_DispStringLine_EN(LINE(0), "hello!my deer friend!");
    delay_ms(500);
    ILI9341_DispStringLine_EN(LINE(1), "HERE IS .......");
    LCD_SetColors(WHITE, GREY);
    LCD_SetFont(&Font24x32);
    delay_ms(500);
    ILI9341_DispStringLine_EN(LINE(1), "   ZZY   ");
    LCD_SetFont(&Font8x16);
    LCD_SetColors(RED, GREEN);
    ILI9341_DispStringLine_EN(LINE(4), "   from UESTC   ");
    delay_ms(500);
}

void recongization_LCD(void)
{
    ILI9341_Clear(0, 0, LCD_X_LENGTH, LCD_Y_LENGTH);
    ILI9341_DispStringLine_EN(LINE(5), "let's start the program!");
    delay_ms(500);
    ILI9341_DispStringLine_EN(LINE(6), "the result of recognization:");
    delay_ms(500);
}

void result_LCD(int flag)
{

    if (flag == SUCCESS_recognization)
    {
        LCD_SetColors(RED, GREEN);
        LCD_SetFont(&Font24x32);
        ILI9341_Clear(0, 0, LCD_X_LENGTH, LCD_Y_LENGTH); /* 清屏*/
        delay_ms(300);
        ILI9341_DispStringLine_EN(LINE(4), "  PASS!   ");
        delay_ms(500);
    }
    else if (flag == FALSE_recognization)
    {
        LCD_SetColors(RED, GREEN);
        LCD_SetFont(&Font24x32);
      
        ILI9341_Clear(0, 0, LCD_X_LENGTH, LCD_Y_LENGTH); /* 清屏 */
        delay_ms(300);
        ILI9341_DispStringLine_EN(LINE(4), "  DENIED!   ");
      delay_ms(300);
    }
    else if (flag == TURNOFF)
    {

        LCD_SetColors(WHITE, BLACK);
        ILI9341_Clear(0, 0, LCD_X_LENGTH, LCD_Y_LENGTH);
        LCD_SetFont(&Font24x32);
      delay_ms(300);
        ILI9341_DispStringLine_EN(LINE(4), "  BYE!   ");
        delay_ms(300);
      return;
    }
    
    LCD_SetFont(&Font8x16);
    LCD_SetColors(RED, GREEN);
    ILI9341_Clear(0, 0, LCD_X_LENGTH, LCD_Y_LENGTH); /* 清屏，显示绿色背景*/
}