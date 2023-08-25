#ifndef __MY_LCD_H
#define __MY_LCD_H

#include "stm32f10x.h"
#include "bsp_ili9341_lcd.h"
#include "fonts.h"
void start_LCD(void);
void recongization_LCD(void);
void result_LCD(int flag);

#endif