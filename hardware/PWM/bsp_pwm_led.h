#ifndef __BSP_PWM_LED_H
#define __BSP_PWM_LED_H

#include "stm32f10x.h"




#define TIM_BASIC TIM6


typedef enum{
  FAST = 3,
  MIDDLE =2,
  SLOW = 1,
  STOP = 0
}OPEN_BOX_SPEED;


#define GREENBOX 0
#define REDBOX 1

void TIM_basic_config(void);
void TIM_COM_config(void);
void TIM_COM2_config(void);
void SetRedbox(uint32_t compare);
void SetGreenbox(uint32_t compare);
void openbox(uint32_t box);
void closebox(uint32_t box);
void breathLED_B_config(void);
void pwm_set_compare(uint32_t compare);

#endif