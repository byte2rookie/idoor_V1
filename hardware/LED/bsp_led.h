#ifndef  __BSP_LED_H //防止stm32f10x.h被重复定义
#define __BSP_LED_H

#include "stm32f10x.h"
#define LED_G_GPIO_PIN          GPIO_Pin_0
#define LED_G_GPIO_PORT       GPIOB
#define LED_G_GPIO_CLK      RCC_APB2Periph_GPIOB

#define LED_R_GPIO_PIN          GPIO_Pin_5
#define LED_R_GPIO_PORT       GPIOB
#define LED_R_GPIO_CLK      RCC_APB2Periph_GPIOB

#define LED_B_GPIO_PIN          GPIO_Pin_1
#define LED_B_GPIO_PORT       GPIOB
#define LED_B_GPIO_CLK      RCC_APB2Periph_GPIOB


//学一下更简明的宏定义
//   "\"是续行符，可以链接不同行
#define ON 0
#define OFF 1
#define LED_G(a)        if(a)\
                                GPIO_SetBits(LED_G_GPIO_PORT, LED_G_GPIO_PIN);\
                                else GPIO_ResetBits(LED_G_GPIO_PORT, LED_G_GPIO_PIN);
#define LED_B(a)        if(a)\
                                GPIO_SetBits(LED_B_GPIO_PORT, LED_B_GPIO_PIN);\
                                else GPIO_ResetBits(LED_B_GPIO_PORT, LED_B_GPIO_PIN);
#define LED_R(a)        if(a)\
                                GPIO_SetBits(LED_R_GPIO_PORT, LED_R_GPIO_PIN);\
                                else GPIO_ResetBits(LED_R_GPIO_PORT, LED_R_GPIO_PIN);



void led_config(void);
void led_reset(void);
#endif


