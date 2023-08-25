#include "bsp_led.h"




void led_config(void)
{
  //1.打开时钟
    RCC_APB2PeriphClockCmd(LED_G_GPIO_CLK, ENABLE);//ENABLE这类数在stm32f10x.h里已经定义好了
  
  //2.设置GPIO
  GPIO_InitTypeDef  LED_structure;
  LED_structure.GPIO_Mode=GPIO_Mode_Out_PP;
  LED_structure.GPIO_Speed=GPIO_Speed_10MHz;
  LED_structure.GPIO_Pin=GPIO_Pin_5;
  //先将ODR初始化为1
  GPIO_SetBits(LED_G_GPIO_PORT, LED_G_GPIO_PIN);
  GPIO_SetBits(LED_B_GPIO_PORT, LED_B_GPIO_PIN);
  GPIO_SetBits(LED_R_GPIO_PORT, LED_R_GPIO_PIN);
  //再初始化
  GPIO_Init(LED_R_GPIO_PORT, &LED_structure);
  LED_structure.GPIO_Pin=GPIO_Pin_0;
  GPIO_Init(LED_G_GPIO_PORT, &LED_structure);
  LED_structure.GPIO_Pin=GPIO_Pin_1;
  GPIO_Init(LED_B_GPIO_PORT, &LED_structure);
}

void led_reset(void)
{
  LED_R(OFF);
  LED_G(OFF);
  LED_B(OFF);
}

