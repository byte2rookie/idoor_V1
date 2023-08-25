#include "delay.h"

// 延时效果太垃圾了，什么情况，看来得用Systic或者TIM延时一下
void delay(int max)
{
    for (int i = max; i > 0; i--)
        for (int j = max; j > 0; j--)
            ;
}

// 微秒级的延时
void delay_us(uint32_t delay_us)
{
    volatile unsigned int num;
    volatile unsigned int t;

    for (num = 0; num < delay_us; num++)
    {
        t = 11;
        while (t != 0)
        {
            t--;
        }
    }
}
// 毫秒级的延时
void delay_ms(uint16_t delay_ms)
{
    volatile unsigned int num;
    for (num = 0; num < delay_ms; num++)
    {
        delay_us(1000);
    }
}