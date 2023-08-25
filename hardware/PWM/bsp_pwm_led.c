#include "bsp_pwm_led.h"


//先学习怎么用初始化的计时器(1s的)
void TIM_basic_config(void)
{
  //先打开时钟
  RCC_APB1PeriphClockCmd(RCC_APB1Periph_TIM6,ENABLE);
  //设置基本单元部分
  TIM_TimeBaseInitTypeDef TIM_TimeBaseInitStruct;
  TIM_TimeBaseStructInit(&TIM_TimeBaseInitStruct);//将这个结构体初始化，这样就不用填那些可能用不上的参数了
  TIM_TimeBaseInitStruct.TIM_ClockDivision=TIM_CKD_DIV2;//36MHz
  TIM_TimeBaseInitStruct.TIM_CounterMode=TIM_CounterMode_Up;//向上计数模式
  TIM_TimeBaseInitStruct.TIM_Period=1000-1;//每次重载中断都是1000个周期
  TIM_TimeBaseInitStruct.TIM_Prescaler=36000-1;//分频为36000-1=1000Hz,周期就是1ms
  //TIM_TimeBaseInitStruct.TIM_RepetitionCounter=;//只有高级定时器才需要写
  TIM_TimeBaseInit(TIM_BASIC,&TIM_TimeBaseInitStruct);
  //这样就初始化好了基本定时器
  //设置一下中断，方便重置
  
  NVIC_InitTypeDef NVIC_InitStructure;
  NVIC_InitStructure.NVIC_IRQChannel = TIM6_IRQn;  //TIM6中断
	NVIC_InitStructure.NVIC_IRQChannelPreemptionPriority = 0;  //抢占优先级0级
	NVIC_InitStructure.NVIC_IRQChannelSubPriority = 3;  //从优先级3级
	NVIC_InitStructure.NVIC_IRQChannelCmd = ENABLE; //IRQ通道被使能
	NVIC_Init(&NVIC_InitStructure);  //初始化NVIC寄存器

  
  TIM_Cmd(TIM_BASIC,ENABLE);//打开TIM6  
}

//学习使用通用计时器
void TIM_COM_config(void)
{     
  //先打开时钟
  RCC_APB1PeriphClockCmd(RCC_APB1Periph_TIM2,ENABLE);
  TIM_InternalClockConfig(TIM2);
  RCC_APB2PeriphClockCmd(RCC_APB2Periph_GPIOA,ENABLE);
  //设置基本单元部分
  TIM_TimeBaseInitTypeDef TIM_TimeBaseInitStruct;
  TIM_TimeBaseStructInit(&TIM_TimeBaseInitStruct);//将这个结构体初始化，这样就不用填那些可能用不上的参数了
  TIM_TimeBaseInitStruct.TIM_ClockDivision=TIM_CKD_DIV1;//72MHz
  TIM_TimeBaseInitStruct.TIM_CounterMode=TIM_CounterMode_Up;//向上计数模式
  TIM_TimeBaseInitStruct.TIM_Period=20000-1;//20000次计数周期            ARR
  TIM_TimeBaseInitStruct.TIM_Prescaler=72-1;//分频为72-1=1MHz,周期就是10^-6 s         PSC
  TIM_TimeBaseInit(TIM2,&TIM_TimeBaseInitStruct);
  //PWM频率为 72M/72/0.02M=50Hz,周期为0.02s也就是20ms,与舵机要求的周期一致
  //查看了开发板的原理图，发现PA0被KEY1占用了，所以就先用PA2，也就是OC3频道
  //接下来初始化输出比较单元（配置好基本单元后就可以开始进行输出比较）
  TIM_OCInitTypeDef TIM_OCInitStruct;
  TIM_OCStructInit(&TIM_OCInitStruct);
  TIM_OCInitStruct.TIM_OCMode=TIM_OCMode_PWM1;//采用PWM1模式
  TIM_OCInitStruct.TIM_OCPolarity=TIM_OCPolarity_High;//选取有效电平
  TIM_OCInitStruct.TIM_OutputState=TIM_OutputState_Enable;//打开输出开关
  TIM_OCInitStruct.TIM_Pulse=0;//      CCR:对应500~2500,对应也就是对应0.5ms~2.5ms(-90~+90)
  TIM_OC3Init(TIM2,&TIM_OCInitStruct);
  TIM_Cmd(TIM2,ENABLE);
  
    //PWM也是通过GPIO来输出的，所以要把对应PWM的GPIO给初始化好
  GPIO_InitTypeDef GPIO_InitStructure;
  GPIO_InitStructure.GPIO_Pin = GPIO_Pin_2;//PA2,对应TIM2_OC3
	GPIO_InitStructure.GPIO_Mode = GPIO_Mode_AF_PP;//推挽复用MODE
	GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
	GPIO_Init(GPIOA, &GPIO_InitStructure);    
}

void TIM_COM2_config(void)
{     
  //先打开时钟
  RCC_APB1PeriphClockCmd(RCC_APB1Periph_TIM4,ENABLE);
  TIM_InternalClockConfig(TIM4);
  RCC_APB2PeriphClockCmd(RCC_APB2Periph_GPIOB,ENABLE);
  //设置基本单元部分
  TIM_TimeBaseInitTypeDef TIM_TimeBaseInitStruct;
  TIM_TimeBaseStructInit(&TIM_TimeBaseInitStruct);//将这个结构体初始化，这样就不用填那些可能用不上的参数了
  TIM_TimeBaseInitStruct.TIM_ClockDivision=TIM_CKD_DIV1;//72MHz
  TIM_TimeBaseInitStruct.TIM_CounterMode=TIM_CounterMode_Up;//向上计数模式
  TIM_TimeBaseInitStruct.TIM_Period=20000-1;//20000次计数周期            ARR
  TIM_TimeBaseInitStruct.TIM_Prescaler=72-1;//分频为72-1=1MHz,周期就是10^-6 s         PSC
  TIM_TimeBaseInit(TIM4,&TIM_TimeBaseInitStruct);
  //PWM频率为 72M/72/0.02M=50Hz,周期为0.02s也就是20ms,与舵机要求的周期一致
  //查看了开发板的原理图，发现PA0被KEY1占用了，所以就先用PA2，也就是OC3频道
  //接下来初始化输出比较单元（配置好基本单元后就可以开始进行输出比较）
  TIM_OCInitTypeDef TIM_OCInitStruct;
  TIM_OCStructInit(&TIM_OCInitStruct);
  TIM_OCInitStruct.TIM_OCMode=TIM_OCMode_PWM1;//采用PWM1模式
  TIM_OCInitStruct.TIM_OCPolarity=TIM_OCPolarity_High;//选取有效电平
  TIM_OCInitStruct.TIM_OutputState=TIM_OutputState_Enable;//打开输出开关
  TIM_OCInitStruct.TIM_Pulse=0;//      CCR:对应500~2500,对应也就是对应0.5ms~2.5ms(-90~+90)
  TIM_OC3Init(TIM4,&TIM_OCInitStruct);
  TIM_Cmd(TIM4,ENABLE);
  
    //PWM也是通过GPIO来输出的，所以要把对应PWM的GPIO给初始化好
  GPIO_InitTypeDef GPIO_InitStructure;
  GPIO_InitStructure.GPIO_Pin = GPIO_Pin_8;//PA2,对应TIM4_OC3
	GPIO_InitStructure.GPIO_Mode = GPIO_Mode_AF_PP;//推挽复用MODE
	GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
	GPIO_Init(GPIOB, &GPIO_InitStructure);    
}



//呼吸灯效果
void breathLED_B_config(void)
{
//先打开时钟
  RCC_APB1PeriphClockCmd(RCC_APB1Periph_TIM3,ENABLE);
  TIM_InternalClockConfig(TIM3);
  RCC_APB2PeriphClockCmd(RCC_APB2Periph_GPIOB,ENABLE);
  
  //设置基本单元部分
  TIM_TimeBaseInitTypeDef TIM_TimeBaseInitStruct;
  TIM_TimeBaseStructInit(&TIM_TimeBaseInitStruct);//将这个结构体初始化，这样就不用填那些可能用不上的参数了
  TIM_TimeBaseInitStruct.TIM_ClockDivision=TIM_CKD_DIV1;//72MHz
  TIM_TimeBaseInitStruct.TIM_CounterMode=TIM_CounterMode_Up;//向上计数模式
  TIM_TimeBaseInitStruct.TIM_Period=100-1;//10000次计数周期            ARR
  TIM_TimeBaseInitStruct.TIM_Prescaler=720-1;//分频为720-1=0.1MHz,周期就是10^-5 s         PSC
  TIM_TimeBaseInit(TIM3,&TIM_TimeBaseInitStruct);
  //PWM频率为 72M/720/0.01M=10Hz,周期为0.1s也就是100ms
  //接下来初始化输出比较单元（配置好基本单元后就可以开始进行输出比较）
  TIM_OCInitTypeDef TIM_OCInitStruct;
  TIM_OCStructInit(&TIM_OCInitStruct);
  TIM_OCInitStruct.TIM_OCMode=TIM_OCMode_PWM1;//采用PWM1模式
  TIM_OCInitStruct.TIM_OCPolarity=TIM_OCPolarity_Low;//选取有效电平
  TIM_OCInitStruct.TIM_OutputState=TIM_OutputState_Enable;//打开输出开关
  TIM_OCInitStruct.TIM_Pulse=0;//      CCR
  TIM_OC4Init(TIM3,&TIM_OCInitStruct);
  
  //PWM也是通过GPIO来输出的，所以要把对应PWM的GPIO给初始化好
  GPIO_InitTypeDef GPIO_InitStructure;
  GPIO_InitStructure.GPIO_Pin = GPIO_Pin_1;//PB1,对应TIM3_CH4
	GPIO_InitStructure.GPIO_Mode = GPIO_Mode_AF_PP;//推挽复用MODE
	GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
	GPIO_Init(GPIOB, &GPIO_InitStructure);
  
  
  TIM_Cmd(TIM3,ENABLE);
  
  
}

void pwm_set_compare(uint32_t compare)
{
  TIM_SetCompare4(TIM3,compare);
}





void SetRedbox(uint32_t compare)
{
  
  TIM_SetCompare3(TIM2,compare);//compare在500~2500之间
}
void SetGreenbox(uint32_t compare)
{
  
  TIM_SetCompare3(TIM4,compare);//compare在500~2500之间
}





void openbox(uint32_t box)
{
  switch(box)
{
    case GREENBOX:
  {   for(int i=2500;i>500;i-=60)//控制开盖子的速度（慢开）
    {
      delay_ms(20);
      SetGreenbox(i);
    }
    //delay_ms(1000);    
      break;
  }
    case REDBOX:
  {    
      for(int i=2500;i>500;i-=60)//控制开盖子的速度（慢开）
    {
      delay_ms(20);
      SetRedbox(i);
    }
    //delay_ms(1000);
      break;
  }
}
}

void closebox(uint32_t box)
{
  switch(box)
{
  case GREENBOX:
  {
  for(int i=500;i<2500;i+=50)//控制关盖子的速度（慢关）
    {
      SetGreenbox(i);
      delay_ms(20);
    }
    break;
  }
  case REDBOX:
  {
      for(int i=500;i<2500;i+=200)//控制关盖子的速度（快关）
    {
      SetRedbox(i);
      delay_ms(10);
    }
    break;
  }  
}
}


//NMD，必须手动设置一个5v电源
//void VCC_set(void)
//{
//  //就拿PB9当输出引脚吧！
//  RCC_APB2PeriphClockCmd(RCC_APB2Periph_GPIOB,ENABLE);
//  GPIO_InitTypeDef GPIO_InitStructure;
//  GPIO_InitStructure.GPIO_Pin = GPIO_Pin_9;//PB9
//	GPIO_InitStructure.GPIO_Mode = GPIO_Mode_Out_OD;//开漏输出MODE
//	GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
//	GPIO_Init(GPIOB, &GPIO_InitStructure);
//  GPIO_SetBits(GPIOB,GPIO_Pin_9);//试试看呢？
//}
