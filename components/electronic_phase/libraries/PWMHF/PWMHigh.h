/*
PWMHigh.h - Library for flashing PWMHi code.
*/

#ifndef PWMHigh_h
#define PWMHigh_h

#ifdef __cplusplus
extern "C" 
{
#endif

extern void analogWritef(uint32_t ulPin, uint32_t ulValue , uint32_t freqPWM) ;

extern void analogWriteResol(int res);

extern void WriteCLKPWM(int clko) ;

extern void analogWriteDutyMax(int dt) ;

extern void WriteResolPWM(int res) ;

#ifdef __cplusplus
}
#endif

#endif /* PWMHigh_h */
