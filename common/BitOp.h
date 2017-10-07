#ifndef BITOP_H
#define BITOP_H

#define HEX0(V) ((0x0000000f & V)>>0)
#define HEX1(V) ((0x000000f0 & V)>>4)
#define HEX2(V) ((0x00000f00 & V)>>8)
#define HEX3(V) ((0x0000f000 & V)>>12)
#define HEX4(V) ((0x000f0000 & V)>>16)
#define HEX5(V) ((0x00f00000 & V)>>20)
#define HEX6(V) ((0x0f000000 & V)>>24)
#define HEX7(V) ((0xf0000000 & V)>>28)


#define SET(N,BIT) (N|=(1 << BIT))
#define GET(N,BIT) ((N >> BIT) & 1)

#define ASGN(N,BIT,F) (N ^= (-F ^ N) & (1 << BIT))
#define ASGNC(N,BIT,F) (N|=(F << BIT))

#endif
