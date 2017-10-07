#ifndef CONSTANTS_H
#define CONSTANTS_H

/*
 * @author Vasileios Zois
 * @email vzois@usc.edu
 *
 * Timing constants.
 */


#include<ratio>
#include<climits>
#include<chrono>
#include<ctime>

typedef std::pair<uint64_t, uint64_t> arr2D;

typedef std::chrono::duration<double, std::ratio<1, 1000000000>> nanos;
typedef std::chrono::duration<double, std::ratio<1,1000>> millis;
typedef std::chrono::duration<double, std::ratio<1,1> > secs;
typedef std::chrono::duration<double, std::ratio<60, 1> > mins;
typedef std::chrono::duration<double, std::ratio<3600, 1> > hours;
typedef std::chrono::duration<double, std::ratio<60 * 60 * 24, 1> > days;
typedef std::chrono::duration<double, std::ratio<60 * 60 * 24 * 30, 1> > months;

#endif
