/****************************************************** 
 *	Code by Utkarsh Sinha
 *	Based on JIFT by Jun Liu
 *	Visit http://aishack.in/ for more indepth articles and tutorials
 *	on artificial intelligence
 * Use, reuse, modify, hack, kick. Do whatever you want with
 * this code :)
 ******************************************************/

#ifndef _KEYPOINT_H
#define _KEYPOINT_H

#include "stdafx.h"
#include <vector>

using namespace std;

class Keypoint
{
public:
	float			xi;
	float			yi;	// It's location
	vector<double>	mag;	// The list of magnitudes at this point
	vector<double>	orien;	// The list of orientations detected
	unsigned int	scale;	// The scale where this was detected

	Keypoint() { }
	Keypoint(float x, float y) { xi=x; yi=y; }
	Keypoint(float x, float y, vector<double> const& m, vector<double> const& o, unsigned int s)
	{
		xi = x;
		yi = y;
		mag = m;
		orien = o;
		scale = s;
	}
};

#endif