/****************************************************** 
 *	Code by Utkarsh Sinha
 *	Based on JIFT by Jun Liu
 *	Visit http://aishack.in/ for more indepth articles and tutorials
 *	on artificial intelligence
 * Use, reuse, modify, hack, kick. Do whatever you want with
 * this code :)
 ******************************************************/

#ifndef DESCRIPTOR_H
#define DESCRIPTOR_H

using namespace std;

class Descriptor
{
public:
	float			xi, yi;		// The location
	vector<double>	fv;			// The feature vector

	Descriptor()
	{
	}

	Descriptor(float x, float y, vector<double> const& f)
	{
		xi = x;
		yi = y;
		fv = f;
	}
};

#endif