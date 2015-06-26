#pragma once
#include "VariableSize.h"

// See definitions of VBA type definitions in Excel worksheets

struct VBARate{
	long RateType;
	long RateSize;

    double Ratet[RateTMax];
    double Rater[RateTMax];
};

struct VBADiv{
	long DivType;
	long DivSize;

    double Divt[DivTMax];
    double Divq[DivTMax];
};

struct VBAVol{
	long VolType;
    long VolSizet;
    long VolSizeK;
    
    double Volt[VolTMax];
    double VolK[VolKMax];
    double Volv[VolTMax * VolKMax];
};

struct VBAUnderlying{
    double S;
	double BasePrice;
    struct VBARate Rf;
    struct VBADiv Div;
    struct VBAVol Vol;
	double Quanto;
	double Corr_row[StockSizeMax];
};

struct VBAPayoff{
	long T_exp;
	long T_pay;

    long BermudanType;
    long PayoffType;
	long RefPriceType;
    
	double K;
    double UpBarrier;
    double DownBarrier;
    double TotalUpBarrier;
    double TotalDownBarrier;
    double Participation;

    double Coupon;
	double Dummy;
};

struct VBACalcOption{
	long StockNum;
	long ScheduleNum;
	long SimN;
	long SimMode;
	long blockN;
	long threadN;
	long isStrikePriceQuote;
};

struct VBAResult{
    double price;
    double delta[StockSizeMax];
    double gamma[StockSizeMax];
	double vega[StockSizeMax];		// Vega
	double rho[StockSizeMax];
	double theta;
	double vanna[StockSizeMax];		// Vanna
	double volga[StockSizeMax];		// Volga

	double prob[ScheduleSizeMax];

	double coupon;
};