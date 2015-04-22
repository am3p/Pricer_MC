#pragma once
#include "VariableSize.h"

// Underlying structure
struct Underlying{
	double S;
	
	long RateType;	// Rf type for the underlying: 0 - Fixed, 1 - Term
	long RateSize;	// Rf size

	long DivType;	// Div type for the underlying: 0 - Fixed, 1 - Term
	long DivSize;	// Div size

	long VolType;	// Vol type for the underlying: 0 - Fixed, 1 - Term, 2 - Surf
	long VolSizet;	// Vol size along time axis
	long VolSizeK;	// Vol size along price axis
};

// Payoff structure
struct Payoff{
	long T;					// Time to expiry date
	long T_pay;				// Time to payment date

	long BermudanType;		// Bermudan Type: 0 - Final / 1 - Bermudan / 2 - Coupon (in Monthly Redemption Type)

	long PayoffType;			// Payoff Type
							// Vanilla: 0 - Call, 1 - Put (Modified to be compatible in ELS payoffs)
							// Digital: 2 - DigitCall, 3 - DigitPut
							// KO/KI: 4 - KOCall, 5 - KICall, 6 - KOPut, 7 - KIPut
	
	long ObsPriceType;		// Reference price observation option: 0 - Close / 1 - Lowest

	long RefPriceType;		// Reference price setting (btw assets) option: 0 - Minimum / 1 - Average

	double K;				// Strike
	double UpBarrier;		// Up barrier (only in this schedule)
	double DownBarrier;		// Down barrier (only in this schedule)
	double TotalUpBarrier;	// Total up barrier (globally effective)
	double TotalDownBarrier;	// Total down barrier (globally effective)
	double Coupon;			// Coupon amount
	double Dummy;			// Dummy amount, if any
	double Participation;	// Participation rate
};

// YTM structure
struct YTM{
	long YTMType;	// YTM Type	
	long YTMSize;	// YTM Size
};

// Price result
struct Result{
	double price;		// Product price
	double delta[StockSizeMax];		// Delta
	double gamma[StockSizeMax];		// Gamma
	double vega[StockSizeMax];		// Vega
	double rho[StockSizeMax];		// Rho
	double theta;		// Theta
	double vanna[StockSizeMax];		// Vanna
	double volga[StockSizeMax];		// Volga

	long prob;
};