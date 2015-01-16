
#pragma once

// Underlying structure
struct Underlying{
	float S;
	
	int RateType;	// Rf type for the underlying: 0 - Fixed, 1 - Term
	int RateSize;	// Rf size

	int DivType;	// Div type for the underlying: 0 - Fixed, 1 - Term
	int DivSize;	// Div size

	int VolType;	// Vol type for the underlying: 0 - Fixed, 1 - Term, 2 - Surf
	int VolSizet;	// Vol size along time axis
	int VolSizeK;	// Vol size along price axis
};

// Payoff structure
struct Payoff{
	int T;					// Time to expiry date
	int T_pay;				// Time to payment date

	int BermudanType;		// Bermudan Type: 0 - Final / 1 - Bermudan / 2 - Coupon (in Monthly Redemption Type)

	int PayoffType;			// Payoff Type
							// Vanilla: 0 - Call, 1 - Put (Modified to be compatible in ELS payoffs)
							// Digital: 2 - DigitCall, 3 - DigitPut
							// KO/KI: 4 - KOCall, 5 - KICall, 6 - KOPut, 7 - KIPut
	
	int ObsPriceType;		// Reference price observation option: 0 - Close / 1 - Lowest

	int RefPriceType;		// Reference price setting (btw assets) option: 0 - Minimum / 1 - Average

	float K;				// Strike
	float UpBarrier;		// Up barrier (only in this schedule)
	float DownBarrier;		// Down barrier (only in this schedule)
	float TotalUpBarrier;	// Total up barrier (globally effective)
	float TotalDownBarrier;	// Total down barrier (globally effective)
	float Coupon;			// Coupon amount
	float Dummy;			// Dummy amount, if any
	float Participation;	// Participation rate
};

// YTM structure
struct YTM{
	int YTMType;	// YTM Type	
	int YTMSize;	// YTM Size
};

// Price result
struct Result{
	float price;		// Product price
	float delta[3];		// Delta
	float gamma[3];		// Gamma
	float vega[3];		// Vega
	float rho[3];		// Rho
	float theta;		// Theta
	float vanna[3];		// Vanna
	float volga[3];		// Volga

	int prob;
};