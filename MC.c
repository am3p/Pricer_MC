///////////////////////////////////////////////////////////////////////////////////////
//
// MC Pricer ver 0.1
// 2014. 11. 28 by Boram Hwang
//
// Main features
//  1. Conducting MC pricing for user-defined payoff
//     (This version only contains knock-out call, and composites of step-down ELS,
//		but payoffs are quite easily extendable)
//  2. MC is done by GPU-wise sample construction
//     (Each sample is constructed per a GPU so that by upgrading VGA with more SM
//		will reduce calculation time)
//  3. Adapting constant, term-structured, or surface parameters
//	   (longerpolation/Extrapolation of parameters can be done linearly only, in this
//		version)
//
///////////////////////////////////////////////////////////////////////////////////////

#include <stdio.h> 
#include <stdlib.h> 
#include <math.h>

#include "MCwrapper_fcn.h"
#include "MCstruct_VBA.h"
#include "VariableSize.h"

__declspec(dllexport) struct VBAResult __stdcall Pricer_MC(long NStock, double* StockPrice, double* BasePrice,
														   long* RateType, long* RateTNum, double* RateT, double* Rate,
														   long* DivType, long* DivTNum, double* DivT, double* Div,
														   long* VolType, long* VolTNum, long* VolKNum, double* VolT, double* VolK, double* Vol,
														   long* FXVolType, long* FXVolTNum, double* FXVolT, double* FXVol,
														   double* StockCorr,  double* FXCorr,
														   long NSchedule, long* T_exp, long* T_pay, long* BermudanType, long* PayoffType, long* RefPriceType,
														   double* Strike, double* UpBarrier, double* DownBarrier, double* Coupon, double* Dummy, double* Participation,
														   double TotalUpBarrier, double TotalDownBarrier,
														   long Mode, long SimN, long blockN, long threadN, long isStrikePriceQuote){

	long i, j, k; double s;

	// Stock: Current stock prices
	double StockPrice_[StockSizeMax] = {0};
	// Stock: Base prices for the product
	double BasePrice_[StockSizeMax] = {0};

	
	// YTM rate: fixed case and term structure
	// Time axis size: Max 20 (just my purpose)
	long YTMType_ = 0, YTMSize_ = 0;
	double YTMt_[RateTMax] = {0}, YTM_[RateTMax] = {0};
	long YTMInd_src = 0, YTMInd_dest = 0;
	
	// Rate info: type and size (type: 0 - Fixed / 1 - Term)
	long RateType_[StockSizeMax] = {0}, RateSize_[StockSizeMax] = {0};
	// Rate info: fixed case and term structure
	// Time axis size: Max 20 (just my purpose)
	double Ratet_[StockSizeMax * RateTMax] = {0}, Rate_[StockSizeMax * RateTMax] = {0};
	long RateInd_src = 0, RateInd_dest = 0;

	// Div info: type and size (type: 0 - Fixed / 1 - Term)
	long DivType_[StockSizeMax] = {0}, DivSize_[StockSizeMax] = {0};
	// Div info: fixed case and term structure
	// Time axis size: Max 20 (just my purpose)
	double Divt_[StockSizeMax * DivTMax] = {0}, Div_[StockSizeMax * DivTMax] = {0};
	long DivInd_src = 0, DivInd_dest = 0;

	// Vol info: type and size (type: 0 - Fixed / 1 - Term / 2 - Surface)
	long VolType_[StockSizeMax] = {0}, VolSize_t_[StockSizeMax] = {0}, VolSize_K_[StockSizeMax] = {0};
	// Vol info: fixed case, term structure, and surface
	// Time axis size: Max 40 (Covering NICE Full data)
	// Price axis size for vol surface: Max 21 (Covering NICE Full data)
	double Volt_[StockSizeMax * VolTMax] = {0}, VolK_[StockSizeMax * VolKMax] = {0}, Vol_[StockSizeMax * VolTMax * VolKMax] = {0};
	long VolTInd_src = 0, VolTInd_dest = 0, VolKInd_src = 0, VolKInd_dest = 0, VolInd_src = 0, VolInd_dest = 0;

	// Correlation: Cholesky decomposed (LD), Max 4x4 (just my purpose)
	double StockCorr_LD[StockSizeMax * StockSizeMax] = {0};

	// Quanto adjustment
	double QuantoAdj[StockSizeMax] = {0};
	long QuantoInd_src = 0, QuantoInd_dest = 0;

	// Schedule info: TTM
	long PayoffT_[ScheduleSizeMax] = {0}, PayoffT_pay[ScheduleSizeMax] = {0};
	// Schedule info: Types (Bermudan / Payoff type / Reference price type)
	long BermudanType_[ScheduleSizeMax] = {0}, PayoffType_[ScheduleSizeMax] = {0}, RefPriceType_[ScheduleSizeMax] = {0};
	// Schedule info: Exercise and barrier
	double PayoffK_[ScheduleSizeMax] = {0}, UpBarrier_[ScheduleSizeMax] = {0}, DownBarrier_[ScheduleSizeMax] = {0};
	// Schedule info: Payout coupon, dummy
	double Coupon_[ScheduleSizeMax] = {0}, Dummy_[ScheduleSizeMax] = {0};
	// Schedule info: Participation rate
	double Participation_[ScheduleSizeMax] = {0};

	// Result format
	struct VBAResult* result = (struct VBAResult *) malloc(sizeof(struct VBAResult));
	struct VBAResult result_VBA;
	result->price = 0;
	result->theta = 0;
	for (i = 0; i < NStock; i++){
		result->delta[i] = 0;
		result->gamma[i] = 0;
		result->vega[i] = 0;
		result->rho[i] = 0;
		result->vanna[i] = 0;
		result->volga[i] = 0;
	}
	for (i = 0; i < 100; i++){
		result->prob[i] = 0;
	}

	// Copying product info for CUDA function
	for (i = 0; i < NStock; i++){
		// Current stock price (normalized by base price)
		StockPrice_[i] = StockPrice[i];
		BasePrice_[i] = BasePrice[i];
	}

	// YTM info
	YTMType_ = RateType[0];
	YTMSize_ = RateTNum[0];
	switch(YTMType_){
		case 0:
			{
				YTM_[YTMInd_dest] = Rate[YTMInd_src];
				YTMInd_src++;
				YTMInd_dest = RateTMax;
				break;
			}
		case 1:
			{
				for (i = 0; i < YTMSize_; i++){
					YTMt_[YTMInd_dest] = RateT[YTMInd_src];
					YTM_[YTMInd_dest] = Rate[YTMInd_src];
					YTMInd_src++;
					YTMInd_dest++;
				}
				YTMInd_dest = RateTMax;
				break;
			}
		default:
			break;
	}

	// Risk free rate info
	RateInd_src = YTMInd_src;
	for (i = 0; i < NStock; i++){
		RateType_[i] = RateType[i+1];
		RateSize_[i] = RateTNum[i+1];
		switch(RateType_[i]){
			case 0:
				{
					Rate_[RateInd_dest] = Rate[RateInd_src];
					RateInd_src++;
					RateInd_dest = (i+1)*RateTMax;
					break;
				}
			case 1:
				{
					for (j = 0; j < RateSize_[i]; j++){
						Ratet_[RateInd_dest] = RateT[RateInd_src];
						Rate_[RateInd_dest] = Rate[RateInd_src];
						RateInd_src++;
						RateInd_dest++;
					}
					RateInd_dest = (i+1)*RateTMax;
					break;
				}
			default:
				break;
		}
	}

	// Dividend info
	for (i = 0; i < NStock; i++){
		DivType_[i] = DivType[i];
		DivSize_[i] = DivTNum[i];
		switch(DivType_[i]){
			case 0:
				{
					Div_[DivInd_dest] = Div[DivInd_src];
					DivInd_src++;
					DivInd_dest = (i+1)*DivTMax;
					break;
				}
			case 1:
				{
					for (j = 0; j < DivSize_[i]; j++){
						Divt_[DivInd_dest] = DivT[DivInd_src];
						Div_[DivInd_dest] = Div[DivInd_src];
						DivInd_src++;
						DivInd_dest++;
					}
					DivInd_dest = (i+1)*DivTMax;
					break;
				}
			default:
				break;
		}
	}

	// Vol info
	for (i = 0; i < NStock; i++){
		VolType_[i] = VolType[i];
		VolSize_t_[i] = VolTNum[i];
		VolSize_K_[i] = VolKNum[i];
		switch(VolType_[i]){
			case 0:
				{
					Vol_[VolInd_dest] = Vol[VolInd_src];
					VolInd_src++;
					VolInd_dest = (i+1)*VolTMax*VolKMax;
					break;
				}
			case 1:
				{
					for (j = 0; j < VolTNum[i]; j++){
						Volt_[VolTInd_dest] = VolT[VolTInd_src];
						Vol_[VolInd_dest] = Vol[VolInd_src];
						VolInd_src++; VolInd_dest++;
						VolTInd_src++; VolTInd_dest++;
					}
					VolTInd_dest = (i+1)*VolTMax;
					VolInd_dest = (i+1)*VolTMax*VolKMax;
					break;
				}
			case 2:
				{
					for (j = 0; j < VolTNum[i]; j++){
						Volt_[VolTInd_dest] = VolT[VolTInd_src];
						VolTInd_src++; VolTInd_dest++;
					}
					VolTInd_dest = (i+1)*VolTMax;

					for (j = 0; j < VolKNum[i]; j++){
						VolK_[VolKInd_dest] = VolK[VolKInd_src];
						VolKInd_src++; VolKInd_dest++;
					}
					VolKInd_dest = (i+1)*VolKMax;

					for (j = 0; j < VolTNum[i]; j++){
						for (k = 0; k < VolKNum[i]; k++){
							Vol_[VolInd_dest] = Vol[VolInd_src];
							VolInd_src++; VolInd_dest++;
						}
						VolInd_dest = i*VolTMax*VolKMax + (j+1)*VolKMax;
					}
					VolInd_dest = (i+1)*VolTMax*VolKMax;
					break;
				}
			default:
				break;
		}
	}

	// Correlation: Cholesky decomposition (LD)
	for (i = 0; i < NStock; i++){
		for (j = 0; j < (i+1); j++) {
			s = 0;
			for (k = 0; k < j; k++) 
				s += StockCorr[i*NStock+k] * StockCorr[j*NStock+k];
			StockCorr_LD[i*NStock+j] = (i == j) ? sqrt(StockCorr[i*NStock+i]-s) : (1.0/StockCorr_LD[j*NStock+j] * (StockCorr[i*NStock+j]-s));
		}
	}	

	// Quanto adjustment factor
	QuantoInd_src = 0; QuantoInd_dest = 0;
	for (i = 0; i < NStock; i++){
		if (FXVolType[i] == 0){
			QuantoAdj[QuantoInd_dest] = FXCorr[QuantoInd_src] * FXVol[QuantoInd_src];
			QuantoInd_src++;
			QuantoInd_dest++;
		}
		else if (FXVolType[i] == 1){
			// not yet defined
		}
	}

	// Schedule info: copying relevant information
	for (i = 0; i < NSchedule; i++){
		PayoffT_[i] = T_exp[i];
		PayoffT_pay[i] = T_pay[i];

		BermudanType_[i] = BermudanType[i];
		PayoffType_[i] = PayoffType[i];
		RefPriceType_[i] = RefPriceType[i];

		PayoffK_[i] = Strike[i];
		UpBarrier_[i] = UpBarrier[i];
		DownBarrier_[i] = DownBarrier[i];

		Participation_[i] = Participation[i];
		Coupon_[i] = Coupon[i];
		Dummy_[i] = Dummy[i];
	}

	// MC function
	CalcMC(NStock, StockPrice_, BasePrice_,
		   NSchedule,	
		   PayoffT_, PayoffT_pay, BermudanType_, PayoffType_, RefPriceType_,
		   PayoffK_, Coupon_, Dummy_,
		   UpBarrier_, DownBarrier_, TotalUpBarrier, TotalDownBarrier,
		   Participation_,
		   RateType_, RateSize_, Ratet_, Rate_,
		   DivType_, DivSize_, Divt_, Div_,
		   VolType_, VolSize_t_, VolSize_K_, Volt_, VolK_, Vol_,
		   YTMType_, YTMSize_, YTMt_, YTM_,
		   StockCorr_LD, QuantoAdj,
		   isStrikePriceQuote, SimN, Mode, blockN, threadN,
		   result);

	// Arrange result
	result_VBA.price = result->price;
	for (i = 0; i < ScheduleSizeMax; i++){
		result_VBA.prob[i] = result->prob[i];
	}
	if (Mode > 0){
		for (i = 0; i < NStock; i++)
			result_VBA.delta[i] = result->delta[i];
		for (i = 0; i < NStock; i++)
			result_VBA.gamma[i] = result->gamma[i];
		for (i = 0; i < NStock; i++)
			result_VBA.vega[i] = result->vega[i];
	}
	if (Mode > 1){
		for (i = 0; i < NStock; i++)
			result_VBA.rho[i] = result->rho[i];
		result_VBA.theta = result->theta;
	}
	if (Mode > 2){
		for (i = 0; i < NStock; i++)
			result_VBA.vanna[i] = result->vanna[i];
		for (i = 0; i < NStock; i++)
			result_VBA.volga[i] = result->volga[i];
	}

	free(result);
	return result_VBA;
}



