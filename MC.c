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
//	   (Interpolation/Extrapolation of parameters can be done linearly only, in this
//		version)
//
///////////////////////////////////////////////////////////////////////////////////////

#include <stdio.h> 
#include <stdlib.h> 
#include <math.h>

#include "MCwrapper_fcn.h"
#include "MCstruct_VBA.h"

__declspec(dllexport) struct VBAResult __stdcall Pricer_MC(struct VBAData* Data){

	int i, j, k; float s;

	// Stock size: Max 3 (just my purpose)
	int StockSize_ = Data->StockSize;
	// Stock: Current stock prices
	float StockPrice_[3] = {0};
	// Stock: Base prices for the product
	float BasePrice_[3] = {0};
	
	// Rate info: type and size (type: 0 - Fixed / 1 - Term)
	int RateType_[3] = {0}, RateSize_[3] = {0};
	// Rate info: fixed case and term structure
	// Time axis size: Max 20 (just my purpose)
	float Ratet_[60] = {0}, RateFixed_[3] = {0}, RateCurve_[60] = {0};

	// Div info: type and size (type: 0 - Fixed / 1 - Term)
	int DivType_[3] = {0}, DivSize_[3] = {0};
	// Div info: fixed case and term structure
	// Time axis size: Max 20 (just my purpose)
	float Divt_[60] = {0}, DivFixed_[3] = {0}, DivCurve_[60] = {0};

	// Vol info: type and size (type: 0 - Fixed / 1 - Term / 2 - Surface)
	int VolType_[3] = {0}, VolSize_t_[3] = {0}, VolSize_K_[3] = {0};
	// Vol info: fixed case, term structure, and surface
	// Time axis size: Max 20 (just my purpose)
	// Price axis size for vol surface: Max 13 (assumed to be fixed, just my purpose)
	float Volt_[60] = {0}, VolK_[60] = {0}, VolFixed_[3] = {0}, VolCurve_[60] = {0}, VolSurf_[1200] = {0};
	
	// Correlation: raw matrix and Cholesky decomposed (LD)
	// Correlation size: Max 3x3 (just my purpose)
	float correl_raw[9] = {0}, correl_[9] = {0};

	// Quanto adjustment
	float Quanto_[3] = {0};
	
	// Schedule size: Max 60 (just my purpose)
	int ScheduleSize_ = Data->ScheduleSize;
	// Schedule info: TTM
	int PayoffT_[60] = {0}, PayoffT_pay[60] = {0};
	// Schedule info: Types (Bermudan / Payoff type / Reference price type)
	int BermudanType_[60] = {0}, PayoffType_[60] = {0}, RefPriceType_[60] = {0};
	// Schedule info: Exercise and barrier
	float PayoffK_[60] = {0}, UpBarrier_[60] = {0}, DownBarrier_[60] = {0}, TotalUpBarrier_[60] = {0}, TotalDownBarrier_[60] = {0};
	// Schedule info: Payout coupon, dummy
	float Coupon_[60] = {0}, Dummy_[60] = {0};
	// Schedule info: Participation rate
	float Participation_[60] = {0};

	// YTM info: type and size
	int YTMType_, YTMSize_;
	// YTM rate: fixed case and term structure
	// Time axis size: Max 20 (just my purpose)
	float YTMt_[20] = {0}, YTMFixed_, YTMCurve_[20] = {0};

	int isStrikePriceQuote_ = Data->isStrikePriceQuote;
	int SimN_ = Data->SimN;
	int SimMode_ = Data->SimMode;
	int blockN_ = Data->blockN;
	int threadN_ = Data->threadN;

	// Result format
	struct VBAResult* result = (struct VBAResult *) malloc(sizeof(struct VBAResult));
	struct VBAResult result_VBA;
	result->price = 0;
	result->theta = 0;
	for (i = 0; i < StockSize_; i++){
		result->delta[i] = 0;
		result->gamma[i] = 0;
		result->vega[i] = 0;
		result->rho[i] = 0;
		result->vanna[i] = 0;
		result->volga[i] = 0;
	}
	for (i = 0; i < 60; i++){
		result->prob[i] = 0;
	}

	// Copying product info for CUDA function
	for (i = 0; i < StockSize_; i++){
		// Current stock price (normalized by base price)
		StockPrice_[i] = Data->Stock[i].S;
		BasePrice_[i] = Data->BasePrice[i];
		
		// Rate info
		RateType_[i] = Data->Stock[i].Rf.RateType; RateSize_[i] = Data->Stock[i].Rf.RateSize;
		switch(RateType_[i]){
			case 0:
				{
					RateFixed_[i] = Data->Stock[i].Rf.Rater[0];
					break;
				}
			case 1:
				{
					for (j = 0; j < RateSize_[i]; j++){
						Ratet_[20*i+j] = Data->Stock[i].Rf.Ratet[j];
						RateCurve_[20*i+j] = Data->Stock[i].Rf.Rater[j];
					}
					break;
				}
			default:
				break;
		}

		// Dividend info
		DivType_[i] = Data->Stock[i].Div.DivType; DivSize_[i] = Data->Stock[i].Div.DivSize;
		switch(DivType_[i]){
			case 0:
				{
					DivFixed_[i] = Data->Stock[i].Div.Divq[0];
					break;
				}
			case 1:
				{
					for (j = 0; j < DivSize_[i]; j++){
						Divt_[20*i+j] = Data->Stock[i].Div.Divt[j];
						DivCurve_[20*i+j] = Data->Stock[i].Div.Divq[j];
					}
					break;
				}
			default:
				break;
		}

		// Vol info
		VolType_[i] = Data->Stock[i].Vol.VolType; VolSize_t_[i] = Data->Stock[i].Vol.VolSizet; VolSize_K_[i] = Data->Stock[i].Vol.VolSizeK;
		switch(VolType_[i]){
			case 0:
				{
					VolFixed_[i] = Data->Stock[i].Vol.Volv[0];
					break;
				}
			case 1:
				{
					for (j = 0; j < VolSize_t_[i]; j++){
						Volt_[20*i+j] = Data->Stock[i].Vol.Volt[j];
						VolCurve_[20*i+j] = Data->Stock[i].Vol.Volv[20*i+j];
					}
					break;
				}
			case 2:
				{
					for (j = 0; j < VolSize_t_[i]; j++){
						Volt_[20*i+j] = Data->Stock[i].Vol.Volt[j];
						for (k = 0; k < VolSize_K_[i]; k++){
							VolK_[20*i+k] = Data->Stock[i].Vol.VolK[k];
							VolSurf_[400*i+20*j+k] = Data->Stock[i].Vol.Volv[VolSize_K_[i]*j+k];
						}
					}
					break;
				}
			default:
				break;
		}
	}

	// Correlation: copying raw matrix
	for (i = 0; i < StockSize_; i++){
		for (j = 0; j < StockSize_; j++){
			correl_raw[i*StockSize_+j] = Data->Correl[i*StockSize_+j];
		}
	}

	// Correlation: Cholesky decomposition (LD)
	for (i = 0; i < StockSize_; i++){
		for (j = 0; j < (i+1); j++) {
			s = 0;
			for (k = 0; k < j; k++) 
				s += correl_[i*StockSize_+k] * correl_[j*StockSize_+k];
			correl_[i*StockSize_+j] = (i == j) ? (float)sqrt(correl_raw[i*StockSize_+i]-s) : (1.0f/correl_[j*StockSize_+j] * (correl_raw[i*StockSize_+j]-s));
		}
	}	

	// Correlation: copying raw matrix
	for (i = 0; i < StockSize_; i++){
		Quanto_[i] = Data->Quanto[i];
	}

	// Schedule info: copying relevant information
	for (i = 0; i < ScheduleSize_; i++){
		PayoffT_[i] = Data->Schedule[i].T;
		PayoffT_pay[i] = Data->Schedule[i].T_pay;

		BermudanType_[i] = Data->Schedule[i].BermudanType;
		PayoffType_[i] = Data->Schedule[i].PayoffType;
		RefPriceType_[i] = Data->Schedule[i].RefPriceType;

		PayoffK_[i] = Data->Schedule[i].K;
		UpBarrier_[i] = Data->Schedule[i].UpBarrier;
		DownBarrier_[i] = Data->Schedule[i].DownBarrier;
		TotalUpBarrier_[i] = Data->Schedule[i].TotalUpBarrier;
		TotalDownBarrier_[i] = Data->Schedule[i].TotalDownBarrier;

		Participation_[i] = Data->Schedule[i].Participation;
		Coupon_[i] = Data->Schedule[i].Coupon;
		Dummy_[i] = Data->Schedule[i].Dummy;
	}

	// YTM info
	YTMType_ = Data->YTM.YTMType;
	YTMSize_ = Data->YTM.YTMSize;
	switch(YTMType_){
		case 0:
			{
				YTMFixed_ = Data->YTM.YTMr[0];
				break;
			}
		case 1:
			{
				for (j = 0; j < YTMSize_; j++){
					YTMt_[j] = Data->YTM.YTMt[j];
					YTMCurve_[j] = Data->YTM.YTMr[j];
				}
				break;
			}
		default:
			break;
	}

	// MC function
	CalcMC(StockSize_, StockPrice_, BasePrice_,
		   ScheduleSize_,	
		   PayoffT_, PayoffT_pay, BermudanType_, PayoffType_, RefPriceType_,
		   PayoffK_, Coupon_, Dummy_,
		   UpBarrier_, DownBarrier_, TotalUpBarrier_, TotalDownBarrier_,
		   Participation_,
		   RateType_, RateSize_, Ratet_, RateFixed_, RateCurve_,
		   DivType_, DivSize_, Divt_, DivFixed_, DivCurve_,
		   VolType_, VolSize_t_, VolSize_K_, Volt_, VolK_, VolFixed_, VolCurve_, VolSurf_,
		   YTMType_, YTMSize_, YTMt_, YTMFixed_, YTMCurve_,
		   correl_, Quanto_,
		   isStrikePriceQuote_, SimN_, SimMode_, blockN_, threadN_,
		   result);

	// Arrange result
	result_VBA.price = result->price;
	for (i = 0; i < 60; i++){
		result_VBA.prob[i] = result->prob[i];
	}
	if (SimMode_ > 0){
		for (i = 0; i < StockSize_; i++)
			result_VBA.delta[i] = result->delta[i];
		for (i = 0; i < StockSize_; i++)
			result_VBA.gamma[i] = result->gamma[i];
		for (i = 0; i < StockSize_; i++)
			result_VBA.vega[i] = result->vega[i];
	}
	if (SimMode_ > 1){
		for (i = 0; i < StockSize_; i++)
			result_VBA.rho[i] = result->rho[i];
		result_VBA.theta = result->theta;
	}
	if (SimMode_ > 2){
		for (i = 0; i < StockSize_; i++)
			result_VBA.vanna[i] = result->vanna[i];
		for (i = 0; i < StockSize_; i++)
			result_VBA.volga[i] = result->volga[i];
	}

	free(result);
	return result_VBA;
}



