#pragma once

#ifdef __cplusplus
extern "C"
{
#endif

// MC function
void CalcMC(int StockSize_, float* StockPrice_, float* BasePrice_,
			int ScheduleSize_,	
			int* PayoffT_, int* PayoffT_pay, int* BermudanType_, int* PayoffType_, int* RefPriceType_,
			float* PayoffK_, float* Coupon_, float* Dummy_,
			float* UpBarrier_, float* DownBarrier_, float* TotalUpBarrier_, float* TotalDownBarrier_,
			float* Participation_,
 			int* RateType_, int* RateSize_, float* Ratet_, float* RateFixed_, float* RateCurve_,
			int* DivType_, int* DivSize_, float* Divt_, float* DivFixed_, float* DivCurve_,
 			int* VolType_, int* VolSizet_, int* VolSizeK_, float* Volt_, float* VolK_, float* VolFixed_, float* VolCurve_, float* VolSurf_,
			int YTMType_, int YTMSize_, float* YTMt_, float YTMFixed_, float* YTMCurve_,
			float* correl_, float* Quanto_,
			int SimN_, int SimMode_, int blockN_, int threadN_, 
			struct VBAResult* res);


#ifdef __cplusplus
}
#endif