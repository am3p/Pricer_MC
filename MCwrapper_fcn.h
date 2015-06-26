#pragma once

#ifdef __cplusplus
extern "C"
{
#endif

// MC function
void CalcMC(long StockSize_, double* StockPrice_, double* BasePrice_,
			long ScheduleSize_,	
			long* PayoffT_, long* PayoffT_pay, long* BermudanType_, long* PayoffType_, long* RefPriceType_,
			double* PayoffK_, double* Coupon_, double* Dummy_,
			double* UpBarrier_, double* DownBarrier_, double TotalUpBarrier_, double TotalDownBarrier_,
			double* Participation_,
			long isUpTouched_, long isDownTouched_,
 			long* RateType_, long* RateSize_, double* Ratet_, double* Rate_,
			long* DivType_, long* DivSize_, double* Divt_, double* Div_,
 			long* VolType_, long* VolSizet_, long* VolSizeK_, double* Volt_, double* VolK_, double* Vol_,
			long YTMType_, long YTMSize_, double* YTMt_, double* YTM_,
			double* correl_, double* Quanto_,
			long isStrikePriceQuote_, long SimN_, long SimMode_, long blockN_, long threadN_,
			struct VBAResult* result);


#ifdef __cplusplus
}
#endif