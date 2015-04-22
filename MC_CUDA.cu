//
#include <cuda.h>
#include <curand_kernel.h>

#include "MCstruct.h"
#include "MCstruct_VBA.h"
#include "MCwrapper_fcn.h"
#include "VariableSize.h"

// Global variables for MC
// Underlying: Max 3
__constant__ Underlying Stock[StockSizeMax];
__constant__ double BasePrice[StockSizeMax];

// Schedule: Max 60
__constant__ Payoff Schedule[ScheduleSizeMax];

// YTMinfo: t size 20
__constant__ double YTMt[RateTMax];
__constant__ double YTM[RateTMax];

// Rate: t size 20 per each asset
__constant__ double Ratet[StockSizeMax * RateTMax];
__constant__ double Rate[StockSizeMax * RateTMax];

// Div: t size 20 per each asset
__constant__ double Divt[StockSizeMax * DivTMax];
__constant__ double Div[StockSizeMax * DivTMax];

// Vol: t size 20, K size 13 per each asset
__constant__ double Volt[StockSizeMax * VolTMax];
__constant__ double VolK[StockSizeMax * VolKMax];
__constant__ double Vol[StockSizeMax * VolTMax * VolKMax];

// Correlation
__constant__ double correl[StockSizeMax * StockSizeMax];

// Quanto
__constant__ double Quanto[StockSizeMax];

// Global Functions: functions are called in CalcMC
__global__ void InitSeed(curandState *state, const long threadN);
__global__ void MC(curandState *state, 
				   const long StockSize, const long ScheduleSize, 
				   const long YTMType, const long YTMSize, 
				   const long SimMode, const long isStrikePriceQuote, const long threadN, 
				   Result *result);

// Device functions: functions are called in global functions
__device__ double YTMInterp(double t, long YTMType, long YTMSize);	// YTM rate longerp/extrapolation
__device__ double RfInterp(double t, long stocknum);					// Rf spot rate longerp/extrapolation
__device__ double DivInterp(double t, long stocknum);				// Dividend longerp/extrapolation
__device__ double VolInterp(double t, double K, long stocknum);		// Volatility longerp/extrapolationlation

__device__ double SMin(double S_min[][StockSizeMax], long StockSize, long casenum);
__device__ double SMax(double S_max[][StockSizeMax], long StockSize, long casenum);

__device__ double RefPriceCalc(double S, long StockSize, long sched_ind, long casenum);
__device__ bool PayoffCheck(double S[][StockSizeMax], double* S_min, double* S_max, long StockSize, long sched_ind, long casenum);
__device__ double PayoffCalc(double S[][StockSizeMax], double* S_min, double* S_max, long StockSize, long sched_ind, long casenum);

// Main function
void CalcMC(long StockSize_, double* StockPrice_, double* BasePrice_,
			long ScheduleSize_,	
			long* PayoffT_, long* PayoffT_pay, long* BermudanType_, long* PayoffType_, long* RefPriceType_,
			double* PayoffK_, double* Coupon_, double* Dummy_,
			double* UpBarrier_, double* DownBarrier_, double TotalUpBarrier_, double TotalDownBarrier_,
			double* Participation_,
 			long* RateType_, long* RateSize_, double* Ratet_, double* Rate_,
			long* DivType_, long* DivSize_, double* Divt_, double* Div_,
 			long* VolType_, long* VolSizet_, long* VolSizeK_, double* Volt_, double* VolK_, double* Vol_,
			long YTMType_, long YTMSize_, double* YTMt_, double* YTM_,
			double* correl_, double* Quanto_,
			long isStrikePriceQuote_, long SimN_, long SimMode_, long blockN_, long threadN_,
			struct VBAResult* result){

	// GPU parallelization: block/thread for CUDA cores
	long blockN = blockN_;
	long threadN = threadN_;

	// Pseudorandom number state: most simple one provided in CUDA
	curandState *devStates;
	
	// Result vector
	Result *devResults, *hostResults; 
	
	// Allocate space for host vector (4 * 160 is developer's pleases)
	hostResults = (Result *)calloc(blockN * threadN, sizeof(Result)); 
	
	// Allocate space for device vector
	cudaMalloc((void **)&devResults, blockN * threadN * sizeof(Result));  
	cudaMemset(devResults, 0, blockN * threadN * sizeof(Result));	

	// Allocate space for pseudorng states (device) 
	cudaMalloc((void **)&devStates, blockN * threadN * sizeof(curandState));

	// Seed initialization (fixed seed: set to each thread id)
	InitSeed<<<blockN, threadN>>>(devStates, threadN);

	// Copying product info to global variables
	// Global variable: Stock
	Underlying stock_[StockSizeMax];
	for (long i = 0; i < StockSize_; i++)
	{
		stock_[i].S = StockPrice_[i];
		
		stock_[i].RateType = RateType_[i];
		stock_[i].RateSize = RateSize_[i];

		stock_[i].DivType = DivType_[i];
		stock_[i].DivSize = DivSize_[i];

		stock_[i].VolType = VolType_[i];
		stock_[i].VolSizet = VolSizet_[i];
		stock_[i].VolSizeK = VolSizeK_[i];
	}
	Underlying* stock_ptr;
	cudaGetSymbolAddress((void**) &stock_ptr, Stock);
	cudaMemcpy(stock_ptr, stock_, StockSizeMax * sizeof(Underlying), cudaMemcpyHostToDevice);

	// Global variable: YTM
	double* YTMt_ptr;
	cudaGetSymbolAddress((void**) &YTMt_ptr, YTMt);
	cudaMemcpy(YTMt_ptr, YTMt_, RateTMax * sizeof(double), cudaMemcpyHostToDevice);
	double* YTM_ptr;
	cudaGetSymbolAddress((void**) &YTM_ptr, YTM);
	cudaMemcpy(YTM_ptr, YTM_, RateTMax * sizeof(double), cudaMemcpyHostToDevice);

	// Global variable: Schedule
	Payoff schedule_[ScheduleSizeMax];
	for (long i = 0; i < ScheduleSize_; i++)
	{
		schedule_[i].T = PayoffT_[i];
		schedule_[i].T_pay = PayoffT_pay[i];
		schedule_[i].BermudanType = BermudanType_[i];
		schedule_[i].PayoffType = PayoffType_[i];
		schedule_[i].RefPriceType = RefPriceType_[i];

		schedule_[i].K = PayoffK_[i];
		schedule_[i].UpBarrier = UpBarrier_[i];
		schedule_[i].DownBarrier = DownBarrier_[i];
		schedule_[i].TotalUpBarrier = TotalUpBarrier_;
		schedule_[i].TotalDownBarrier = TotalDownBarrier_;

		schedule_[i].Coupon = Coupon_[i];
		schedule_[i].Dummy = Dummy_[i];

		schedule_[i].Participation = Participation_[i];
	}
	Payoff* sched_ptr;
	cudaGetSymbolAddress((void**) &sched_ptr, Schedule);
	cudaMemcpy(sched_ptr, schedule_, ScheduleSizeMax * sizeof(Payoff), cudaMemcpyHostToDevice);

	double* BasePrice_ptr;
	cudaGetSymbolAddress((void**) &BasePrice_ptr, BasePrice);
	cudaMemcpy(BasePrice_ptr, BasePrice_, StockSizeMax * sizeof(double), cudaMemcpyHostToDevice);

	// Global variable: Rate
	double* Ratet_ptr;
	cudaGetSymbolAddress((void**) &Ratet_ptr, Ratet);
	cudaMemcpy(Ratet_ptr, Ratet_, StockSizeMax * RateTMax * sizeof(double), cudaMemcpyHostToDevice);
	double* Rate_ptr;
	cudaGetSymbolAddress((void**) &Rate_ptr, Rate);
	cudaMemcpy(Rate_ptr, Rate_, StockSizeMax * RateTMax * sizeof(double), cudaMemcpyHostToDevice);

	// Global variable: Dividend
	double* Divt_ptr;
	cudaGetSymbolAddress((void**) &Divt_ptr, Divt);
	cudaMemcpy(Divt_ptr, Divt_, StockSizeMax * DivTMax * sizeof(double), cudaMemcpyHostToDevice);
	double* Div_ptr;
	cudaGetSymbolAddress((void**) &Div_ptr, Div);
	cudaMemcpy(Div_ptr, Div_, StockSizeMax * DivTMax * sizeof(double), cudaMemcpyHostToDevice);

	// Global variable: Volatility
	double* Volt_ptr;
	cudaGetSymbolAddress((void**) &Volt_ptr, Volt);
	cudaMemcpy(Volt_ptr, Volt_, StockSizeMax * VolTMax * sizeof(double), cudaMemcpyHostToDevice);
	double* VolK_ptr;
	cudaGetSymbolAddress((void**) &VolK_ptr, VolK);
	cudaMemcpy(VolK_ptr, VolK_, StockSizeMax * VolKMax * sizeof(double), cudaMemcpyHostToDevice);
	double* Vol_ptr;
	cudaGetSymbolAddress((void**) &Vol_ptr, Vol);
	cudaMemcpy(Vol_ptr, Vol_, StockSizeMax * VolTMax * VolKMax * sizeof(double), cudaMemcpyHostToDevice);

	// Global variable: correlation
	double* correl_ptr;
	cudaGetSymbolAddress((void **) &correl_ptr, correl);
	cudaMemcpy(correl_ptr, correl_, StockSizeMax * StockSizeMax * sizeof(double), cudaMemcpyHostToDevice);

	// Global variable: Quanto
	double* Quanto_ptr;
	cudaGetSymbolAddress((void **) &Quanto_ptr, Quanto);
	cudaMemcpy(Quanto_ptr, Quanto_, StockSizeMax * sizeof(double), cudaMemcpyHostToDevice);

	// Main MC part (the repeat number is just own purpose)
	for (long i = 0; i < SimN_; i++){
		MC<<<blockN, threadN>>>(devStates, StockSize_, ScheduleSize_, YTMType_, YTMSize_, SimMode_, isStrikePriceQuote_, threadN, devResults);
		cudaMemcpy(hostResults, devResults, blockN * threadN * sizeof(Result), cudaMemcpyDeviceToHost);

		// Copying MC results
		for (long j = 0; j < blockN * threadN; j++){
			result->price += hostResults[j].price / ((double)(blockN * threadN * SimN_));
			result->prob[hostResults[j].prob] += 1.0 / ((double)(blockN * threadN * SimN_));
			if (SimMode_ > 0){
				for (long k = 0; k < StockSize_; k++){
					result->delta[k] += hostResults[j].delta[k] / ((double)(blockN * threadN * SimN_));
					result->gamma[k] += hostResults[j].gamma[k] / ((double)(blockN * threadN * SimN_));
					result->vega[k] += hostResults[j].vega[k] / ((double)(blockN * threadN * SimN_));
				}
			}
			if (SimMode_ > 1){
				for (long k = 0; k < StockSize_; k++){
					result->rho[k] += hostResults[j].rho[k] / ((double)(blockN * threadN * SimN_));
				}
				result->theta += hostResults[j].theta / ((double)(blockN * threadN * SimN_));
			}
			if (SimMode_ > 2){
				for (long k = 0; k < StockSize_; k++){
					result->vanna[k] += hostResults[j].vanna[k] / ((double)(blockN * threadN * SimN_));
					result->volga[k] += hostResults[j].volga[k] / ((double)(blockN * threadN * SimN_));
				}
			}
		}
	}
	cudaFree(devStates);
	cudaFree(devResults);
	free(hostResults);
}

// Seed initialization
__global__ void InitSeed(curandState *state, const long threadN)
{
	long id = threadIdx.x + blockIdx.x * threadN;
	curand_init(id, 0, 0, &state[id]);
}

// Main Monte Carlo part
__global__ void MC(curandState *state, 
				   const long StockSize, const long ScheduleSize, 
				   const long YTMType, const long YTMSize, 
				   const long SimMode, const long isStrikePriceQuote, const long threadN,
				   Result *result){ 
	long id = threadIdx.x + blockIdx.x * threadN; 
	long t = 0; double dt = 1.0/365.0;
	long CFnum = (long)(pow(2.0, (double)(StockSize+1))-1);
	long adjnum = (long)(pow(2.0, (double)(StockSize)));

	// Price variables
	double logS_MC[StockSizeMax], logS_MCmin[StockSizeMax], logS_MCmax[StockSizeMax];
	double logS_MC_Sp[StockSizeMax], logS_MCmin_Sp[StockSizeMax], logS_MCmax_Sp[StockSizeMax];
	double logS_MC_Sm[StockSizeMax], logS_MCmin_Sm[StockSizeMax], logS_MCmax_Sm[StockSizeMax];
	double logS_MC_vp[StockSizeMax], logS_MCmin_vp[StockSizeMax], logS_MCmax_vp[StockSizeMax];
	double logS_MC_vpSp[StockSizeMax], logS_MCmin_vpSp[StockSizeMax], logS_MCmax_vpSp[StockSizeMax];
	double logS_MC_vpSm[StockSizeMax], logS_MCmin_vpSm[StockSizeMax], logS_MCmax_vpSm[StockSizeMax];
	double logS_MC_vm[StockSizeMax], logS_MCmin_vm[StockSizeMax], logS_MCmax_vm[StockSizeMax];
	double logS_MC_vmSp[StockSizeMax], logS_MCmin_vmSp[StockSizeMax], logS_MCmax_vmSp[StockSizeMax];
	double logS_MC_vmSm[StockSizeMax], logS_MCmin_vmSm[StockSizeMax], logS_MCmax_vmSm[StockSizeMax];
	double logS_MC_rp[StockSizeMax], logS_MCmin_rp[StockSizeMax], logS_MCmax_rp[StockSizeMax];
	double logS_MC_tm[StockSizeMax], logS_MCmin_tm[StockSizeMax], logS_MCmax_tm[StockSizeMax];

	for (long j = 0; j < StockSize; j++){
		logS_MC[j] = logS_MCmin[j] = logS_MCmax[j] = log(Stock[j].S);
		logS_MC_Sp[j] = logS_MCmin_Sp[j] = logS_MCmax_Sp[j] = log(Stock[j].S * 1.01f);
		logS_MC_Sm[j] = logS_MCmin_Sm[j] = logS_MCmax_Sm[j] = log(Stock[j].S * 0.99f);

		logS_MC_vp[j] = logS_MCmin_vp[j] = logS_MCmax_vp[j] = log(Stock[j].S);
		logS_MC_vpSp[j] = logS_MCmin_vpSp[j] = logS_MCmax_vpSp[j] = log(Stock[j].S * 1.01f);
		logS_MC_vpSm[j] = logS_MCmin_vpSm[j] = logS_MCmax_vpSm[j] = log(Stock[j].S * 0.99f);

		logS_MC_vm[j] = logS_MCmin_vm[j] = logS_MCmax_vm[j] = log(Stock[j].S);
		logS_MC_vmSp[j] = logS_MCmin_vmSp[j] = logS_MCmax_vmSp[j] = log(Stock[j].S * 1.01f);
		logS_MC_vmSm[j] = logS_MCmin_vmSm[j] = logS_MCmax_vmSm[j] = log(Stock[j].S * 0.99f);

		logS_MC_rp[j] = logS_MCmin_rp[j] = logS_MCmax_rp[j] = log(Stock[j].S);
		logS_MC_tm[j] = logS_MCmin_tm[j] = logS_MCmax_tm[j] = log(Stock[j].S);
	}

	// Price information for payoff calculation (current price, min/max along path)
	double S_MC_CF[StockSizeMax], S_MCmin_CF[StockSizeMax], S_MCmax_CF[StockSizeMax];
	double S_MC_CF_Sp[StockSizeMax], S_MCmin_CF_Sp[StockSizeMax], S_MCmax_CF_Sp[StockSizeMax];
	double S_MC_CF_Sm[StockSizeMax], S_MCmin_CF_Sm[StockSizeMax], S_MCmax_CF_Sm[StockSizeMax];
	double S_MC_CF_vp[StockSizeMax], S_MCmin_CF_vp[StockSizeMax], S_MCmax_CF_vp[StockSizeMax];
	double S_MC_CF_vpSp[StockSizeMax], S_MCmin_CF_vpSp[StockSizeMax], S_MCmax_CF_vpSp[StockSizeMax];
	double S_MC_CF_vpSm[StockSizeMax], S_MCmin_CF_vpSm[StockSizeMax], S_MCmax_CF_vpSm[StockSizeMax];
	double S_MC_CF_vm[StockSizeMax], S_MCmin_CF_vm[StockSizeMax], S_MCmax_CF_vm[StockSizeMax];
	double S_MC_CF_vmSp[StockSizeMax], S_MCmin_CF_vmSp[StockSizeMax], S_MCmax_CF_vmSp[StockSizeMax];
	double S_MC_CF_vmSm[StockSizeMax], S_MCmin_CF_vmSm[StockSizeMax], S_MCmax_CF_vmSm[StockSizeMax];
	double S_MC_CF_rp[StockSizeMax], S_MCmin_CF_rp[StockSizeMax], S_MCmax_CF_rp[StockSizeMax];
	double S_MC_CF_tm[StockSizeMax], S_MCmin_CF_tm[StockSizeMax], S_MCmax_CF_tm[StockSizeMax];

	double S_Payoff[12][StockSizeMax], S_Payoffmin[12][StockSizeMax], S_Payoffmax[12][StockSizeMax];
	
	// Global min/max among all underlyings
	double Smin[12], Smax[12];
	// Parameter
	double rf, rfp, ytm, ytmp, ytmtm, div, vol, volp, volm;

	// Brownian motion variable
	double W_MC_indep[StockSizeMax], W_MC[StockSizeMax];

	// Cash flow status (redeemed or not)
	long price_status = 0;						double price_tmp = 0;
	long delta_status[2 * StockSizeMax] = {0};	double delta_tmp[2 * StockSizeMax] = {0};
	long gamma_status[2 * StockSizeMax] = {0};	double gamma_tmp[2 * StockSizeMax] = {0};
	long vega_status[2 * StockSizeMax] = {0};	double vega_tmp[2 * StockSizeMax] = {0};
	long rho_status[StockSizeMax] = {0};		double rho_tmp[StockSizeMax] = {0};
	long theta_status = 0;						double theta_tmp = 0;
	long vanna_status[4 * StockSizeMax] = {0};	double vanna_tmp[4 * StockSizeMax] = {0};
	long volga_status[2 * StockSizeMax] = {0};	double volga_tmp[2 * StockSizeMax] = {0};

	// Simulation part
	for(long i = 0; i < ScheduleSize; i++){ 
		// Innovate until next redemption schedule
		while (t <= Schedule[i].T){
			// Generate independent Brownian motion
			for (long j = 0; j < StockSize; j++){
				W_MC_indep[j] = curand_normal(&state[id])*sqrt(dt);
			}
			// Incorporating correlation
			for (long j = StockSize-1; j >= 0; j--){
				W_MC[j] = correl[j*StockSize + j] * W_MC_indep[j];
				for (long k = j-1; k >= 0; k--){
					W_MC[j] += correl[j*StockSize + k] * W_MC_indep[k];
				}
			}
			// Innovation
			for (long j = 0; j < StockSize; j++){

				if (SimMode > 1){
					logS_MC_tm[j] = logS_MC[j];
					logS_MCmin_tm[j] = logS_MCmin[j];
					logS_MCmax_tm[j] = logS_MCmax[j];
				}

				rf = RfInterp((double)(t)*dt, j);								// longerp/extrap Risk-free rate at t
				
				div = DivInterp((double)(t)*dt, j);								// longerp/extrap Dividend rate at t

				// original path
				vol = VolInterp((double)(t)*dt, expf(logS_MC[j]), j);
				logS_MC[j] += (rf - div + Quanto[j]*vol - vol*vol/2.0f)*dt + vol*W_MC[j];	// Innovation
				logS_MCmin[j] = (logS_MC[j] < logS_MCmin[j]) ? logS_MC[j] : logS_MCmin[j];	// Updating minimum
				logS_MCmax[j] = (logS_MC[j] > logS_MCmax[j]) ? logS_MC[j] : logS_MCmax[j];	// Updating maximum

				if (SimMode > 0){
					// up-shifting price
					vol = VolInterp((double)(t)*dt, expf(logS_MC_Sp[j]), j);
					logS_MC_Sp[j] += (rf - div + Quanto[j]*vol - vol*vol/2.0f)*dt + vol*W_MC[j];				// Innovation
					logS_MCmin_Sp[j] = (logS_MC_Sp[j] < logS_MCmin_Sp[j]) ? logS_MC_Sp[j] : logS_MCmin_Sp[j];	// Updating minimum
					logS_MCmax_Sp[j] = (logS_MC_Sp[j] > logS_MCmax_Sp[j]) ? logS_MC_Sp[j] : logS_MCmax_Sp[j];	// Updating maximum

					// down-shifting price
					vol = VolInterp((double)(t)*dt, expf(logS_MC_Sm[j]), j);
					logS_MC_Sm[j] += (rf - div + Quanto[j]*vol - vol*vol/2.0f)*dt + vol*W_MC[j];				// Innovation
					logS_MCmin_Sm[j] = (logS_MC_Sm[j] < logS_MCmin_Sm[j]) ? logS_MC_Sm[j] : logS_MCmin_Sm[j];	// Updating minimum
					logS_MCmax_Sm[j] = (logS_MC_Sm[j] > logS_MCmax_Sm[j]) ? logS_MC_Sm[j] : logS_MCmax_Sm[j];	// Updating maximum

					// up-shifting volatility
					volp = VolInterp((double)(t)*dt, expf(logS_MC_vp[j]), j) + 0.01f;
					logS_MC_vp[j] += (rf - div + Quanto[j]*volp - volp*volp/2.0f)*dt + volp*W_MC[j];												// Innovation
					logS_MCmin_vp[j] = (logS_MC_vp[j] < logS_MCmin_vp[j]) ? logS_MC_vp[j] : logS_MCmin_vp[j];	// Updating minimum
					logS_MCmax_vp[j] = (logS_MC_vp[j] > logS_MCmax_vp[j]) ? logS_MC_vp[j] : logS_MCmax_vp[j];	// Updating maximum

					// down-shifting volatility
					volm = VolInterp((double)(t)*dt, expf(logS_MC_vm[j]), j) - 0.01f;	
					logS_MC_vm[j] += (rf - div + Quanto[j]*volm - volm*volm/2.0f)*dt + volm*W_MC[j];												// Innovation
					logS_MCmin_vm[j] = (logS_MC_vm[j] < logS_MCmin_vm[j]) ? logS_MC_vm[j] : logS_MCmin_vm[j];	// Updating minimum
					logS_MCmax_vm[j] = (logS_MC_vm[j] > logS_MCmax_vm[j]) ? logS_MC_vm[j] : logS_MCmax_vm[j];	// Updating maximum
				}

				if (SimMode > 1){
					// up-shifting risk free rate
					rfp = rf + 0.001;
					vol = VolInterp((double)(t)*dt, expf(logS_MC_rp[j]), j);
					logS_MC_rp[j] += (rfp - div + Quanto[j]*vol - vol*vol/2.0f)*dt + vol*W_MC[j];
					logS_MCmin_rp[j] = (logS_MC_rp[j] < logS_MCmin_rp[j]) ? logS_MC_rp[j] : logS_MCmin_rp[j];	// Updating minimum
					logS_MCmax_rp[j] = (logS_MC_rp[j] > logS_MCmax_rp[j]) ? logS_MC_rp[j] : logS_MCmax_rp[j];	// Updating maximum
				}

				if (SimMode > 2){
					volp = VolInterp((double)(t)*dt, expf(logS_MC_vpSp[j]), j) + 0.01f;
					// up-shifting volatility, up-shifting price
					logS_MC_vpSp[j] += (rf - div + Quanto[j]*volp - volp*volp/2.0f)*dt + volp*W_MC[j];						// Innovation
					logS_MCmin_vpSp[j] = (logS_MC_vpSp[j] < logS_MCmin_vpSp[j]) ? logS_MC_vpSp[j] : logS_MCmin_vpSp[j];	// Updating minimum
					logS_MCmax_vpSp[j] = (logS_MC_vpSp[j] > logS_MCmax_vpSp[j]) ? logS_MC_vpSp[j] : logS_MCmax_vpSp[j];	// Updating maximum

					volp = VolInterp((double)(t)*dt, expf(logS_MC_vpSm[j]), j) + 0.01f;
					// up-shifting volatility, down-shifting price
					logS_MC_vpSm[j] += (rf - div + Quanto[j]*volp - volp*volp/2.0f)*dt + volp*W_MC[j];						// Innovation
					logS_MCmin_vpSm[j] = (logS_MC_vpSm[j] < logS_MCmin_vpSm[j]) ? logS_MC_vpSm[j] : logS_MCmin_vpSm[j];	// Updating minimum
					logS_MCmax_vpSm[j] = (logS_MC_vpSm[j] > logS_MCmax_vpSm[j]) ? logS_MC_vpSm[j] : logS_MCmax_vpSm[j];	// Updating maximum

					volm = VolInterp((double)(t)*dt, expf(logS_MC_vmSp[j]), j) - 0.01f;
					// up-shifting volatility, up-shifting price
					logS_MC_vmSp[j] += (rf - div + Quanto[j]*volm - volm*volm*2.0f)*dt + volm*W_MC[j];						// Innovation
					logS_MCmin_vmSp[j] = (logS_MC_vmSp[j] < logS_MCmin_vmSp[j]) ? logS_MC_vmSp[j] : logS_MCmin_vmSp[j];	// Updating minimum
					logS_MCmax_vmSp[j] = (logS_MC_vmSp[j] > logS_MCmax_vmSp[j]) ? logS_MC_vmSp[j] : logS_MCmax_vmSp[j];	// Updating maximum

					volm = VolInterp((double)(t)*dt, expf(logS_MC_vmSm[j]), j) - 0.01f;
					// up-shifting volatility, down-shifting price
					logS_MC_vmSm[j] += (rf - div + Quanto[j]*volm - volm*volm/2.0f)*dt + volm*W_MC[j];						// Innovation
					logS_MCmin_vmSm[j] = (logS_MC_vmSm[j] < logS_MCmin_vmSm[j]) ? logS_MC_vmSm[j] : logS_MCmin_vmSm[j];	// Updating minimum
					logS_MCmax_vmSm[j] = (logS_MC_vmSm[j] > logS_MCmax_vmSm[j]) ? logS_MC_vmSm[j] : logS_MCmax_vmSm[j];	// Updating maximum
				}
			}
			__syncthreads();
			t++;
		}
		ytm = YTMInterp((double)(Schedule[i].T_pay)*dt, YTMType, YTMSize);
		ytmtm = YTMInterp((double)(Schedule[i].T_pay-1)*dt, YTMType, YTMSize);
		ytmp = ytm + 0.001;

		for(long j = 0; j < StockSize; j++){
			if (isStrikePriceQuote == 1){
				S_MC_CF[j] = exp(logS_MC[j]);
				S_MCmin_CF[j] = exp(logS_MCmin[j]);
				S_MCmax_CF[j] = exp(logS_MCmax[j]);
			}
			else if (isStrikePriceQuote == 0){
				S_MC_CF[j] = exp(logS_MC[j])/BasePrice[j] * 100.0f;
				S_MCmin_CF[j] = exp(logS_MCmin[j])/BasePrice[j] * 100.0f;
				S_MCmax_CF[j] = exp(logS_MCmax[j])/BasePrice[j] * 100.0f;
			}
		}

		if (SimMode > 0){
			if (isStrikePriceQuote == 1){
				for (long j = 0; j < StockSize; j++){
					S_MC_CF_Sp[j] = exp(logS_MC_Sp[j]);
					S_MCmin_CF_Sp[j] = exp(logS_MCmin_Sp[j]);
					S_MCmax_CF_Sp[j] = exp(logS_MCmax_Sp[j]);

					S_MC_CF_Sm[j] = exp(logS_MC_Sm[j]);
					S_MCmin_CF_Sm[j] = exp(logS_MCmin_Sm[j]);
					S_MCmax_CF_Sm[j] = exp(logS_MCmax_Sm[j]);

					S_MC_CF_vp[j] = exp(logS_MC_vp[j]);
					S_MCmin_CF_vp[j] = exp(logS_MCmin_vp[j]);
					S_MCmax_CF_vp[j] = exp(logS_MCmax_vp[j]);

					S_MC_CF_vm[j] = exp(logS_MC_vm[j]);
					S_MCmin_CF_vm[j] = exp(logS_MCmin_vm[j]);
					S_MCmax_CF_vm[j] = exp(logS_MCmax_vm[j]);
				}
			}
			else if (isStrikePriceQuote == 0){
				for (long j = 0; j < StockSize; j++){
					S_MC_CF_Sp[j] = exp(logS_MC_Sp[j])/BasePrice[j] * 100.0f;
					S_MCmin_CF_Sp[j] = exp(logS_MCmin_Sp[j])/BasePrice[j] * 100.0f;
					S_MCmax_CF_Sp[j] = exp(logS_MCmax_Sp[j])/BasePrice[j] * 100.0f;

					S_MC_CF_Sm[j] = exp(logS_MC_Sm[j])/BasePrice[j] * 100.0f;
					S_MCmin_CF_Sm[j] = exp(logS_MCmin_Sm[j])/BasePrice[j] * 100.0f;
					S_MCmax_CF_Sm[j] = exp(logS_MCmax_Sm[j])/BasePrice[j] * 100.0f;

					S_MC_CF_vp[j] = exp(logS_MC_vp[j])/BasePrice[j] * 100.0f;
					S_MCmin_CF_vp[j] = exp(logS_MCmin_vp[j])/BasePrice[j] * 100.0f;
					S_MCmax_CF_vp[j] = exp(logS_MCmax_vp[j])/BasePrice[j] * 100.0f;

					S_MC_CF_vm[j] = exp(logS_MC_vm[j])/BasePrice[j] * 100.0f;
					S_MCmin_CF_vm[j] = exp(logS_MCmin_vm[j])/BasePrice[j] * 100.0f;
					S_MCmax_CF_vm[j] = exp(logS_MCmax_vm[j])/BasePrice[j] * 100.0f;
				}
			}
		}

		if (SimMode > 1){
			if (isStrikePriceQuote == 1){
				for (long j = 0; j < StockSize; j++){
					S_MC_CF_rp[j] = exp(logS_MC_rp[j]);
					S_MCmin_CF_rp[j] = exp(logS_MCmin_rp[j]);
					S_MCmax_CF_rp[j] = exp(logS_MCmax_rp[j]);

					S_MC_CF_tm[j] = exp(logS_MC_tm[j]);
					S_MCmin_CF_tm[j] = exp(logS_MCmin_tm[j]);
					S_MCmax_CF_tm[j] = exp(logS_MCmax_tm[j]);
				}
			}
			else if (isStrikePriceQuote == 0){
				for (long j = 0; j < StockSize; j++){
					S_MC_CF_rp[j] = exp(logS_MC_rp[j])/BasePrice[j] * 100.0f;
					S_MCmin_CF_rp[j] = exp(logS_MCmin_rp[j])/BasePrice[j] * 100.0f;
					S_MCmax_CF_rp[j] = exp(logS_MCmax_rp[j])/BasePrice[j] * 100.0f;

					S_MC_CF_tm[j] = exp(logS_MC_tm[j])/BasePrice[j] * 100.0f;
					S_MCmin_CF_tm[j] = exp(logS_MCmin_tm[j])/BasePrice[j] * 100.0f;
					S_MCmax_CF_tm[j] = exp(logS_MCmax_tm[j])/BasePrice[j] * 100.0f;
				}
			}
		}

		if (SimMode > 2){
			if (isStrikePriceQuote == 1){
				for (long j = 0; j < StockSize; j++){
					S_MC_CF_vpSp[j] = exp(logS_MC_vpSp[j]);
					S_MCmin_CF_vpSp[j] = exp(logS_MCmin_vpSp[j]);
					S_MCmax_CF_vpSp[j] = exp(logS_MCmax_vpSp[j]);

					S_MC_CF_vpSm[j] = exp(logS_MC_vpSm[j]);
					S_MCmin_CF_vpSm[j] = exp(logS_MCmin_vpSm[j]);
					S_MCmax_CF_vpSm[j] = exp(logS_MCmax_vpSm[j]);

					S_MC_CF_vmSp[j] = exp(logS_MC_vmSp[j]);
					S_MCmin_CF_vmSp[j] = exp(logS_MCmin_vmSp[j]);
					S_MCmax_CF_vmSp[j] = exp(logS_MCmax_vmSp[j]);

					S_MC_CF_vmSm[j] = exp(logS_MC_vmSm[j]);
					S_MCmin_CF_vmSm[j] = exp(logS_MCmin_vmSm[j]);
					S_MCmax_CF_vmSm[j] = exp(logS_MCmax_vmSm[j]);
				}
			}
			else if (isStrikePriceQuote == 0){
				for (long j = 0; j < StockSize; j++){
					S_MC_CF_vpSp[j] = exp(logS_MC_vpSp[j])/BasePrice[j] * 100.0f;
					S_MCmin_CF_vpSp[j] = exp(logS_MCmin_vpSp[j])/BasePrice[j] * 100.0f;
					S_MCmax_CF_vpSp[j] = exp(logS_MCmax_vpSp[j])/BasePrice[j] * 100.0f;

					S_MC_CF_vpSm[j] = exp(logS_MC_vpSm[j])/BasePrice[j] * 100.0f;
					S_MCmin_CF_vpSm[j] = exp(logS_MCmin_vpSm[j])/BasePrice[j] * 100.0f;
					S_MCmax_CF_vpSm[j] = exp(logS_MCmax_vpSm[j])/BasePrice[j] * 100.0f;

					S_MC_CF_vmSp[j] = exp(logS_MC_vmSp[j])/BasePrice[j] * 100.0f;
					S_MCmin_CF_vmSp[j] = exp(logS_MCmin_vmSp[j])/BasePrice[j] * 100.0f;
					S_MCmax_CF_vmSp[j] = exp(logS_MCmax_vmSp[j])/BasePrice[j] * 100.0f;

					S_MC_CF_vmSm[j] = exp(logS_MC_vmSm[j])/BasePrice[j] * 100.0f;
					S_MCmin_CF_vmSm[j] = exp(logS_MCmin_vmSm[j])/BasePrice[j] * 100.0f;
					S_MCmax_CF_vmSm[j] = exp(logS_MCmax_vmSm[j])/BasePrice[j] * 100.0f;
				}
			}
		}
			
		// Price
		for (long j = 0; j < StockSize; j++){
			S_Payoff[0][j] = S_MC_CF[j];
			S_Payoffmin[0][j] = S_MCmin_CF[j];
			S_Payoffmax[0][j] = S_MCmax_CF[j];
		}
		Smin[0] = SMin(S_Payoffmin, StockSize, 0);
		Smax[0] = SMax(S_Payoffmax, StockSize, 0);
		if (price_status == 0){
			if(PayoffCheck(S_Payoff, Smin, Smax, StockSize, i, 0)){					// Checking Redemption
				price_tmp = PayoffCalc(S_Payoff, Smin, Smax, StockSize, i, 0) * expf(-ytm*(double)(Schedule[i].T_pay)*dt);
				price_status++;
				result[id].prob = i;
			}
		}

			
		if (SimMode > 0){
			// Delta & Gamma
			for (long j = 0; j < 2 * StockSize; j++){
				for (long k = 0; k < StockSize; k++){
					S_Payoff[j][k] = S_MC_CF[k];
					S_Payoffmin[j][k] = S_MCmin_CF[k];
					S_Payoffmax[j][k] = S_MCmax_CF[k];
				}
				switch (j){
					case 0:
						S_Payoff[0][0] = S_MC_CF_Sp[0]; S_Payoffmin[0][0] = S_MCmin_CF_Sp[0]; S_Payoffmax[0][0] = S_MCmax_CF_Sp[0];
					case 1:
						S_Payoff[1][0] = S_MC_CF_Sm[0]; S_Payoffmin[1][0] = S_MCmin_CF_Sm[0]; S_Payoffmax[1][0] = S_MCmax_CF_Sm[0];
					case 2:
						S_Payoff[2][1] = S_MC_CF_Sp[1]; S_Payoffmin[2][1] = S_MCmin_CF_Sp[1]; S_Payoffmax[2][1] = S_MCmax_CF_Sp[1];
					case 3:
						S_Payoff[3][1] = S_MC_CF_Sm[1]; S_Payoffmin[3][1] = S_MCmin_CF_Sm[1]; S_Payoffmax[3][1] = S_MCmax_CF_Sm[1];
					case 4:
						S_Payoff[4][2] = S_MC_CF_Sp[2]; S_Payoffmin[4][2] = S_MCmin_CF_Sp[2]; S_Payoffmax[4][2] = S_MCmax_CF_Sp[2];
					case 5:
						S_Payoff[5][2] = S_MC_CF_Sm[2]; S_Payoffmin[5][2] = S_MCmin_CF_Sm[2]; S_Payoffmax[5][2] = S_MCmax_CF_Sm[2];
					default:
						break;
				}
			}
			for (long j = 0; j < 2 * StockSize; j++){
				Smin[j] = SMin(S_Payoffmin, StockSize, j);
				Smax[j] = SMax(S_Payoffmax, StockSize, j);
			}
			for (long j = 0; j < 2*StockSize; j++){
				if (delta_status[j] == 0){
					if(PayoffCheck(S_Payoff, Smin, Smax, StockSize, i, j)){					// Checking Redemption
						delta_tmp[j] = PayoffCalc(S_Payoff, Smin, Smax, StockSize, i, j) * expf(-ytm*(double)(Schedule[i].T_pay)*dt);
						(delta_status[j])++;
					}
				}

				if (gamma_status[j] == 0){
					if(PayoffCheck(S_Payoff, Smin, Smax, StockSize, i, j)){					// Checking Redemption
						gamma_tmp[j] = PayoffCalc(S_Payoff, Smin, Smax, StockSize, i, j) * expf(-ytm*(double)(Schedule[i].T_pay)*dt);
						(gamma_status[j])++;
					}
				}
			}		

			// Vega
			for (long j = 0; j < 2 * StockSize; j++){
				for (long k = 0; k < StockSize; k++){
					S_Payoff[j][k] = S_MC_CF[k];
					S_Payoffmin[j][k] = S_MCmin_CF[k];
					S_Payoffmax[j][k] = S_MCmax_CF[k];
				}
				switch (j){
					case 0:
						S_Payoff[0][0] = S_MC_CF_vp[0]; S_Payoffmin[0][0] = S_MCmin_CF_vp[0]; S_Payoffmax[0][0] = S_MCmax_CF_vp[0];
					case 1:
						S_Payoff[1][0] = S_MC_CF_vm[0]; S_Payoffmin[1][0] = S_MCmin_CF_vm[0]; S_Payoffmax[1][0] = S_MCmax_CF_vm[0];
					case 2:
						S_Payoff[2][1] = S_MC_CF_vp[1]; S_Payoffmin[2][1] = S_MCmin_CF_vp[1]; S_Payoffmax[2][1] = S_MCmax_CF_vp[1];
					case 3:
						S_Payoff[3][1] = S_MC_CF_vm[1]; S_Payoffmin[3][1] = S_MCmin_CF_vm[1]; S_Payoffmax[3][1] = S_MCmax_CF_vm[1];
					case 4:
						S_Payoff[4][2] = S_MC_CF_vp[2]; S_Payoffmin[4][2] = S_MCmin_CF_vp[2]; S_Payoffmax[4][2] = S_MCmax_CF_vp[2];
					case 5:
						S_Payoff[5][2] = S_MC_CF_vm[2]; S_Payoffmin[5][2] = S_MCmin_CF_vm[2]; S_Payoffmax[5][2] = S_MCmax_CF_vm[2];
					default:
						break;
				}
			}
			for (long j = 0; j < 2 * StockSize; j++){
				Smin[j] = SMin(S_Payoffmin, StockSize, j);
				Smax[j] = SMax(S_Payoffmax, StockSize, j);
			}
			for (long j = 0; j < 2*StockSize; j++){
				if (vega_status[j] == 0){
					if(PayoffCheck(S_Payoff, Smin, Smax, StockSize, i, j)){					// Checking Redemption
						vega_tmp[j] = PayoffCalc(S_Payoff, Smin, Smax, StockSize, i, j) * expf(-ytm*(double)(Schedule[i].T_pay)*dt);
						(vega_status[j])++;
					}
				}
			}
		}

		if (SimMode > 1){
			// Rho
			for (long j = 0; j < StockSize; j++){
				for (long k = 0; k < StockSize; k++){
					S_Payoff[j][k] = S_MC_CF[k];
					S_Payoffmin[j][k] = S_MCmin_CF[k];
					S_Payoffmax[j][k] = S_MCmax_CF[k];
				}
				switch (j){
					case 0:
						S_Payoff[0][0] = S_MC_CF_rp[0]; S_Payoffmin[0][0] = S_MCmin_CF_rp[0]; S_Payoffmax[0][0] = S_MCmax_CF_rp[0];
					case 1:
						S_Payoff[1][1] = S_MC_CF_rp[1]; S_Payoffmin[1][1] = S_MCmin_CF_rp[1]; S_Payoffmax[1][1] = S_MCmax_CF_rp[1];
					case 2:
						S_Payoff[2][2] = S_MC_CF_rp[2]; S_Payoffmin[2][2] = S_MCmin_CF_rp[2]; S_Payoffmax[2][2] = S_MCmax_CF_rp[2];
					default:
						break;
				}
			}
			for (long j = 0; j < StockSize; j++){
				Smin[j] = SMin(S_Payoffmin, StockSize, j);
				Smax[j] = SMax(S_Payoffmax, StockSize, j);
			}
			for (long j = 0; j < StockSize; j++){
				if (rho_status[j] == 0){
					if(PayoffCheck(S_Payoff, Smin, Smax, StockSize, i, j)){					// Checking Redemption
						if (j == 0){
							rho_tmp[j] = PayoffCalc(S_Payoff, Smin, Smax, StockSize, i, j) * expf(-ytmp*(double)(Schedule[i].T_pay)*dt);
						}
						else{
							rho_tmp[j] = PayoffCalc(S_Payoff, Smin, Smax, StockSize, i, j) * expf(-ytm*(double)(Schedule[i].T_pay)*dt);
						}
						(rho_status[j])++;
					}
				}
			}

			// Theta
			for (long j = 0; j < StockSize; j++){
				S_Payoff[0][j] = S_MC_CF_tm[j];
				S_Payoffmin[0][j] = S_MCmin_CF_tm[j];
				S_Payoffmax[0][j] = S_MCmax_CF_tm[j];
			}
			for (long j = 0; j < StockSize; j++){
				Smin[j] = SMin(S_Payoffmin, StockSize, j);
				Smax[j] = SMax(S_Payoffmax, StockSize, j);
			}
			if (theta_status < 1){
				if(PayoffCheck(S_Payoff, Smin, Smax, StockSize, i, 0)){					// Checking Redemption
					theta_tmp = PayoffCalc(S_Payoff, Smin, Smax, StockSize, i, 0) * expf(-ytmtm*(double)(Schedule[i].T_pay-1)*dt);
					theta_status++;
				}
			}
		}

		if (SimMode > 2){
			// Vanna
			for (long j = 0; j < 4 * StockSize; j++){
				for (long k = 0; k < StockSize; k++){
					S_Payoff[j][k] = S_MC_CF[k];
					S_Payoffmin[j][k] = S_MCmin_CF[k];
					S_Payoffmax[j][k] = S_MCmax_CF[k];
				}
				switch (j){
					case 0:
						S_Payoff[0][0] = S_MC_CF_vpSp[0]; S_Payoffmin[0][0] = S_MCmin_CF_vpSp[0]; S_Payoffmax[0][0] = S_MCmax_CF_vpSp[0];
					case 1:
						S_Payoff[1][0] = S_MC_CF_vpSm[0]; S_Payoffmin[1][0] = S_MCmin_CF_vpSm[0]; S_Payoffmax[1][0] = S_MCmax_CF_vpSm[0];
					case 2:
						S_Payoff[2][0] = S_MC_CF_vmSp[0]; S_Payoffmin[2][0] = S_MCmin_CF_vmSp[0]; S_Payoffmax[2][0] = S_MCmax_CF_vmSp[0];
					case 3:
						S_Payoff[3][0] = S_MC_CF_vmSm[0]; S_Payoffmin[3][0] = S_MCmin_CF_vmSm[0]; S_Payoffmax[3][0] = S_MCmax_CF_vmSm[0];
					case 4:
						S_Payoff[4][1] = S_MC_CF_vpSp[1]; S_Payoffmin[4][1] = S_MCmin_CF_vpSp[1]; S_Payoffmax[4][1] = S_MCmax_CF_vpSp[1];
					case 5:
						S_Payoff[5][1] = S_MC_CF_vpSm[1]; S_Payoffmin[5][1] = S_MCmin_CF_vpSm[1]; S_Payoffmax[5][1] = S_MCmax_CF_vpSm[1];
					case 6:
						S_Payoff[6][1] = S_MC_CF_vmSp[1]; S_Payoffmin[6][1] = S_MCmin_CF_vmSp[1]; S_Payoffmax[6][1] = S_MCmax_CF_vmSp[1];
					case 7:
						S_Payoff[7][1] = S_MC_CF_vmSm[1]; S_Payoffmin[7][1] = S_MCmin_CF_vmSm[1]; S_Payoffmax[7][1] = S_MCmax_CF_vmSm[1];
					case 8:
						S_Payoff[8][2] = S_MC_CF_vpSp[2]; S_Payoffmin[8][2] = S_MCmin_CF_vpSp[2]; S_Payoffmax[8][2] = S_MCmax_CF_vpSp[2];
					case 9:
						S_Payoff[9][2] = S_MC_CF_vpSm[2]; S_Payoffmin[9][2] = S_MCmin_CF_vpSm[2]; S_Payoffmax[9][2] = S_MCmax_CF_vpSm[2];
					case 10:
						S_Payoff[10][2] = S_MC_CF_vmSp[2]; S_Payoffmin[10][2] = S_MCmin_CF_vmSp[2]; S_Payoffmax[10][2] = S_MCmax_CF_vmSp[2];
					case 11:
						S_Payoff[11][2] = S_MC_CF_vmSm[2]; S_Payoffmin[11][2] = S_MCmin_CF_vmSm[2]; S_Payoffmax[11][2] = S_MCmax_CF_vmSm[2];
					default:
						break;
				}
			}
			for (long j = 0; j < 4 * StockSize; j++){
				Smin[j] = SMin(S_Payoffmin, StockSize, j);
				Smax[j] = SMax(S_Payoffmax, StockSize, j);
			}					
			for (long j = 0; j < 4*StockSize; j++){
				if (vanna_status[j] == 0){
					if(PayoffCheck(S_Payoff, Smin, Smax, StockSize, i, j)){					// Checking Redemption
						vanna_tmp[j] = PayoffCalc(S_Payoff, Smin, Smax, StockSize, i, j) * expf(-ytm*(double)(Schedule[i].T_pay)*dt);
						(vanna_status[j])++;
					}
				}	
			}

			// Volga
			for (long j = 0; j < 2 * StockSize; j++){
				for (long k = 0; k < StockSize; k++){
					S_Payoff[j][k] = S_MC_CF[k];
					S_Payoffmin[j][k] = S_MCmin_CF[k];
					S_Payoffmax[j][k] = S_MCmax_CF[k];
				}
				switch (j){
					case 0:
						S_Payoff[0][0] = S_MC_CF_vp[0]; S_Payoffmin[0][0] = S_MCmin_CF_vp[0]; S_Payoffmax[0][0] = S_MCmax_CF_vp[0];
					case 1:
						S_Payoff[1][0] = S_MC_CF_vm[0]; S_Payoffmin[1][0] = S_MCmin_CF_vm[0]; S_Payoffmax[1][0] = S_MCmax_CF_vm[0];
					case 2:
						S_Payoff[2][1] = S_MC_CF_vp[1]; S_Payoffmin[2][1] = S_MCmin_CF_vp[1]; S_Payoffmax[2][1] = S_MCmax_CF_vp[1];
					case 3:
						S_Payoff[3][1] = S_MC_CF_vm[1]; S_Payoffmin[3][1] = S_MCmin_CF_vm[1]; S_Payoffmax[3][1] = S_MCmax_CF_vm[1];
					case 4:
						S_Payoff[4][2] = S_MC_CF_vp[2]; S_Payoffmin[4][2] = S_MCmin_CF_vp[2]; S_Payoffmax[4][2] = S_MCmax_CF_vp[2];
					case 5:
						S_Payoff[5][2] = S_MC_CF_vm[2]; S_Payoffmin[5][2] = S_MCmin_CF_vm[2]; S_Payoffmax[5][2] = S_MCmax_CF_vm[2];
					default:
						break;
				}
			}
			for (long j = 0; j < 2 * StockSize; j++){
				Smin[j] = SMin(S_Payoffmin, StockSize, j);
				Smax[j] = SMax(S_Payoffmax, StockSize, j);
			}
			for (long j = 0; j < 2*StockSize; j++){
				if (volga_status[j] == 0){
					if(PayoffCheck(S_Payoff, Smin, Smax, StockSize, i, j)){					// Checking Redemption
						volga_tmp[j] = PayoffCalc(S_Payoff, Smin, Smax, StockSize, i, j) * expf(-ytm*(double)(Schedule[i].T_pay)*dt);
						(volga_status[j])++;
					}
				}
			}
		}
	}

	result[id].price = price_tmp;
	if (SimMode > 0){
		for (long i = 0; i < StockSize; i++)
			result[id].delta[i] = (delta_tmp[2*i] - delta_tmp[2*i+1]) / (2.0f * 0.01f * Stock[i].S);
		for (long i = 0; i < StockSize; i++)
			result[id].gamma[i] = (gamma_tmp[2*i] - 2.0f * price_tmp + gamma_tmp[2*i+1]) / (0.01f * Stock[i].S * 0.01f * Stock[i].S);
		for (long i = 0; i < StockSize; i++)
			result[id].vega[i] = (vega_tmp[2*i] - vega_tmp[2*i+1]) / 2.0f;
	}
	if (SimMode > 1){
		for (long i = 0; i < StockSize; i++)
			result[id].rho[i] = (rho_tmp[i] - price_tmp) / 0.001f;
		result[id].theta = price_tmp - theta_tmp;
	}
	if (SimMode > 2){
		for (long i = 0; i < StockSize; i++)
			result[id].vanna[i] = ((vanna_tmp[4*i] - vanna_tmp[4*i+1]) - (vanna_tmp[4*i+2] - vanna_tmp[4*i+3]))/ (2.0f * 2.0f * 0.01f * Stock[i].S);
		for (long i = 0; i < StockSize; i++)
			result[id].volga[i] = (volga_tmp[2*i] - 2.0f * price_tmp + volga_tmp[2*i+1]) / (2.0f * 2.0f);
	}

}


// YTM longerp/extrapolation
__device__ double YTMInterp(double t, long YTMType, long YTMSize){
	double r = 0; 
	double t_prior = 0; double r_prior; 
	double YTM_longerp;
	long tind = 0;
	if (YTMType == 0)
		r = YTM[0];
	else if (YTMType == 1){
		r_prior = YTM[0];
		while (t > YTMt[tind]){
			r += (r_prior + YTM[tind]) / 2.0 * (YTMt[tind] - t_prior);
			t_prior = YTMt[tind];
			r_prior = YTM[tind];
			tind++;
		}
		YTM_longerp = YTM[tind-1] + (YTM[tind] - YTM[tind-1])/(YTMt[tind] - YTMt[tind-1])*(t-YTMt[tind]);
		r += (r_prior + YTM_longerp) / 2.0 * (t - t_prior);
		r /= t;
	}
	return r;
}

// Risk-free rate longerp/extrpolation
__device__ double RfInterp(double t, long stocknum){
	double Rf;
	long tind = 0;

	// Fixed case
	if (Stock[stocknum].RateType == 0)
		Rf = Rate[stocknum*RateTMax];

	// Term-structure case
	else if (Stock[stocknum].RateType == 1){
		while (t > Ratet[RateTMax*stocknum + tind] && tind < Stock[stocknum].RateSize){
			tind++;
			if (tind == Stock[stocknum].RateSize)	break;
		}
		
		// nearest extrapolation
		if (tind == 0)								Rf = Rate[RateTMax*stocknum];
		else if (tind == Stock[stocknum].RateSize)	Rf = Rate[RateTMax*stocknum + Stock[stocknum].RateSize-1];
		else{
			// linear longerpolation
			Rf = Rate[RateTMax*stocknum + tind-1] + 
				 (Rate[RateTMax*stocknum + tind] - Rate[RateTMax*stocknum + tind-1])/(Ratet[RateTMax*stocknum + tind] - Ratet[RateTMax*stocknum + tind-1]) *
				 (t-Ratet[RateTMax*stocknum + tind-1]);
		}
	}
	return Rf;
}

// Dividend longerp/extrapolation
__device__ double DivInterp(double t, long stocknum){
	double Divq;
	long tind = 0;

	// Fixed case
	if (Stock[stocknum].DivType == 0)
		Divq = Div[stocknum*DivTMax];

	// Term structure case
	else if (Stock[stocknum].DivType == 1){
		while (t > Divt[DivTMax*stocknum + tind] && tind < Stock[stocknum].DivSize){
			tind++;
			if (tind == Stock[stocknum].DivSize)	break;
		}
		// nearest extrapolation
		if (tind == 0)								Divq = Div[DivTMax*stocknum];
		else if (tind == Stock[stocknum].DivSize)	Divq = Div[DivTMax*stocknum + Stock[stocknum].DivSize-1];
		else{
			// linear longerpolation
			Divq = Div[DivTMax*stocknum + tind-1] +
				   (Div[DivTMax*stocknum + tind] - Div[DivTMax*stocknum + tind-1])/(Divt[DivTMax*stocknum + tind] - Divt[DivTMax*stocknum + tind-1]) *
				   (t-Divt[20*stocknum + tind-1]);
		}
	}
	return Divq;
}

__device__ double VolInterp(double t, double K, long stocknum){
	double v;
	double Vol1, Vol2, Vol11, Vol12, Vol21, Vol22;
	long tind = 0, Kind = 0;

	// Fixed case
	if (Stock[stocknum].VolType == 0)
		v = Vol[stocknum*VolTMax*VolKMax];

	// Term structure case (need to be mended!)
	else if (Stock[stocknum].VolType == 1){
		if (t > Volt[VolTMax*stocknum + tind] && tind < Stock[stocknum].VolSizet)
			tind++;
		// nearest extrapolation
		if (tind == 0)								v = Vol[VolTMax*stocknum + 0];
		else if (tind == Stock[stocknum].VolSizet)	v = Vol[VolTMax*stocknum + Stock[stocknum].VolSizet-1];
		else{
			// linear longerpolation
			v = Vol[VolTMax*stocknum + tind-1] + 
				(Vol[VolTMax*stocknum + tind] - Vol[VolTMax*stocknum + tind-1])/(Volt[VolTMax*stocknum + tind] - Volt[VolTMax*stocknum + tind-1]) *
				(t-Volt[VolTMax*stocknum + tind-1]);
		}
	}

	// Surface case
	else if (Stock[stocknum].VolType == 2){
		if (t > Volt[VolTMax*stocknum + tind] && tind < Stock[stocknum].VolSizet){
				while (t > Volt[VolTMax*stocknum + tind] && tind < Stock[stocknum].VolSizet){
					tind++;
					if (tind == Stock[stocknum].VolSizet)	break;
			}
	}

	if (K > VolK[VolKMax*stocknum + Kind]){
		while (K > VolK[VolKMax*stocknum + Kind] && Kind < Stock[stocknum].VolSizeK){
			Kind++;
			if (Kind == Stock[stocknum].VolSizeK)	break;
		}
	}

	if (tind == 0){
		if (Kind == 0)								v = Vol[VolTMax*VolKMax*stocknum + 0];
		else if (Kind == Stock[stocknum].VolSizeK)	v = Vol[VolTMax*VolKMax*stocknum + Stock[stocknum].VolSizeK - 1];
		else{
			v = Vol[VolTMax*VolKMax*stocknum + Kind-1] + 
			    (Vol[VolTMax*VolKMax*stocknum + Kind] - Vol[VolTMax*VolKMax*stocknum + Kind-1])/(VolK[VolKMax*stocknum + Kind] - VolK[VolKMax*stocknum + Kind-1]) *
			    (K-VolK[VolKMax*stocknum + Kind-1]);
		}
	}
	else if (tind == Stock[stocknum].VolSizet){
		if (Kind == 0)								v = Vol[VolTMax*VolKMax*stocknum + VolKMax*(Stock[stocknum].VolSizet-1)];
		else if (Kind == Stock[stocknum].VolSizeK)	v = Vol[VolTMax*VolKMax*stocknum + VolKMax*(Stock[stocknum].VolSizet-1)+Stock[stocknum].VolSizeK - 1];
		else{
			v = Vol[VolTMax*VolKMax*stocknum + (VolKMax*(Stock[stocknum].VolSizet-1)) + Kind-1] + 
				(Vol[VolTMax*VolKMax*stocknum + (VolKMax*(Stock[stocknum].VolSizet-1)) + Kind] - Vol[VolTMax*VolKMax*stocknum + (VolKMax*(Stock[stocknum].VolSizet-1)) + Kind-1])/(VolK[VolKMax*stocknum + Kind] - VolK[VolKMax*stocknum + Kind-1]) *
				(K-VolK[VolKMax*stocknum + Kind-1]);
		}
	}
	else{
		if (Kind == 0){
			Vol1 = Vol[VolTMax*VolKMax*stocknum + VolKMax*(tind-1)];
			Vol2 = Vol[VolTMax*VolKMax*stocknum + VolKMax*tind];
			v = Vol1 + (Vol2-Vol1)/(Volt[VolTMax*stocknum + tind] - Volt[VolTMax*stocknum + tind-1]) * (t-Volt[VolTMax*stocknum + tind-1]);
		}
		else if (Kind == Stock[stocknum].VolSizeK){
			Vol1 = Vol[VolTMax*VolKMax*stocknum + VolKMax*(tind-1) + Stock[stocknum].VolSizeK-1];
			Vol2 = Vol[VolTMax*VolKMax*stocknum + VolKMax*(tind) + Stock[stocknum].VolSizeK-1];
			v = Vol1 + (Vol2-Vol1)/(Volt[VolTMax*stocknum + tind] - Volt[VolTMax*stocknum + tind-1]) * (t-Volt[VolTMax*stocknum + tind-1]);
		}
		else{
			Vol11 = Vol[VolTMax*VolKMax*stocknum + VolKMax*(tind-1) + Kind-1];
			Vol12 = Vol[VolTMax*VolKMax*stocknum + VolKMax*(tind-1) + Kind];
			Vol21 = Vol[VolTMax*VolKMax*stocknum + VolKMax*(tind) + Kind-1];
			Vol22 = Vol[VolTMax*VolKMax*stocknum + VolKMax*(tind) + Kind];

			Vol1 = Vol11 + (Vol12-Vol11)/(VolK[VolKMax*stocknum + Kind] - VolK[VolKMax*stocknum + Kind-1]) * (K-VolK[VolKMax*stocknum + Kind-1]);
			Vol2 = Vol21 + (Vol22-Vol21)/(VolK[VolKMax*stocknum + Kind] - VolK[VolKMax*stocknum + Kind-1]) * (K-VolK[VolKMax*stocknum + Kind-1]);

			v = Vol1 + (Vol2-Vol1)/(Volt[VolTMax*stocknum + tind] - Volt[VolTMax*stocknum + tind-1]) * (t-Volt[VolTMax*stocknum + tind-1]);
		}
	}

	}
	return v;
}

// Minimum among stock prices
__device__ double SMin(double S_min[][StockSizeMax], long StockSize, long casenum){
	double Min = S_min[casenum][0];
	for (long i = 1; i < StockSize; i++){
		Min = (S_min[casenum][i] < Min) ? S_min[casenum][i] : Min;
	}
	return Min;
}

// Maximum among stock prices
__device__ double SMax(double S_max[][StockSizeMax], long StockSize, long casenum){
	double Max = S_max[casenum][0];
	for (long i = 1; i < StockSize; i++){
		Max = (S_max[casenum][i] > Max) ? S_max[casenum][i] : Max;
	}
	return Max;
}

// Reference price
__device__ double RefPriceCalc(double S[][StockSizeMax], long StockSize, long sched_ind, long casenum){
	double RefPrice = 0;
	switch(Schedule[sched_ind].RefPriceType){
		// Minimum case
		case 0:
			{
				RefPrice = SMin(S, StockSize, casenum);
				break;
			}
		// Average case
		case 1:
			{
				for (long i = 0; i < StockSize; i++){					
					RefPrice += S[casenum][i]/(double)(StockSize);
				}
				break;
			}
		default:
			break;
	}
	return RefPrice;
}

// Checking redemption
__device__ bool PayoffCheck(double S[][StockSizeMax], double* S_min, double* S_max, long StockSize, long sched_ind, long casenum){
	bool result = false;
	switch(Schedule[sched_ind].BermudanType){
		// Final case
		case 0:
			{
				result = true;
				break;
			}
		// Bermudan case
		case 1:
			{
				switch(Schedule[sched_ind].PayoffType){
					case 2:
						{
							if (RefPriceCalc(S, StockSize, sched_ind, casenum) > Schedule[sched_ind].K)	result = true;
							else																		result = false;
							break;
						}
					default:
						break;
				}
				break;
			}
		// Coupon case
		case 2:
			{
				result = true;
				break;
			}
		default:
			break;
	}
	return result;
}

// Payoff amount calculation (if redeem)
__device__ double PayoffCalc(double S[][StockSizeMax], double* S_min, double* S_max, long StockSize, long sched_ind, long casenum){
	double result = 0;
	switch(Schedule[sched_ind].BermudanType){
		// Final case
		case 0:
			{
				switch(Schedule[sched_ind].PayoffType){
					// PUT
					case 1:
						{
							double PayoffPrice = RefPriceCalc(S, StockSize, sched_ind, casenum);
							if (PayoffPrice > Schedule[sched_ind].K)						result = 100.0 + Schedule[sched_ind].Coupon;
							else if (S_min[casenum] > Schedule[sched_ind].TotalDownBarrier)	result = 100.0 + Schedule[sched_ind].Dummy;
							else															result = SMin(S, StockSize, casenum);
							break;
						}
					// DIGITCALL
					case 2:
						{
							double PayoffPrice = RefPriceCalc(S, StockSize, sched_ind, casenum);
							if (PayoffPrice > Schedule[sched_ind].K)
								result = 100.0 + Schedule[sched_ind].Coupon;
							else if (S_min[casenum] > Schedule[sched_ind].DownBarrier)
								result = 100.0 + Schedule[sched_ind].Dummy;
							else
								result = 100.0;
							break;
						}
					// KO CALL (coupon acts as a principal value)
					case 4:
						{
							float PayoffPrice = RefPriceCalc(S, StockSize, sched_ind, casenum);
							if (S_max[casenum] < Schedule[sched_ind].TotalUpBarrier)
							{
								if (PayoffPrice > Schedule[sched_ind].K)
									result = Schedule[sched_ind].Participation * (PayoffPrice - Schedule[sched_ind].K) + Schedule[sched_ind].Coupon;
								else
									result = Schedule[sched_ind].Coupon;
							}
							else
							{
								result = Schedule[sched_ind].Coupon;
							}
							break;
						}
					// KO PUT (coupon acts as a principal value)
					case 6:
						{
							float PayoffPrice = RefPriceCalc(S, StockSize, sched_ind, casenum);
							if (S_max[casenum] < Schedule[sched_ind].TotalUpBarrier)
							{
								if (PayoffPrice < Schedule[sched_ind].K)
									result = Schedule[sched_ind].Participation * (Schedule[sched_ind].K - PayoffPrice) + Schedule[sched_ind].Coupon;
								else
									result = Schedule[sched_ind].Coupon;
							}
							else
							{
								result = Schedule[sched_ind].Coupon;
							}
							break;
						}
					default:
						break;
				}
				break;
			}
		// Bermudan Case
		case 1:
			{
				switch(Schedule[sched_ind].PayoffType){
					// DIGITCALL
					case 2:
						{
							result = 100.0 + Schedule[sched_ind].Coupon;
							break;
						}
					default:
						break;
				}
				break;
			}
		default:
			break;
	}
	return result;
}