//
#include <cuda.h>
#include <curand_kernel.h>

#include "MCstruct.h"
#include "MCstruct_VBA.h"
#include "MCwrapper_fcn.h"

// Global variables for MC
// Underlying: Max 3
__constant__ Underlying Stock[3];
__constant__ float BasePrice[3];

// Schedule: Max 60
__constant__ Payoff Schedule[60];

// YTMinfo: t size 20
__constant__ YTM YTMInfo;
__constant__ float YTMt[20];
__constant__ float YTMFixed;
__constant__ float YTMCurve[20];

// Rate: t size 20 per each asset
__constant__ float Ratet[60];
__constant__ float RateFixed[3];
__constant__ float RateCurve[60];

// Div: t size 20 per each asset
__constant__ float Divt[60];
__constant__ float DivFixed[3];
__constant__ float DivCurve[60];

// Vol: t size 20, K size 13 per each asset
__constant__ float Volt[60];
__constant__ float VolK[60];
__constant__ float VolFixed[3];
__constant__ float VolCurve[60];
__constant__ float VolSurf[1200];

// Correlation
__constant__ float correl[9];

// Quanto
__constant__ float Quanto[3];

// Global Functions: functions are called in CalcMC
__global__ void InitSeed(curandState *state, const int threadN);
__global__ void MC(curandState *state, const int StockSize, const int ScheduleSize, const int SimMode, const int isStrikePriceQuote, const int threadN, Result *result);

// Device functions: functions are called in global functions
__device__ float YTMInterp(float t);												// YTM rate interp/extrapolation
__device__ float RfInterp(float t, int* tind, int stocknum);						// Rf spot rate interp/extrapolation
__device__ float DivInterp(float t, int* tind, int stocknum);						// Dividend interp/extrapolation
__device__ float VolInterp(float t, float K, int* tind, int* Kind, int stocknum);	// Volatility interp/extrapolationlation

__device__ float SMin(float S_min[][3], int StockSize, int casenum);
__device__ float SMax(float S_max[][3], int StockSize, int casenum);

__device__ float RefPriceCalc(float S, int StockSize, int sched_ind, int casenum);
__device__ bool PayoffCheck(float S[][3], float* S_min, float* S_max, int StockSize, int sched_ind, int casenum);
__device__ float PayoffCalc(float S[][3], float* S_min, float* S_max, int StockSize, int sched_ind, int casenum);

// Main function
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
			int isStrikePriceQuote_, int SimN_, int SimMode_, int blockN_, int threadN_, 
			struct VBAResult *result){

	// GPU parallelization: block/thread for CUDA cores
	int blockN = blockN_;
	int threadN = threadN_;

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
	Underlying stock_[3];
	for (int i = 0; i < StockSize_; i++)
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
	cudaMemcpy(stock_ptr, stock_, 3 * sizeof(Underlying), cudaMemcpyHostToDevice);

	// Global variable: YTM
	YTM YTMInfo_;
	YTM* YTMInfo_ptr;
	YTMInfo_.YTMType = YTMType_;
	YTMInfo_.YTMSize = YTMSize_;
	cudaGetSymbolAddress((void**) &YTMInfo_ptr, YTMInfo);
	cudaMemcpy(YTMInfo_ptr, &YTMInfo_, sizeof(YTM), cudaMemcpyHostToDevice);	
	float* YTMt_ptr;
	cudaGetSymbolAddress((void**) &YTMt_ptr, YTMt);
	cudaMemcpy(YTMt_ptr, YTMt_, 20 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(YTMFixed, &YTMFixed_, sizeof(float));
	float* YTMCurve_ptr;
	cudaGetSymbolAddress((void**) &YTMCurve_ptr, YTMCurve);
	cudaMemcpy(YTMCurve_ptr, YTMCurve_, 20 * sizeof(float), cudaMemcpyHostToDevice);

	// Global variable: Schedule
	Payoff schedule_[60];
	for (int i = 0; i < ScheduleSize_; i++)
	{
		schedule_[i].T = PayoffT_[i];
		schedule_[i].T_pay = PayoffT_pay[i];
		schedule_[i].BermudanType = BermudanType_[i];
		schedule_[i].PayoffType = PayoffType_[i];
		schedule_[i].RefPriceType = RefPriceType_[i];

		schedule_[i].K = PayoffK_[i];
		schedule_[i].UpBarrier = UpBarrier_[i];
		schedule_[i].DownBarrier = DownBarrier_[i];
		schedule_[i].TotalUpBarrier = TotalUpBarrier_[i];
		schedule_[i].TotalDownBarrier = TotalDownBarrier_[i];

		schedule_[i].Coupon = Coupon_[i];
		schedule_[i].Dummy = Dummy_[i];

		schedule_[i].Participation = Participation_[i];
	}
	Payoff* sched_ptr;
	cudaGetSymbolAddress((void**) &sched_ptr, Schedule);
	cudaMemcpy(sched_ptr, schedule_, 60 * sizeof(Payoff), cudaMemcpyHostToDevice);
	float* BasePrice_ptr;
	cudaGetSymbolAddress((void**) &BasePrice_ptr, BasePrice);
	cudaMemcpy(BasePrice_ptr, BasePrice_, 3 * sizeof(float), cudaMemcpyHostToDevice);

	// Global variable: Rate
	float* Ratet_ptr;
	cudaGetSymbolAddress((void**) &Ratet_ptr, Ratet);
	cudaMemcpy(Ratet_ptr, Ratet_, 60 * sizeof(float), cudaMemcpyHostToDevice);
	float* RateFixed_ptr;
	cudaGetSymbolAddress((void**) &RateFixed_ptr, RateFixed);
	cudaMemcpy(RateFixed_ptr, RateFixed_, 3 * sizeof(float), cudaMemcpyHostToDevice);
	float* RateCurve_ptr;
	cudaGetSymbolAddress((void**) &RateCurve_ptr, RateCurve);
	cudaMemcpy(RateCurve_ptr, RateCurve_, 60 * sizeof(float), cudaMemcpyHostToDevice);

	// Global variable: Dividend
	float* Divt_ptr;
	cudaGetSymbolAddress((void**) &Divt_ptr, Divt);
	cudaMemcpy(Divt_ptr, Divt_, 60 * sizeof(float), cudaMemcpyHostToDevice);
	float* DivFixed_ptr;
	cudaGetSymbolAddress((void**) &DivFixed_ptr, DivFixed);
	cudaMemcpy(DivFixed_ptr, DivFixed_, 3 * sizeof(float), cudaMemcpyHostToDevice);
	float* DivCurve_ptr;
	cudaGetSymbolAddress((void**) &DivCurve_ptr, DivCurve);
	cudaMemcpy(DivCurve_ptr, DivCurve_, 60 * sizeof(float), cudaMemcpyHostToDevice);

	// Global variable: Volatility
	float* Volt_ptr;
	cudaGetSymbolAddress((void**) &Volt_ptr, Volt);
	cudaMemcpy(Volt_ptr, Volt_, 60 * sizeof(float), cudaMemcpyHostToDevice);
	float* VolK_ptr;
	cudaGetSymbolAddress((void**) &VolK_ptr, VolK);
	cudaMemcpy(VolK_ptr, VolK_, 60 * sizeof(float), cudaMemcpyHostToDevice);
	float* VolFixed_ptr;
	cudaGetSymbolAddress((void**) &VolFixed_ptr, VolFixed);
	cudaMemcpy(VolFixed_ptr, VolFixed_, 3 * sizeof(float), cudaMemcpyHostToDevice);
	float* VolCurve_ptr;
	cudaGetSymbolAddress((void**) &VolCurve_ptr, VolCurve);
	cudaMemcpy(VolCurve_ptr, VolCurve_, 60 * sizeof(float), cudaMemcpyHostToDevice);
	float* VolSurf_ptr;
	cudaGetSymbolAddress((void**) &VolSurf_ptr, VolSurf);
	cudaMemcpy(VolSurf_ptr, VolSurf_, 1200 * sizeof(float), cudaMemcpyHostToDevice);

	// Global variable: correlation
	float* correl_ptr;
	cudaGetSymbolAddress((void **) &correl_ptr, correl);
	cudaMemcpy(correl_ptr, correl_, 9 * sizeof(float), cudaMemcpyHostToDevice);

	// Global variable: Quanto
	float* Quanto_ptr;
	cudaGetSymbolAddress((void **) &Quanto_ptr, Quanto);
	cudaMemcpy(Quanto_ptr, Quanto_, 3 * sizeof(float), cudaMemcpyHostToDevice);

	// Main MC part (the repeat number is just own purpose)
	for (int i = 0; i < SimN_; i++){
		MC<<<blockN, threadN>>>(devStates, StockSize_, ScheduleSize_, SimMode_, isStrikePriceQuote_, threadN, devResults);
		cudaMemcpy(hostResults, devResults, blockN * threadN * sizeof(Result), cudaMemcpyDeviceToHost);
		int resind = 0;
		// Copying MC results
		for (int j = 0; j < blockN * threadN; j++){
			result->price += hostResults[j].price / ((float)(blockN * threadN) * float(SimN_));
			result->prob[hostResults[j].prob] += 1.0 / ((float)(blockN * threadN) * float(SimN_));
			if (SimMode_ > 0){
				for (int k = 0; k < StockSize_; k++){
					result->delta[k] += hostResults[j].delta[k] / ((float)(blockN * threadN) * float(SimN_));
					result->gamma[k] += hostResults[j].gamma[k] / ((float)(blockN * threadN) * float(SimN_));
					result->vega[k] += hostResults[j].vega[k] / ((float)(blockN * threadN) * float(SimN_));
				}
			}
			if (SimMode_ > 1){
				for (int k = 0; k < StockSize_; k++){
					result->rho[k] += hostResults[j].rho[k] / ((float)(blockN * threadN) * float(SimN_));
				}
				result->theta += hostResults[j].theta / ((float)(blockN * threadN) * float(SimN_)); 
			}
			if (SimMode_ > 2){
				for (int k = 0; k < StockSize_; k++){
					result->vanna[k] += hostResults[j].vanna[k] / ((float)(blockN * threadN) * float(SimN_));
					result->volga[k] += hostResults[j].volga[k] / ((float)(blockN * threadN) * float(SimN_));
				}
			}
		}
	}
	cudaFree(devStates);
	cudaFree(devResults);
	free(hostResults);
}

// Seed initialization
__global__ void InitSeed(curandState *state, const int threadN)
{
	int id = threadIdx.x + blockIdx.x * threadN;
	curand_init(id, 0, 0, &state[id]);
}

// Main Monte Carlo part
__global__ void MC(curandState *state, 
				   const int StockSize, const int ScheduleSize, const int SimMode, const int isStrikePriceQuote, const int threadN,
				   Result *result){ 
	int id = threadIdx.x + blockIdx.x * threadN; 
	int t = 0; float dt = 1.0f/365.0f;
	int CFnum = (int)(pow(2.0, (double)(StockSize+1))-1);
	int adjnum = (int)(pow(2.0, (double)(StockSize)));

	// Price variables
	float logS_MC[3], logS_MCmin[3], logS_MCmax[3];
	float logS_MC_Sp[3], logS_MCmin_Sp[3], logS_MCmax_Sp[3];
	float logS_MC_Sm[3], logS_MCmin_Sm[3], logS_MCmax_Sm[3];
	float logS_MC_vp[3], logS_MCmin_vp[3], logS_MCmax_vp[3];
	float logS_MC_vpSp[3], logS_MCmin_vpSp[3], logS_MCmax_vpSp[3];
	float logS_MC_vpSm[3], logS_MCmin_vpSm[3], logS_MCmax_vpSm[3];
	float logS_MC_vm[3], logS_MCmin_vm[3], logS_MCmax_vm[3];
	float logS_MC_vmSp[3], logS_MCmin_vmSp[3], logS_MCmax_vmSp[3];
	float logS_MC_vmSm[3], logS_MCmin_vmSm[3], logS_MCmax_vmSm[3];
	float logS_MC_rp[3], logS_MCmin_rp[3], logS_MCmax_rp[3];
	float logS_MC_tm[3], logS_MCmin_tm[3], logS_MCmax_tm[3];

	for (int j = 0; j < StockSize; j++){
		logS_MC[j] = logS_MCmin[j] = logS_MCmax[j] = logf(Stock[j].S);
		logS_MC_Sp[j] = logS_MCmin_Sp[j] = logS_MCmax_Sp[j] = logf(Stock[j].S * 1.01f);
		logS_MC_Sm[j] = logS_MCmin_Sm[j] = logS_MCmax_Sm[j] = logf(Stock[j].S * 0.99f);

		logS_MC_vp[j] = logS_MCmin_vp[j] = logS_MCmax_vp[j] = logf(Stock[j].S);
		logS_MC_vpSp[j] = logS_MCmin_vpSp[j] = logS_MCmax_vpSp[j] = logf(Stock[j].S * 1.01f);
		logS_MC_vpSm[j] = logS_MCmin_vpSm[j] = logS_MCmax_vpSm[j] = logf(Stock[j].S * 0.99f);

		logS_MC_vm[j] = logS_MCmin_vm[j] = logS_MCmax_vm[j] = logf(Stock[j].S);
		logS_MC_vmSp[j] = logS_MCmin_vmSp[j] = logS_MCmax_vmSp[j] = logf(Stock[j].S * 1.01f);
		logS_MC_vmSm[j] = logS_MCmin_vmSm[j] = logS_MCmax_vmSm[j] = logf(Stock[j].S * 0.99f);

		logS_MC_rp[j] = logS_MCmin_rp[j] = logS_MCmax_rp[j] = logf(Stock[j].S);
		logS_MC_tm[j] = logS_MCmin_tm[j] = logS_MCmax_tm[j] = logf(Stock[j].S);
	}

	// Price information for payoff calculation (current price, min/max along path)
	float S_MC_CF[3], S_MCmin_CF[3], S_MCmax_CF[3];
	float S_MC_CF_Sp[3], S_MCmin_CF_Sp[3], S_MCmax_CF_Sp[3];
	float S_MC_CF_Sm[3], S_MCmin_CF_Sm[3], S_MCmax_CF_Sm[3];
	float S_MC_CF_vp[3], S_MCmin_CF_vp[3], S_MCmax_CF_vp[3];
	float S_MC_CF_vpSp[3], S_MCmin_CF_vpSp[3], S_MCmax_CF_vpSp[3];
	float S_MC_CF_vpSm[3], S_MCmin_CF_vpSm[3], S_MCmax_CF_vpSm[3];
	float S_MC_CF_vm[3], S_MCmin_CF_vm[3], S_MCmax_CF_vm[3];
	float S_MC_CF_vmSp[3], S_MCmin_CF_vmSp[3], S_MCmax_CF_vmSp[3];
	float S_MC_CF_vmSm[3], S_MCmin_CF_vmSm[3], S_MCmax_CF_vmSm[3];
	float S_MC_CF_rp[3], S_MCmin_CF_rp[3], S_MCmax_CF_rp[3];
	float S_MC_CF_tm[3], S_MCmin_CF_tm[3], S_MCmax_CF_tm[3];

	float S_Payoff[12][3], S_Payoffmin[12][3], S_Payoffmax[12][3];
	
	// Global min/max among all underlyings
	float Smin[12], Smax[12];
	// Parameter
	float rf, rfp, ytm, ytmp, ytmtm, div, vol, volp, volm;
	// Parameter index (used in term structure/surface interp/extrapolation)
	int rft_ind[3] = {0};
	int divt_ind[3] = {0};
	int volt_ind[3] = {0}, volK_ind[3] = {0};
	int volt_Sp_ind[3] = {0}, volK_Sp_ind[3] = {0}, volt_Sm_ind[3] = {0}, volK_Sm_ind[3] = {0};
	int volt_vp_ind[3] = {0}, volK_vp_ind[3] = {0}, volt_vpSp_ind[3] = {0}, volK_vpSp_ind[3] = {0}, volt_vpSm_ind[3] = {0}, volK_vpSm_ind[3] = {0};
	int volt_vm_ind[3] = {0}, volK_vm_ind[3] = {0}, volt_vmSp_ind[3] = {0}, volK_vmSp_ind[3] = {0}, volt_vmSm_ind[3] = {0}, volK_vmSm_ind[3] = {0};
	int volt_rp_ind[3] = {0}, volK_rp_ind[3] = {0};

	// Brownian motion variable
	float W_MC_indep[3], W_MC[3];

	// Cash flow status (redeemed or not)
	int price_status = 0;		float price_tmp = 0;
	int delta_status[6] = {0};	float delta_tmp[6] = {0};
	int gamma_status[6] = {0};	float gamma_tmp[6] = {0};
	int vega_status[6] = {0};	float vega_tmp[6] = {0};
	int rho_status[3] = {0};	float rho_tmp[3] = {0};
	int theta_status = 0;		float theta_tmp = 0;
	int vanna_status[12] = {0};	float vanna_tmp[12] = {0};
	int volga_status[6] = {0};	float volga_tmp[6] = {0};

	// Simulation part
	for(int i = 0; i < ScheduleSize; i++){ 
		// Innovate until next redemption schedule
		while (t <= Schedule[i].T){
			// Generate independent Brownian motion
			for (int j = 0; j < StockSize; j++){
				W_MC_indep[j] = curand_normal(&state[id])*sqrt(dt);
			}
			// Incorporating correlation
			for (int j = StockSize-1; j >= 0; j--){
				W_MC[j] = correl[j*StockSize + j] * W_MC_indep[j];
				for (int k = j-1; k >= 0; k--){
					W_MC[j] += correl[j*StockSize + k] * W_MC_indep[k];
				}
			}
			// Innovation
			for (int j = 0; j < StockSize; j++){

				if (SimMode > 1){
					logS_MC_tm[j] = logS_MC[j];
					logS_MCmin_tm[j] = logS_MCmin[j];
					logS_MCmax_tm[j] = logS_MCmax[j];
				}

				rf = RfInterp((float)(t)*dt, rft_ind, j);									// Interp/extrap Risk-free rate at t
				
				div = DivInterp((float)(t)*dt, divt_ind, j);								// Interp/extrap Dividend rate at t

				// original path
				vol = VolInterp((float)(t)*dt, expf(logS_MC[j]), volt_ind, volK_ind, j);
				logS_MC[j] += (rf - div + Quanto[j]*vol - vol*vol/2.0f)*dt + vol*W_MC[j];	// Innovation
				logS_MCmin[j] = (logS_MC[j] < logS_MCmin[j]) ? logS_MC[j] : logS_MCmin[j];	// Updating minimum
				logS_MCmax[j] = (logS_MC[j] > logS_MCmax[j]) ? logS_MC[j] : logS_MCmax[j];	// Updating maximum

				if (SimMode > 0){
					// up-shifting price
					vol = VolInterp((float)(t)*dt, expf(logS_MC_Sp[j]), volt_Sp_ind, volK_Sp_ind, j);
					logS_MC_Sp[j] += (rf - div + Quanto[j]*vol - vol*vol/2.0f)*dt + vol*W_MC[j];				// Innovation
					logS_MCmin_Sp[j] = (logS_MC_Sp[j] < logS_MCmin_Sp[j]) ? logS_MC_Sp[j] : logS_MCmin_Sp[j];	// Updating minimum
					logS_MCmax_Sp[j] = (logS_MC_Sp[j] > logS_MCmax_Sp[j]) ? logS_MC_Sp[j] : logS_MCmax_Sp[j];	// Updating maximum

					// down-shifting price
					vol = VolInterp((float)(t)*dt, expf(logS_MC_Sm[j]), volt_Sm_ind, volK_Sm_ind, j);
					logS_MC_Sm[j] += (rf - div + Quanto[j]*vol - vol*vol/2.0f)*dt + vol*W_MC[j];				// Innovation
					logS_MCmin_Sm[j] = (logS_MC_Sm[j] < logS_MCmin_Sm[j]) ? logS_MC_Sm[j] : logS_MCmin_Sm[j];	// Updating minimum
					logS_MCmax_Sm[j] = (logS_MC_Sm[j] > logS_MCmax_Sm[j]) ? logS_MC_Sm[j] : logS_MCmax_Sm[j];	// Updating maximum

					// up-shifting volatility
					volp = VolInterp((float)(t)*dt, expf(logS_MC_vp[j]), volt_vp_ind, volK_vp_ind, j) + 0.01f;
					logS_MC_vp[j] += (rf - div + Quanto[j]*volp - volp*volp/2.0f)*dt + volp*W_MC[j];												// Innovation
					logS_MCmin_vp[j] = (logS_MC_vp[j] < logS_MCmin_vp[j]) ? logS_MC_vp[j] : logS_MCmin_vp[j];	// Updating minimum
					logS_MCmax_vp[j] = (logS_MC_vp[j] > logS_MCmax_vp[j]) ? logS_MC_vp[j] : logS_MCmax_vp[j];	// Updating maximum

					// down-shifting volatility
					volm = VolInterp((float)(t)*dt, expf(logS_MC_vm[j]), volt_vm_ind, volK_vm_ind, j) - 0.01f;	
					logS_MC_vm[j] += (rf - div + Quanto[j]*volm - volm*volm/2.0f)*dt + volm*W_MC[j];												// Innovation
					logS_MCmin_vm[j] = (logS_MC_vm[j] < logS_MCmin_vm[j]) ? logS_MC_vm[j] : logS_MCmin_vm[j];	// Updating minimum
					logS_MCmax_vm[j] = (logS_MC_vm[j] > logS_MCmax_vm[j]) ? logS_MC_vm[j] : logS_MCmax_vm[j];	// Updating maximum
				}

				if (SimMode > 1){
					// up-shifting risk free rate
					rfp = rf + 0.001;
					vol = VolInterp((float)(t)*dt, expf(logS_MC_rp[j]), volt_rp_ind, volK_rp_ind, j);
					logS_MC_rp[j] += (rfp - div + Quanto[j]*vol - vol*vol/2.0f)*dt + vol*W_MC[j];
					logS_MCmin_rp[j] = (logS_MC_rp[j] < logS_MCmin_rp[j]) ? logS_MC_rp[j] : logS_MCmin_rp[j];	// Updating minimum
					logS_MCmax_rp[j] = (logS_MC_rp[j] > logS_MCmax_rp[j]) ? logS_MC_rp[j] : logS_MCmax_rp[j];	// Updating maximum
				}

				if (SimMode > 2){
					volp = VolInterp((float)(t)*dt, expf(logS_MC_vpSp[j]), volt_vpSp_ind, volK_vpSp_ind, j) + 0.01f;
					// up-shifting volatility, up-shifting price
					logS_MC_vpSp[j] += (rf - div + Quanto[j]*volp - volp*volp/2.0f)*dt + volp*W_MC[j];						// Innovation
					logS_MCmin_vpSp[j] = (logS_MC_vpSp[j] < logS_MCmin_vpSp[j]) ? logS_MC_vpSp[j] : logS_MCmin_vpSp[j];	// Updating minimum
					logS_MCmax_vpSp[j] = (logS_MC_vpSp[j] > logS_MCmax_vpSp[j]) ? logS_MC_vpSp[j] : logS_MCmax_vpSp[j];	// Updating maximum

					volp = VolInterp((float)(t)*dt, expf(logS_MC_vpSm[j]), volt_vpSm_ind, volK_vpSm_ind, j) + 0.01f;
					// up-shifting volatility, down-shifting price
					logS_MC_vpSm[j] += (rf - div + Quanto[j]*volp - volp*volp/2.0f)*dt + volp*W_MC[j];						// Innovation
					logS_MCmin_vpSm[j] = (logS_MC_vpSm[j] < logS_MCmin_vpSm[j]) ? logS_MC_vpSm[j] : logS_MCmin_vpSm[j];	// Updating minimum
					logS_MCmax_vpSm[j] = (logS_MC_vpSm[j] > logS_MCmax_vpSm[j]) ? logS_MC_vpSm[j] : logS_MCmax_vpSm[j];	// Updating maximum

					volm = VolInterp((float)(t)*dt, expf(logS_MC_vmSp[j]), volt_vmSp_ind, volK_vmSp_ind, j) + 0.01f;
					// up-shifting volatility, up-shifting price
					logS_MC_vmSp[j] += (rf - div + Quanto[j]*volm - volm*volm*2.0f)*dt + volm*W_MC[j];						// Innovation
					logS_MCmin_vmSp[j] = (logS_MC_vmSp[j] < logS_MCmin_vmSp[j]) ? logS_MC_vmSp[j] : logS_MCmin_vmSp[j];	// Updating minimum
					logS_MCmax_vmSp[j] = (logS_MC_vmSp[j] > logS_MCmax_vmSp[j]) ? logS_MC_vmSp[j] : logS_MCmax_vmSp[j];	// Updating maximum

					volm = VolInterp((float)(t)*dt, expf(logS_MC_vmSm[j]), volt_vmSm_ind, volK_vmSm_ind, j) + 0.01f;
					// up-shifting volatility, down-shifting price
					logS_MC_vmSm[j] += (rf - div + Quanto[j]*volm - volm*volm/2.0f)*dt + volm*W_MC[j];						// Innovation
					logS_MCmin_vmSm[j] = (logS_MC_vmSm[j] < logS_MCmin_vmSm[j]) ? logS_MC_vmSm[j] : logS_MCmin_vmSm[j];	// Updating minimum
					logS_MCmax_vmSm[j] = (logS_MC_vmSm[j] > logS_MCmax_vmSm[j]) ? logS_MC_vmSm[j] : logS_MCmax_vmSm[j];	// Updating maximum
				}
			}
			__syncthreads();
			t++;
		}
		ytm = YTMInterp((float)(Schedule[i].T_pay)*dt);
		ytmtm = YTMInterp((float)(Schedule[i].T_pay-1)*dt);
		ytmp = ytm + 0.001;

		for(int j = 0; j < StockSize; j++){
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
				for (int j = 0; j < StockSize; j++){
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
				for (int j = 0; j < StockSize; j++){
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
				for (int j = 0; j < StockSize; j++){
					S_MC_CF_rp[j] = exp(logS_MC_rp[j]);
					S_MCmin_CF_rp[j] = exp(logS_MCmin_rp[j]);
					S_MCmax_CF_rp[j] = exp(logS_MCmax_rp[j]);

					S_MC_CF_tm[j] = exp(logS_MC_tm[j]);
					S_MCmin_CF_tm[j] = exp(logS_MCmin_tm[j]);
					S_MCmax_CF_tm[j] = exp(logS_MCmax_tm[j]);
				}
			}
			else if (isStrikePriceQuote == 0){
				for (int j = 0; j < StockSize; j++){
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
				for (int j = 0; j < StockSize; j++){
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
				for (int j = 0; j < StockSize; j++){
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
		for (int j = 0; j < StockSize; j++){
			S_Payoff[0][j] = S_MC_CF[j];
			S_Payoffmin[0][j] = S_MCmin_CF[j];
			S_Payoffmax[0][j] = S_MCmax_CF[j];
		}
		Smin[0] = SMin(S_Payoffmin, StockSize, 0);
		Smax[0] = SMax(S_Payoffmax, StockSize, 0);
		if (price_status == 0){
			if(PayoffCheck(S_Payoff, Smin, Smax, StockSize, i, 0)){					// Checking Redemption
				price_tmp = PayoffCalc(S_Payoff, Smin, Smax, StockSize, i, 0) * expf(-ytm*(float)(Schedule[i].T_pay)*dt);
				price_status++;
				result[id].prob = i;
			}
		}

			
		if (SimMode > 0){
			// Delta & Gamma
			for (int j = 0; j < 2 * StockSize; j++){
				for (int k = 0; k < StockSize; k++){
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
			for (int j = 0; j < 2 * StockSize; j++){
				Smin[j] = SMin(S_Payoffmin, StockSize, j);
				Smax[j] = SMax(S_Payoffmax, StockSize, j);
			}
			for (int j = 0; j < 2*StockSize; j++){
				if (delta_status[j] == 0){
					if(PayoffCheck(S_Payoff, Smin, Smax, StockSize, i, j)){					// Checking Redemption
						delta_tmp[j] = PayoffCalc(S_Payoff, Smin, Smax, StockSize, i, j) * expf(-ytm*(float)(Schedule[i].T_pay)*dt);
						(delta_status[j])++;
					}
				}

				if (gamma_status[j] == 0){
					if(PayoffCheck(S_Payoff, Smin, Smax, StockSize, i, j)){					// Checking Redemption
						gamma_tmp[j] = PayoffCalc(S_Payoff, Smin, Smax, StockSize, i, j) * expf(-ytm*(float)(Schedule[i].T_pay)*dt);
						(gamma_status[j])++;
					}
				}
			}		

			// Vega
			for (int j = 0; j < 2 * StockSize; j++){
				for (int k = 0; k < StockSize; k++){
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
			for (int j = 0; j < 2 * StockSize; j++){
				Smin[j] = SMin(S_Payoffmin, StockSize, j);
				Smax[j] = SMax(S_Payoffmax, StockSize, j);
			}
			for (int j = 0; j < 2*StockSize; j++){
				if (vega_status[j] == 0){
					if(PayoffCheck(S_Payoff, Smin, Smax, StockSize, i, j)){					// Checking Redemption
						vega_tmp[j] = PayoffCalc(S_Payoff, Smin, Smax, StockSize, i, j) * expf(-ytm*(float)(Schedule[i].T_pay)*dt);
						(vega_status[j])++;
					}
				}
			}
		}

		if (SimMode > 1){
			// Rho
			for (int j = 0; j < StockSize; j++){
				for (int k = 0; k < StockSize; k++){
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
			for (int j = 0; j < StockSize; j++){
				Smin[j] = SMin(S_Payoffmin, StockSize, j);
				Smax[j] = SMax(S_Payoffmax, StockSize, j);
			}
			for (int j = 0; j < StockSize; j++){
				if (rho_status[j] == 0){
					if(PayoffCheck(S_Payoff, Smin, Smax, StockSize, i, j)){					// Checking Redemption
						if (j == 0){
							rho_tmp[j] = PayoffCalc(S_Payoff, Smin, Smax, StockSize, i, j) * expf(-ytmp*(float)(Schedule[i].T_pay)*dt);
						}
						else{
							rho_tmp[j] = PayoffCalc(S_Payoff, Smin, Smax, StockSize, i, j) * expf(-ytm*(float)(Schedule[i].T_pay)*dt);
						}
						(rho_status[j])++;
					}
				}
			}

			// Theta
			for (int j = 0; j < StockSize; j++){
				S_Payoff[0][j] = S_MC_CF_tm[j];
				S_Payoffmin[0][j] = S_MCmin_CF_tm[j];
				S_Payoffmax[0][j] = S_MCmax_CF_tm[j];
			}
			for (int j = 0; j < StockSize; j++){
				Smin[j] = SMin(S_Payoffmin, StockSize, j);
				Smax[j] = SMax(S_Payoffmax, StockSize, j);
			}
			if (theta_status < 1){
				if(PayoffCheck(S_Payoff, Smin, Smax, StockSize, i, 0)){					// Checking Redemption
					theta_tmp = PayoffCalc(S_Payoff, Smin, Smax, StockSize, i, 0) * expf(-ytmtm*(float)(Schedule[i].T_pay-1)*dt);
					theta_status++;
				}
			}
		}

		if (SimMode > 2){
			// Vanna
			for (int j = 0; j < 4 * StockSize; j++){
				for (int k = 0; k < StockSize; k++){
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
			for (int j = 0; j < 4 * StockSize; j++){
				Smin[j] = SMin(S_Payoffmin, StockSize, j);
				Smax[j] = SMax(S_Payoffmax, StockSize, j);
			}					
			for (int j = 0; j < 4*StockSize; j++){
				if (vanna_status[j] == 0){
					if(PayoffCheck(S_Payoff, Smin, Smax, StockSize, i, j)){					// Checking Redemption
						vanna_tmp[j] = PayoffCalc(S_Payoff, Smin, Smax, StockSize, i, j) * expf(-ytm*(float)(Schedule[i].T_pay)*dt);
						(vanna_status[j])++;
					}
				}	
			}

			// Volga
			for (int j = 0; j < 2 * StockSize; j++){
				for (int k = 0; k < StockSize; k++){
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
			for (int j = 0; j < 2 * StockSize; j++){
				Smin[j] = SMin(S_Payoffmin, StockSize, j);
				Smax[j] = SMax(S_Payoffmax, StockSize, j);
			}
			for (int j = 0; j < 2*StockSize; j++){
				if (volga_status[j] == 0){
					if(PayoffCheck(S_Payoff, Smin, Smax, StockSize, i, j)){					// Checking Redemption
						volga_tmp[j] = PayoffCalc(S_Payoff, Smin, Smax, StockSize, i, j) * expf(-ytm*(float)(Schedule[i].T_pay)*dt);
						(volga_status[j])++;
					}
				}
			}
		}
	}

	result[id].price = price_tmp;
	if (SimMode > 0){
		for (int i = 0; i < StockSize; i++)
			result[id].delta[i] = (delta_tmp[2*i] - delta_tmp[2*i+1]) / (2.0f * 0.01f * Stock[i].S);
		for (int i = 0; i < StockSize; i++)
			result[id].gamma[i] = (gamma_tmp[2*i] - 2.0f * price_tmp + gamma_tmp[2*i+1]) / (0.01f * Stock[i].S * 0.01f * Stock[i].S);
		for (int i = 0; i < StockSize; i++)
			result[id].vega[i] = (vega_tmp[2*i] - vega_tmp[2*i+1]) / 2.0f;
	}
	if (SimMode > 1){
		for (int i = 0; i < StockSize; i++)
			result[id].rho[i] = (rho_tmp[i] - price_tmp) / 0.001f;
		result[id].theta = price_tmp - theta_tmp;
	}
	if (SimMode > 2){
		for (int i = 0; i < StockSize; i++)
			result[id].vanna[i] = ((vanna_tmp[4*i] - vanna_tmp[4*i+1]) - (vanna_tmp[4*i+2] - vanna_tmp[4*i+3]))/ (2.0f * 2.0f * 0.01f * Stock[i].S);
		for (int i = 0; i < StockSize; i++)
			result[id].volga[i] = (volga_tmp[2*i] - 2.0f * price_tmp + volga_tmp[2*i+1]) / (2.0f * 2.0f);
	}

}


// YTM Interp/extrapolation
__device__ float YTMInterp(float t){
	float YTM = 0; 
	float t_prior = 0; float r_prior; 
	float YTM_interp;
	int tind = 0;
	if (YTMInfo.YTMType == 0)
		YTM = YTMFixed;
	else if (YTMInfo.YTMType == 1){
		r_prior = YTMCurve[0];
		while (t > YTMt[tind]){
			YTM += (r_prior + YTMCurve[tind]) / 2.0 * (YTMt[tind] - t_prior);
			t_prior = YTMt[tind];
			r_prior = YTMCurve[tind];
			tind++;
		}
		YTM_interp = YTMCurve[tind-1] + (YTMCurve[tind] - YTMCurve[tind-1])/(YTMt[tind] - YTMt[tind-1])*(t-YTMt[tind]);
		YTM += (r_prior + YTM_interp) / 2.0 * (t - t_prior);
		YTM /= t;
	}
	return YTM;
}

// Risk-free rate Interp/extrpolation
__device__ float RfInterp(float t, int* tind, int stocknum){
	float Rf;

	// Fixed case
	if (Stock[stocknum].RateType == 0)
		Rf = RateFixed[stocknum];

	// Term-structure case
	else if (Stock[stocknum].RateType == 1){
		if (t > Ratet[20*stocknum + (tind[stocknum])] && tind[stocknum] < Stock[stocknum].RateSize)
			(tind[stocknum])++;
		// nearest extrapolation
		if (tind[stocknum] == 0)								Rf = RateCurve[20*stocknum + 0];
		else if (tind[stocknum] == Stock[stocknum].RateSize)	Rf = RateCurve[20*stocknum + Stock[stocknum].RateSize-1];
		else{
			// linear interpolation
			Rf = RateCurve[20*stocknum + (tind[stocknum])-1] + 
				 (RateCurve[20*stocknum + (tind[stocknum])] - RateCurve[20*stocknum + (tind[stocknum])-1])/(Ratet[20*stocknum + (tind[stocknum])] - Ratet[20*stocknum + (tind[stocknum])-1]) *
				 (t-Ratet[20*stocknum + (tind[stocknum])-1]);
		}
	}
	return Rf;
}

// Dividend interp/extrapolation
__device__ float DivInterp(float t, int* tind, int stocknum){
	float Div;

	// Fixed case
	if (Stock[stocknum].DivType == 0)
		Div = DivFixed[stocknum];

	// Term structure case
	else if (Stock[stocknum].DivType == 1){
		if (t > Divt[20*stocknum + (tind[stocknum])] && tind[stocknum] < Stock[stocknum].DivSize)
			(tind[stocknum])++;
		// nearest extrapolation
		if (tind[stocknum] == 0)								Div = DivCurve[20*stocknum + 0];
		else if ((tind[stocknum]) == Stock[stocknum].DivSize)	Div = DivCurve[20*stocknum + Stock[stocknum].DivSize-1];
		else{
			// linear interpolation
			Div = DivCurve[20*stocknum + (tind[stocknum])-1] +
				  (DivCurve[20*stocknum + (tind[stocknum])] - DivCurve[20*stocknum + (tind[stocknum])-1])/(Divt[20*stocknum + (tind[stocknum])] - Divt[20*stocknum + (tind[stocknum])-1]) *
				  (t-Divt[20*stocknum + (tind[stocknum])-1]);
		}
	}
	return Div;
}

__device__ float VolInterp(float t, float K, int* tind, int* Kind, int stocknum){
	float Vol;
	float Vol1, Vol2, Vol11, Vol12, Vol21, Vol22;

	// Fixed case
	if (Stock[stocknum].VolType == 0)
		Vol = VolFixed[stocknum];

	// Term structure case
	else if (Stock[stocknum].VolType == 1){
		if (t > Volt[20*stocknum + (tind[stocknum])] && tind[stocknum] < Stock[stocknum].VolSizet)
			(tind[stocknum])++;
		// nearest extrapolation
		if ((tind[stocknum]) == 0)								Vol = VolCurve[20*stocknum + 0];
		else if ((tind[stocknum]) == Stock[stocknum].VolSizet)	Vol = VolCurve[20*stocknum + Stock[stocknum].VolSizet-1];
		else{
			// linear interpolation
			Vol = VolCurve[20*stocknum + (tind[stocknum])-1] + 
				  (VolCurve[20*stocknum + (tind[stocknum])] - VolCurve[20*stocknum + (tind[stocknum])-1])/(Volt[20*stocknum + (tind[stocknum])] - Volt[20*stocknum + (tind[stocknum])-1]) *
				  (t-Volt[20*stocknum + (tind[stocknum])-1]);
		}
	}

	// Surface case
	else if (Stock[stocknum].VolType == 2){
		if (t > Volt[20*stocknum + (tind[stocknum])] && tind[stocknum] < Stock[stocknum].VolSizet)
			(tind[stocknum])++;

		if (K < VolK[20*stocknum + (Kind[stocknum])]){
			while (K < VolK[20*stocknum + (Kind[stocknum])]){
				if (Kind[stocknum] == 0)	break;
				(Kind[stocknum])--;
			}
		}
		else if (K > VolK[20*stocknum + (Kind[stocknum])+1]){
			while (K > VolK[20*stocknum + (Kind[stocknum])+1]){
				if (Kind[stocknum] == Stock[stocknum].VolSizeK)	break;
				(Kind[stocknum])++;
			}
		}

		if ((tind[stocknum]) == 0){
			if ((Kind[stocknum]) == 0)								Vol = VolSurf[400*stocknum + 0];
			else if ((Kind[stocknum]) == Stock[stocknum].VolSizeK)	Vol = VolSurf[400*stocknum + Stock[stocknum].VolSizeK - 1];
			else{
				Vol = VolSurf[400*stocknum + (Kind[stocknum])-1] + 
					  (VolSurf[400*stocknum + (Kind[stocknum])] - VolSurf[400*stocknum + (Kind[stocknum])-1])/(VolK[20*stocknum + (Kind[stocknum])] - VolK[20*stocknum + (Kind[stocknum])-1]) *
					  (K-VolK[20*stocknum + (Kind[stocknum])-1]);
			}
		}
		else if ((tind[stocknum]) == Stock[stocknum].VolSizet){
			if ((Kind[stocknum]) == 0)								Vol = VolSurf[400*stocknum + 20*(Stock[stocknum].VolSizet-1)];
			else if ((Kind[stocknum]) == Stock[stocknum].VolSizeK)	Vol = VolSurf[400*stocknum + 20*Stock[stocknum].VolSizet - 1];
			else{
				Vol = VolSurf[400*stocknum + (20*(Stock[stocknum].VolSizet-1)) + (Kind[stocknum])-1] + 
					  (VolSurf[400*stocknum + (20*(Stock[stocknum].VolSizet-1)) + (Kind[stocknum])] - VolSurf[400*stocknum + (20*(Stock[stocknum].VolSizet-1)) + (Kind[stocknum])-1])/(VolK[20*stocknum + Kind[stocknum]] - VolK[20*stocknum + Kind[stocknum]-1]) *
					  (K-VolK[20*stocknum + (Kind[stocknum])-1]);
			}
		}
		else{
			if ((Kind[stocknum]) == 0){
				Vol1 = VolSurf[400*stocknum + 20*((tind[stocknum])-1)];
				Vol2 = VolSurf[400*stocknum + 20*(tind[stocknum])];
				Vol = Vol1 + (Vol2-Vol1)/(Volt[20*stocknum + (tind[stocknum])] - Volt[20*stocknum + (tind[stocknum])-1]) * (t-Volt[20*stocknum + (tind[stocknum])-1]);
			}
			else if ((Kind[stocknum]) == Stock[stocknum].VolSizeK){
				Vol1 = VolSurf[400*stocknum + 20*(tind[stocknum])-1];
				Vol2 = VolSurf[400*stocknum + 20*((tind[stocknum])+1)-1];
				Vol = Vol1 + (Vol2-Vol1)/(Volt[20*stocknum + (tind[stocknum])] - Volt[20*stocknum + (tind[stocknum])-1]) * (t-Volt[20*stocknum + (tind[stocknum])-1]);
			}
			else{
				Vol11 = VolSurf[400*stocknum + 20*((tind[stocknum])-1) + (Kind[stocknum])-1];
				Vol12 = VolSurf[400*stocknum + 20*((tind[stocknum])-1) + (Kind[stocknum])];
				Vol21 = VolSurf[400*stocknum + 20*((tind[stocknum])) + (Kind[stocknum])-1];
				Vol22 = VolSurf[400*stocknum + 20*((tind[stocknum])) + (Kind[stocknum])];
				Vol1 = Vol11 + (Vol12-Vol11)/(VolK[20*stocknum + (Kind[stocknum])] - VolK[20*stocknum + (Kind[stocknum])-1]) * (K-VolK[20*stocknum + (Kind[stocknum])-1]);
				Vol2 = Vol21 + (Vol22-Vol21)/(VolK[20*stocknum + (Kind[stocknum])] - VolK[20*stocknum + (Kind[stocknum])-1]) * (K-VolK[20*stocknum + (Kind[stocknum])-1]);
				Vol = Vol1 + (Vol2-Vol1)/(Volt[20*stocknum + (tind[stocknum])] - Volt[20*stocknum + (tind[stocknum])-1]) * (t-Volt[20*stocknum + (tind[stocknum])-1]);
			}
		}

	}
	return Vol;
}

// Minimum among stock prices
__device__ float SMin(float S_min[][3], int StockSize, int casenum){
	float Min = S_min[casenum][0];
	for (int i = 1; i < StockSize; i++){
		Min = (S_min[casenum][i] < Min) ? S_min[casenum][i] : Min;
	}
	return Min;
}

// Maximum among stock prices
__device__ float SMax(float S_max[][3], int StockSize, int casenum){
	float Max = S_max[casenum][0];
	for (int i = 1; i < StockSize; i++){
		Max = (S_max[casenum][i] > Max) ? S_max[casenum][i] : Max;
	}
	return Max;
}

// Reference price
__device__ float RefPriceCalc(float S[][3], int StockSize, int sched_ind, int casenum){
	float RefPrice = 0;
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
				for (int i = 0; i < StockSize; i++){					
					RefPrice += S[casenum][i]/(float)(StockSize);
				}
				break;
			}
		default:
			break;
	}
	return RefPrice;
}

// Checking redemption
__device__ bool PayoffCheck(float S[][3], float* S_min, float* S_max, int StockSize, int sched_ind, int casenum){
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
__device__ float PayoffCalc(float S[][3], float* S_min, float* S_max, int StockSize, int sched_ind, int casenum){
	float result = 0;
	switch(Schedule[sched_ind].BermudanType){
		// Final case
		case 0:
			{
				switch(Schedule[sched_ind].PayoffType){
					// PUT
					case 1:
						{
							float PayoffPrice = RefPriceCalc(S, StockSize, sched_ind, casenum);
							if (PayoffPrice > Schedule[sched_ind].K)						result = 100.0f + Schedule[sched_ind].Coupon;
							else if (S_min[casenum] > Schedule[sched_ind].TotalDownBarrier)	result = 100.0f + Schedule[sched_ind].Dummy;
							else															result = SMin(S, StockSize, casenum);
							break;
						}
					// KO CALL
					case 4:
						{
							float PayoffPrice = RefPriceCalc(S, StockSize, sched_ind, casenum);
							if (PayoffPrice > Schedule[sched_ind].K && PayoffPrice < Schedule[sched_ind].UpBarrier)
								result = Schedule[sched_ind].Participation * (PayoffPrice - Schedule[sched_ind].K);
							else
								result = 0;
							break;
						}
					// KO PUT
					case 6:
						{
							float PayoffPrice = RefPriceCalc(S, StockSize, sched_ind, casenum);
							if (PayoffPrice < Schedule[sched_ind].K && PayoffPrice > Schedule[sched_ind].DownBarrier)
								result = Schedule[sched_ind].Participation * (Schedule[sched_ind].K - PayoffPrice);
							else
								result = 0;
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
							result = 100.0f + Schedule[sched_ind].Coupon;
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