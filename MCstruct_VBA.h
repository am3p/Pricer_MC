// See definitions of VBA type definitions in Excel worksheets

struct VBARate{
	int RateType;
	int RateSize;

    float Ratet[20];
    float Rater[20];
};

struct VBADiv{
	int DivType;
	int DivSize;

    float Divt[20];
    float Divq[20];
};

struct VBAVol{
	int VolType;
    int VolSizet;
    int VolSizeK;
    
    float Volt[20];
    float VolK[20];
    float Volv[400];
};

struct VBAYTM{
	int YTMType;
    int YTMSize;

    float YTMt[20];
    float YTMr[20];
};

struct VBAUnderlying{
    float S;
    struct VBARate Rf;
    struct VBADiv Div;
    struct VBAVol Vol;
};

struct VBAPayoff{
	int T;
	int T_pay;

    int BermudanType;
    int PayoffType;
	int RefPriceType;
    
	float K;
    float UpBarrier;
    float DownBarrier;
    float TotalUpBarrier;
    float TotalDownBarrier;
    float Participation;

    float Coupon;
	float Dummy;
};

struct VBAData{
    int StockSize;
    int ScheduleSize;
    
    struct VBAUnderlying Stock[3];
    float BasePrice[3];
    float Correl[9];
	float Quanto[3];
    struct VBAPayoff Schedule[60];
    struct VBAYTM YTM;

	int SimN;
	int SimMode;
	int blockN;
	int threadN;
};

struct VBAResult{
    float price;
    float delta[3];
    float gamma[3];
	float vega[3];		// Vega
	float rho[3];
	float theta;
	float vanna[3];		// Vanna
	float volga[3];		// Volga

	float prob[60];
};