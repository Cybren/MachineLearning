package algorithms;

public class FFT {
	private double[] real;
	private double[] imag;
	private int n;
	public double[][] frequenzySpectrum;
	public double[] power;
	
    public FFT(double[]real,double[]imag){
    	this.real = real.clone();
    	this.imag = imag.clone();
    	this.n = real.length;
    	int m=(int) (Math.log(this.n)/Math.log(2));
    	
    	if(this.n!=this.imag.length){
    		throw new RuntimeException("Lengths of the arrays dont match");
    	}else if(this.n!=(1<<m)){
    		throw new RuntimeException("Length has to be a power of 2");
    	}
    	frequenzySpectrum = setFrequenzySpectrum(real,imag);
    	setPower();
    }
    
    private double[][] setFrequenzySpectrum(double[] real,double[] imag){//checked!
    	
    	double [][]h = new double[2][real.length];//h[0] contains the real part h[1] the imaginary part
    	double sin, cos, real1, imag1;
    	
    	if(real.length == 1){
    		h[0][0] = real[0];
    		h[1][0] = imag[0];
    		return h;
    	}else {
    		double[] realEven = divideArray(real,real.length,true);
    		double[] imagEven = divideArray(imag,imag.length,true);
    		double[][] even = setFrequenzySpectrum(realEven,imagEven);
    		
    		double[] realOdd = divideArray(real,real.length,false);
    		double[] imagOdd = divideArray(imag,imag.length,false);
    		double[][] odd = setFrequenzySpectrum(realOdd,imagOdd);
    		
    		for (int i = 0; i < real.length/2; i++) {
				sin = Math.sin(-2*Math.PI*i/real.length);
				cos = Math.cos(-2*Math.PI*i/real.length);
				real1 = cos*odd[0][i]-sin*odd[1][i];
				imag1 = sin*odd[0][i]+cos*odd[1][i];
				h[0][i] = even[0][i]+real1;
				h[1][i] = even[1][i]+imag1;
				h[0][i+real.length/2] = even[0][i]-real1;
				h[1][i+real.length/2] = even[1][i]-imag1;
    		}
    	return h;
    	}
    }
    
    private void setPower() {
    	if(frequenzySpectrum == null) {
    		frequenzySpectrum = setFrequenzySpectrum(real,imag);
    	}
    	
    	power = new double[n];
    	double[] absolute = getAbsolute();
    	for (int i = 0; i < power.length; i++) {
			power[i] = absolute[i]*absolute[i]/((double)n);
		}
    }
    
    //returns the absolute of an imaginary number a[0] contains the real parts, 
    public double[] getAbsolute(){
    	double[]absolute = new double[frequenzySpectrum[0].length];
    	for (int i = 0; i < absolute.length; i++) {
    		absolute[i]=(int) Math.sqrt((Math.pow(frequenzySpectrum[0][i], 2)+Math.pow(frequenzySpectrum[1][i], 2)));
		}
    	return absolute;
    }
    
  //divide the array in to two and return the even indexes
    public double[] divideArray (double[] a, int length, boolean even){
    	double[] d1 = new double[length/2];
    	
		if(even){
			for (int i = 0; i < a.length; i+=2) {
				d1 [i/2]
						=a[i];
			}
		}else{
			for (int i = 1; i < a.length; i+=2) {
				d1 [i/2]=a[i];
			}
		}
		return d1;
    }
    
    /*public double [][]inversefft(double [] real,double []imag){
    	int n = real.length;
    	double [][]h = new double[2][n];//h[0] contains the real part h[1] the imaginary part
    	double sin,cos,real1,imag1;
    	
    	if(n == 1){
    		h[0][0] = real[0];
    		h[1][0] = imag[0];
    		return h;
    	}else {
    		double [] realg = divideArray(real,real.length,true);
    		double [] imagg = divideArray(imag,imag.length,true);
    		double[] []g = inversefft(realg,imagg);
    		
    		double	[] realu = divideArray(real,real.length,false);
    		double []imagu = divideArray(imag,imag.length,false);
    		double [][]u = inversefft(realu,imagu);
    		
    		for (int i = 0; i < n/2; i++) {
				sin = Math.sin(2*Math.PI*i/n);
				cos = Math.cos(2*Math.PI*i/n);
				real1 = cos*u[0][i]-sin*u[1][i];
				imag1 = sin*u[0][i]+cos*u[1][i];
				h[0][i] = (g[0][i]+real1);
				h[1][i] = (g[1][i]+imag1);
				h[0][i+n/2] = (g[0][i]-real1);
				h[1][i+n/2] = (g[1][i]-imag1);
    		}
    	return h;
    	}
    }
    
    public double[][] completeIFFT(int n, double[] real, double[] imag){//???????????
    	double IFFT[][] = inversefft(real, imag);
    	int length = IFFT[0].length;
    	System.out.println("length: "+length);
    	for (int i = 0; i < length; i++) {
			IFFT[0][i]=Math.round(IFFT[0][i]*(1.0/length));
			IFFT[1][i]=Math.round(IFFT[1][i]*(1.0/length));
		}
    	return IFFT;
    }

    public static void main(String[] args) {
    	double []real =  new double [64];
    	double []imag =  new double [64];
    	for (int i = 0; i < real.length; i++) {
			real[i]=Math.sin(i+Math.PI/2);
			imag[i]=0;
		}
    	FFT fft = new FFT(real, imag);
    	System.out.println(Arrays.toString(fft.getAbsolute()));
    }*/
}
