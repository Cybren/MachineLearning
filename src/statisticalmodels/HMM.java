package statisticalmodels;

import java.util.Arrays;

public class HMM {
	private String[] states;//the set of teh states the HMM can be in
	private String [] v;//set of observation symbols for each state
	
	private int n;//|states|
	public int getN() {return n;}
	public void setN(int n) {this.n = n;}
	
	private double[][] a;//a[i][j]state transition probabilities: probability that we enter state Sj in t+1 given, that we are in state Si at time t
	public double[][] getA() {return a;}
	public void setA(double[][] a) {this.a = a;}

	private int m;//|v|
	public int getM() {return m;}
	public void setM(int m) {this.m = m;}
	
	private double[][] b;//b[i][j]observation symbol probability distribution: probability that that we observe j, given we are in state Si at time t
	public double[][] getB() {return b;}
	public void setB(double[][] b) {this.b = b;}
	
	private double[] ini;//initial state probabilities
	public double[] getIni() {return ini;}
	public void setIni(double[] ini) {this.ini = ini;}

	public HMM(String[] states, double[][] a, String[] v,  double[][] b, double[] ini) {
		this.states = states;
		this.n = a.length;
		this.a = a;
		
		this.v = v;
		this.m = v.length;
		this.b = b;
		
		this.ini = ini;

	}
	
	public HMM(double[][] a, double[][] b, double[] ini) {
		this.n = a.length;
		this.a = a;
		
		this.m = b[0].length;
		this.b = b;
		
		this.ini = ini;

	}
	
	public double eval(int[] observations) {
		int t = observations.length;
		ForwardBackwardCalculator fbc = new ForwardBackwardCalculator(this, observations);
		double[] scales = fbc.scales.clone();
		double res = 1;
		for (int k = 0; k < t; k++) {
			res *= scales[k];
		}
		return 1/res;
	}
	
	public double evalForward(int[] observations) {
		int t = observations.length;
		ForwardBackwardCalculator fbc = new ForwardBackwardCalculator(this, observations);
		double[][] forward = fbc.forwardUnscaled.clone();
		double res = 0;
		for (int i = 0; i < n; i++) {
			res += forward[t-1][i];
		}
		return res;
	}
	
	public double evalLog(int[] observations) {
		int t = observations.length;
		ForwardBackwardCalculator fbc = new ForwardBackwardCalculator(this, observations);
		double[] scale = fbc.scales;
		double res = 1;
		for (int k = 0; k < t; k++) {
			res += Math.log(scale[k]);
		}
		return -res;
	}
	
	public int[] viterbi(int[] observations) {
		int t = observations.length;
		double[][] theta = new double[t][n];
		int[][] psi = new int[t][n];
		//initilize
		for (int i = 0; i < n; i++) {
			theta[0][i] = ini[i]*b[i][observations[0]];
			psi[0][i] = 0;
		}
		//recursion
		double max;
		double temp;
		for (int k = 1; k < t; k++) {
			for (int i = 0; i < n; i++) {
				max = 0;
				for (int j = 0; j < n; j++) {
					temp = a[j][i]*theta[k-1][j];
					//System.out.println(a[j][i]+"*"+theta[k-1][j]);
					if(temp>max) {
						max = temp;
						psi[k][i] = j;
					}
				}
				theta[k][i] = b[i][observations[k]]*max;
				//System.out.println("theta: "+theta[k][i]);
			}
		}
		//termination
		double pMax = 0;
		int[] ret = new int[t];
		for (int i = 0; i < n; i++) {
			if(theta[t-1][i]>pMax) {
				pMax = theta[t-1][i];
				ret[t-1] = i;
			}
		}
		System.out.println("pMax: "+pMax);
		for (int i = t-2; i > -1; i--) {
			ret[i] = psi[i+1][ret[i+1]];
		}
		
		printStates(ret);
		return ret;
	}
	
	public void baumWelch(int[] observations) {
		int t = observations.length;
		ForwardBackwardCalculator fbc = new ForwardBackwardCalculator(this, observations);
		double[][] forward = fbc.forward;
		double[][] backward = fbc.backward;
		
		double[][][] xi = new double[t-1][n][n];
		double[][] gamma = new double[t][n];
		
		double temp;
		double[] sumGamma = new double[n];
		
		double[] iniNew = new double[n];
		double[][] aNew = new double[n][n];
		double[][] bNew = new double[n][m];
		
		for (int k = 0; k < t-1; k++) {
			temp = 0;
			for (int i = 0; i < n; i++) {
				for (int j = 0; j < n; j++) {
					temp += forward[k][i]*a[i][j]*b[j][observations[k+1]]*backward[k+1][j];
				}
				
			}
			for (int i = 0; i < n; i++) {
				gamma[k][i] = 0;
				for (int j = 0; j < n; j++) {
					xi[k][i][j] = (forward[k][i]*a[i][j]*b[j][observations[k+1]]*backward[k+1][j])/temp;
					gamma[k][i] += xi[k][i][j];
				}
			}
		}
		
		iniNew = gamma[0].clone();
		//aNew
		for (int i = 0; i < n; i++) {
			sumGamma[i] = 0;
			for (int k = 0; k < t-1; k++) {
				sumGamma[i] += gamma[k][i];
			}
			for (int j = 0; j < n; j++) {
				temp = 0;
				for (int k = 0; k < t-1; k++) {
					temp += xi[k][i][j];
				}
				aNew[i][j] = temp/sumGamma[i];
			}
			//bNew
			sumGamma[i] = 0;
			for (int k = 0; k < t; k++) {
				sumGamma[i] += gamma[k][i];
			}
			for (int l = 0; l < m; l++) {
				bNew[i][l] = 0;
				for (int k = 0; k < t; k++) {
					if(observations[k] == l) {
						bNew[i][l] += gamma[k][i];
					}
				}
				bNew[i][l]/=sumGamma[i];
			}
		}
		
		ini = iniNew.clone();
		a = aNew.clone();
		b = bNew.clone();
	}
	
	public void baumWelch(int[][] observations) {
		double[][] aNew = new double[n][n];
		double[][] bNew = new double[n][m];
		
		double[][] tempACounter = new double[n][n];
		double[][] tempADenominator = new double[n][n];
		
		double[] tempBCounter = new double[observations.length];
		double[] tempBDenominator = new double[observations.length];
		
		double[] p = new double[observations.length];
		

		double[][][] forward = new double[observations.length][][];
		double[][][] backward =  new double[observations.length][][];
		
		//precompute
		for (int o = 0; o < observations.length; o++) {
			p[o] = evalForward(observations[o]);
			ForwardBackwardCalculator fbc = new ForwardBackwardCalculator(this, observations[o]);
			forward[o] = fbc.forward.clone();
			backward[o] = fbc.backward.clone();
		}
		
		//calc aNew
		for (int o = 0; o < observations.length; o++) {
			ANewHelper a = new ANewHelper(observations[o], this);
			for (int i = 0; i < n; i++) {
				for (int j = 0; j < n; j++) {
					tempACounter[i][j] += a.aCounter[i][j]/p[o];
					tempADenominator[i][j]+= a.aDenominator[i][j]/p[o];
				}
			}
		}
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < n; j++) {
				aNew[i][j] = tempACounter[i][j]/tempADenominator[i][j];
			}
		}
		
		//calc bNew
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < m; j++) {
				for (int o = 0; o < observations.length; o++) {
					
					tempBCounter[o] = 0;
					tempBDenominator[o] = 0;

					int t = observations[o].length;
					
					double tempSumCounter = 0;
					double tempSumDenominator = 0;

					//calc sums
					for (int k = 0; k < t-1; k++) {
						tempSumDenominator += forward[o][k][i]*backward[o][k][i];

					}
					for (int k = 0; k < t-1; k++) {
						if(observations[o][k] == j) {
							tempSumCounter += forward[o][k][i]*backward[o][k][i];
						}
					}
					
					tempBCounter[o] = tempSumCounter/p[o];
					tempBDenominator[o] = tempSumDenominator/p[o];
				}
				double bCounter = 0;
				double bDenominator = 0;
				for (int o = 0; o < observations.length; o++) {
							
					bCounter += tempBCounter[o];
					bDenominator += tempBDenominator[o];
				}
				//System.out.println("b:"+bCounter+"/"+bDenominator);
				bNew[i][j] = bCounter/bDenominator;
			}
		}
		a = aNew.clone();
		b = bNew.clone();
	}
	
	public void printStates(int[] states) {
		if(this.states != null) {
			for (int i = 0; i < states.length; i++) {
			System.out.print(this.states[states[i]]+" ");
			}
		}
		System.out.println();
	}
	
	public void printObservations(int[] obs) {
		if(v != null) {
			for (int i = 0; i < obs.length; i++) {
			System.out.print(v[i]+" ");
			}
		}
		System.out.println();
	}
	
	private class ANewHelper{
		private int[] obs;
		public double[][] aCounter;
		public double[][] aDenominator;
		private HMM hmm;
		public ANewHelper(int observations[], HMM hmm) {
			this.obs = observations;
			this.hmm = hmm;
			this.aCounter = new double[n][n];
			this.aDenominator = new double[n][n];
			set();
		}
		public void set() {
			int t = obs.length;
			ForwardBackwardCalculator fbc = new ForwardBackwardCalculator(hmm, obs);
			double[][] forward = fbc.forward;
			double[][] backward = fbc.backward;
			
			double[][][] xi = new double[t-1][n][n];
			double[][] gamma = new double[t][n];
			
			double temp;
			double[] sumGamma = new double[n];
			
			for (int k = 0; k < t-1; k++) {
				temp = 0;
				for (int i = 0; i < n; i++) {
					for (int j = 0; j < n; j++) {
						temp += forward[k][i]*a[i][j]*b[j][obs[k+1]]*backward[k+1][j];
					}
					
				}
				for (int i = 0; i < n; i++) {
					gamma[k][i] = 0;
					for (int j = 0; j < n; j++) {
						xi[k][i][j] = (forward[k][i]*a[i][j]*b[j][obs[k+1]]*backward[k+1][j])/temp;
						gamma[k][i] += xi[k][i][j];
					}
				}
			}

			//aNew
			for (int i = 0; i < n; i++) {
				sumGamma[i] = 0;
				for (int k = 0; k < t-1; k++) {
					sumGamma[i] += gamma[k][i];
				}
				for (int j = 0; j < n; j++) {
					temp = 0;
					for (int k = 0; k < t-1; k++) {
						temp += xi[k][i][j];
					}
					aCounter[i][j] = temp;
					aDenominator[i][j] = sumGamma[i];
				}
			}
		}
	}
	
	public static void tests() {
		String[] states = {"sunny","rainy"};
		double[][] a = {{0.8,0.2},
						{0.4,0.6}};
		String[] v = {"happy", "sad"};
		double[][] b = {{0.8,0.2},
						{0.4,0.6}};
		double[] ini = {2.0/3.0,1.0/3.0};
		HMM hmm = new HMM(states, a, v, b, ini);
		int[] obs = {0,0,1,1,1,0};
		System.out.println(hmm.eval(obs));
		System.out.println(Arrays.toString(hmm.viterbi(obs)));
		int[] obs2 = {0,1,1,1,1,0};
		System.out.println(hmm.eval(obs2));
		System.out.println(Arrays.toString(hmm.viterbi(obs2)));
		
		String[] states2 = {"H", "L"};//sunny rainy
		double[][] a2 = {{0.5,0.5},
						{0.4,0.6}};
		String[] v2 = {"A", "C", "G", "T"};//A, C, G, T
		double[][] b2 = {{0.2, 0.3, 0.3, 0.2},
						{0.3, 0.2, 0.2, 0.3}};
		double[] ini2 = {0.5,0.5};
		HMM hmm2 = new HMM(states2, a2, v2, b2, ini2);
		int[] obs22 = {3,3,2,1};
		System.out.println(hmm2.eval(obs22));
		System.out.println(Arrays.toString(hmm2.viterbi(obs22)));
		
		String[] states3 = {"Healthy", "Feaver"};
		double[][] a3 = {{0.7,0.3},
						{0.4,0.6}};
		String[] v3 = {"Dizzy", "Cold", "Normal"};//Dizzy, Cold, Normal
		double[][] b3 = {{0.1, 0.4, 0.5},
						{0.6, 0.3, 0.1}};
		double[] ini3 = {0.6,0.4};
		HMM hmm3 = new HMM(states3, a3, v3, b3, ini3);
		int[] obs3 = {2,1,0};
		System.out.println("eval: "+hmm3.eval(obs3));
		System.out.println("evalForward: "+hmm3.evalForward(obs3));
		System.out.println("evalLog"+ hmm3.evalLog(obs3)+"/"+Math.pow(Math.E, hmm3.evalLog(obs3)));
		System.out.println(Arrays.toString(hmm3.viterbi(obs3)));
		
		double[][] a4 = {{0.6,0.2,0.0},
						 {0,0.5,0.5},
						 {0.0,0.0,1.0}};
		double[][] b4 = {{0.3, 0.4, 0.2,0.1},
						  {0.4,0.3, 0.2 ,0.1},
						  {0.3,0.3, 0.3,0.1}};					
		double[] ini4 = {1.0,0.0,0.0};
		HMM hmm4 = new HMM(a4,b4,ini4);//0 = No Eggs 1 = Eggs
		int[] z ={0,1,2,2};
		int[][] z2 = {{0,1,2,2},{2,1,0,2},{0,1,2,2},{0,1,2,2},{2,1,0,2},{2,1,0,2}};
		System.out.println("-----");
		for (int i = 0; i < 10; i++) {
			System.out.println(i);
			hmm4.baumWelch(z);
			System.out.println(Arrays.toString(hmm4.viterbi(z)));
			System.out.println(Arrays.deepToString(hmm4.a));
			System.out.println(Arrays.deepToString(hmm4.b));
			System.out.println(hmm4.evalLog(z));
		}
		hmm4 = new HMM(a4,b4,ini4);
		int[] zz = {2,1,0,2};
		for (int i = 0; i < 100; i++) {
			hmm4.baumWelch(z2);
			System.out.println(Arrays.toString(hmm4.viterbi(z)));
			System.out.println(Arrays.deepToString(hmm4.a));
			System.out.println(Arrays.deepToString(hmm4.b));
			System.out.println("eval: "+hmm4.eval(z));
			System.out.println("evalForward: "+hmm4.evalForward(z));
			System.out.println("evalLog"+ hmm4.evalLog(z)+"/"+Math.pow(Math.E, hmm4.evalLog(zz)));
			System.out.println("eval: "+hmm4.eval(zz));
			System.out.println("evalForward: "+hmm4.evalForward(zz));
			System.out.println("evalLog"+ hmm4.evalLog(zz)+"/"+Math.pow(Math.E, hmm4.evalLog(zz)));
		}
		
		
	}
//	public static void main(String args[]) {
//		HMM.tests();
//	}
}

