package statisticalmodels;

public class ForwardBackwardCalculator{
	public double[][] forward;
	public double[][] forwardUnscaled;
	public double[] scales;
	public double[][] backward;
	public double[][] backwardUnscaled;
	private HMM hmm;
	private int[] obs;
	private int t;
	public ForwardBackwardCalculator(HMM hmm, int[] observations) {
		this.hmm = hmm;
		this.obs = observations.clone();
		t = observations.length;
		this.forward = setForward();
		this.backward = setBackward();
	}
	
	
	private double[][] setForward(){
		double forward[][] = new double[t][hmm.getN()];
		forwardUnscaled = new double[t][hmm.getN()];
		this.scales = new double[t];
		double sum = 0;
		//initilize: calculate for every forward i the probability P(O1|q1 = Si)
		for (int i = 0; i < hmm.getN(); i++) {
			forward[0][i]= hmm.getIni()[i]* hmm.getB()[i][obs[0]];
			sum += forward[0][i];
		}
		scales[0] = 1/sum;
		
		for (int i = 0; i <  hmm.getN(); i++) {
			forward[0][i] *= scales[0];
		}
		
		//induction: probability that state Sj reached * probability that state Si emmits observation k
		double temp;
		for (int k = 1; k < t; k++) {
			sum = 0;
			for (int j = 0; j <  hmm.getN(); j++) {
				temp = 0;
				
				for (int i = 0; i <  hmm.getN(); i++) {//probability that State Sj is reached
					temp += forward[k-1][i]* hmm.getA()[i][j];
				}
				forward[k][j] = temp* hmm.getB()[j][obs[k]];
				sum += forward[k][j];
			}
			scales[k] = 1/sum;
			
			for (int i = 0; i <  hmm.getN(); i++) {
				forwardUnscaled[k][i] = forward[k][i];
				forward[k][i] *= scales[k];
			}
		}
		
		return forward;
	}
	
	private double[][] setBackward(){
		if(scales == null) {
			setForward();
		}
		double backward[][] = new double[t][hmm.getN()];
		backwardUnscaled = new double[t][hmm.getN()];

		//initialization
		for (int i = 0; i <  hmm.getN(); i++) {
			backward[t-1][i] = 1*scales[t-1];
		}
		//induction
		for (int k = t-2; k > -1; k--) {
			for (int i = 0; i <  hmm.getN(); i++) {
				backward[k][i] = 0;
				for (int j = 0; j <  hmm.getN(); j++) {
					backward[k][i] +=  hmm.getA()[i][j]* hmm.getB()[j][obs[k+1]]*backward[k+1][j];
				}
				backwardUnscaled[k][i] = backward[k][i];
				backward[k][i] *= scales[k];
			}
		}
		return backward;
	}
}
