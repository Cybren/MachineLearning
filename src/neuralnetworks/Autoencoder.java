package neuralnetworks;

public class Autoencoder {
	String path = "C:/Users/Cybren/Desktop/HW/";
	int[] layer;
	Layer[] layers;
	double learningrate = 0.00095;
	
	public Autoencoder(int[]layer){// nur den Input, die Hidden und den Codevektor übergeben, der Rest wird symmetrisch erstellt
		this.layer = layer;
		layers = new Layer[layer.length*2-2];// für 2: 2 3: 4 4: 6 5: 
		System.out.println(layers.length);
		for (int i = 0; i < layer.length-1; i++) {
			layers[i]= new Layer(layer[i],layer[i+1],learningrate);
			System.out.println(layers[i]);
		}
		for (int i = 0; i < layer.length-1; i++) {
			int x = layer.length-i-1;
			layers[layer.length+i-1]=new Layer(layer[x],layer[x-1],learningrate);
		}
	}

	public double[] feedForward(double[]inputs){
	
		layers[0].feedForward(inputs);
		
		for (int i = 1; i < layers.length; i++) {
			layers[i].feedForward(layers[i-1].outputs);
		}
		return layers[layers.length-1].outputs;
	}
	
	public void backProp(double []expected){
		for (int i = layers.length-1; i > -1; i--) {
			if(i==layers.length-1){
				layers[i].backPropOutput(expected);
			}else{
				layers[i].backPropHidden(layers[i+1].gamma, layers[i+1].weights);
			}
		}
		for (int i = 0; i < layers.length; i++) {
			layers[i].update();
		}
	}
	
	public [] getDecoder() {
		
	}
	
	public class Layer{
		
		int numberOfInputs; // number of neurons in prev layer
		int numberOfOutputs;// number of nurons in curent layer
		double learningrate;
		double [] outputs;
		double [] inputs;
		double [][] weights;
		double [][] weightsDelta;
		double [] gamma;
		double [] error;
		double [] bias;
		double [] biasDelta;
		
		public Layer(int numberOfInputs, int numberOfOutputs,double rate){
			this.learningrate = rate;
			this.numberOfInputs = numberOfInputs;
			this.numberOfOutputs = numberOfOutputs;
			
			outputs = new double[numberOfOutputs];
			inputs = new double [numberOfInputs];
			
			weights = new double[numberOfOutputs][numberOfInputs];
			weightsDelta = new double[numberOfOutputs][numberOfInputs];
			
			gamma = new double[numberOfOutputs];
			error = new double[numberOfOutputs];
			
			bias = new double[numberOfOutputs];
			biasDelta = new double[numberOfOutputs];
			
			ini();
		}
		
		public void ini(){
			for (int i = 0; i < weights.length; i++) {
				for (int j = 0; j < weights[i].length; j++) {
					weights[i][j] = Math.random()-0.5;
				}
				bias[i]= Math.random()-0.5;
			}
		}
		
		public double[] feedForward(double []inputs){
			this.inputs = inputs;
			
			for (int i = 0; i < numberOfOutputs; i++) {
				outputs[i] = 0;
				for (int j = 0; j < numberOfInputs; j++) {
					outputs[i]+= inputs[j]*weights[i][j];
				}
				outputs[i]+=bias[i];
				outputs[i] = Math.tanh(outputs[i]);
			}
			
			return outputs;
		}
		
		public void backPropOutput(double[] expected){
			for (int i = 0; i < numberOfOutputs; i++) {
				error[i] = outputs[i]-expected[i];
				gamma[i] = error[i]*tanhPrime(outputs[i]);
			}
			for (int i = 0; i < weightsDelta.length; i++) {
				for (int j = 0; j < weightsDelta[i].length; j++) {
					weightsDelta[i][j]= gamma[i]*inputs[j];
				}
				biasDelta[i]=gamma[i];
			}
		}
		public void backPropHidden(double[]gammaForward, double [][] weightsForward){
			for (int i = 0; i < numberOfOutputs; i++) {
				gamma[i] = 0;
				for (int j = 0; j < gammaForward.length; j++) {
					gamma[i] += gammaForward[j]*weightsForward[j][i];
				}
				gamma[i]*= tanhPrime(outputs[i]);
			}
			for (int i = 0; i < numberOfOutputs; i++) {
				for (int j = 0; j < numberOfInputs; j++) {
					weightsDelta[i][j]= gamma[i]*inputs[j];
				}
				biasDelta[i] = gamma[i];
			}
		}
		
		public void update(){
			for (int i = 0; i < weights.length; i++) {
				for (int j = 0; j < weights[i].length; j++) {
					weights[i][j] -= weightsDelta[i][j]*learningrate;
				}
				bias[i]-=biasDelta[i];
			}
		}
		
		public double sigmoid(double x){
			return(1/(1+Math.exp(-x)));
		}
		
		public double tanhPrime(double x){
			return 1-(x*x);
		}
		
		public double relu(double x) {
			if(x<0) {
				return 0;
			}else {
				return x;
			}
		}
		
		public double reluPrime(double x) {
			if(x<0) {
				return 0;
			}else {
				return 1;
			}
		}
		
	}
	
	public static void main(String[] args) {
		
		int layers[] = {25,20,12};
		Autoencoder a = new Autoencoder(layers);
		for (int i = 0; i < a.layers.length; i++) {
			System.out.println(a.layers[i].numberOfInputs);
			System.out.println(a.layers[i].numberOfOutputs);
			
		}
	}

}
