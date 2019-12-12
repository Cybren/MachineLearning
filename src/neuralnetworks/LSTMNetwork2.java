package neuralnetworks;

import java.io.IOException;
import java.nio.file.FileSystems;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;

public class LSTMNetwork2 {
	String path = "C:/Users/ben-g/Desktop/HW/";
	int[] layer;
	OutputLayer[] layers;
	double learningrate = 1;
	int times;
	
	public LSTMNetwork2(int[]layer,int times){
		this.layer = layer;
		this.times=times;
		
		layers = new OutputLayer[layer.length-1];
		System.out.println(Arrays.toString(layer));
		System.out.println(layers.length);
		for (int i = 0; i < layers.length-1; i++) {
			layers[i]= new HiddenLayer(layer[i],layer[i+1],learningrate,times);
		}
		layers[layers.length-1]=new OutputLayer(layer[layer.length-2],layer[layer.length-1],learningrate,times);
	}

	public double[][] feedForward(double[][]inputs){
//		System.out.println(times);
		for (int h = 0; h < times; h++) {
			layers[0].feedForward(inputs[h],h);
			for (int i = 1; i < layers.length; i++) {
				layers[i].feedForward(layers[i-1].outputs[h+1],h);
			}
//			System.out.println();
		}
		
		return layers[layers.length-1].outputs;
	}
	
	public void backProp(double [][]expected){
		for (int i = 0; i < times; i++) {
//			System.out.println("expected at time "+i +": "+Arrays.toString(expected[i]));
			layers[layers.length-1].backPropOutput(expected[i],i);
			for (int j = layers.length-2; j > -1; j--) {
					layers[j].backPropHidden(layers[j+1].gamma,layers[j+1].weights,i);
			}
//			System.out.println("----------------------------------------------------------------------------------------");
		}
		for (int j = 0; j < layers.length; j++) {
			layers[j].update();
			layers[j].reset();
		}
	}
	
	public void saveANN(){// alle Weights/ biases rein
		String out ="";
		List<String> lines= new LinkedList<String>(Arrays.asList(""));
		for (int i = 0; i < layer.length; i++) {
			out+=layer[i]+" ";
		}
		out+=System.lineSeparator();
		for (int i = 0; i < layers.length; i++) {
			for (int j = 0; j < layers[i].numberOfOutputs; j++) {
				for (int j2 = 0; j2 < layers[i].numberOfInputs; j2++) {
					out+=layers[i].weights[j][j2]+" ";
				}
				out+=System.lineSeparator();
			}
			out+=System.lineSeparator();
		}
		for (int i = 0; i < layers.length; i++) {
			for (int j = 0; j < layers[i].numberOfOutputs; j++) {
				out+=layers[i].bias[j]+" ";
			}
			out+=System.lineSeparator();
		}
		lines.add(out);
		Path p = FileSystems.getDefault().getPath(path,"Handwrite0055.ann");
		try {
			Files.delete(p);
			Files.write(p, lines,StandardOpenOption.CREATE_NEW,StandardOpenOption.WRITE);
		} catch (IOException e) {e.printStackTrace();}
	}
	
	public double[]zero(double[]in){
		for (int i = 0; i < in.length; i++) {
			in[i]=0;
		}
		return in;
	}
	public double[][]zero(double[][]in){
		for (int i = 0; i < in.length; i++) {
			for (int j = 0; j < in[i].length; j++) {
				in[i][j]=0;
			}
		}
		return in;
	}
	public double [][][]zero(double[][][]in){
		for (int i = 0; i < in.length; i++) {
			for (int j = 0; j < in[i].length; j++) {
				for (int j2 = 0; j2 < in[i][j].length; j2++) {
					in[i][j][j2]=0;
				}
			}
		}
		return in;
	}
	
	public double[] concetanate(double[]a,double []b) {
		double[] newArray = new double [a.length+b.length];
		for (int i = 0; i < a.length; i++) {
			newArray[i]=a[i];
		}
		for (int i = 0; i < b.length; i++) {
			newArray[i+a.length]=b[i];
		}
		
		return newArray;
		
	}
	
	public class OutputLayer{
		public String type ="Output";

		int numberOfInputs; // number of neurons in prev layer
		int numberOfOutputs;// number of nurons in curent layer
		double learningrate;
		double [][] outputs;//{time}{ouputs}
		double [][] z;
		double [][] inputs;//{time}{inputs}
		double [][] weights;
		double [][] weightsDelta;
		double [] gamma;
		double [] error;
		double [] bias;
		double [] biasDelta;
		int times;
		
		public OutputLayer(int numberOfInputs, int numberOfOutputs,double rate, int times){
			this.learningrate = rate;
			this.numberOfInputs = numberOfInputs;
			this.numberOfOutputs = numberOfOutputs;
			this.times = times;
			
			outputs = new double[times][numberOfOutputs];
			z = new double[times][numberOfOutputs];
			inputs = new double [times][numberOfInputs];
			
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
					weightsDelta[i][j]=0;
				}
				bias[i]= Math.random()-0.5;
				biasDelta[i]=0;
			}
		}
		public void reset(){
			for (int i = 0; i < weights.length; i++) {
				for (int j = 0; j < weights[i].length; j++) {
					weightsDelta[i][j]=0;
				}
				biasDelta[i]=0;
			}
		}
		
		public double[] feedForward(double []inputs,int time){
			this.inputs[time] = inputs;
//			System.out.println("Inputs of Outputlayer at time "+time+": "+Arrays.toString(inputs));
			
			for (int i = 0; i < numberOfOutputs; i++) {
				z[time][i] = 0;
				for (int j = 0; j < numberOfInputs; j++) {
					z[time][i]+= inputs[j]*weights[i][j];
				}
				z[time][i]+=bias[i];
				outputs[time][i] = sigmoid(z[time][i]);
			}
//			System.out.println("Outputs of Outputlayer at time "+time+": "+Arrays.toString(outputs[time]));
			return outputs[time];
		}
		
		public void backPropOutput(double[] expected,int time){
//			System.out.println("Expected in PropOut at time "+ time+ " : "+Arrays.toString(expected));
//			System.out.println("Output in PropOut at time "+ time+ " : "+Arrays.toString(outputs[time]));
			for (int i = 0; i < numberOfOutputs; i++) {
				error[i] = outputs[time][i]-expected[i];
				gamma[i] = error[i]*sigmoidPrime(z[time][i]);
			}
//			System.out.println("error in Output at time "+time+" : "+Arrays.toString(error));
//			System.out.println("gamma in Output at time "+time+" : "+Arrays.toString(gamma));
			for (int i = 0; i < numberOfOutputs; i++) {
				for (int j = 0; j < numberOfInputs; j++) {
					weightsDelta[i][j]+= gamma[i]*inputs[time][j];
//					System.out.println("weightsdelta "+i+","+j+" :"+weightsDelta[i][j]+"= gamma"+i+" "+gamma[i]+"* inputs "+time+", "+i+" "+inputs[time][j]);
				}
//				System.out.println("weightsDelta in Output at time "+time+" : "+Arrays.toString(weightsDelta[i]));
				biasDelta[i]+=gamma[i];
			}
//			System.out.println("biasDelta in Output at time "+ time+ " : "+Arrays.toString(biasDelta));
		}
		
		public void update(){
			for (int i = 0; i < weights.length; i++) {
				for (int j = 0; j < weights[i].length; j++) {
					weights[i][j] -= weightsDelta[i][j]*learningrate/times;
				}
				bias[i]-=biasDelta[i]*learningrate/times;
			}
		}
		
		public double sigmoid(double x){
			return(1/(1+Math.exp(-x)));
		}
		
		public double sigmoidPrime(double x){
			return(sigmoid(x)*(1-sigmoid(x)));
		}
		
		public double tanhPrime(double x){
			return 1-(Math.tanh(x)*Math.tanh(x));
		}

		public void setPast(double[] past) {
			System.out.println("parentcall setPast()");
		}
		public void backPropHidden(double[] gamma2,double [][] weightsForward, int i) {
			System.out.println("parentcall backPropHidden()");
		}
	}
	
	public class HiddenLayer extends OutputLayer{
		
		public String type ="Hidden";
		
		double[][] weightsF;
		double[][] weightsI;
		double[][] weightsA;
		double[][] weightsO;
		
		double [][] weightsFDelta;
		double [][] weightsIDelta;
		double [][] weightsADelta;
		double [][] weightsODelta;
		
		double [][] rF;//recurrents für forget;
		double [][] rFDelta;
		double [][] rI;//recurrents für input;
		double [][] rIDelta;
		double [][] rA;//recurrents für Add;
		double [][] rADelta;
		double [][] rO;//recurrents für Output;
		double [][] rODelta;
		
		double [] biasF;//bias für forget;
		double [] biasFDelta;
		double [] biasI;//bias für input;
		double [] biasIDelta;
		double [] biasA;//bias für Add;
		double [] biasADelta;
		double [] biasO;//bias für Output;
		double [] biasODelta;
		
		double [][] forget;
		double [][] ingate;
		double [][] addgate;
		double [][] outgate;
		
		double [][] past;
		double [][] cellstate;
		
		public HiddenLayer(int numberOfInputs, int numberOfOutputs, double rate,int times) {
			super(numberOfInputs, numberOfOutputs, rate,times);
			outputs = new double[times+1][numberOfOutputs];
			
			weightsF = new double [numberOfOutputs][numberOfInputs];
			weightsI= new double [numberOfOutputs][numberOfInputs];
			weightsA= new double [numberOfOutputs][numberOfInputs];
			weightsO= new double [numberOfOutputs][numberOfInputs];
			
			weightsFDelta= new double [numberOfOutputs][numberOfInputs];
			weightsIDelta= new double [numberOfOutputs][numberOfInputs];
			weightsADelta= new double [numberOfOutputs][numberOfInputs];
			weightsODelta= new double [numberOfOutputs][numberOfInputs];
			
			rF = new double [numberOfOutputs][numberOfOutputs];
			rFDelta = new double [numberOfOutputs][numberOfOutputs];
			rI = new double [numberOfOutputs][numberOfOutputs];
			rIDelta = new double [numberOfOutputs][numberOfOutputs];
			rA = new double [numberOfOutputs][numberOfOutputs];
			rADelta = new double [numberOfOutputs][numberOfOutputs];
			rO = new double [numberOfOutputs][numberOfOutputs];
			rODelta = new double [numberOfOutputs][numberOfOutputs];
			
			biasF = new double[numberOfOutputs];
			biasFDelta = new double[numberOfOutputs];
			biasI = new double[numberOfOutputs];
			biasIDelta = new double[numberOfOutputs];
			biasA = new double[numberOfOutputs];
			biasADelta = new double[numberOfOutputs];
			biasO = new double[numberOfOutputs];
			biasODelta = new double[numberOfOutputs];
			
			//entspricht z
			forget = new double[times][numberOfOutputs];
			ingate = new double[times][numberOfOutputs];
			addgate = new double[times][numberOfOutputs];
			outgate = new double[times][numberOfOutputs];
			
			cellstate = new double[times+1][numberOfOutputs];
			for (int i = 0; i < weightsF.length; i++) {
				for (int j = 0; j < weightsF[i].length; j++) {
					weightsF[i][j]=Math.random()-0.5;
					weightsI[i][j]=Math.random()-0.5;
					weightsA[i][j]=Math.random()-0.5;
					weightsO[i][j]=Math.random()-0.5;
					
					weightsFDelta[i][j]=0;
					weightsIDelta[i][j]=0;
					weightsADelta[i][j]=0;
					weightsODelta[i][j]=0;
					
				}
				biasF[i]= Math.random()-0.5;
				biasI[i]= Math.random()-0.5;
				biasA[i]= Math.random()-0.5;
				biasO[i]= Math.random()-0.5;
				
				biasFDelta[i]=0;
				biasIDelta[i]=0;
				biasADelta[i]=0;
				biasODelta[i]=0;
			}
			for (int i = 0; i < numberOfOutputs; i++) {
				for (int j = 0; j < numberOfOutputs; j++) {
					rF[i][j]=Math.random()-0.5;
					rI[i][j]=Math.random()-0.5;
					rA[i][j]=Math.random()-0.5;
					rO[i][j]=Math.random()-0.5;
					
					rFDelta[i][j]=0;
					rIDelta[i][j]=0;
					rADelta[i][j]=0;
					rODelta[i][j]=0;
				}
			outputs[0][i]=0;
			cellstate[0][i]=0;
			}
		}
		
		public void reset(){
			for (int i = 0; i < numberOfOutputs; i++) {
				for (int j = 0; j < numberOfInputs; j++) {
					weightsFDelta[i][j]=0;
					weightsIDelta[i][j]=0;
					weightsADelta[i][j]=0;
					weightsODelta[i][j]=0;
				}
				for (int j = 0; j < numberOfOutputs; j++) {
					rFDelta[i][j]=0;
					rIDelta[i][j]=0;
					rADelta[i][j]=0;
					rODelta[i][j]=0;
				}
				biasFDelta[i]=0;
				biasIDelta[i]=0;
				biasADelta[i]=0;
				biasODelta[i]=0;
			}
		}
		
		public double[] feedForward(double []inputs,int time){
			this.inputs[time] = inputs;
//			System.out.println("Inputs of Hiddenlayer at time "+time+": "+Arrays.toString(inputs));
//			System.out.println("Recurrentinputs of Hiddenlayer at time "+time+": "+Arrays.toString(outputs[time]));
			for (int i = 0; i < numberOfOutputs; i++) {//Iterate over outputs from this time
				forget[time][i] = 0;
				ingate[time][i] = 0;
				addgate[time][i] = 0;
				outgate[time][i] = 0;
				for (int j = 0; j < numberOfInputs; j++) {//Iterate over the inputs
					forget[time][i]+= inputs[j]*weightsF[i][j];
					ingate[time][i]+= inputs[j]*weightsI[i][j];
					addgate[time][i]+= inputs[j]*weightsA[i][j];
					outgate[time][i]+= inputs[j]*weightsO[i][j];
				}
				for (int j = 0; j < numberOfOutputs; j++) {//Iterate over the inputs
					forget[time][i]+= past[time][j]*rF[i][j];
					ingate[time][i]+= past[time][j]*rI[i][j];
					addgate[time][i]+= past[time][j]*rA[i][j];
					outgate[time][i]+= past[time][j]*rO[i][j];
				}
				forget[time][i]+= biasF[i];
				ingate[time][i]+= biasI[i];
				addgate[time][i]+= biasA[i];
				outgate[time][i]+= biasO[i];
				
				cellstate[time+1][i]=cellstate[time][i]*sigmoid(forget[time][i])+sigmoid(ingate[time][i])*Math.tanh(addgate[time][i]);
				outputs[time+1][i] = sigmoid(outgate[time][i])*Math.tanh(cellstate[time+1][i]);
			}
//			System.out.println("Outputs of Hiddenlayer at time "+time+": "+Arrays.toString(outputs[time+1]));
			return outputs[time+1];
		}
		public void backPropHidden(double[]gammaForward, double [][] weightsForward,int time){//returns the delta for weights, recurrentweights and bias from timestamp time
			//normal
			for (int i = 0; i < numberOfOutputs; i++) {
				gamma[i] = 0;
				for (int j = 0; j < gammaForward.length; j++) {
					gamma[i] += gammaForward[j]*weightsForward[j][i];//Vektor * Matrix= Vektor +
				}
				gamma[i]*= tanhPrime(z[time][i]);//Vektor * Vektor = Vektor *
			}
			for (int i = 0; i < numberOfOutputs; i++) {
				for (int j = 0; j < numberOfInputs; j++) {
					weightsDelta[i][j]+= gamma[i]*inputs[time][j];//Vektor * Vektor = Matrix
				}
				biasDelta[i]+= gamma[i];//Vektor = Vektor
			}
			
			//Time
			double [][][]temp= new double[time+1][numberOfOutputs][numberOfOutputs];
			for (int i = 0; i < temp.length; i++) {
				for (int j = 0; j < temp[i].length; j++) {
					for (int j2 = 0; j2 < temp[i][j].length; j2++) {
						temp[i][j][j2]=0;
					}
				}
			}
		}
		
		public void update(){
			//updateweights
			for (int i = 0; i < weights.length; i++) {
				for (int j = 0; j < weights[i].length; j++) {
					weightsF[i][j] -= weightsFDelta[i][j]*learningrate;
					weightsA[i][j] -= weightsADelta[i][j]*learningrate;
					weightsI[i][j] -= weightsIDelta[i][j]*learningrate;
					weightsO[i][j] -= weightsODelta[i][j]*learningrate;
				}
				biasF[i]-=biasFDelta[i];
				biasA[i]-=biasFDelta[i];
				biasI[i]-=biasFDelta[i];
				biasO[i]-=biasFDelta[i];
			}
			//update recursive
			for (int i = 0; i < rF.length; i++) {
				for (int j = 0; j < rF[i].length; j++) {
					rF[i][j] -= rFDelta[i][j]*learningrate;
					rA[i][j] -= rADelta[i][j]*learningrate;
					rI[i][j] -= rIDelta[i][j]*learningrate;
					rO[i][j] -= rODelta[i][j]*learningrate;
				}
			}
//			System.out.println(check(rF));
//			System.out.println(check(rA));
//			System.out.println(check(rI));
//			System.out.println(check(rO));
//			System.out.println();
//			
//			System.out.println(check(weightsA));
//			System.out.println(check(weightsF));
//			System.out.println(check(weightsI));
//			System.out.println(check(weightsO));
			
			
		}
		
		public void setPast(double[] past) {
			outputs[0]=past;
		}
		
		public double sigmoid(double x){
			return(1/(1+Math.exp(-x)));
		}
		
		public double tanhPrime(double x){
			return 1-(Math.tanh(x)*Math.tanh(x));
		}
		
		public double sigmoidPrime(double x){
			return(sigmoid(x)*(1-sigmoid(x)));
		}
	}
	
	public static void main(String[] args) {
		int[] a = {2,100,200,4};
		double [][]input = {{1,0,0,0},{0,1,0,0},{0,0,1,0},{0,0,1,0}};
		double [][]expected = {{0,1,0,0},{0,0,1,0},{0,0,1,0},{0,0,0,1}};
		LSTMNetwork2 rn2 = new LSTMNetwork2(a, 4);
		System.out.println("input: "+Arrays.toString(input));
		double[][] test = rn2.feedForward(input);
		for (int i = 0; i < test.length; i++) {
			System.out.println("output"+Arrays.toString(test[i]));	
		}
		System.out.println();
		rn2.layers[0].setPast(rn2.layers[0].outputs[rn2.layers[0].outputs.length-1]);
		System.out.println("set Past to: "+Arrays.toString(rn2.layers[0].outputs[0]));
		test = rn2.feedForward(input);
		for (int i = 0; i < test.length; i++) {
			System.out.println("output"+Arrays.toString(test[i]));	
		}
		System.out.println();
		System.out.println("---------------------------------------------------------------------------------------");
		for (int i = 0; i < 2000; i++) {
			rn2.feedForward(input);
			rn2.backProp(expected);
		}
		test = rn2.feedForward(input);
		for (int j = 0; j < test.length; j++) {
			System.out.println("output test"+Arrays.toString(test[j]));	
		}
		
	}

}
