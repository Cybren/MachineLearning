package neuralnetworks;

import java.io.IOException;
import java.nio.file.FileSystems;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;

public class LSTMNetwork {//Cellstate safen und an next Batch weiter geben
	double max = 0;
	double dx =0;
	String path = "C:/Users/Cybren/Desktop/HW/";
	int[] layer;//structure {{5},{100},{2}}
	OutputLayer[] layers;//layers 2 hidden 1 output
	double learningrate = 0.1;
	int times;// how many times the Network does remember
	
	public LSTMNetwork(int[]layer,int times,double learningrate){
		this.layer = layer;
		this.times=times;
		this.learningrate = learningrate;
		layers = new OutputLayer[layer.length-1];
		for (int i = 0; i < layers.length-1; i++) {
			layers[i]= new HiddenLayer(layer[i],layer[i+1],learningrate,times);
		}
		layers[layers.length-1]= new OutputLayer(layer[layer.length-2],layer[layer.length-1],learningrate,times);
	}
	
	public double[][] feedForward(double[][]inputs){
		for (int i = 0; i < times; i++) {
			layers[0].feedForward(inputs[i],i);
			for (int j = 1; j < layers.length; j++) {
				layers[j].feedForward(layers[j-1].outputs[i],i);
			}
		}
		return layers[layers.length-1].outputs;
	}
	
	public void backProp(double [][]expected){
		for (int i = times-1; i > -1; i--) {
			layers[layers.length-1].backProp(expected[i],i);//outputlayer
//			System.out.println(check(layers[layers.length-1].weights));
//			System.out.println(check(layers[layers.length-1].bias));
//			System.out.println(check(layers[layers.length-1].gamma));
			//hiddenlayer under outputlayer
			layers[layers.length-2].backProp(layers[layers.length-1].gamma,layers[layers.length-1].weights,layers[layers.length-1].gamma,layers[layers.length-1].weights,layers[layers.length-1].gamma,layers[layers.length-1].weights,layers[layers.length-1].gamma,layers[layers.length-1].weights,i);
//			layers[layers.length-2].backPropTime(layers[layers.length-1].gamma,layers[layers.length-1].gamma,layers[layers.length-1].gamma,layers[layers.length-1].gamma,i);
			//the hiddenlayers below the last hiddenlayer
			for (int j = layers.length-3; j > -1; j--) {
				layers[j].backProp(layers[j+1].gammaF,layers[j+1].weightsF,layers[j+1].gammaA,layers[j+1].weightsA,layers[j+1].gammaI,layers[j+1].weightsI,layers[j+1].gammaO,layers[j+1].weightsO,i);
//				layers[j].backPropTime(layers[j].gammaFR,layers[j].gammaAR,layers[j].gammaIR,layers[j].gammaOR,i);
			}
		}
		for (int i = 0; i < layers.length; i++) {
			layers[i].update();
		}
	}
	
	public void saveANN(){//doesnt work yet
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
		Path p = FileSystems.getDefault().getPath(path,"Handwrite0055.lstm");
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
	
	public class HiddenLayer extends OutputLayer{
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
		double [][] add;
		double [][] outgate;
		
		double [][] past;
		double [][] cellstate;
		
		public HiddenLayer(int numberOfInputs, int numberOfOutputs,double rate,int times){
			super(numberOfInputs,numberOfOutputs,rate,times);
			type = "hidden";
			weightsF = new double [numberOfOutputs][numberOfInputs];
			weightsFDelta = new double [numberOfOutputs][numberOfInputs];
			weightsI = new double [numberOfOutputs][numberOfInputs];
			weightsIDelta = new double [numberOfOutputs][numberOfInputs];
			weightsA = new double [numberOfOutputs][numberOfInputs];
			weightsADelta = new double [numberOfOutputs][numberOfInputs];
			weightsO = new double [numberOfOutputs][numberOfInputs];
			weightsODelta = new double [numberOfOutputs][numberOfInputs];
			
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
			
			gammaF=new double[numberOfOutputs];
			gammaI=new double[numberOfOutputs];
			gammaA=new double[numberOfOutputs];
			gammaO=new double[numberOfOutputs];
			
			gammaFR=new double[numberOfOutputs];
			gammaIR=new double[numberOfOutputs];
			gammaAR=new double[numberOfOutputs];
			gammaOR=new double[numberOfOutputs];
			
			forget = new double[times][numberOfOutputs];
			ingate = new double[times][numberOfOutputs];
			add = new double[times][numberOfOutputs];
			outgate = new double[times][numberOfOutputs];
			
			past = new double[times+1][numberOfOutputs];
			cellstate = new double[times+1][numberOfOutputs];//warum +1, weil 0 der intial ist und danach immer time+1 für speicher
			iniH();
		}
		
		public void iniH(){
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
				
				past[0][i]=1;
				cellstate[0][i]=1;
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
			}
		}
		
		public HiddenLayer(int numberOfInputs, int numberOfOutputs,double rate,int times, double[]cellstate, double[] past){//to transfer the last cellstate and output to the present
			super(numberOfInputs,numberOfOutputs,rate,times);
			
			weightsF = new double [numberOfOutputs][numberOfInputs+numberOfOutputs];
			weightsFDelta = new double [numberOfOutputs][numberOfInputs+numberOfOutputs];
			weightsI = new double [numberOfOutputs][numberOfInputs+numberOfOutputs];
			weightsIDelta = new double [numberOfOutputs][numberOfInputs+numberOfOutputs];
			weightsA = new double [numberOfOutputs][numberOfInputs+numberOfOutputs];
			weightsADelta = new double [numberOfOutputs][numberOfInputs+numberOfOutputs];
			weightsO = new double [numberOfOutputs][numberOfInputs+numberOfOutputs];
			weightsODelta = new double [numberOfOutputs][numberOfInputs+numberOfOutputs];
			
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
			
			gammaF=new double[numberOfOutputs];
			gammaI=new double[numberOfOutputs];
			gammaA=new double[numberOfOutputs];
			gammaO=new double[numberOfOutputs];
			
			gammaFR=new double[numberOfOutputs];
			gammaIR=new double[numberOfOutputs];
			gammaAR=new double[numberOfOutputs];
			gammaOR=new double[numberOfOutputs];
			
			forget = new double[times][numberOfOutputs];
			ingate = new double[times][numberOfOutputs];
			add = new double[times][numberOfOutputs];
			outgate = new double[times][numberOfOutputs];
			
			this.past = new double[times+1][numberOfOutputs];
			this.cellstate = new double[times+1][numberOfOutputs];
			iniH(cellstate,past);
		}
		
		public void iniH(double[]cellstate,double[]past){//to set the Cellstate
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
			}
			this.past[0]=past;
			this.cellstate[0]=cellstate;
		}
		
		
		public double[] feedForward(double []inputs,int time){
			this.inputs[time]=inputs;
			
			for (int i = 0; i < numberOfOutputs; i++) {//start feedforward
				forget[time][i]=0;
				ingate[time][i]=0;
				add[time][i]=0;
				outgate[time][i]=0;
//				System.out.println("checkWF: "+check(weightsF));
//				System.out.println("checkWA: "+check(weightsA));
//				System.out.println("checkWI: "+check(weightsI));
//				System.out.println("checkWO: "+check(weightsO));
				for (int j = 0; j < inputs.length; j++) {
					forget [time][i]+=inputs[j]*weightsF[i][j];
					ingate [time][i]+=inputs[j]*weightsI[i][j];
					add    [time][i]+=inputs[j]*weightsA[i][j];
					if(add[time][i]>1000) {
//						System.out.println(i+" "+j+" "+time);
//						System.out.println(add[time][i]+"+="+inputs[j]+"*"+weightsA[i][j]+"+"+past[time][j]+"*"+rA[i][j]);
//						System.out.println("stop");
					}
					outgate[time][i]+=inputs[j]*weightsO[i][j];
				}
				for (int j = 0; j < numberOfOutputs; j++) {
					forget[time][i]+=past[time][j]*rF[i][j];
					add   [time][i]+=past[time][j]*rA[i][j];
					ingate[time][i]+=past[time][j]*rI[i][j];
					outgate[time][i]+=past[time][j]*rO[i][j];
				}
				forget[time][i]+=biasF[i];
				ingate[time][i]+=biasI[i];
				add[time][i]+=biasA[i];
				outgate[time][i]+=biasO[i];
				
				cellstate[time+1][i]=cellstate[time][i]*sigmoid(forget[time][i])+(sigmoid(ingate[time][i])*Math.tanh(add[time][i]));
				
				outputs[time][i]=sigmoid(outgate[time][i])*Math.tanh(cellstate[time+1][i]);
				if(outputs[time][i]!=outputs[time][i]) {
//					System.out.println("Fehler"+time+";"+i);
//					System.out.println("check cellstate: "+check(cellstate));
//					System.out.println("check forget: "+check(forget));
//					System.out.println("check ingate: "+check(ingate));
//					System.out.println("check add: "+check(add));
//					System.out.println("check outgate: "+check(outgate));
					
				}
				past[time+1][i]=outputs[time][i];
			}
			return outputs[time];
		}
		public void backProp(double[]gammaFForward,double[][]weightsFForward,double[]gammaAForward,double[][]weightsAForward,double[]gammaIForward,double[][]weightsIForward,double[]gammaOForward,double[][]weightsOForward,int time){
			for (int i = 0; i < numberOfOutputs; i++) {
				gammaF[i] = 0;
				gammaA[i] = 0;
				gammaI[i] = 0;
				gammaO[i] = 0;
				for (int j = 0; j < weightsFForward.length; j++) {
					gammaF[i] += gammaFForward[j]*weightsFForward[j][i];
					gammaA[i] += gammaAForward[j]*weightsAForward[j][i];
					gammaI[i] += gammaIForward[j]*weightsIForward[j][i];
					gammaO[i] += gammaOForward[j]*weightsOForward[j][i];
				}
				
				gammaF[i]*= tanhPrime(cellstate[time+1][i])*sigmoid(outgate[time][i])*sigmoidPrime(forget[time][i])*cellstate[time][i];
				gammaA[i]*= tanhPrime(cellstate[time+1][i])*sigmoid(outgate[time][i])*sigmoid(ingate[time][i])*tanhPrime(add[time][i]);
				
				if(gammaA[i]!=gammaA[i]||Double.isInfinite(gammaA[i])||Double.isNaN(gammaA[i]*5)) {
//					System.out.println(gammaA[i]+"*="+ tanhPrime(cellstate[time+1][i])+"*"+sigmoid(outgate[time][i])+"*"+sigmoid(ingate[time][i])+"*"+tanhPrime(add[time][i]));
//					System.out.println("stpo");
				}
				gammaI[i]*= tanhPrime(cellstate[time+1][i])*sigmoid(outgate[time][i])*Math.tanh(add[time][i])*sigmoidPrime(ingate[time][i]);
				gammaO[i]*= Math.tanh(cellstate[time+1][i])*sigmoidPrime(outgate[time][i]);
			}
			if(time==times-1) {
				gammaFR=gammaF.clone();
				gammaAR=gammaA.clone();
				gammaIR=gammaI.clone();
				gammaOR=gammaO.clone();
			}else{
				double [] gammaFRnew= new double[gammaFR.length];
				double [] gammaARnew=new double[gammaAR.length];
				double [] gammaIRnew=new double[gammaIR.length];
				double [] gammaORnew=new double[gammaOR.length];
				gammaFRnew = zero(gammaFRnew);
				gammaARnew = zero(gammaARnew);
				gammaIRnew = zero(gammaIRnew);
				gammaORnew = zero(gammaORnew);
				for (int j = 0; j < numberOfOutputs; j++) {
					for (int k = 0; k < numberOfOutputs; k++) {
						gammaFRnew[j] += gammaFR[k]*rF[k][j];
						gammaARnew[j] += gammaAR[k]*rA[k][j];
//						System.out.println(gammaARnew[j] +"+="+ gammaAR[k]+"*"+rA[k][j]);
//						if(gammaARnew[j]!=gammaARnew[j]||Double.isInfinite(gammaARnew[j])||Double.isNaN(gammaARnew[j]*5)) {
//							System.out.println(gammaARnew[j] +"+="+ gammaAR[j]+"*"+rA[j][j]);
//							System.out.println("stpo");
//						}
						gammaIRnew[j] += gammaIR[k]*rI[k][j];
						gammaORnew[j] += gammaOR[k]*rO[k][j];
					}
					gammaFR[j]=gammaFRnew[j]*sigmoid(outgate[time+1][j])*tanhPrime(cellstate[time+2][j])*cellstate[time+1][j]*sigmoidPrime(forget[time+1][j]);
					gammaAR[j]=gammaARnew[j]*sigmoid(outgate[time+1][j])*tanhPrime(cellstate[time+2][j])*sigmoid(ingate[time+1][j])*tanhPrime(add[time+1][j]);
					if(gammaAR[j]!=gammaAR[j]||Double.isInfinite(gammaAR[j])||Double.isNaN(gammaAR[j]*5)) {
//						System.out.println(gammaAR[j]+"="+gammaARnew[j]+"*"+sigmoid(outgate[time+1][j])+"*"+tanhPrime(cellstate[time+2][j])+"*"+sigmoid(ingate[time+1][j])+"*"+tanhPrime(add[time+1][j]));
//						System.out.println(j+" "+time);
//						System.out.println("stop");
					}
					gammaIR[j]=gammaIRnew[j]*sigmoid(outgate[time+1][j])*tanhPrime(cellstate[time+2][j])*Math.tanh(add[time][j])*sigmoidPrime(ingate[time+1][j]);
					gammaOR[j]=gammaORnew[j]*Math.tanh(cellstate[time+2][j])*sigmoidPrime(outgate[time+1][j]);
				}
			}
			for (int i = 0; i < numberOfOutputs; i++) {
				for (int j = 0; j < numberOfOutputs; j++) {
					rFDelta[i][j]+=gammaFR[i]*past[time][j];
					rADelta[i][j]+=gammaAR[i]*past[time][j];
					rIDelta[i][j]+=gammaIR[i]*past[time][j];
					rODelta[i][j]+=gammaOR[i]*past[time][j];
				}
			}
			
			for (int i = 0; i < numberOfOutputs; i++) {
				for (int j = 0; j < numberOfInputs; j++) {
					weightsFDelta[i][j]+= gammaF[i]*inputs[time][j];
					weightsADelta[i][j]+= gammaA[i]*inputs[time][j];
					weightsIDelta[i][j]+= gammaI[i]*inputs[time][j];
					weightsODelta[i][j]+= gammaO[i]*inputs[time][j];
					if(weightsADelta[i][j]!=weightsADelta[i][j]) {
//						System.out.println(i+" "+j);
//						System.out.println("stop");
					}
				}
				biasFDelta[i]+= gammaF[i];
				biasADelta[i]+= gammaA[i];
				biasIDelta[i]+= gammaI[i];
				biasODelta[i]+= gammaO[i];
			}
		}
		
		public void backPropTime(double[] gammaFRForward,double[] gammaARForward,double[] gammaIRForward,double[] gammaORForward,int time) {
			gammaFR=zero(gammaFR);
			gammaAR=zero(gammaAR);
			gammaIR=zero(gammaIR);
			gammaOR=zero(gammaOR);
			for (int h = time; h < times; h++) {
				for (int i = 0; i < numberOfOutputs; i++) {
					for (int j = 0; j < gammaFRForward.length; j++) {
						gammaFR[i]+=gammaFRForward[j]*rF[j][i];
						gammaAR[i]+=gammaARForward[j]*rF[j][i];
						gammaIR[j]+=gammaIRForward[j]*rF[j][i];
						gammaOR[j]+=gammaORForward[j]*rF[j][i];
					}
					gammaFR[i]*= tanhPrime(cellstate[time+1][i])*sigmoid(outgate[time][i])*sigmoidPrime(forget[time][i])*cellstate[time][i];
					gammaAR[i]*= tanhPrime(cellstate[time+1][i])*sigmoid(outgate[time][i])*sigmoid(ingate[time][i])*tanhPrime(add[time][i]);
					gammaIR[i]*= tanhPrime(cellstate[time+1][i])*sigmoid(outgate[time][i])*Math.tanh(add[time][i])*sigmoidPrime(ingate[time][i]);
					gammaOR[i]*= Math.tanh(cellstate[time+1][i])*sigmoidPrime(outgate[time][i]);
				}
			}
			
			for (int i = 0; i < numberOfOutputs; i++) {
				for (int j = 0; j < gammaFRForward.length; j++) {
					gammaFR[i]+=gammaFRForward[j]*rF[j][i];
					gammaAR[i]+=gammaARForward[j]*rF[j][i];
					gammaIR[j]+=gammaIRForward[j]*rF[j][i];
					gammaOR[j]+=gammaORForward[j]*rF[j][i];
				}
				gammaFR[i]*= tanhPrime(cellstate[time+1][i])*sigmoid(outgate[time][i])*sigmoidPrime(forget[time][i])*cellstate[time][i];
				gammaAR[i]*= tanhPrime(cellstate[time+1][i])*sigmoid(outgate[time][i])*sigmoid(ingate[time][i])*tanhPrime(add[time][i]);
				gammaIR[i]*= tanhPrime(cellstate[time+1][i])*sigmoid(outgate[time][i])*Math.tanh(add[time][i])*sigmoidPrime(ingate[time][i]);
				gammaOR[i]*= Math.tanh(cellstate[time+1][i])*sigmoidPrime(outgate[time][i]);
			}
			for (int i = 0; i < numberOfOutputs; i++) {
				for (int j = 0; j < numberOfOutputs; j++) {
					rFDelta[i][j]= gammaFR[i]*past[time][j];
					rADelta[i][j]= gammaAR[i]*past[time][j];
					rIDelta[i][j]= gammaIR[i]*past[time][j];
					rODelta[i][j]= gammaOR[i]*past[time][j];
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
	}
public class OutputLayer{
		String type = "output";
		public double[] gammaFR;
		public double[] gammaAR;
		public double[] gammaIR;
		public double[] gammaOR;
		public double[][] weightsO;
		public double[] gammaO;
		public double[][] weightsI;
		public double[] gammaI;
		public double[][] weightsA;
		public double[] gammaA;
		public double[][] weightsF;
		public double[] gammaF;
		
		int numberOfInputs; // number of neurons in prev layer
		int numberOfOutputs;// number of neurons in curent layer
		double learningrate;
		double [][] outputs;
		double [][] inputs;
		double [][] weights;
		double [][] weightsDelta;
		double [] gamma;
		double [] error;
		double [] bias;
		double [] biasDelta;
		
		public OutputLayer(int numberOfInputs, int numberOfOutputs,double rate,int times){
			this.learningrate = rate;
			this.numberOfInputs = numberOfInputs;
			this.numberOfOutputs = numberOfOutputs;
			
			outputs = new double[times][numberOfOutputs];
			inputs = new double [times][numberOfInputs];
			
			weights = new double[numberOfOutputs][numberOfInputs];
			weightsDelta = new double[numberOfOutputs][numberOfInputs];
			
			gamma = new double[numberOfOutputs];
			error = new double[numberOfOutputs];
			
			bias = new double[numberOfOutputs];
			biasDelta = new double[numberOfOutputs];
			
			ini();
		}
		
		public void backPropTime(double[] gammaFR2, double[] gammaAR2, double[] gammaIR2, double[] gammaOR2, int i) {
			System.out.println("This is the prototype"+i);	
		}

		public void backProp(double[] gamma2, double[][] weights2, double[] gamma3, double[][] weights3,double[] gamma4, double[][] weights4, double[] gamma5, double[][] weights5, int i) {
			System.out.println("This is the prototype"+i);
		}

		public void ini(){
			for (int i = 0; i < weights.length; i++) {
				for (int j = 0; j < weights[i].length; j++) {
					weights[i][j] = Math.random()-0.5;
					weightsDelta[i][j] =0;
				}
				bias[i]= Math.random()-0.5;
				biasDelta[i]=0;
			}
		}
		
		public double[] feedForward(double []inputs,int time){
			this.inputs[time] = inputs;
			
			for (int i = 0; i < numberOfOutputs; i++) {
				outputs[time][i] = 0;
//				System.out.println("check: "+check(weights)+" ;"+time);
				for (int j = 0; j < numberOfInputs; j++) {
					if(inputs[j]!=inputs[j]) {
//						System.out.println("stop"+j);
					}
					if(outputs[time][i]==outputs[time][i]) {
//						System.out.println("abcd"+outputs[time][i]+"+="+ inputs[j]+"*"+weights[i][j]+";"+j);
					}
					outputs[time][i]+= inputs[j]*weights[i][j];
				}
				outputs[time][i]+=bias[i];
//				System.out.println("outputs: "+Arrays.toString(outputs[time]));
			}
			outputs[time] = softmax(outputs[time]);
//			for (int i = 0; i < outputs[time].length; i++) {
//				if(outputs[time][i])
//			}
			return outputs[time];
		}
		
		public void backProp(double[] expected,int time){
			for (int i = 0; i < numberOfOutputs; i++) {
//				System.out.println(outputs[time][i]+"-"+expected[i]);
				error[i] = outputs[time][i]-expected[i];
//				System.out.println(error[i]+"="+outputs[time][i]+"-"+expected[i]);
				gamma[i] = error[i];
			}
			for (int i = 0; i < weightsDelta.length; i++) {
				for (int j = 0; j < weightsDelta[i].length; j++) {
					weightsDelta[i][j]+= error[i]*inputs[time][j];
					if(weightsDelta[i][j]!=weightsDelta[i][j]) {
//						System.out.println("weightsD: "+weightsDelta[i][j]+"+="+ error[i]+"*"+inputs[time][j]);
					}
				}
				biasDelta[i]+=error[i];
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
			double out = 1/(1+Math.exp(-x));
			if(1+Math.exp(-x)<max) {
				max=1+Math.exp(-x);
				dx=x;
			}
			return out;
		}
		
		public double sigmoidPrime(double x){
			return(sigmoid(x)*(1-sigmoid(x)));
		}
		
		public double tanhPrime(double x){
			return 1-(Math.tanh(x)*Math.tanh(x));
		}
		public double[] softmax (double[] x) {
			double [] back = new double[x.length];
			double sum =0;
			for (int i = 0; i < x.length; i++) {
				sum+=Math.exp(x[i]);
			}
			if(sum==0||sum!=sum) {
				sum=0.1;
			}
//			System.out.println("sum: "+sum);
//			System.out.println("x: "+Arrays.toString(x));
			for (int i = 0; i < back.length; i++) {
				back[i]=Math.exp(x[i])/sum;
			}
//			System.out.println("back"+Arrays.toString(back));
			return back;
		}
	}
	public boolean check(double[]a) {
		for (int i = 0; i < a.length; i++) {
			if(a[i]!=a[i]||a[i]>1000) {
				return true;
			}
		}
		return false;
	}
	public boolean check(double[][]a) {
		for (int i = 0; i < a.length; i++) {
			for (int j = 0; j < a[i].length; j++) {
				if(a[i][j]!=a[i][j]||a[i][j]>1000) {
					return true;
				}
			}
		}
		return false;
	}
	public boolean check(double[][][]a) {
		for (int i = 0; i < a.length; i++) {
			for (int j = 0; j < a[i].length; j++) {
				for (int j2 = 0; j2 < a[i][j].length; j2++) {
					if(a[i][j][j2]!=a[i][j][j2]||a[i][j][j2]>1000) {
						return true;
					}
				}
			}
		}
		return false;
	}
	public static void main(String[]args){
		//h:0 e:1 l:2 o:3;
		int []layer = {4,500,500,250,4};
		double [][]input = {{1,0,0,0},{0,1,0,0},{0,0,1,0},{0,0,1,0},{0,0,1,0}};
		double [][]expected = {{0,1,0,0},{0,0,1,0},{0,0,1,0},{0,0,0,1},{0,1,0,0}};
		LSTMNetwork r = new LSTMNetwork(layer, input.length,0.02);
		double [][]out= {{}};
		for (int i = 0; i < 1000; i++) {
				out=r.feedForward(input);
				r.backProp(expected);
			for (int j = 0; j < r.layers.length; j++) {
				r.layers[j].update();
			}
			for (int h = 0; h < out.length; h++) {
				System.out.println(Arrays.toString(out[h]));
			}
			System.out.println();
		}
			out=r.feedForward(input);
			for (int i = 0; i < out.length; i++) {
				System.out.println(Arrays.toString(out[i]));
			}
	}
}
