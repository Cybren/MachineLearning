package features;

public class FeatureBuilder {
	double[][] frames;
	double[][] features;
	public FeatureBuilder(double[][] frames) {
		this.frames = frames;
		setFeatures();
	}
	private void setFeatures() {
		features = new double[frames.length][39];
		MFCC mfccs[] = new MFCC[frames.length];
		
		for (int i = 0; i < frames.length; i++) {
			mfccs[i] = new MFCC(frames[i], 16000);
		}
		
		double [][]cepstralCoef = new double[mfccs.length][12];
		for (int i = 0; i < cepstralCoef.length; i++) {
			for (int j = 0; j < cepstralCoef[0].length; j++) {
				cepstralCoef[i][j] = mfccs[i].mfccs[j];
			}
		}
		double [][] deltaC = getDeltas(cepstralCoef);
		double [][] deltaC2 = getDeltas(deltaC);
		
		double [][]energys = new double[mfccs.length][1];
		for (int i = 0; i < cepstralCoef.length; i++) {
				energys[i][0] = mfccs[i].energy;
		}
		double [][] deltaE = getDeltas(energys);
		double [][] deltaE2 = getDeltas(deltaE);
		
		for (int i = 0; i < cepstralCoef.length; i++) {
			for (int j = 0; j < cepstralCoef[i].length; j++) {
				features[i][j] = cepstralCoef[i][j];
				features[i][j+cepstralCoef[i].length] = deltaC[i][j];
				features[i][j+cepstralCoef[i].length*2] = deltaC2[i][j];
			}
			features[i][cepstralCoef[i].length*3] = energys[i][0];
			features[i][cepstralCoef[i].length*3+1] = deltaE[i][0];
			features[i][cepstralCoef[i].length*3+2] = deltaE2[i][0];
		}
	}
	
	private double[][] getDeltas(double[][] features){
		int n = 1;
		double tempCounter, tempDenominator;
		double[][] deltas = features.clone();
		for (int i = n; i < deltas.length-n; i++) {
			deltas[i] = new double[features[i].length];
			for (int j = 0; j < deltas[i].length; j++) {
				tempCounter = 0;
				tempDenominator = 0;
				for (int j2 = 1; j2 < n+1; j2++) {
					tempCounter += j2*(features[i+j2][j]-features[i-j2][j]);
					tempDenominator += j2*j2;
				}
				deltas[i][j] = tempCounter/(2*tempDenominator);
			}
		}
		return deltas;
	}
}
