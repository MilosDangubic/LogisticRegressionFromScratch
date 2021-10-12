using System;

namespace LogisticRegressionFromScratch
{
    class Program
    {
        static void Main(string[] args)
        {
          
            Console.WriteLine("======LOGISTIC REGRESSION CLASSIFICATION============");
            string fileName = "../../../winequality-red.txt";
            //Ucitavanje podataka
            Console.WriteLine("Loading data..");
            double[][] data = Regression.MatrixLoad(fileName, true, ';');
            //Fisher-Yates algoritam za permutaciju observacija u skupu podataka
            Regression.randomize(data);
            //iz trening i test skupa izvlacimo poslednju kolonu jer se u njoj nalaze labele(klase) kojima observacije pripadaju
            double[] allLabels = Regression.ExtractLabels(data);
            data = Regression.RemoveLabels(data);
            Console.WriteLine("Feature selection...");
            data=FeatureSelection.selectFeatures(data, allLabels, 3);
			//ubacivanje nove kolone na prvu poziciju, svaki element kolone ima vrednost 1
            data = Regression.InsertBiasColumn(data);
            //70% posto podataka koristimo za trening skup, 30% podataka koristimo za test skup
            int n = data.Length;
            int nTrain = n / 100 * 70;
            double[][] trainData = new double[nTrain][];
            double[] trainLabels = new double[nTrain];
            double[][] testData = new double[n - nTrain][];
            double[] testLabels = new double[n - nTrain];
            //Podela skupa na trening i test skup
            Regression.SplitTrainTestData(data, allLabels, trainData, testData, trainLabels, testLabels, n);

			//Kako se logisticka regresija koristi za binarnu klasifikaciju, a mi imamo 11 mogucih klasa, moramo napraviti 11 modela
			// gde za svaki model jedna klasa se gleda kao pozitivna klasa, a sve ostale kao negativne. One vs Rest metod
			
			//(1)Funkcija CreateLabelClasses od  niza trening labela, pravi 11 novih nizova trening labela, za svaki model po jedan
            double[][] trainLabelClases = Regression.CreateLabelClasses(trainLabels);
            //Kako je logisticka regresija opisana sa nizom tezina, mi pravimo 11 nizova tezina, za svaki model jedan niz tezina
            double[][] weights = Regression.createWeights(11, trainData[0].Length);
			
            //Treniranje svih 11 modela pomocu metode Gradijentnog spusta
            Console.WriteLine("Model training...");
            Regression.GradientDescent(trainData, trainLabelClases, weights, 0.02, 1500);
			//Nakon treniranja modela merimo preciznost na trening skupu
            int[] trainPredictions = Regression.Predict(weights, trainData);
            double trainAcc = Regression.CalculateAccuracy(trainPredictions, trainLabels);
            Console.WriteLine("Preciznost na trening skupu je {0} %", trainAcc * 100);
            //Merimo preciznost na test skupu
            int[] testPrediction = Regression.Predict(weights, testData);
            double testAcc = Regression.CalculateAccuracy(testPrediction, testLabels);
            Console.WriteLine("Preciznost na test skupu je {0} %", testAcc * 100);
            
        }
    }
}
