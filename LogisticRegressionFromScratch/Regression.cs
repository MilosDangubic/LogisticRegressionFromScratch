using System;
using System.Collections.Generic;
using System.Text;

namespace LogisticRegressionFromScratch
{
    public class Regression
    {
        public static void SplitTrainTestData(double[][] data, double[] allLabels,double[][] trainData,double[][] testData,double[] trainLabels,double[] testLabels,int n) 
        {
            int nTrain = n / 100 * 70;
            for (int i = 0; i < nTrain; i++)
            {
                trainLabels[i] = allLabels[i];
                trainData[i] = new double[data[i].Length];
                for (int j = 0; j < data[i].Length; j++)
                {
                    trainData[i][j] = data[i][j];
                }
            }
            for (int i = 0; i < n - nTrain; i++)
            {
                testLabels[i] = allLabels[i + nTrain];
                testData[i] = new double[data[i].Length];
                for (int j = 0; j < data[i].Length; j++)
                {
                    testData[i][j] = data[i + nTrain][j];
                }
            }

        }
        public static double CalculateAccuracy(int[] predicted,double[] actual) 
        {
            double acc = 0;
            for (int i = 0; i < predicted.Length; i++)
            {
                if (predicted[i] == actual[i])
                {
                    acc++;
                }
            }
            acc = acc / predicted.Length;
            return acc;
        }
        public static void randomize(double[][] data)
        {
            int n;
            n = data.Length;
            Random r = new Random();
            for (int i = n - 1; i > 0; i--)
            {
                int j = r.Next(0, i + 1);
                for(int k=0; k<data[i].Length; k++) 
                {
                    double temp = data[i][k];
                    data[i][k] = data[j][k];
                    data[j][k] = temp;
                }

              
            }
        }

        public static double[][] createWeights(int numOfClasses, int numOfFeatures) 
        {
            double[][] weights = new double[numOfClasses][];
            for (int i = 0; i < weights.Length; i++)
            {
                weights[i] = new double[numOfFeatures];
                for (int j = 0; j < weights[i].Length; j++)
                {
                    weights[i][j] = 0;
                }
            }
            return weights;
        }
        public static double[][]  RemoveLabels(double[][] datawithLabel) 
        {
            double[][] data = new double[datawithLabel.Length][];
            for (int i = 0; i < data.Length; i++) 
            {
                data[i] = new double[datawithLabel[i].Length - 1];
            }

            for (int i = 0; i < data.Length; i++) 
            {
                for (int j = 0; j < data[i].Length; j++) 
                {
                    data[i][j] = datawithLabel[i][j];
                }
            }
            return data;
        }
        public static double[] ExtractLabels(double[][] data) 
        {
            double[] labels = new double[data.Length];
            for (int i = 0; i < data.Length; i++) 
            {
                labels[i] = data[i][data[i].Length - 1];
            }
            return labels;
        
        } 
        public static double[][] MatrixLoad(string file, bool header, char sep)
        {
            string line = "";
            string[] tokens = null;
            int ct = 0;
            int rows, cols;
            System.IO.FileStream ifs =
              new System.IO.FileStream(file, System.IO.FileMode.Open);
            System.IO.StreamReader sr =
              new System.IO.StreamReader(ifs);
            while ((line = sr.ReadLine()) != null)
            {
                ++ct;
                tokens = line.Split(sep); 
            }
            sr.Close(); ifs.Close();
            if (header == true)
                rows = ct - 1;
            else
                rows = ct;
            cols = tokens.Length;
            double[][] result = MatrixCreate(rows, cols);

            // load
            int i = 0; // row index
            ifs = new System.IO.FileStream(file, System.IO.FileMode.Open);
            sr = new System.IO.StreamReader(ifs);

            if (header == true)
                line = sr.ReadLine();  // consume header
            while ((line = sr.ReadLine()) != null)
            {
                tokens = line.Split(sep);
                for (int j = 0; j < cols; ++j)
                    result[i][j] = double.Parse(tokens[j], System.Globalization.CultureInfo.InvariantCulture);
                ++i; // next row
            }
            sr.Close(); ifs.Close();
            return result;
        }
        private static double[][] MatrixCreate(int rows, int cols)
        {
            double[][] result = new double[rows][];
            for (int i = 0; i < rows; ++i)
                result[i] = new double[cols];
            return result;
        }
        public static double[][] CreateLabelClasses(double[] labels) 
        {
            double[][] labelClases = new double[labels.Length][];
            for (int i = 0; i < labelClases.Length; i++) 
            {
                labelClases[i] = new double[11];
            }
            for (int i = 0; i < 11; i++) 
            {
                for (int j = 0; j < labelClases.Length; j++) 
                {
                    if (labels[j] == i)
                    {
                        labelClases[j][i] = 1;
                    }
                    else
                    {
                        labelClases[j][i] = 0;
                    }
                }
            
            }
            return labelClases;
        
        }

        public static double[][] InsertBiasColumn(double[][] data)
        {
            int rows = data.Length;
            int cols = data[0].Length;
            double[][] result = MatrixCreate(rows, cols + 1);
            for (int i = 0; i < rows; ++i)
                result[i][0] = 1.0;

            for (int i = 0; i < rows; ++i)
                for (int j = 0; j < cols; ++j)
                    result[i][j + 1] = data[i][j];

            return result;
        }

        private static double DotProduct(double[] A, double[] B) 
        {
            double rez = 0;
            for (int i = 0; i < A.Length; i++) 
            {
                rez += A[i] * B[i];
            }
            return rez;
        }
   
        public static double[] MatrixVectorDotProduct(double[][] A, double[] weights) 
        {
            double[] rez = new double[A.Length];
            for (int i = 0; i < rez.Length; i++)
            {
                rez[i] = DotProduct(A[i], weights);
            }
            return rez;
        }

        public static  double[][] MatrixTranspose(double[][] matrix)
        {
            int rows = matrix.Length;
            int cols = matrix[0].Length;
            double[][] result = MatrixCreate(cols, rows); 
            for (int i = 0; i < rows; ++i)
            {
                for (int j = 0; j < cols; ++j)
                {
                    result[j][i] = matrix[i][j];
                }
            }
            return result;
        } 

        public static double[] Hypothesis(double[] weights, double[][] X)
        { 
            double[] z = MatrixVectorDotProduct(X, weights);
            double[] resault = new double[z.Length];
            for (int i = 0; i < resault.Length; i++) 
            {
                resault[i] = 1.0 / (1.0 + Math.Exp(-z[i]));
            }
            return resault;     
        }

        public static double[] SubtractVectors(double[] A, double[] B )
        {
            if (A.Length != B.Length)
            {
                Console.WriteLine("ERROR");
            }
            double[] rez = new double[A.Length];
            for (int i = 0; i < rez.Length; i++) 
            {
                rez[i] = A[i] - B[i];
            }
            return rez;
        }

        public static double SumOfVector(double[] A) 
        {
            double sum = 0;
            for (int i = 0; i < A.Length; i++)
            {
                sum += A[i];
            }
            return sum;
        }
        public static double[] ProdCor(double[] A, double[] B) 
        {
            double[] rez = new double[A.Length];
            for (int i = 0; i < rez.Length; i++) 
            {
                rez[i] = A[i] * B[i];
            }
            return rez;
        }

         
        public static void GradientDescent(double[][] X, double[][] Y, double[][] weights,double lr,int epochs) 
        {
            int m = X.Length;
            double[][] YT = MatrixTranspose(Y);
            double[][] XT = MatrixTranspose(X);
			//Za svaku epohu
            for(int i=0; i<epochs; i++) 
            {
				//Za svaki model
                for (int j = 0; j < 11; j++)
                {
                    double[] h = Hypothesis(weights[j], X);
                    double[] subtract = SubtractVectors(h, YT[j]);
					//Azuriramo vrednost za svaku od tezina
                    for(int k=0; k<weights[j].Length; k++) 
                    {
                        double[] dotProd = ProdCor(subtract, XT[k]);
                        double sum = SumOfVector(dotProd);
                        weights[j][k]-= (lr / m) * sum;
                      
                    }
                }
            }
            
        }





        public static int MaxNiz(double[] niz) 
        {
            double maks = niz[0];
            int poz = 0;
            for (int i = 0; i < niz.Length; i++) 
            {
                if (maks < niz[i]) 
                {
                    maks = niz[i];
                    poz = i;
                }
            }
            return poz;
        }

        public static int[] Predict(double[][] weights, double[][] X) 
        {
            int [] predic = new int[X.Length];
            for (int i = 0; i < X.Length; i++) 
            {
                double[] temp = new double[weights.Length];
                for(int j=0; j<temp.Length; j++) 
                {
                    temp[j] = DotProduct(weights[j], X[i]);
                }
                int poz = MaxNiz(temp);
                predic[i] = poz;
            }
            return predic;
        }
        public static List<string> extractFeatureNames(string file, char sep)
        {
            List<string> featureNames = new List<string>();
            System.IO.FileStream ifs = new System.IO.FileStream(file, System.IO.FileMode.Open);
            System.IO.StreamReader sr = new System.IO.StreamReader(ifs);
            string line = sr.ReadLine();
            string[] lineParts = line.Split(sep);

            for (int i = 0; i < lineParts.Length - 1; i++)
            {
                featureNames.Add(lineParts[i]);
            }

            return featureNames;

        }




        public static double EvaluateModel(double[][] X,double[] Y) 
        {
            double[][] weights = Regression.createWeights(11, X[0].Length);
            double[][] labelClases = Regression.CreateLabelClasses(Y);
            Regression.GradientDescent(X, labelClases, weights, 0.02, 1500);

            double acc = 0;
            int[] trainPredictions = Regression.Predict(weights, X);
            for (int i = 0; i < trainPredictions.Length; i++)
            {
                if (trainPredictions[i] == Y[i])
                {
                    acc++;
                }
            }
            acc = acc / trainPredictions.Length;
            return acc;

        }
    }
}
