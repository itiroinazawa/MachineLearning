using Encog.App.Analyst;
using Encog.App.Analyst.CSV.Normalize;
using Encog.App.Analyst.CSV.Segregate;
using Encog.App.Analyst.CSV.Shuffle;
using Encog.App.Analyst.Wizard;
using Encog.Engine.Network.Activation;
using Encog.Neural.Networks;
using Encog.Neural.Networks.Layers;
using Encog.Neural.Networks.Training.Propagation.Resilient;
using Encog.Persist;
using Encog.Util.CSV;
using Encog.Util.Simple;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MachineLearning
{
    public class ResilientPropagationClassification
    {
        /// <summary>
        /// Metodo Principal da classe
        /// </summary>
        public static void MainExecute()
        {
            Console.WriteLine("Mistura os dados do arquivo CSV.");
            Shuffle(Config.ClassificationFile);
            Console.WriteLine("Segrega os dados originais e cria os arquivos para treinamento e avaliação.");
            Segregate(Config.ShuffledClassificationFile);
            Console.WriteLine("Normaliza os dados de treinamento e de avaliação.");
            Normalization();
            Console.WriteLine("Cria a rede neural.");
            CreateNetwork(Config.TrainedNetworkClassificationFile);
            Console.WriteLine("Treina a rede neural com o arquivo de treinamento.");
            TrainNetwork();
            Console.WriteLine("Avalia a rede neural com o arquivo de avaliação.");
            Evaluate();
            Console.WriteLine("Pressione uma tecla para sair...");
            Console.ReadLine();
        }

        /// <summary>
        /// Metodo responsavel por misturar as informacoes do dataset
        /// </summary>
        /// <param name="source">FileInfo com o caminho do Dataset original</param>
        private static void Shuffle(FileInfo source)
        {
            var shuffle = new ShuffleCSV();
            shuffle.Analyze(source, true, CSVFormat.English);
            shuffle.ProduceOutputHeaders = true;
            shuffle.Process(Config.ShuffledClassificationFile);
        }

        /// <summary>
        /// Metodo responsavel por segregar as informacoes do dataset em infos para Treinamento do Modelo e para Teste do Modelo 
        /// </summary>
        /// <param name="source">FileInfo com o caminho do Dataset misturado</param>
        private static void Segregate(FileInfo source)
        {
            var seg = new SegregateCSV();
            seg.Targets.Add(new SegregateTargetPercent(Config.TrainingClassificationFile, 75));
            seg.Targets.Add(new SegregateTargetPercent(Config.EvaluateClassificationFile, 25));
            seg.ProduceOutputHeaders = true;
            seg.Analyze(source, true, CSVFormat.English);
            seg.Process();
        }

        /// <summary>
        /// Metodo responsavel por normalizar as informacoes para adequar a execucao da rede neural
        /// </summary>
        private static void Normalization()
        {
            var analyst = new EncogAnalyst();

            var wizard = new AnalystWizard(analyst);
            wizard.Wizard(Config.ClassificationFile, true, AnalystFileFormat.DecpntComma);

            var norm = new AnalystNormalizeCSV();
            norm.Analyze(Config.TrainingClassificationFile, true, CSVFormat.English, analyst);
            norm.ProduceOutputHeaders = true;
            norm.Normalize(Config.NormalizedTrainingClassificationFile);

            norm.Analyze(Config.EvaluateClassificationFile, true, CSVFormat.English, analyst);
            norm.Normalize(Config.NormalizedEvaluateClassificationFile);

            analyst.Save(Config.AnalystClassificationFile);
        }

        /// <summary>
        /// Metodo responsavel por criar a rede neural
        /// </summary>
        /// <param name="source">FileInfo com o path do network</param>
        private static void CreateNetwork(FileInfo source)
        {
            var network = new BasicNetwork();
            network.AddLayer(new BasicLayer(new ActivationLinear(), true, 4));
            network.AddLayer(new BasicLayer(new ActivationTANH(), true, 6));
            network.AddLayer(new BasicLayer(new ActivationTANH(), false, 2));
            network.Structure.FinalizeStructure();
            network.Reset();
            EncogDirectoryPersistence.SaveObject(source, (BasicNetwork)network);
        }

        /// <summary>
        /// Metodo responsavel por treinar a rede neural a uma taxa de erro de 1%
        /// </summary>
        private static void TrainNetwork()
        {
            var network = (BasicNetwork)EncogDirectoryPersistence.LoadObject(Config.TrainedNetworkClassificationFile);
            var trainingSet = EncogUtility.LoadCSV2Memory(Config.NormalizedTrainingClassificationFile.ToString(),
                network.InputCount, network.OutputCount, true, CSVFormat.English, false);


            var train = new ResilientPropagation(network, trainingSet);
            int epoch = 1;
            do
            {
                train.Iteration();
                Console.WriteLine("Epoch : {0} Error : {1}", epoch, train.Error);
                epoch++;
            } while (train.Error > 0.01);

            EncogDirectoryPersistence.SaveObject(Config.TrainedNetworkClassificationFile, (BasicNetwork)network);
        }

        /// <summary>
        /// Metodo responsavel por avaliar a rede neural treinada com a massa de testes criada no metodo Segregate e normalizada no metodo Normalization
        /// </summary>
        private static void Evaluate()
        {
            var network = (BasicNetwork)EncogDirectoryPersistence.LoadObject(Config.TrainedNetworkClassificationFile);
            var analyst = new EncogAnalyst();
            analyst.Load(Config.AnalystClassificationFile.ToString());
            var evaluationSet = EncogUtility.LoadCSV2Memory(Config.NormalizedEvaluateClassificationFile.ToString(),
                network.InputCount, network.OutputCount, true, CSVFormat.English, false);

            int count = 0;
            int CorrectCount = 0;
            foreach (var item in evaluationSet)
            {
                count++;
                var output = network.Compute(item.Input);

                var sepal_l = analyst.Script.Normalize.NormalizedFields[0].DeNormalize(item.Input[0]);
                var sepal_w = analyst.Script.Normalize.NormalizedFields[1].DeNormalize(item.Input[1]);
                var petal_l = analyst.Script.Normalize.NormalizedFields[2].DeNormalize(item.Input[2]);
                var petal_w = analyst.Script.Normalize.NormalizedFields[3].DeNormalize(item.Input[3]);

                int classCount = analyst.Script.Normalize.NormalizedFields[4].Classes.Count;
                double normalizationHigh = analyst.Script.Normalize.NormalizedFields[4].NormalizedHigh;
                double normalizationLow = analyst.Script.Normalize.NormalizedFields[4].NormalizedLow;

                var eq = new Encog.MathUtil.Equilateral(classCount, normalizationHigh, normalizationLow);
                var predictedClassInt = eq.Decode(output);
                var predictedClass = analyst.Script.Normalize.NormalizedFields[4].Classes[predictedClassInt].Name;
                var idealClassInt = eq.Decode(item.Ideal);
                var idealClass = analyst.Script.Normalize.NormalizedFields[4].Classes[idealClassInt].Name;

                if (predictedClassInt == idealClassInt)
                {
                    CorrectCount++;
                }
                Console.WriteLine("Count :{0} Properties [{1},{2},{3},{4}] ,Ideal : {5} Predicted : {6} ",
                    count, sepal_l, sepal_w, petal_l, petal_w, idealClass, predictedClass);
            }

            Console.WriteLine("Quantidade de itens: {0}", count);
            Console.WriteLine("Quantidade de acertos: {0}", CorrectCount);
            Console.WriteLine("Porcentagem de acertos: {0}", ((CorrectCount * 100.0) / count));
        }
    }
}
