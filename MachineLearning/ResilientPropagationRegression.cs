using Encog.App.Analyst;
using Encog.App.Analyst.CSV.Normalize;
using Encog.App.Analyst.CSV.Segregate;
using Encog.App.Analyst.CSV.Shuffle;
using Encog.App.Analyst.Wizard;
using Encog.Engine.Network.Activation;
using Encog.ML.Data.Basic;
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
    public class ResilientPropagationRegression
    {
        /// <summary>
        /// Metodo Principal da classe
        /// </summary>
        public static void MainExecute()
        {
            Console.WriteLine("Mistura os dados do arquivo CSV.");
            Shuffle(Config.RegressionFile);
            Console.WriteLine("Segrega os dados originais e cria os arquivos para treinamento e avaliação.");
            Segregate(Config.ShuffledRegressionFile);
            Console.WriteLine("Normaliza os dados de treinamento e de avaliação.");
            Normalization();
            Console.WriteLine("Cria a rede neural.");
            CreateNetwork(Config.TrainedNetworkRegressionFile);
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
            shuffle.Process(Config.ShuffledRegressionFile);
        }

        /// <summary>
        /// Metodo responsavel por segregar as informacoes do dataset em infos para Treinamento do Modelo e para Teste do Modelo 
        /// </summary>
        /// <param name="source">FileInfo com o caminho do Dataset misturado</param>
        private static void Segregate(FileInfo source)
        {
            var seg = new SegregateCSV();
            seg.Targets.Add(new SegregateTargetPercent(Config.TrainingRegressionFile, 75));
            seg.Targets.Add(new SegregateTargetPercent(Config.EvaluateRegressionFile, 25));
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

            //Wizard
            var wizard = new AnalystWizard(analyst);
            wizard.Wizard(Config.RegressionFile, true, AnalystFileFormat.DecpntComma);

            //Cilindros
            analyst.Script.Normalize.NormalizedFields[0].Action = Encog.Util.Arrayutil.NormalizationAction.Equilateral;
            //displacement
            analyst.Script.Normalize.NormalizedFields[1].Action = Encog.Util.Arrayutil.NormalizationAction.Normalize;
            //HorsePower
            analyst.Script.Normalize.NormalizedFields[2].Action = Encog.Util.Arrayutil.NormalizationAction.Normalize;
            //Peso
            analyst.Script.Normalize.NormalizedFields[3].Action = Encog.Util.Arrayutil.NormalizationAction.Normalize;
            //Aceleração
            analyst.Script.Normalize.NormalizedFields[4].Action = Encog.Util.Arrayutil.NormalizationAction.Normalize;
            //Ano
            analyst.Script.Normalize.NormalizedFields[5].Action = Encog.Util.Arrayutil.NormalizationAction.Equilateral;
            //Origem
            analyst.Script.Normalize.NormalizedFields[6].Action = Encog.Util.Arrayutil.NormalizationAction.Equilateral;
            //Nome
            analyst.Script.Normalize.NormalizedFields[7].Action = Encog.Util.Arrayutil.NormalizationAction.Ignore;
            //MPG
            analyst.Script.Normalize.NormalizedFields[8].Action = Encog.Util.Arrayutil.NormalizationAction.Normalize;

            var norm = new AnalystNormalizeCSV();
            norm.ProduceOutputHeaders = true;

            norm.Analyze(Config.TrainingRegressionFile, true, CSVFormat.English, analyst);
            norm.Normalize(Config.NormalizedTrainingRegressionFile);

            //Norm of evaluation
            norm.Analyze(Config.EvaluateRegressionFile, true, CSVFormat.English, analyst);
            norm.Normalize(Config.NormalizedEvaluateRegressionFile);

            //save the analyst file
            analyst.Save(Config.AnalystRegressionFile);
        }

        /// <summary>
        /// Metodo responsavel por criar a rede neural
        /// </summary>
        /// <param name="source">FileInfo com o path do network</param>
        private static void CreateNetwork(FileInfo networkFile)
        {
            var network = new BasicNetwork();
            network.AddLayer(new BasicLayer(new ActivationLinear(), true, 22));
            network.AddLayer(new BasicLayer(new ActivationTANH(), true, 6));
            network.AddLayer(new BasicLayer(new ActivationTANH(), false, 1));
            network.Structure.FinalizeStructure();
            network.Reset();
            EncogDirectoryPersistence.SaveObject(networkFile, (BasicNetwork)network);
        }

        /// <summary>
        /// Metodo responsavel por treinar a rede neural a uma taxa de erro de 1%
        /// </summary>
        static void TrainNetwork()
        {
            var network = (BasicNetwork)EncogDirectoryPersistence.LoadObject(Config.TrainedNetworkRegressionFile);
            var trainingSet = EncogUtility.LoadCSV2Memory(Config.NormalizedTrainingRegressionFile.ToString(),
                network.InputCount, network.OutputCount, true, CSVFormat.English, false);


            var train = new ResilientPropagation(network, trainingSet);
            int epoch = 1;
            do
            {
                train.Iteration();
                Console.WriteLine("Epoch : {0} Error : {1}", epoch, train.Error);
                epoch++;
            } while (train.Error > 0.01);

            EncogDirectoryPersistence.SaveObject(Config.TrainedNetworkRegressionFile, (BasicNetwork)network);

        }

        /// <summary>
        /// Metodo responsavel por avaliar a rede neural treinada com a massa de testes criada no metodo Segregate e normalizada no metodo Normalization
        /// </summary>
        private static void Evaluate()
        {
            var network = (BasicNetwork)EncogDirectoryPersistence.LoadObject(Config.TrainedNetworkRegressionFile);
            var analyst = new EncogAnalyst();
            analyst.Load(Config.AnalystRegressionFile.ToString());
            var evaluationSet = EncogUtility.LoadCSV2Memory(Config.NormalizedEvaluateRegressionFile.ToString(),
                network.InputCount, network.OutputCount, true, CSVFormat.English, false);

            using (var file = new System.IO.StreamWriter(Config.ValidationRegressionResult.ToString()))
            {
                foreach (var item in evaluationSet)
                {

                    var NormalizedActualoutput = (BasicMLData)network.Compute(item.Input);
                    var Actualoutput = analyst.Script.Normalize.NormalizedFields[8].DeNormalize(NormalizedActualoutput.Data[0]);
                    var IdealOutput = analyst.Script.Normalize.NormalizedFields[8].DeNormalize(item.Ideal[0]);

                    //Write to File
                    var resultLine = IdealOutput.ToString() + "," + Actualoutput.ToString();
                    file.WriteLine(resultLine);
                    Console.WriteLine("Ideal : {0}, Actual : {1}", IdealOutput, Actualoutput);

                }
            }
        }
    }
}
