using Encog.Util.File;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MachineLearning
{
    public class Config
    {
        #region Base
        public static FileInfo ClassificationPath = new FileInfo(@"C:\Users\r.inazawa.araujo\Desktop\Machine Learning\Projeto\MachineLearning\DataClassification\");
        public static FileInfo RegressionPath = new FileInfo(@"C:\Users\r.inazawa.araujo\Desktop\Machine Learning\Projeto\MachineLearning\DataRegression\");
        #endregion

        #region ClassificationTask
        public static FileInfo ClassificationFile = FileUtil.CombinePath(ClassificationPath, "IrisData.csv");
        public static FileInfo ShuffledClassificationFile = FileUtil.CombinePath(ClassificationPath, "Iris_Shuffled.csv");
        public static FileInfo TrainingClassificationFile = FileUtil.CombinePath(ClassificationPath, "Iris_Train.csv");
        public static FileInfo EvaluateClassificationFile = FileUtil.CombinePath(ClassificationPath, "Iris_Eval.csv");
        public static FileInfo NormalizedTrainingClassificationFile = FileUtil.CombinePath(ClassificationPath, "Iris_Train_Norm.csv");
        public static FileInfo NormalizedEvaluateClassificationFile = FileUtil.CombinePath(ClassificationPath, "Iris_Eval_Norm.csv");
        public static FileInfo AnalystClassificationFile = FileUtil.CombinePath(ClassificationPath, "Iris_Analyst.ega");
        public static FileInfo TrainedNetworkClassificationFile = FileUtil.CombinePath(ClassificationPath, "Iris_Train.eg");
        #endregion

        #region RegressionTask
        public static FileInfo RegressionFile = FileUtil.CombinePath(RegressionPath, "AutoMPG.csv");
        public static FileInfo ShuffledRegressionFile = FileUtil.CombinePath(RegressionPath, "AutoMPG_Shuffled.csv");
        public static FileInfo TrainingRegressionFile = FileUtil.CombinePath(RegressionPath, "AutoMPG_Train.csv");
        public static FileInfo EvaluateRegressionFile = FileUtil.CombinePath(RegressionPath, "AutoMPG_Eval.csv");
        public static FileInfo NormalizedTrainingRegressionFile = FileUtil.CombinePath(RegressionPath, "AutoMPG_Train_Norm.csv");
        public static FileInfo NormalizedEvaluateRegressionFile = FileUtil.CombinePath(RegressionPath, "AutoMPG_Eval_Norm.csv");
        public static FileInfo AnalystRegressionFile = FileUtil.CombinePath(RegressionPath, "AutoMPG_Analyst.ega");
        public static FileInfo TrainedNetworkRegressionFile = FileUtil.CombinePath(RegressionPath, "AutoMPG_Train.eg");
        public static FileInfo ValidationRegressionResult = FileUtil.CombinePath(RegressionPath, "AutoMPG_ValidationResult.csv");
        #endregion
    }
}
