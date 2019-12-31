using System;
using System.Linq;
using ImageClassification.DataModels;
using System.IO;
using Microsoft.ML;
using System.Collections.Generic;
using Microsoft.ML.Data;

namespace ImageClassification.Model
{
    public class ModelScorer
    {
        private readonly string imagesFolder;
        private readonly string modelLocation;
        private readonly MLContext mlContext;

        public ModelScorer(string imagesFolder, string modelLocation)
        {
            this.imagesFolder = imagesFolder;
            this.modelLocation = modelLocation;
            mlContext = new MLContext(seed: 1);
        }

        public ImagePrediction ClassifyImages()
        {
            
            // Load the model
            ITransformer loadedModel = mlContext.Model.Load(modelLocation,out var modelInputSchema);

            // Make prediction engine (input = ImageDataForScoring, output = ImagePrediction)
            var predictionEngine = mlContext.Model.CreatePredictionEngine<ImageData, ImagePrediction>(loadedModel);

            IEnumerable<ImageData> imagesToPredict = LoadImagesFromDirectory(imagesFolder, true);

            //Predict the first image in the folder
            //
            ImageData imageToPredict = new ImageData
            {
                ImagePath = imagesToPredict.First().ImagePath
            };

            var prediction = predictionEngine.Predict(imageToPredict);

            return prediction;
            //Console.WriteLine($"ImageFile : [{Path.GetFileName(imageToPredict.ImagePath)}], " +
            //                  $"Scores : [{string.Join(",", prediction.Score)}], " +
            //                  $"Predicted Label : {prediction.PredictedLabelValue}");
                      
            //////

            
        }

        public ImagePrediction ClassifySingleImage()
        {

            // Load the model
            ITransformer loadedModel = mlContext.Model.Load(modelLocation, out var modelInputSchema);

            // Make prediction engine (input = ImageDataForScoring, output = ImagePrediction)
            var predictionEngine = mlContext.Model.CreatePredictionEngine<ImageData, ImagePrediction>(loadedModel);

            //Predict the image loaded in form
            //
            ImageData imageToPredict = new ImageData
            {
                ImagePath = imagesFolder
            };

            var prediction = predictionEngine.Predict(imageToPredict);

            return prediction;
            
        }

        public static IEnumerable<ImageData> LoadImagesFromDirectory(string folder, bool useFolderNameasLabel = true)
        {
            var files = Directory.GetFiles(folder, "*",
                searchOption: SearchOption.AllDirectories);

            foreach (var file in files)
            {
                if ((Path.GetExtension(file) != ".jpg") && (Path.GetExtension(file) != ".png"))
                    continue;

                var label = Path.GetFileName(file);
                if (useFolderNameasLabel)
                    label = Directory.GetParent(file).Name;
                else
                {
                    for (int index = 0; index < label.Length; index++)
                    {
                        if (!char.IsLetter(label[index]))
                        {
                            label = label.Substring(0, index);
                            break;
                        }
                    }
                }

                yield return new ImageData()
                {
                    ImagePath = file,
                    Label = label
                };

            }
        }
    }
}
