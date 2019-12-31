using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using ImageClassification.DataModels;
using ImageClassification.Model;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using Microsoft.AspNetCore.Mvc.RazorPages;
using Microsoft.Extensions.Logging;

namespace ImageClassification.PredictWeb.Pages
{
    public class IndexModel : PageModel
    {
        private readonly ILogger<IndexModel> _logger;

        public string Message { get; set; }
        [BindProperty(SupportsGet = true)]
        public IFormFile ImageFile { get; set; }
        public ImagePrediction Predict { get; set; }

        public IndexModel(ILogger<IndexModel> logger)
        {
            _logger = logger;
        }

        public void OnGet()
        {
            Message = ".";
        }

        public async Task OnPostAsync()
        {
            Message = "LOADING...";


            string assetsRelativePath = @"../../../assets";
            string assetsPath = GetAbsolutePath(assetsRelativePath);


            var imagesFolder = Path.Combine(assetsPath, "inputs", "images-for-predictions", "tmp-web-image");
            var imageClassifierZip = Path.Combine(assetsPath, "inputs", "MLNETModel", "imageClassifier.zip");
            imagesFolder += @"/" + ImageFile.FileName;

            try
            {
                using (var stream = System.IO.File.Create(imagesFolder))
                {
                    await ImageFile.CopyToAsync(stream);
                }


                var modelScorer = new ModelScorer(imagesFolder, imageClassifierZip);
                Predict = modelScorer.ClassifySingleImage();
            }
            catch (Exception ex)
            {
                Message = ex.Message;
            }

            Message = "...";
            //return RedirectToPage("./Index");
        }

        public static string GetAbsolutePath(string relativePath)
        {
            FileInfo _dataRoot = new FileInfo(typeof(Program).Assembly.Location);
            string assemblyFolderPath = _dataRoot.Directory.FullName;

            string fullPath = Path.Combine(assemblyFolderPath, relativePath);

            return fullPath;
        }
    }
}
