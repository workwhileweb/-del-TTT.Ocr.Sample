using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.OCR;
using Emgu.CV.Structure;
using Emgu.CV.Text;
using Emgu.CV.UI;
using Emgu.CV.Util;
using TTT.Gui.Builder;

namespace TTT.Ocr;

internal class Program
{
    private static readonly HttpClient Client = new();
    private Tesseract? _ocr;

    private readonly Form _form;
    private readonly ComboBox _comboBox;
    private readonly ImageBox _imageBox;
    private readonly TextBox _textBoxFile;
    private readonly TextBox _textBoxHOcr;
    private readonly TextBox _textBoxOcr;

    protected Program()
    {
        var lineImage = LineLabelTextButton.Create("hình ảnh", "<file hình>", "đọc");
        lineImage.ButtonControl.Click += LoadImageClick;
        _textBoxFile = lineImage.TextControl;

        var buttonLoadLanguage = Line.CreateButton("ngôn ngữ", 2, LoadLanguageClick);
        _comboBox = GuiBuilder.CreateComboBox("Full Page OCR", "Text Region Detection");
        var lineLanguage = GuiBuilder.CreateLine(buttonLoadLanguage, Line.Create(_comboBox, 6));

        _imageBox = GuiBuilder.CreateControl<ImageBox>();
        var pageImage = GuiBuilder.CreateTabPage("hình ảnh", _imageBox);

        _textBoxOcr = GuiBuilder.CreateTextBox("", true, false);
        var pageOcr = GuiBuilder.CreateTabPage("ocr", _textBoxOcr);

        _textBoxHOcr = GuiBuilder.CreateTextBox("", true, false);
        var pageHOcr = GuiBuilder.CreateTabPage("hOcr", _textBoxHOcr);

        var tab = GuiBuilder.CreateTabControl(pageImage, pageOcr, pageHOcr);
        var lineTab = GuiBuilder.CreateLine(Line.Create(tab, 10, 10));

        var table = GuiBuilder.CreateTable(lineImage, lineLanguage, lineTab);
        _form = GuiBuilder.CreateForm("OCR", 400, 600, table);

        const string lang = "eng";
        if (!InitOcr(ref _ocr, Tesseract.DefaultTesseractDirectory, lang, OcrEngineMode.TesseractLstmCombined)) return;
        _form.Text = @$"{lang}-{OcrEngineMode.TesseractLstmCombined}-v{Tesseract.VersionString}";

        var img = new Mat(200, 400, DepthType.Cv8U, 3);
        img.SetTo(new Bgr(255, 0, 0).MCvScalar);

        CvInvoke.PutText(
            img,
            "Hello, world",
            new Point(10, 80),
            FontFace.HersheyComplex,
            1.0,
            new Bgr(0, 255, 0).MCvScalar);

        OcrImage(img);
    }

    private OcrMode Mode => _comboBox.SelectedIndex == 0 ? OcrMode.FullPage : OcrMode.TextDetection;

    private static async Task DownloadFileAsync(string fileUrl, string pathToSave)
    {
        var httpResult = await Client.GetAsync(fileUrl);
        await using var resultStream = await httpResult.Content.ReadAsStreamAsync();
        await using var fileStream = File.Create(pathToSave);
        await resultStream.CopyToAsync(fileStream).ConfigureAwait(false);
    }

    private static void TesseractRequire(string folder, string lang)
    {
        if (!Directory.Exists(folder)) Directory.CreateDirectory(folder);
        var dest = Path.Combine(folder, $"{lang}.traineddata");
        if (File.Exists(dest)) return;
        var source = Tesseract.GetLangFileUrl(lang);
        DownloadFileAsync(source, dest).Wait();
    }

    private static bool InitOcr(ref Tesseract? tesseract, string path, string lang, OcrEngineMode mode)
    {
        try
        {
            tesseract?.Dispose();
            if (string.IsNullOrEmpty(path)) path = Tesseract.DefaultTesseractDirectory;
            TesseractRequire(path, lang);
            TesseractRequire(path, "osd");
            tesseract = new Tesseract(path, lang, mode);
            return true;
        }
        catch (Exception e)
        {
            tesseract = null;
            MessageBox.Show(e.Message, nameof(InitOcr), MessageBoxButtons.OK);
            return false;
        }
    }

    private static Rectangle ScaleRectangle(Rectangle r, double scale)
    {
        var centerX = r.Location.X + r.Width / 2.0;
        var centerY = r.Location.Y + r.Height / 2.0;
        var newWidth = Math.Round(r.Width * scale);
        var newHeight = Math.Round(r.Height * scale);
        return new Rectangle(
            (int)Math.Round(centerX - newWidth / 2.0),
            (int)Math.Round(centerY - newHeight / 2.0),
            (int)newWidth,
            (int)newHeight);
    }

    private static string OcrImage(Tesseract ocr, Mat image, OcrMode mode, Mat imageColor, bool checkInvert = true)
    {
        var drawCharColor = new Bgr(Color.Red);

        if (image.NumberOfChannels == 1)
            CvInvoke.CvtColor(image, imageColor, ColorConversion.Gray2Bgr);
        else
            image.CopyTo(imageColor);

        if (mode == OcrMode.FullPage)
        {
            ocr.SetImage(imageColor);

            if (ocr.Recognize() != 0) throw new Exception(nameof(ocr.Recognize));
            
            var characters = ocr.GetCharacters();
            if (characters.Length == 0)
            {
                var imgGrey = new Mat();
                CvInvoke.CvtColor(image, imgGrey, ColorConversion.Bgr2Gray);
                var thresholded = new Mat();
                CvInvoke.Threshold(imgGrey, thresholded, 65, 255, ThresholdType.Binary);
                ocr.SetImage(thresholded);
                characters = ocr.GetCharacters();
                imageColor = thresholded;
                if (characters.Length == 0)
                {
                    CvInvoke.Threshold(image, thresholded, 190, 255, ThresholdType.Binary);
                    ocr.SetImage(thresholded);
                    characters = ocr.GetCharacters();
                    imageColor = thresholded;
                }
            }

            foreach (var c in characters) 
                CvInvoke.Rectangle(imageColor, c.Region, drawCharColor.MCvScalar);

            return ocr.GetUTF8Text();
        }

        Rectangle[] regions;

        using (var er1 = new ERFilterNM1("trained_classifierNM1.xml", 8))
        using (var er2 = new ERFilterNM2("trained_classifierNM2.xml"))
        {
            var channelCount = image.NumberOfChannels;
            var channels = new UMat[checkInvert ? channelCount * 2 : channelCount];

            for (var i = 0; i < channelCount; i++)
            {
                var c = new UMat();
                CvInvoke.ExtractChannel(image, c, i);
                channels[i] = c;
            }

            if (checkInvert)
                for (var i = 0; i < channelCount; i++)
                {
                    var c = new UMat();
                    CvInvoke.BitwiseNot(channels[i], c);
                    channels[i + channelCount] = c;
                }

            var vectorOfErStats = new VectorOfERStat[channels.Length];
            for (var i = 0; i < vectorOfErStats.Length; i++)
                vectorOfErStats[i] = new VectorOfERStat();

            try
            {
                for (var i = 0; i < channels.Length; i++)
                {
                    er1.Run(channels[i], vectorOfErStats[i]);
                    er2.Run(channels[i], vectorOfErStats[i]);
                }

                using var vm = new VectorOfUMat(channels);
                regions = ERFilter.ERGrouping(image, vm, vectorOfErStats,
                    ERFilter.GroupingMethod.OrientationHoriz,
                    "trained_classifier_erGrouping.xml");
            }
            finally
            {
                foreach (var tmp in channels) tmp.Dispose();
                foreach (var tmp in vectorOfErStats) tmp.Dispose();
            }

            var imageRegion = new Rectangle(Point.Empty, imageColor.Size);
            for (var i = 0; i < regions.Length; i++)
            {
                var r = ScaleRectangle(regions[i], 1.1);
                r.Intersect(imageRegion);
                regions[i] = r;
            }
        }

        var allChars = new List<Tesseract.Character>();
        var allText = string.Empty;
        foreach (var rect in regions)
        {
            using var region = new Mat(image, rect);
            ocr.SetImage(region);
            if (ocr.Recognize() != 0) throw new Exception(nameof(ocr.Recognize));
            var characters = ocr.GetCharacters();

            //convert the coordinates from the local region to global
            for (var i = 0; i < characters.Length; i++)
            {
                var charRegion = characters[i].Region;
                charRegion.Offset(rect.Location);
                characters[i].Region = charRegion;
            }

            allChars.AddRange(characters);

            allText += ocr.GetUTF8Text() + Environment.NewLine;
        }

        var drawRegionColor = new Bgr(Color.Red);
        foreach (var rect in regions) CvInvoke.Rectangle(imageColor, rect, drawRegionColor.MCvScalar);
        foreach (var c in allChars) CvInvoke.Rectangle(imageColor, c.Region, drawCharColor.MCvScalar);

        return allText;
    }

    private void OcrImage(Mat source)
    {
        _imageBox.Image = null;
        _textBoxOcr.Text = string.Empty;
        _textBoxHOcr.Text = string.Empty;
        var result = new Mat();
        var ocrImage = OcrImage(_ocr ?? throw new InvalidOperationException(), source, Mode, result);
        _imageBox.Image = result;
        _textBoxOcr.Text = ocrImage;
        if (Mode == OcrMode.FullPage) _textBoxHOcr.Text = _ocr.GetHOCRText();
    }

    private void LoadImageClick(object? sender, EventArgs e)
    {
        var ofd = new OpenFileDialog();
        if (ofd.ShowDialog() != DialogResult.OK) return;
        _textBoxFile.Text = ofd.FileName;
        var source = new Mat(_textBoxFile.Text);
        OcrImage(source);
    }

    private void LoadLanguageClick(object? sender, EventArgs e)
    {
        var ofd = new OpenFileDialog
        {
            DefaultExt = @"traineddata",
            Filter = @"tesseract language file|*.traineddata|All files|*.*"
        };
        if (ofd.ShowDialog() != DialogResult.OK) return;
        var path = Path.GetDirectoryName(ofd.FileName);
        var lang = Path.GetFileNameWithoutExtension(ofd.FileName).Split('.')[0];
        InitOcr(ref _ocr, path ?? throw new InvalidOperationException(), lang, OcrEngineMode.Default);

        _form.Text = @$"{lang}-{OcrEngineMode.Default}-v{Tesseract.VersionString}";
    }

    /// <summary>
    ///     The main entry point for the application.
    /// </summary>
    [STAThread]
    private static void Main()
    {
        Application.EnableVisualStyles();
        Application.SetCompatibleTextRenderingDefault(false);
        Application.Run(new Program()._form);
    }

    /// <summary>
    ///     The OCR mode
    /// </summary>
    private enum OcrMode
    {
        /// <summary>
        ///     Perform a full page OCR
        /// </summary>
        FullPage,

        /// <summary>
        ///     Detect the text region before applying OCR.
        /// </summary>
        TextDetection
    }
}