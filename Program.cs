using System;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;

class Program
{
    static void Main()
    {
        // Skapa ML-kontext (hjärnan i ML.NET)
        // seed: 1 ska göra att resultatet blir samma varje gång man kör den

        MLContext mlContext = new MLContext(seed: 1);

        string csvPath = "Titanic-Dataset.csv";

        Console.ForegroundColor = ConsoleColor.Cyan;
        Console.WriteLine("==== TITANIC SURVIVOR PREDICTOR ====");

        //Läs in CSV-filen till IDataView
        // hasHeader: true: första raden är kolumnnamn
        // separatorChar: ,:  kommaseparerad
        // allowQuoting: true: citattecken i namn 

        IDataView data = mlContext.Data.LoadFromTextFile<PassengerData>(
            path: csvPath,
            hasHeader: true,
            separatorChar: ',',
            allowQuoting: true
        );


        //https://learn.microsoft.com/en-us/dotnet/machine-learning/resources/transforms
        //https://learn.microsoft.com/en-us/dotnet/machine-learning/mldotnet-api

        //Bygg pipeline (förbehandling och modell)

        //OneHotEncoding av Sex (male/female -> siffror)
        // fyll i saknade värden (Age, Pclass, Parch, Fare)
        //Slå ihop alla features till en vektor som heter Features
        //Normalisera värdena (MinMax)
        //träna en regresionsmodell
        var pipeline =
            //Gör om texten male/female till siffror
            mlContext.Transforms.Categorical.OneHotEncoding(
                outputColumnName: "SexEncoded",
                inputColumnName: nameof(PassengerData.Sex))

            // Fyll i saknade värden med medelvärde eller standardvärde
            .Append(mlContext.Transforms.ReplaceMissingValues(
                outputColumnName: "AgeFilled",
                inputColumnName: nameof(PassengerData.Age),
                replacementMode: MissingValueReplacingEstimator.ReplacementMode.Mean))

            .Append(mlContext.Transforms.ReplaceMissingValues(
                outputColumnName: "PclassFilled",
                inputColumnName: nameof(PassengerData.Pclass),
                replacementMode: MissingValueReplacingEstimator.ReplacementMode.Mean))

            .Append(mlContext.Transforms.ReplaceMissingValues(
                outputColumnName: "ParchFilled",
                inputColumnName: nameof(PassengerData.Parch),
                replacementMode: MissingValueReplacingEstimator.ReplacementMode.DefaultValue))

            .Append(mlContext.Transforms.ReplaceMissingValues(
                outputColumnName: "FareFilled",
                inputColumnName: nameof(PassengerData.Fare),
                replacementMode: MissingValueReplacingEstimator.ReplacementMode.Mean))

            // Slå ihop alla features till en enda kolumn
            .Append(mlContext.Transforms.Concatenate(
                "Features",
                "AgeFilled",
                "SexEncoded",
                "PclassFilled",
                "ParchFilled",
                "FareFilled"))

            // Normalisera
            .Append(mlContext.Transforms.NormalizeMinMax("Features"))

            //träna modellen (BinaryClassification = överlever / överlever inte)
            .Append(mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(
                labelColumnName: nameof(PassengerData.Survived),
                featureColumnName: "Features"));


        //5-faldig korsvalidering (för att se hur bra modellen är)

        Console.WriteLine("Kör 5-faldig korsvalidering...");
        var cvResults = mlContext.BinaryClassification.CrossValidate(
            data: data,
            estimator: pipeline,
            numberOfFolds: 5,
            labelColumnName: nameof(PassengerData.Survived)
        );

        double avgAccuracy = cvResults.Average(r => r.Metrics.Accuracy);

        Console.WriteLine($"Genomsnittlig träffsäkerhet: {avgAccuracy:P2}");


        //Dela upp i tränings- och testdata (80/20)
        var split = mlContext.Data.TrainTestSplit(data, testFraction: 0.2, seed: 1);

        //träna
        var model = pipeline.Fit(split.TrainSet);

        // Utvärdera på testdelen
        var predictions = model.Transform(split.TestSet);
        var metrics = mlContext.BinaryClassification.Evaluate(
            data: predictions,
            labelColumnName: nameof(PassengerData.Survived)
        );

        Console.WriteLine("Utvärdering på 80/20 train-test split:");
        Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}");
        Console.WriteLine($"AUC: {metrics.AreaUnderRocCurve:P2}");
        Console.WriteLine($"F1-Score: {metrics.F1Score:P2}");
        Console.WriteLine();
        Console.WriteLine(metrics.ConfusionMatrix.GetFormattedConfusionTable());
        Console.WriteLine();


        //Träna slutgiltig modell på ALL data

        var finalModel = pipeline.Fit(data);

        // Skapa en PredictionEngine
        var predictionEngine = mlContext.Model.CreatePredictionEngine<PassengerData, PassengerPrediction>(finalModel);


        //Interaktiva delen. frågar användaren efter uppgifter

        Console.ForegroundColor = ConsoleColor.Yellow;
        Console.WriteLine("=== Testa med en ny passagerare ===\n");

        //Fråga efter ålder
        Console.Write("Ålder (t.ex. 25): ");
        float age = float.Parse(Console.ReadLine() ?? "25");

        //kön
        Console.Write("Kön (male/female): ");
        string sex = (Console.ReadLine() ?? "male").ToLower().Trim();
        if (sex != "male" && sex != "female")
            sex = "male";   // fallback

        //Hyttklass (1 2 eller 3)
        Console.Write("Hyttklass (1, 2 eller 3): ");
        float pclass = float.Parse(Console.ReadLine() ?? "3");

        //antal föräldrar eller barn ombord
        Console.Write("Antal föräldrar/barn ombord: ");
        float parch = float.Parse(Console.ReadLine() ?? "0");

        //biljettpris
        Console.Write("Biljettpris (t.ex. 7,25) (Se till att använd , och inte .) : ");
        float fare = float.Parse(Console.ReadLine() ?? "7.25");

        //passagerarobjektet
        var newPassenger = new PassengerData
        {
            Age = age,
            Sex = sex,
            Pclass = pclass,
            Parch = parch,
            Fare = fare
        };

        // Gör förutsägelsen
        var prediction = predictionEngine.Predict(newPassenger);

        //visa resultatet
        Console.WriteLine();
        Console.ForegroundColor = ConsoleColor.Green;
        Console.WriteLine("=== Resultat ===");
        Console.WriteLine($"Ålder:          {newPassenger.Age}");
        Console.WriteLine($"Kön:            {newPassenger.Sex}");
        Console.WriteLine($"Klass:          {newPassenger.Pclass}");
        Console.WriteLine($"Föräldrar/barn: {newPassenger.Parch}");
        Console.WriteLine($"Biljettpris:    {newPassenger.Fare}");
        Console.WriteLine();

        if (prediction.Survived)
            Console.WriteLine("passageraren kommer förmodligen att överleva");
        else
            Console.WriteLine("passageraren kommer förmodligen INTE att överleva");

        Console.WriteLine($"Sannolikhet: {prediction.Probability:P1}");
        Console.ResetColor();
    }
}


// Klass som beskriver en rad i CSV-filen
//[LoadColumn(x)] talar om vilken kolumn som ska läsas

public class PassengerData
{
    [LoadColumn(1)] public bool Survived { get; set; }   // 0 = dog, 1 = överlevde
    [LoadColumn(2)] public float Pclass { get; set; }    // 1, 2 eller 3
    [LoadColumn(4)] public string Sex { get; set; } = ""; // "male" / "female"
    [LoadColumn(5)] public float Age { get; set; }
    [LoadColumn(7)] public float Parch { get; set; }     // antal föräldrar/barn
    [LoadColumn(9)] public float Fare { get; set; }      // biljettpris
}


//klass som beskriver vad modellen returnerar

public class PassengerPrediction
{
    [ColumnName("PredictedLabel")]
    public bool Survived { get; set; }   // true = överlever

    public float Probability { get; set; } // hur säker modellen är (0-1)
    public float Score { get; set; }
}