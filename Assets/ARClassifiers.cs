using UnityEngine;
using UnityEngine.XR.ARFoundation;
using UnityEngine.XR.ARSubsystems;
using TMPro;
using Unity.InferenceEngine;
using System.Collections;
using System.Linq;
using UnityEngine.UI;

public class ARClassifier : MonoBehaviour
{
    // ================= STATE =================
    enum AppState { Ready, Capturing, Loading, ShowingResult }
    private AppState currentState = AppState.Ready;

    // ================= MODEL =================
    [Header("AI Model")]
    public ModelAsset modelAsset;
    private Worker worker;
    private Tensor<float> inputTensor;

    // ================= AR =================
    private ARCameraManager camManager;
    private Texture2D cameraTexture;

    // ================= UI =================
    [Header("UI Elements")]
    public Button captureButton;
    public GameObject loadingPanel;
    public TextMeshProUGUI loadingText;
    public GameObject resultPanel;
    
    [Header("Result UI")]
    public TextMeshProUGUI nutrientNameText;
    public TextMeshProUGUI confidenceText;
    public TextMeshProUGUI recommendationText;
    public Image progressCircle;

    // ================= CONFIG =================
    [Header("Settings")]
    [Range(64, 224)]
    public int inputSize = 224;

    private readonly string[] classLabels = { "nitrogen", "phosphorus", "potassium" };
    private System.Collections.Generic.Dictionary<string, NutrientInfo> nutrientDB;

    // ================= START =================
    void Start()
    {
        camManager = GetComponent<ARCameraManager>();

        worker = new Worker(ModelLoader.Load(modelAsset), BackendType.GPUCompute);
        inputTensor = new Tensor<float>(new TensorShape(1, 3, inputSize, inputSize));
        cameraTexture = new Texture2D(inputSize, inputSize, TextureFormat.RGB24, false);

        InitDatabase();
        SetState(AppState.Ready);

        if (captureButton != null)
            captureButton.onClick.AddListener(OnCapturePressed);
    }

    // ================= UPDATE =================
    void Update()
    {
        // TAP DI MANA SAJA SAAT HASIL MUNCUL → RESET KE READY
        if (currentState == AppState.ShowingResult && Input.GetMouseButtonDown(0))
        {
            SetState(AppState.Ready);
        }
    }

    // ================= STATE MANAGEMENT =================
    void SetState(AppState newState)
    {
        currentState = newState;

        // Reset semua UI
        if (captureButton != null)
            captureButton.gameObject.SetActive(false);
        if (loadingPanel != null)
            loadingPanel.SetActive(false);
        if (resultPanel != null)
            resultPanel.SetActive(false);

        // Set UI sesuai state
        switch (newState)
        {
            case AppState.Ready:
                if (captureButton != null)
                {
                    captureButton.gameObject.SetActive(true);
                    captureButton.interactable = true;
                }
                break;

            case AppState.Capturing:
                if (captureButton != null)
                    captureButton.interactable = false;
                break;

            case AppState.Loading:
                if (loadingPanel != null)
                    loadingPanel.SetActive(true);
                if (loadingText != null)
                    loadingText.text = "Menganalisis...";
                break;

            case AppState.ShowingResult:
                if (resultPanel != null)
                    resultPanel.SetActive(true);
                break;
        }
    }

    // ================= CAPTURE =================
    void OnCapturePressed()
    {
        if (currentState != AppState.Ready)
            return;

        StartCoroutine(CaptureAndAnalyze());
    }

    IEnumerator CaptureAndAnalyze()
    {
        SetState(AppState.Capturing);
        
        yield return new WaitForSeconds(0.1f); // Simulasi capture

        SetState(AppState.Loading);

        // Ambil gambar dari AR Camera
        if (!camManager.TryAcquireLatestCpuImage(out var image))
        {
            Debug.LogError("Failed to capture image");
            SetState(AppState.Ready);
            yield break;
        }

        // Convert image
        var conv = new XRCpuImage.ConversionParams
        {
            inputRect = new RectInt(0, 0, image.width, image.height),
            outputDimensions = new Vector2Int(inputSize, inputSize),
            outputFormat = TextureFormat.RGB24,
            transformation = XRCpuImage.Transformation.None
        };

        var raw = cameraTexture.GetRawTextureData<byte>();
        image.Convert(conv, raw);
        cameraTexture.Apply();
        image.Dispose();

        // Convert to tensor
        TextureConverter.ToTensor(cameraTexture, inputTensor);

        // Run AI inference
        worker.Schedule(inputTensor);
        yield return new WaitForSeconds(0.5f); // Tunggu inference selesai

        // Get results
        var output = worker.PeekOutput() as Tensor<float>;
        
        if (output == null || output.shape.length == 0)
        {
            Debug.LogError("Invalid output from model");
            SetState(AppState.Ready);
            yield break;
        }

        float[] scores = output.DownloadToArray();
        
        Debug.Log($"Raw scores: [{string.Join(", ", scores.Select(s => s.ToString("F4")))}]");

        // Apply softmax if needed
        float sum = scores.Sum();
        bool isSoftmax = Mathf.Abs(sum - 1.0f) < 0.01f && scores.All(s => s >= 0 && s <= 1);
        
        if (!isSoftmax)
        {
            Softmax(scores);
            Debug.Log($"After softmax: [{string.Join(", ", scores.Select(s => s.ToString("F4")))}]");
        }

        // Get best prediction
        int bestIdx = ArgMax(scores);
        float confidence = scores[bestIdx] * 100f;
        string predictedClass = classLabels[bestIdx];

        // Show results
        ShowResult(predictedClass, confidence);
    }

    // ================= SHOW RESULT =================
    void ShowResult(string nutrientClass, float confidence)
    {
        if (!nutrientDB.ContainsKey(nutrientClass))
        {
            Debug.LogError($"Nutrient {nutrientClass} not found in database");
            SetState(AppState.Ready);
            return;
        }

        var info = nutrientDB[nutrientClass];

        // Update UI
        if (nutrientNameText != null)
        {
            nutrientNameText.text = info.displayName;
            nutrientNameText.color = info.color;
        }

        if (confidenceText != null)
        {
            confidenceText.text = $"{confidence:F0}%";
        }

        if (recommendationText != null)
        {
            recommendationText.text = $"<b>Rekomendasi:</b>\n{info.recommendation}";
        }

        if (progressCircle != null)
        {
            progressCircle.fillAmount = confidence / 100f;
            progressCircle.color = info.color;
        }

        SetState(AppState.ShowingResult);
    }

    // ================= DATABASE =================
    void InitDatabase()
    {
        nutrientDB = new System.Collections.Generic.Dictionary<string, NutrientInfo>
        {
            { 
                "nitrogen", new NutrientInfo 
                { 
                    displayName = "Nitrogen",
                    color = HexToColor("A8C686"),
                    recommendation = "• Pemupukan sesuai unsur yang kurang.\n• Aplikasi pada waktu yang tepat.\n• Pemupukan seimbang bertahap."
                } 
            },
            { 
                "phosphorus", new NutrientInfo 
                { 
                    displayName = "Fosfor",
                    color = HexToColor("2F5D50"),
                    recommendation = "• Aplikasi pupuk SP-36 atau TSP.\n• Dosis: 100-150 kg/ha.\n• Waktu: Sebelum tanam.\n• Campur dengan tanah secara merata."
                } 
            },
            { 
                "potassium", new NutrientInfo 
                { 
                    displayName = "Kalium",
                    color = HexToColor("F2C94C"),
                    recommendation = "• Aplikasi pupuk KCl (60% K₂O).\n• Dosis: 100-150 kg/ha.\n• Waktu: Fase generatif.\n• Aplikasi merata dan bertahap."
                } 
            }
        };
    }

    // ================= UTILITIES =================
    Color HexToColor(string hex)
    {
        hex = hex.Replace("#", "");
        byte r = byte.Parse(hex.Substring(0, 2), System.Globalization.NumberStyles.HexNumber);
        byte g = byte.Parse(hex.Substring(2, 2), System.Globalization.NumberStyles.HexNumber);
        byte b = byte.Parse(hex.Substring(4, 2), System.Globalization.NumberStyles.HexNumber);
        return new Color32(r, g, b, 255);
    }

    void Softmax(float[] values)
    {
        float max = values.Max();
        float sum = 0;
        
        for (int i = 0; i < values.Length; i++)
        {
            values[i] = Mathf.Exp(values[i] - max);
            sum += values[i];
        }
        
        for (int i = 0; i < values.Length; i++)
            values[i] /= sum;
    }

    int ArgMax(float[] values)
    {
        int idx = 0;
        for (int i = 1; i < values.Length; i++)
            if (values[i] > values[idx]) 
                idx = i;
        return idx;
    }

    // ================= CLEANUP =================
    void OnDestroy()
    {
        worker?.Dispose();
        inputTensor?.Dispose();
        
        if (cameraTexture != null)
            Destroy(cameraTexture);
        
        if (captureButton != null)
            captureButton.onClick.RemoveAllListeners();
    }
}

[System.Serializable]
public class NutrientInfo
{
    public string displayName;
    public Color color;
    public string recommendation;
}