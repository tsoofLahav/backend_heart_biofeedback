<div align="center">
  <h1>Heart Biofeedback</h1>
  <p><strong>Predicting the next heartbeat to bridge processing latency</strong></p>
  <p>Machine learning × cognitive science · University practicum</p>
  <p>Python · PyTorch · Flask · OpenCV · SciPy · Flutter</p>
</div>

![PPG waveform with detected heartbeat peaks and predicted future beat timing](graph.png)

*Predicting ahead: red crosses mark detected beats; green dots mark predicted timings. Their horizontal separation shows timing error in this example.*

## The idea

Developed during a practicum in **Prof. Amir Amedi’s Brain Lab at Reichman University**, Heart Biofeedback brought together two fields I studied at university: **machine learning and cognitive science**. The project explored how camera-based pulse sensing and predictive audio feedback could support interoception—the perception of internal bodily signals.

The central challenge was timing. By the time a fingertip video has been captured, uploaded, and processed, the detected heartbeat is already in the past. Playing that beat immediately would produce delayed feedback. Instead, a model uses recent heartbeat timing to **forecast upcoming beats**, creating audio cues intended to feel live despite the processing delay.

This is predictive biofeedback: the cues estimate future timing rather than guarantee synchronization with each actual heartbeat.

[Watch the app demo →](https://www.youtube.com/shorts/ZayBn7RVBaY)

## From camera to feedback

```mermaid
flowchart TB
    subgraph Phone[Mobile app · capture and playback]
        Capture[Record fingertip video<br/>Repeating 3-second clips]
        Playback[Play returned WAV<br/>Upcoming heartbeat cues]
        Feedback[Loading or unreadable feedback]
    end

    subgraph Signal[Backend · build a readable pulse signal]
        Upload[POST /process_video]
        Extract[Resample to 72 frames at 24 Hz<br/>Circular ROI → mean intensity]
        Buffer[Rolling buffer · latest 3 clips]
        Ready{3 clips available?}
        Join[Join clips and interpolate capture gaps<br/>240-sample analysis window]
        Filter[0.8–3 Hz bandpass<br/>Normalize waveform]
        Quality{Beat-shape correlation<br/>Readable signal?}
        Peaks[Detect pulse peaks<br/>Adaptive peak spacing]
        Upload --> Extract --> Buffer --> Ready
        Ready -->|Yes| Join --> Filter --> Quality
        Quality -->|Yes| Peaks
    end

    subgraph Forecast[Backend · predict beyond the observed signal]
        Features[Last 8 intervals + relative peak times<br/>Pad short history with mean interval]
        Model[Trained MLP · 16 → 128 → 64 → 8<br/>Predict 8 future peak times]
        Align[Align to latest observed beat<br/>Extend with mean interval if needed]
        Window[Select 10.5–14 s forecast window<br/>Shift to 0–3.5 s playback time]
        Boundary[Correct the boundary between chunks<br/>Use previous audio gap]
        Audio[Overlay beeps at forecast times<br/>Return 3.5-second WAV + BPM header]
        Features --> Model --> Align --> Window --> Boundary --> Audio
    end

    subgraph Runtime[Runtime state and inspection]
        State[In-memory buffers<br/>Mean interval · previous audio gap]
        Saved[Saved predictions<br/>Returned by POST /end]
        Inspect[Testing mode<br/>Signal + detected peaks + predictions as JSON]
    end

    Capture -->|Video upload| Upload
    Ready -->|No · loading| Feedback
    Quality -->|No · reset buffers| Feedback
    Peaks --> Features
    State -.-> Buffer
    State -.-> Features
    State -.-> Boundary
    Boundary --> Saved
    Window -. Testing mode .-> Inspect
    Audio -->|Audio response| Playback
    Playback -. Next capture cycle .-> Capture

    classDef mobile fill:#173d38,stroke:#58dbc3,color:#effffb
    classDef processing fill:#182c45,stroke:#779bc5,color:#f2f6fc
    classDef learned fill:#303158,stroke:#b6a0ff,color:#ffffff
    classDef timing fill:#413322,stroke:#e9b86b,color:#fff8ec
    class Capture,Playback,Feedback mobile
    class Upload,Extract,Buffer,Join,Filter,Peaks,Features processing
    class Model learned
    class Align,Window,Boundary,Audio timing
```


The diagram follows the latest backend: signal preparation feeds the predictor, while buffering and audio-boundary logic maintain continuity between requests. The forecast window moves the feedback target beyond the observed signal to accommodate processing latency. Earlier classification and reconstruction experiments are detailed below; the current route does not include a database write.

## Three purpose-built models

Three neural networks were built and trained during the practicum to address **signal quality, missing data, and future beat timing**. The diagrams combine illustrative signals with the actual layer sizes; signal sketches are conceptual examples, not evaluation results.

**Implementation note:** the saved models are multilayer perceptrons (**MLPs**), despite legacy files named “TCN” and “Unet.” Classification and reconstruction belong to earlier experiments; the latest revision uses conventional signal preparation and retains the learned predictor.

### 1 · Signal-quality classification

A window classifier learns to distinguish usable PPG from unreliable segments. Its sigmoid output provides a binary quality decision; a separate processing step can mask the region selected for reconstruction. It was trained on labeled signal windows.

![Quality model: corrupted PPG window, fully connected layers, and a masked signal region](docs/diagrams/quality.svg)

### 2 · Missing-signal reconstruction

The reconstruction network receives a ten-second signal with a two-second region zeroed out and estimates the **48 missing samples** at 24 Hz. Training creates these gaps in clean recordings and uses the original removed segment as the target, optimizing mean squared error. Inserting that estimate completes the waveform for subsequent processing; reconstructed samples remain estimates.

![Reconstruction model: masked PPG through dense layers to an estimated replacement segment](docs/diagrams/reconstruction.svg)

### 3 · Future-peak prediction

The predictor receives **eight recent intervals paired with eight relative peak times**. Its fully connected layers map those 16 values to eight future peak times. Training uses past/future timing pairs with a Smooth L1 loss. The backend aligns the forecast to the latest detected beat and converts the selected future times into audio cues.

![Prediction model: observed beat timing through dense layers to future peak times](docs/diagrams/prediction.svg)

## Explore the implementation

[Video-to-feedback pipeline](video_route.py) · [Prediction network](predict_model.py) · [Signal processing](filter_and_peaks.py) · [Audio generation](create_sound.py) · [Model history and training notes](docs/model-notes.md)
