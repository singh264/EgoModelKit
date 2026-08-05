import { 
    BarChart3,
    Check, 
    ChevronDown,
    ChevronLeft,
    ChevronRight, 
    ChevronUp,
    CircleCheck,
    Cpu,
    FileText,
    Folder,
    Info, 
    Upload,
    Video,
} from "lucide-react";
import {
    type ChangeEvent,
    type DragEvent,
    type RefObject,
    useEffect,
    useRef, 
    useState 
} from "react";

type Step = 
    | "welcome" 
    | "select-model" 
    | "choose-input" 
    | "choose-output"
    | "review"
    | "results"
    | "output-preview";

type StepperStep = Exclude<Step, "welcome" | "output-preview">;

type ReviewMode = "ready" | "dry-run-complete" | "running";

type GuiRunStatus = "ready" | "running" | "completed" | "failed" | "cancelled";

type ModelInfo = {
    id: string;
    name: string;
    description: string;
    acceptedInputLabel: string;
    supportedInputExtensions: string[];
    outputLabel: string;
};

type ModelsResponse = {
    models: ModelInfo[];
}

type SelectOutputFolderResponse = {
    outputRoot: string;
};

type OpenOutputFolderResponse = {
    opened: boolean;
    runId: string;
    outputFolder: string;
};

type CancelRunRequest = {
    runId: string | null;
    operationId: string | null;
};

type CancelRunResponse = {
    cancelled: boolean;
    runId: string | null;
    operationId: string;
};

type OutputPreviewFile = {
    name: string;
    description: string;
}

type OutputPreview = {
    runId: string;
    scenario: string;
    folderTree: string;
    note: string;
    files: OutputPreviewFile[];
};

type RunSummary = {
    modelId: string;
    model: string;
    input: string;
    outputFolder: string;
    status: string;
}

type DryRunResponse = {
    runId: string;
    status: GuiRunStatus;
    scenario: string;
    summary: RunSummary;
    outputPreview: OutputPreview;
}

type StartRunResponse = {
    runId: string;
    status: GuiRunStatus;
    scenario: string;
    summary: RunSummary;
    outputPreview: OutputPreview;
};

type ProgressEvent = {
    stage: string;
    message: string;
    current: number | null;
    total: number | null;
    unit: string | null;
    displayText: string;
};

type RuntimeStatus = {
    modelName: string;
    currentStep: number | null;
    totalSteps: number | null;
};

type RuntimeBuildStage = {
    stageId: string;
    modelName: string;
    current: number;
    total: number;
};

type HandInteractionMetricValues = {
    dominant: number;
    nonDominant: number;
    bilateralTotal: number;
};

type HandInteractionVisualization = {
    kind: "hand-interaction";
    durationSeconds: number;
    metrics: {
        percentInteractionTime: HandInteractionMetricValues;
        interactionDurationSeconds: HandInteractionMetricValues;
        interactionSegmentCount: HandInteractionMetricValues;
    };
    segments: Array<{
        startSeconds: number;
        endSeconds: number;
        handRole: "dominant" | "non_dominant";
    }>;
};

type AdlVisualization = {
    kind: "adl";
    durationSeconds: number;
    analyzedDurationSeconds: number;
    segments: Array<{
        startSeconds: number;
        endSeconds: number;
        activity: string;
    }>;
    activities: Array<{
        activity: string;
        durationSeconds: number;
        sessionPercent: number;
        segmentCount: number;
    }>;
    totalSegmentCount: number;
};

type ResultVisualization = HandInteractionVisualization | AdlVisualization;

type ProgressResponse = {
    runId: string;
    status: GuiRunStatus;
    errorMessage: string | null;
    outputFolder: string;
    events: ProgressEvent[];
    runtimeStatus: RuntimeStatus | null;
    runtimeBuildStages: RuntimeBuildStage[];
    outputPreview: OutputPreview;
    resultVisualization?: ResultVisualization | null;
};

type PersistedAppState = {
    step: Step;
    modelId: string;
    dominantHand: DominantHand;
    inputNames: string[];
    ignoredInputNames: string[];
    outputRoot: string;
    reviewMode: ReviewMode;
    runId: string;
    activeOperationId: string;
    progress: ProgressResponse | null;
    resultSummary: RunSummary | null;
    outputPreview: OutputPreview | null;
};

type DominantHand = "right" | "left";

const HAND_INTERACTION_MODEL_ID = "hand-interaction";
const ADL_MODEL_ID = "adl-recognition";
const DEFAULT_DOMINANT_HAND: DominantHand = "right";

const STEPS: Array<{ id: StepperStep; label: string }> = [
    { id: "select-model", label: "Select model" },
    { id: "choose-input", label: "Choose input" },
    { id: "choose-output", label: "Choose output" },
    { id: "review", label: "Review and run" },
    { id: "results", label: "Results" },
];

const APP_STATE_STORAGE_KEY = "egomodelkit.gui.state.v1";

const buttonBaseClass =
    "inline-flex min-h-12 min-w-[132px] items-center justify-center gap-2 " +
    "rounded-lg px-6 py-3 text-base font-semibold transition-colors " +
    "focus-visible:outline-3 focus-visible:outline-offset-3 " +
    "focus-visible:outline-egm-green disabled:cursor-not-allowed";

const primaryButtonClass =
    `${buttonBaseClass} border border-egm-green bg-egm-green text-white text-lg ` +
    "hover:bg-egm-green-dark disabled:border-egm-disabled " +
    "disabled:bg-egm-disabled disabled:text-white";

const secondaryButtonClass =
    `${buttonBaseClass} border border-egm-border-strong bg-white text-black text-lg ` +
    "hover:bg-egm-hover disabled:border-egm-disabled disabled:bg-white " +
    "disabled:text-egm-disabled-text";

const dangerButtonClass =
    `${buttonBaseClass} border border-egm-danger bg-egm-danger text-white text-lg ` +
    "disabled:border-egm-disabled disabled:bg-egm-disabled disabled:text-white";

const backButtonClass =
    "inline-flex min-h-12 items-center justify-center gap-2 rounded-lg " +
    "border border-transparent bg-transparent pl-0 pr-2 py-3 text-base " +
    "font-medium text-egm-back hover:text-black focus-visible:outline-3 " +
    "focus-visible:outline-offset-3 focus-visible:outline-egm-green";

export function App() {
    const initialStateRef = useRef<PersistedAppState | null | undefined>(undefined);
    if (initialStateRef.current === undefined) {
        initialStateRef.current = readPersistedAppState();
    }

    const initialState = initialStateRef.current;

    const [step, setStep] = useState<Step>(initialState?.step ?? "welcome");
    const [models, setModels] = useState<ModelInfo[]>([]);
    const [modelsLoading, setModelsLoading] = useState<boolean>(true);
    const [modelsError, setModelsError] = useState<string>("");
    const [modelId, setModelId] = useState<string>(initialState?.modelId ?? "");

    const [dominantHand, setDominantHand] = useState<DominantHand>(
        initialState?.dominantHand ?? DEFAULT_DOMINANT_HAND,
    );

    const [files, setFiles] = useState<File[]>([]);
    const [inputNames, setInputNames] = useState<string[]>(initialState?.inputNames ?? []);

    const [ignoredInputNames, setIgnoredInputNames] = useState<string[]>(
        initialState?.ignoredInputNames ?? [],
    );
    
    const fileInputRef = useRef<HTMLInputElement | null>(null);
    const activeAbortControllerRef = useRef<AbortController | null>(null);
    const [outputRoot, setOutputRoot] = useState<string>(initialState?.outputRoot ?? "");    
    const [errorMessage, setErrorMessage] = useState<string>("");
    const [isBusy, setIsBusy] = useState<boolean>(false);
    
    const [reviewMode, setReviewMode] = useState<ReviewMode>(
        initialState?.reviewMode ?? "ready",
    );
    
    const [runId, setRunId] = useState<string>(initialState?.runId ?? "");
    
    const [activeOperationId, setActiveOperationId] = useState<string>(
        initialState?.activeOperationId ?? "",
    );

    const [progress, setProgress] = useState<ProgressResponse | null>(
        initialState?.progress ?? null,
    );

    const [resultSummary, setResultSummary] = useState<RunSummary | null>(
        initialState?.resultSummary ?? null,
    );

    const [outputPreview, setOutputPreview] = useState<OutputPreview | null>(
        initialState?.outputPreview ?? null,
    );

    const selectedModel = models.find((model) => model.id === modelId) ?? null;
    const operationActive = isBusy || reviewMode === "running";

    const stepperCurrentStep: StepperStep =
        step === "output-preview"
            ? "results"
            : step === "welcome"
                ? "select-model"
                : step;

    function startNewRun() {
        setModelId("");
        setDominantHand(DEFAULT_DOMINANT_HAND);
        setFiles([]);
        setInputNames([]);
        setIgnoredInputNames([]);
        setOutputRoot("");
        setErrorMessage("");
        setIsBusy(false);
        clearReviewState();
        setStep("select-model");
    }

    function goHome() {
        setModelId("");
        setDominantHand(DEFAULT_DOMINANT_HAND);
        setFiles([]);
        setInputNames([]);
        setIgnoredInputNames([]);
        setOutputRoot("");
        setErrorMessage("");
        setIsBusy(false);
        clearReviewState();
        setStep("welcome");
    }

    function selectModel(nextModelId: string) {
        if (nextModelId !== modelId) {
            if (!modelUsesDominantHand(nextModelId)) {
                setDominantHand(DEFAULT_DOMINANT_HAND);
            }

            setModelId(nextModelId);
            setFiles([]);
            setInputNames([]);
            setIgnoredInputNames([]);
            setOutputRoot("");
            setErrorMessage("");
            clearReviewState();
        }
    }

    function clearReviewState() {
        setReviewMode("ready");
        setRunId("");
        setProgress(null);
        setResultSummary(null);
        setOutputPreview(null);
        setActiveOperationId("");
        activeAbortControllerRef.current = null;
    }

    useEffect(() => {
        let isMounted = true;

        async function loadModels() {
            try {
                setModelsLoading(true);
                setModelsError("");

                const nextModels = await requestModels();

                if (!isMounted) {
                    return;
                }

                setModels(nextModels);
            } catch {
                if (isMounted) {
                    setModelsError("Unable to load available models.");
                }
            } finally {
                if (isMounted) {
                    setModelsLoading(false);
                }
            }
        }

        void loadModels();

        return () => {
            isMounted = false;
        };
    }, []);

    function selectInput(nextFiles: File[]) {
        const supportedInputExtensions = selectedModel?.supportedInputExtensions ?? [];

        const supportedFiles = filterSupportedInputFiles(
            nextFiles,
            supportedInputExtensions,
        );

        const ignoredNames = nextFiles
            .filter((file) => !isSupportedInputFile(file, supportedInputExtensions))
            .map((file) => file.name);

        setFiles(supportedFiles);
        setInputNames(supportedFiles.map((file) => file.name));
        setIgnoredInputNames(ignoredNames);
        setOutputRoot("");
        setErrorMessage("");
        clearReviewState();
    }

    function handleFilesChange(event: ChangeEvent<HTMLInputElement>) {
        const selectedFiles = event.currentTarget.files
            ? Array.from(event.currentTarget.files)
            : [];
        
        selectInput(selectedFiles);
    }

    function handleDrop(event: DragEvent<HTMLDivElement>) {
        event.preventDefault();

        const droppedFiles = event.dataTransfer.files
            ? Array.from(event.dataTransfer.files)
            : [];
        
        if (droppedFiles.length === 0) {
            return;
        }

        selectInput(droppedFiles);
    }

    async function chooseOutputFolder() {
        try {
            setIsBusy(true);
            setErrorMessage("");

            const backendSelection = await requestNativeOutputFolder();

            if (backendSelection === null) {
                setErrorMessage(
                    "The native output folder picker is not available on this machine.",
                );

                return;
            }

            const selectedOutputRoot = backendSelection.outputRoot.trim();

            if (selectedOutputRoot.length === 0) {
                return;
            }

            setOutputRoot(selectedOutputRoot);
            clearReviewState();
        } catch {
            setErrorMessage("Unable to choose output folder.");
        } finally {
            setIsBusy(false);
        }
    }

    function viewOutputPreview() {
        setErrorMessage("");
        setOutputPreview(progress?.outputPreview ?? outputPreview);
        setStep("output-preview");
    }

    async function openOutputFolder() {
        await openOutputFolderForRun(runId);
    }

    async function openOutputFolderForRun(targetRunId: string) {
        if (targetRunId.length === 0) {
            return;
        }

        try {
            setIsBusy(true);
            setErrorMessage("");

            const outputFolder =
                progress?.outputFolder ?? resultSummary?.outputFolder;

            await requestOpenOutputFolder({
                runId: targetRunId,
                outputFolder,
            });
        } catch (error) {
            setErrorMessage(
                userFacingRequestError(error, "Unable to open output folder."),
            );
        } finally {
            setIsBusy(false);
        }
    }

    function confirmNavigateAwayFromActiveOperation(): boolean {
        if (!operationActive) {
            return true;
        }

        return window.confirm(
            "A model operation is currently in progress. Leaving this page will cancel " +
            "the backend operation and progress will be lost. Continue?",
        );
    }

    async function cancelActiveOperation({
        returnHome,
    }: {
        returnHome: boolean;
    }) {
        const operationId = activeOperationId.length > 0 ? activeOperationId : null;
        const currentRunId = runId.length > 0 ? runId : null;

        try {
            setErrorMessage("");

            if (operationId !== null || currentRunId !== null) {
                await requestCancelRun({
                    runId: currentRunId,
                    operationId,
                });
            }
        } catch {
            setErrorMessage("Unable to cancel the backend operation.");
        } finally {
            activeAbortControllerRef.current?.abort();
            activeAbortControllerRef.current = null;
            setIsBusy(false);
            setActiveOperationId("");
            setProgress(null);
            setReviewMode("ready");

            if (returnHome) {
                goHome();
            }
        }
    }

    function cancelRunFromReview() {
        void cancelActiveOperation({ returnHome: false });
    }

    function navigateHome() {
        if (!confirmNavigateAwayFromActiveOperation()) {
            return;
        }

        if (operationActive) {
            void cancelActiveOperation({ returnHome: true });
            return;
        }

        goHome();
    }

    async function runDryRun() {
        const operationId = buildClientOperationId();
        const abortController = new AbortController();

        activeAbortControllerRef.current = abortController;
        setActiveOperationId(operationId);

        try {
            setIsBusy(true);
            setErrorMessage("");

           const body = await postMultipart<DryRunResponse>("/api/dry-run", {
                modelId,
                outputRoot,
                files,
                dominantHand,
                operationId,
                signal: abortController.signal,
            });

            setRunId(body.runId);
            setResultSummary(body.summary);
            setOutputPreview(body.outputPreview);
            setProgress(null);
            setReviewMode("dry-run-complete");
        } catch (error) {
            if (!isAbortError(error)) {
                setErrorMessage(
                    userFacingRequestError(
                        error, 
                        "Unable to complete dry run.")
                );
                setReviewMode("ready");
            }
        } finally {
            setIsBusy(false);
            setActiveOperationId("");
            activeAbortControllerRef.current = null;
        }
    }

    async function startRun() {
        const operationId = buildClientOperationId();
        const abortController = new AbortController();
        let started = false;

        activeAbortControllerRef.current = abortController;
        setActiveOperationId(operationId);

        try {
            setIsBusy(true);
            setErrorMessage("");

            const body = await postMultipart<StartRunResponse>("/api/runs", {
                modelId,
                outputRoot,
                files,
                dominantHand,
                operationId,
                signal: abortController.signal,
            });


            started = true;

            setRunId(body.runId);
            setResultSummary(body.summary);
            setOutputPreview(body.outputPreview);

            setProgress({
                runId: body.runId,
                status: "running",
                errorMessage: null,
                outputFolder: body.summary.outputFolder,
                events: [],
                runtimeStatus: null,
                runtimeBuildStages: [],
                outputPreview: body.outputPreview,
            });

            setReviewMode("running");
        } catch (error) {
            if (!isAbortError(error)) {
                setErrorMessage(
                    userFacingRequestError(
                        error, 
                        "Unable to start model run.")
                );

                setRunId("");
                setProgress(null);
                setResultSummary(null);
                setOutputPreview(null);
                setReviewMode("ready");
            }
        } finally {
            setIsBusy(false);
            activeAbortControllerRef.current = null;

            if (!started) {
                setActiveOperationId("");
            }
        }
    }

    useEffect(() => {
        if (reviewMode !== "running" || runId.length === 0) {
            return;
        }

        let isMounted = true;
        let timeoutId: number | null = null;
        let consecutiveFailures = 0;

        async function pollProgress() {
            try {
                const body = await requestProgress(runId);

                if (!isMounted) {
                    return;
                }

                consecutiveFailures = 0;
                setErrorMessage("");
                setProgress(body);

                setOutputPreview((currentOutputPreview) => (
                    body.outputPreview ?? currentOutputPreview
                ));

                if (body.status === "completed" || body.status === "failed") {
                    isMounted = false;
                    setActiveOperationId("");
                    setReviewMode("ready");
                    setStep("results");

                    return;
                }

                if (body.status === "cancelled") {
                    isMounted = false;
                    goHome();

                    return;
                }
            } catch {
                if (!isMounted) {
                    return;
                }

                consecutiveFailures += 1;

                if (consecutiveFailures >= 3) {
                    setErrorMessage("Unable to refresh run progress.");
                }
            }

            timeoutId = window.setTimeout(pollProgress, 1000);
        }

        void pollProgress();

        return () => {
            isMounted = false;

            if (timeoutId !== null) {
                window.clearTimeout(timeoutId);
            }
        };
    }, [reviewMode, runId]);

    useEffect(() => {
       writePersistedAppState({
            step,
            modelId,
            dominantHand,
            inputNames,
            ignoredInputNames,
            outputRoot,
            reviewMode,
            runId,
            activeOperationId,
            progress,
            resultSummary,
            outputPreview,
        });
    }, [
        step,
        modelId,
        dominantHand,
        inputNames,
        ignoredInputNames,
        outputRoot,
        reviewMode,
        runId,
        activeOperationId,
        progress,
        resultSummary,
        outputPreview,
    ]);

    return (
        <div className="min-h-screen bg-egm-bg text-black">
            <div className="flex min-h-screen flex-col">
                <header className="border-b border-egm-header-border bg-white">
                    <div
                        className="
                            mx-auto flex h-[68px] w-full max-w-[1040px] items-center px-6
                        "
                    >
                        <button
                            className="
                                rounded-md bg-transparent text-left text-[26px] font-normal
                                leading-none tracking-[0.01em] focus-visible:outline-3 
                                focus-visible:outline-offset-3 focus-visible:outline-egm-green
                            "
                            type="button"
                            onClick={navigateHome}
                        >
                            EgoModelKit
                        </button>
                    </div>
                </header>

                {step === "welcome" ? (
                    <WelcomeScreen onStart={startNewRun} />
                ) : (
                    <main
                        className="
                            mx-auto grid min-h-[calc(100vh-68px)] w-full max-w-[1040px] 
                            grid-cols-1 gap-8 px-6 pt-16 pb-0 
                            md:grid-cols-[220px_minmax(0,1fr)] md:pt-14
                        ">
                        <Stepper currentStep={stepperCurrentStep} />

                        <section aria-live="polite" className="flex min-h-0 min-w-0 flex-col">
                            {errorMessage ? (
                                <div
                                    className="
                                        mb-6 rounded-xl border border-egm-danger-border
                                        bg-egm-danger-soft px-5 py-4 text-base
                                        text-egm-danger
                                    "
                                    role="alert"
                                >
                                    {errorMessage}
                                </div>
                            ) : null}
                            
                            {step === "select-model" ? (
                                <SelectModelScreen
                                    models={models}
                                    modelsLoading={modelsLoading}
                                    modelsError={modelsError}
                                    selectedModelId={modelId}
                                    onSelectModel={selectModel}
                                    dominantHand={dominantHand}
                                    onDominantHandChange={setDominantHand}
                                    canContinue={modelId.length > 0}
                                    onBack={() => setStep("welcome")}
                                    onContinue={() => setStep("choose-input")}
                                />
                            ) : step === "choose-input" && selectedModel !== null ? (
                                <ChooseInputScreen 
                                    selectedModel={selectedModel}
                                    files={files}
                                    ignoredInputNames={ignoredInputNames}
                                    fileInputRef={fileInputRef}
                                    onFilesChange={handleFilesChange}
                                    onDrop={handleDrop}
                                    canContinue={files.length > 0}
                                    onBack={() => setStep("select-model")}
                                    onContinue={() => setStep("choose-output")}
                                />
                            ) : step === "choose-output" ? (
                                <ChooseOutputScreen
                                    outputRoot={outputRoot}
                                    isBusy={isBusy}
                                    onChooseOutputFolder={chooseOutputFolder}
                                    canContinue={outputRoot.trim().length > 0 && !isBusy}
                                    onBack={() => setStep("choose-input")}
                                    onContinue={() => setStep("review")}
                                />
                            ) : step === "review" && selectedModel !== null ? (
                                <ReviewScreen
                                    selectedModel={selectedModel}
                                    dominantHand={
                                        modelUsesDominantHand(selectedModel.id)
                                            ? dominantHand
                                            : null
                                    }
                                    files={files}
                                    outputRoot={outputRoot}
                                    reviewMode={reviewMode}
                                    progress={progress}
                                    runId={runId}
                                    isBusy={isBusy}
                                    onBack={() => setStep("choose-output")}
                                    onDryRun={runDryRun}
                                    onRun={startRun}
                                    inputNames={inputNames}
                                    onCancelRun={cancelRunFromReview}               
                                />
                            ) : step === "results" && selectedModel !== null ? (
                                <ResultsScreen
                                    selectedModel={selectedModel}
                                    dominantHand={
                                        modelUsesDominantHand(selectedModel.id)
                                            ? dominantHand
                                            : null
                                    }
                                    files={files}
                                    runId={runId}
                                    resultSummary={resultSummary}
                                    progress={progress}
                                    isBusy={isBusy}
                                    canViewOutputPreview={
                                        progress?.status === "completed" &&
                                        Boolean(progress?.outputPreview ?? outputPreview)
                                    }
                                    onOpenOutputFolder={openOutputFolder}
                                    onStartNewRun={startNewRun}
                                    onViewOutputPreview={viewOutputPreview}
                                    inputNames={inputNames}
                                />
                            ) : step === "output-preview" && outputPreview !== null ? ( 
                                <OutputPreviewScreen 
                                    outputPreview={outputPreview}
                                    isBusy={isBusy}
                                    onBack={() => setStep("results")}
                                    onOpenOutputFolder={openOutputFolder}
                                />
                            ) : (
                                <SelectModelScreen
                                    models={models}
                                    modelsLoading={modelsLoading}
                                    modelsError={modelsError}
                                    selectedModelId={modelId}
                                    onSelectModel={selectModel}
                                    dominantHand={dominantHand}
                                    onDominantHandChange={setDominantHand}
                                    canContinue={modelId.length > 0}
                                    onBack={() => setStep("welcome")}
                                    onContinue={() => setStep("choose-input")}
                                />
                            )}
                        </section>
                    </main>
                )}
            </div>
        </div>
    );
}

function WelcomeScreen({ onStart }: { onStart: () => void }) {
    const workflowSteps = [
        { label: "Video", Icon: Video },
        { label: "Model", Icon: Cpu },
        { label: "Metrics", Icon: BarChart3 },
        { label: "Research outputs", Icon: FileText },
    ];

    return (
        <main className="mx-auto w-full max-w-[800px] px-6 pt-12 pb-10 text-center sm:pt-14">
            <div
                className="
                    mx-auto flex h-20 w-20 items-center justify-center rounded-3xl
                    bg-egm-green-soft text-egm-green
                "
            >
                <Video aria-hidden="true" size={40} strokeWidth={2.1} />
            </div>

            <h1 className="mt-6 text-[42px] font-semibold leading-[1.1] tracking-[-0.04em]">
                EgoModelKit
            </h1>

            <p className="mt-5 text-2xl font-semibold leading-[1.35] text-egm-subtitle">
                Egocentric video analysis for rehabilitation research
            </p>

            <p className="mx-auto mt-6 max-w-[680px] text-lg leading-7 text-egm-body-copy">
                Analyze daily-activity videos using computer vision models and generate
                structured hand-use metrics, timelines, and reproducible research outputs.
            </p>

            <div className="mt-10 rounded-2xl border border-egm-list-border bg-white px-6 py-6">
                <div className="flex flex-wrap items-start justify-center gap-x-4 gap-y-5">
                    {workflowSteps.map(({ label, Icon }, index) => (
                        <div className="contents" key={label}>
                            <div className="min-w-[112px] text-center">
                                <div
                                    className="
                                        mx-auto flex h-12 w-12 items-center justify-center
                                        rounded-2xl bg-egm-green-tint text-egm-green
                                    "
                                >
                                    <Icon aria-hidden="true" size={25} strokeWidth={2} />
                                </div>
                                <p className="mt-2 text-sm font-semibold text-egm-strong-copy">
                                    {label}
                                </p>
                            </div>

                            {index < workflowSteps.length - 1 ? (
                                <ChevronRight
                                    aria-hidden="true"
                                    className="mt-3 hidden text-egm-step-border sm:block"
                                    size={26}
                                    strokeWidth={1.8}
                                />
                            ) : null}
                        </div>
                    ))}
                </div>
            </div>

            <div
                className="
                    mx-auto mt-10 w-full max-w-[560px] rounded-2xl border
                    border-egm-blue-border bg-egm-blue-soft px-6 py-4 text-left
                    text-base font-medium leading-6 text-blue-800
                "
            >
                <strong>Local processing.</strong> All analysis runs on your computer.
                No data is uploaded or shared.
            </div>

            <button
                className="
                    mt-10 min-h-14 min-w-[220px] rounded-xl border border-egm-green
                    bg-egm-green px-9 py-3 text-lg font-semibold text-white
                    hover:bg-egm-green-dark focus-visible:outline-3
                    focus-visible:outline-offset-3 focus-visible:outline-egm-green
                "
                type="button"
                onClick={onStart}
            >
                Start analysis
            </button>
        </main>
    );
}

function SelectModelScreen({
    models,
    modelsLoading,
    modelsError,
    selectedModelId,
    onSelectModel,
    dominantHand,
    onDominantHandChange,
    canContinue,
    onBack,
    onContinue,
}: {
    models: ModelInfo[];
    modelsLoading: boolean;
    modelsError: string;
    selectedModelId: string;
    onSelectModel: (modelId: string) => void;
    dominantHand: DominantHand;
    onDominantHandChange: (dominantHand: DominantHand) => void;
    canContinue: boolean;
    onBack: () => void;
    onContinue: () => void;
}) {
    const interactionSettingsRef = useRef<HTMLElement | null>(null);

    useEffect(() => {
        if (selectedModelId === HAND_INTERACTION_MODEL_ID) {
            interactionSettingsRef.current!.scrollIntoView({
                behavior: "smooth",
                block: "center",
            });
        }
    }, [selectedModelId]);

    return (
        <>
            <PageHeading
                title="Select a model"
                subtitle="Choose the workflow you want to run."
            />

            {modelsLoading ? (
                <div
                    className="
                        rounded-2xl border border-egm-card-border bg-white
                        px-5 py-4 text-base text-egm-body-copy
                    "
                    role="status"
                >
                    Loading available models...
                </div>
            ) : modelsError ? (
                <div
                    className="
                        mt-8 rounded-xl border border-egm-danger-border
                        bg-egm-danger-soft px-5 py-4 text-base text-egm-danger
                    "
                    role="alert"
                >
                    {modelsError}
                </div>
            ) : models.length === 0 ? (
                <div
                    className="
                        mt-8 rounded-2xl border border-egm-card-border bg-white
                        px-5 py-4 text-base text-egm-body-copy
                    "
                >
                    No models are available from the local backend.
                </div>
            ) : (
                <div
                    aria-label="Available models"
                    className="mt-8 flex flex-col gap-4"
                    role="group"
                >
                    {models.map((model) => {
                        const selected = model.id === selectedModelId;

                        return (
                            <button
                                key={model.id}
                                aria-pressed={selected}
                                className={[
                                    "flex min-h-[168px] w-full items-start rounded-2xl",
                                    "px-6 py-6 text-left transition-colors",
                                    selected
                                        ? `
                                            border-2 border-black bg-egm-green-tint
                                        `
                                        : `
                                            border border-egm-card-border 
                                            hover:bg-egm-hover bg-white
                                        ` 
                                ].join(" ")}
                                type="button"
                                onClick={() => onSelectModel(model.id)}
                            >
                                <span>
                                    <span
                                        className={[
                                            "flex h-7 w-7 shrink-0 items-center",
                                            "justify-center rounded-full border-[1px]",
                                            selected
                                                ? "bg-egm-green"
                                                : "bg-white"
                                        ].join(" ")}
                                    >
                                        <span 
                                            className="
                                                h-4 w-4 rounded-full bg-white
                                            " 
                                        />
                                    </span>
                                </span>
                                <span className="ml-5 flex flex-col">
                                    <span className="text-2xl font-medium leading-none">
                                        {model.name}
                                    </span>

                                    <span 
                                        className="
                                            mt-4 text-base leading-6 text-egm-body-copy
                                        "
                                    >
                                        {model.description}
                                    </span>

                                    <span 
                                        className="
                                            mt-6 text-base leading-6 
                                            text-egm-secondary-copy
                                            "
                                        >
                                        {modelInputLabel(model)}
                                        <br />
                                        {modelOutputLabel(model)}
                                    </span>
                                </span>
                            </button>
                        )
                    })}
                </div>
            )}

           {modelUsesDominantHand(selectedModelId) ? (
                <InteractionSettingsPanel
                    sectionRef={interactionSettingsRef}
                    dominantHand={dominantHand}
                    onDominantHandChange={onDominantHandChange}
                />
            ) : null}

            <FooterActions
                onBack={onBack}
                onContinue={onContinue}
                continueLabel="Continue"
                continueDisabled={!canContinue || modelsLoading || modelsError.length > 0}
            />
        </>
    );
}

function InteractionSettingsPanel({
    sectionRef,
    dominantHand,
    onDominantHandChange,
}: {
    sectionRef: RefObject<HTMLElement | null>;
    dominantHand: DominantHand;
    onDominantHandChange: (dominantHand: DominantHand) => void;
}) {
    const options: DominantHand[] = ["right", "left"];

    return (
        <section
            ref={sectionRef}
            aria-labelledby="interaction-settings-heading"
            className="
                mt-6 rounded-2xl border border-egm-card-border bg-white px-5 py-5
                text-base text-egm-body-copy
            "
        >
            <h2
                id="interaction-settings-heading"
                className="text-base font-semibold text-black"
            >
                Interaction settings
            </h2>

            <fieldset className="mt-5">
                <legend className="font-semibold text-egm-strong-copy">
                    Dominant hand after injury
                </legend>

                <div className="mt-3 grid grid-cols-2 gap-2">
                    {options.map((option) => {
                        const selected = dominantHand === option;

                        return (
                            <label
                                key={option}
                                className={[
                                    "inline-flex min-h-10 cursor-pointer items-center",
                                    "justify-center rounded-lg border px-5 py-2",
                                    "font-semibold transition-colors",
                                    "focus-within:outline-3 focus-within:outline-offset-3",
                                    "focus-within:outline-egm-green",
                                    selected
                                        ? [
                                            "border-egm-green bg-egm-green-tint",
                                            "text-egm-green",
                                        ].join(" ")
                                        : [
                                            "border-egm-radio-border bg-white",
                                            "text-egm-strong-copy hover:bg-egm-hover",
                                        ].join(" "),
                                ].join(" ")}
                            >
                                <input
                                    checked={selected}
                                    className="sr-only"
                                    name="dominant-hand"
                                    type="radio"
                                    value={option}
                                    onChange={() => onDominantHandChange(option)}
                                />
                                {dominantHandLabel(option)}
                            </label>
                        );
                    })}
                </div>
            </fieldset>

            <p className="mt-3 text-sm leading-6 text-egm-secondary-copy">
                Used to label dominant-hand and non-dominant-hand interaction metrics.
            </p>
        </section>
    );
}

function Stepper({ currentStep }: { currentStep: StepperStep }) {
    const currentIndex = STEPS.findIndex((step) => step.id === currentStep);

    return (
        <aside className="w-full md:w-[220px]">
            {STEPS.map((step, index) => {
                const complete = index < currentIndex;
                const current = index === currentIndex;

                return (
                    <div className="mb-4 flex min-h-10 items-center" key={step.id}>
                        <span className={[
                            "flex h-10 w-10 shrink-0 items-center justify-center",
                            "rounded-full text-lg leading-none",
                            complete
                                ? "bg-egm-green-soft text-egm-green"
                                : current
                                    ? "border-[3px] border-egm-green text-egm-green"
                                    : "border-[3px] border-egm-step-border text-egm-step-text",
                        ].join(" ")}
                        >
                            {complete ? (
                                <Check aria-hidden="true" size={22} strokeWidth={2.8} />
                            ) : (
                                index + 1
                            )}
                        </span>

                        <span
                            className={[
                                "flex ml-4 text-base leading-none",
                                current ? "text-black" : "text-egm-step-label",
                            ].join(" ")}
                        >
                            {step.label}
                        </span>
                    </div>
                )
            })}
        </aside>
    );
}

function PageHeading({ title, subtitle }: {title: string; subtitle: string }) {
    return (
        <header>            
            <h1 className="text-[30px] font-normal leading-[1.15] tracking-[-0.03em]">
                {title}
            </h1>

            <p className="mt-3.5 text-lg font-normal leading-[1.45] text-egm-subtitle">
                {subtitle}
            </p>
        </header>
    )
}

function FooterActions({
    onBack,
    onContinue,
    continueLabel,
    continueDisabled,
} : {
    onBack: () => void;
    onContinue: () => void;
    continueLabel: string;
    continueDisabled: boolean;
}) {
    return (
        <div 
            className="
                sticky bottom-0 z-10 mt-auto flex items-center justify-between gap-4
                bg-egm-bg pt-8 pb-4
            "
        >
            <button className={backButtonClass} type="button" onClick={onBack}>
                <ChevronLeft aria-hidden="true" size={22} strokeWidth={2.4} />
                Back
            </button>
            
            <button 
                className={primaryButtonClass} 
                disabled={continueDisabled}
                type="button" 
                onClick={onContinue}
            >
                {continueLabel}
            </button>
        </div>
    )
}

function inputLabelFromNames(inputNames: string[]): string {
    if (inputNames.length === 0) {
        return "Not available";
    }

    if (inputNames.length === 1) {
        return inputNames[0];
    }

    return `${inputNames.length} files`;
}

function filterSupportedInputFiles(
    files: File[],
    supportedInputExtensions: string[],
): File[] {
    return files.filter((file) =>
        isSupportedInputFile(file, supportedInputExtensions),
    );
}

function isSupportedInputFile(
    file: File,
    supportedInputExtensions: string[],
): boolean {
    const lowerCaseName = file.name.toLowerCase();

    return supportedInputExtensions.some((extension) =>
        lowerCaseName.endsWith(extension.toLowerCase()),
    );
}

function supportedInputAccept(model: ModelInfo): string {
    return model.supportedInputExtensions.join(",");
}

async function requestModels(): Promise<ModelInfo[]> {
    const response = await fetch("/api/models");

    if (!response.ok) {
        throw new Error(`Request failed with status ${response.status}.`);
    }

    const body = (await response.json()) as ModelsResponse;

    return body.models;
}

async function requestNativeOutputFolder(): Promise<SelectOutputFolderResponse | null> {
    const response = await fetch("/api/select-output-folder", {
        method: "POST",
    });

    if ([404, 405].includes(response.status)) {
        return null;
    }

    if (!response.ok) {
        throw new Error(`Request failed with status ${response.status}.`);
    }

    return (await response.json()) as SelectOutputFolderResponse;
}

async function requestOpenOutputFolder({
    runId,
    outputFolder,
} : {
    runId: string;
    outputFolder?: string;
}) : Promise<OpenOutputFolderResponse> {
    const response = await fetch("/api/open-output-folder", {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify({ runId, outputFolder }),
    });

    if (!response.ok) {
        throw new ApiRequestError(await responseErrorDetail(response));
    }

    return (await response.json()) as OpenOutputFolderResponse;
}

async function requestCancelRun(request: CancelRunRequest): Promise<CancelRunResponse> {
    const response = await fetch("/api/cancel-run", {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify(request),
    });

    if (!response.ok) {
        throw new Error(`Request failed with status ${response.status}.`);
    }

    return (await response.json()) as CancelRunResponse;
}

function modelInputLabel(model: ModelInfo): string {
    return withLabelPrefix("Input", model.acceptedInputLabel);
}

function modelOutputLabel(model: ModelInfo): string {
    return withLabelPrefix("Output", model.outputLabel);
}

function withLabelPrefix(prefix: string, value: string): string {
    return value.startsWith(`${prefix}:`) ? value : `${prefix}: ${value}`;
}

function selectedLabelFromFiles(files: File[]): string {
    if (files.length === 1) {
        return "Selected: 1 file";
    }

    return `Selected: ${files.length} files`;
}

function ignoredLabelFromFileNames(fileNames: string[]): string {
    if (fileNames.length === 1) {
        return "Ignored: 1 file";
    }

    return `Ignored: ${fileNames.length} files`;
}

function ignorelDescriptionFromFileNames(fileNames: string[]): string {
    if (fileNames.length === 1) {
        return "This file is not supported by the selected model";
    }

    return "These files are not supported by the selected model";
}

function buildClientOperationId(): string {
    if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
        return `operation-${crypto.randomUUID()}`;
    }

    return `operation-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function isAbortError(error: unknown): boolean {
    return error instanceof DOMException && error.name === "AbortError";
}

function readPersistedAppState(): PersistedAppState | null {
    try {
        const rawState = window.localStorage.getItem(APP_STATE_STORAGE_KEY);

        if (rawState === null) {
            return null;
        }

        return normalizePersistedAppState(JSON.parse(rawState));
    } catch {
        return null;
    }
}

function writePersistedAppState(state: PersistedAppState): void {
    try {
        window.localStorage.setItem(APP_STATE_STORAGE_KEY, JSON.stringify(state));
    } catch {
        // Ignore storage failures so the local GUI remains usable.
    }
}

function normalizePersistedAppState(value: unknown): PersistedAppState | null {
    if (!isRecord(value) || !isStep(value.step)) {
        return null;
    }

    return {
        step: value.step,
        modelId: stringValue(value.modelId),
        dominantHand: isDominantHand(value.dominantHand)
            ? value.dominantHand
            : DEFAULT_DOMINANT_HAND,
        inputNames: stringArrayValue(value.inputNames),
        ignoredInputNames: stringArrayValue(value.ignoredInputNames),
        outputRoot: stringValue(value.outputRoot),
        reviewMode: isReviewMode(value.reviewMode) ? value.reviewMode : "ready",
        runId: stringValue(value.runId),
        activeOperationId: stringValue(value.activeOperationId),
        progress: isProgressResponse(value.progress) ? value.progress : null,
        resultSummary: isRunSummary(value.resultSummary) ? value.resultSummary : null,
        outputPreview: isOutputPreview(value.outputPreview) ? value.outputPreview : null,
    };
}

function isStep(value: unknown): value is Step {
    return [
        "welcome",
        "select-model",
        "choose-input",
        "choose-output",
        "review",
        "results",
        "output-preview",
    ].includes(String(value));
}

function isReviewMode(value: unknown): value is ReviewMode {
    return ["ready", "dry-run-complete", "running"].includes(String(value));
}

function isDominantHand(value: unknown): value is DominantHand {
    return value === "right" || value === "left";
}

function modelUsesDominantHand(modelId: string): boolean {
    return modelId === HAND_INTERACTION_MODEL_ID;
}

function dominantHandLabel(value: DominantHand): string {
    return value === "right" ? "Right" : "Left";
}

function isProgressResponse(value: unknown): value is ProgressResponse {
    return isRecord(value) && typeof value.runId === "string";
}

function isOutputPreview(value: unknown): value is OutputPreview {
    return isRecord(value) && typeof value.runId === "string";
}

function isRunSummary(value: unknown): value is RunSummary {
    return isRecord(value) && typeof value.model === "string";
}

function isRecord(value: unknown): value is Record<string, unknown> {
    return typeof value === "object" && value !== null;
}

function stringValue(value: unknown): string {
    return typeof value === "string" ? value : "";
}

function stringArrayValue(value: unknown): string[] {
    return Array.isArray(value)
        ? value.filter((item): item is string => typeof item === "string")
        : [];
}

class ApiRequestError extends Error {
    constructor(public readonly detail: string | null) {
        super(detail ?? "Request failed.");
    }
}

function userFacingRequestError(error: unknown, fallback: string): string {
    if (error instanceof ApiRequestError && error.detail !== null) {
        return error.detail;
    }

    return fallback;
}

async function responseErrorDetail(response: Response): Promise<string | null> {
    try {
        const body = (await response.json()) as { detail?: unknown };

        return typeof body.detail === "string" && body.detail.length > 0
            ? body.detail
            : null;
    } catch {
        return null;
    }
}

async function postMultipart<T>(
    url: string,
    {
        modelId,
        outputRoot,
        files,
        dominantHand,
        operationId,
        signal
    }: {
        modelId: string;
        outputRoot: string;
        files: File[];
        dominantHand: DominantHand;
        operationId: string;
        signal: AbortSignal;
    },

): Promise<T> {
    const formData = new FormData();

    formData.append("modelId", modelId);
    formData.append("outputRoot", outputRoot);

    if (modelUsesDominantHand(modelId)) {
        formData.append("dominantHand", dominantHand);
    }

    for (const file of files) {
        formData.append("files", file, file.name);
    }

    formData.append("operationId", operationId);

    const response = await fetch(url, {
        method: "POST",
        body: formData,
        signal,
    });

    if (!response.ok) {
        throw new ApiRequestError(await responseErrorDetail(response));
    }

    return (await response.json()) as T;
}

function ChooseInputScreen({
    selectedModel,
    files,
    ignoredInputNames,
    fileInputRef,
    onFilesChange,
    onDrop,
    canContinue,
    onBack,
    onContinue,
} : {
    selectedModel: ModelInfo;
    files: File[];
    ignoredInputNames: string[];
    fileInputRef: RefObject<HTMLInputElement | null>;
    onFilesChange: (event: ChangeEvent<HTMLInputElement>) => void;
    onDrop: (event: DragEvent<HTMLDivElement>) => void;
    canContinue: boolean;
    onBack: () => void;
    onContinue: () => void;
}) {
    const subtitle = `Select ${selectedModel.acceptedInputLabel}`;

    return (
        <>
            <PageHeading title="Choose input" subtitle={subtitle} />

            <div 
                className="
                    mt-8 flex min-h-[280px] flex-col items-center justify-center
                    rounded-2xl border-2 border-dashed border-egm-dashed bg-white
                    px-6 py-10 text-center hover:bg-egm-hover
                "
                data-testid="input-drop-zone"
                onDragOver={(event) => event.preventDefault()}
                onDrop={onDrop}
            >
                <div 
                    className="
                        flex h-16 w-16 items-center justify-center rounded-full
                        bg-egm-icon-bg
                    "
                >
                    <Upload aria-hidden="true" size={36} strokeWidth={2.0} />
                </div>

                <h2 className="mt-6 text-2xl font-normal leading-none">
                    Drop input or choose from your computer
                </h2>

                <button
                    className="
                        mt-7 min-h-12 rounded-lg border border-egm-border-strong
                        bg-white px-7 py-3 text-base hover:bg-egm-hover
                        focus-visible:outline-3 focus-visible:outline-offset-3
                        focus-visible:outline-egm-green
                    "
                    type="button"
                    onClick={() => fileInputRef.current?.click()}
                >
                    Choose input
                </button>

                <p className="mt-5 text-sm leading-6 text-egm-secondary-copy">
                    Supported files: {selectedModel.supportedInputExtensions.join(", ")}
                </p>

                <input
                    ref={fileInputRef}
                    accept={supportedInputAccept(selectedModel)}
                    aria-label="Choose input files"
                    className="hidden"
                    multiple
                    type="file"
                    onChange={onFilesChange}
                />
            </div>

            {files.length === 0 ? (
                <p className="mt-6 text-base text-egm-body-copy">
                    No input selected yet.
                </p>
            ) : (
                <div
                    className="
                        mt-6 rounded-2xl border border-egm-card-border 
                        bg-egm-success-soft px-6 py-4 text-base text-egm-body-copy
                    "
                >
                    <p className="font-semibold text-egm-strong-copy">
                        {selectedLabelFromFiles(files)}
                    </p>
                    
                    <ul className="mt-3 list-disc space-y-1 pl-5">
                        {files.map((file, index) => (
                            <li key={`${file.name}-${index}`}>{file.name}</li>
                        ))}
                    </ul>
                </div>
            )}

            {ignoredInputNames.length > 0 ? (
                <div
                    className="
                        mt-4 rounded-2xl border border-egm-card-border bg-white
                        px-6 py-4 text-base text-egm-body-copy
                    "
                >
                    <p className="font-semibold text-egm-strong-copy">
                        {ignoredLabelFromFileNames(ignoredInputNames)}
                    </p>
                    
                    <p className="mt-2 text-sm leading-6 text-egm-secondary-copy">
                        {ignorelDescriptionFromFileNames(ignoredInputNames)}
                    </p>
                    
                    <ul className="mt-3 list-disc space-y-1 pl-5">
                        {ignoredInputNames.map((fileName, index) => (
                            <li key={`${fileName}-${index}`}>{fileName}</li>
                        ))}
                    </ul>
                </div>
            ) : null}

            <FooterActions
                onBack={onBack}
                onContinue={onContinue}
                continueLabel="Continue"
                continueDisabled={!canContinue}
            />
        </>
    )
}

function ChooseOutputScreen({
    outputRoot,
    isBusy,
    onChooseOutputFolder,
    canContinue,
    onBack,
    onContinue,
} : {
    outputRoot: string;
    isBusy: boolean;
    onChooseOutputFolder: () => void;
    canContinue: boolean;
    onBack: () => void;
    onContinue: () => void;
}) {
    return (
        <>
            <PageHeading
                title="Choose output folder"
                subtitle="Select a folder where EgoModelKit should save the results."
            />

            <div
                className="
                    mt-8 rounded-2xl border border-egm-card-border bg-white px-14 py-10
                "
            >
                <div className="flex items-center gap-5">
                    <div className="
                        flex h-14 w-14 shrink-0 items-center justify-center rounded-lg
                        bg-egm-icon-bg
                        "
                    >
                        <Folder aria-hidden="true" size={30} strokeWidth={2.0} />
                    </div>
                    <p className="
                        min-w-0 break-words text-base text-egm-secondary-copy
                        "
                    >
                        {outputRoot.trim()
                            ? outputRoot.trim()
                            : "No output folder selected"}
                    </p>
                </div>

                <button
                    className="
                            mt-7 min-h-12 rounded-lg border border-egm-border-strong
                            bg-white px-6 py-3 text-base hover:bg-egm-hover
                            focus-visible:outline-3 focus-visible:outline-offset-3
                            focus-visible:outline-egm-green disabled:cursor-not-allowed
                            disabled:border-egm-disabled disabled:text-egm-disabled-text
                    "
                    disabled={isBusy}
                    type="button"
                    onClick={onChooseOutputFolder}
                >
                    Choose Output Folder
                </button>
            </div>

            <div
                className="
                    mt-5 rounded-xl border border-egm-blue-border bg-egm-blue-soft
                    px-5 py-4 text-base text-egm-body-copy
                "
            >
                A new run folder will be created inside the selected output folder.
            </div>

           <FooterActions
                onBack={onBack}
                onContinue={onContinue}
                continueLabel="Continue"
                continueDisabled={!canContinue}
           />
        </>
    )
}

function ReviewScreen({
    selectedModel,
    dominantHand,
    files,
    outputRoot,
    reviewMode,
    progress,
    runId,
    isBusy,
    onBack,
    onDryRun,
    onRun,
    inputNames,
    onCancelRun,
} : {
    selectedModel: ModelInfo;
    dominantHand: DominantHand | null;
    files: File[];
    outputRoot: string;
    reviewMode: ReviewMode;
    progress: ProgressResponse | null;
    runId: string;
    isBusy: boolean;
    onBack: () => void;
    onDryRun: () => void;
    onRun: () => void;
    inputNames: string[];
    onCancelRun: () => void;
}) {
    const operationActive = isBusy || reviewMode === "running";

    return (
        <>
            <PageHeading 
                title="Review and run"
                subtitle="Confirm the model, input, and output location before starting."
            />

            <SummaryPanel
                selectedModel={selectedModel}
                inputLabel={inputLabelFromNames(inputNames)}
                outputRoot={outputRoot}
                dominantHand={dominantHand}
            />

            <div
                className="
                    mt-5 rounded-xl border border-egm-blue-border bg-egm-blue-soft
                    px-5 py-4 text-base text-egm-body-copy
                "
            >
                Dry run checks the input, output folder, and local
                setup without running the full model.
            </div>

            {reviewMode === "ready" ? <ReadyPanel /> : null}

            {reviewMode === "dry-run-complete" ? (
                <DryRunCompletePanel runId={runId} />
            ) : null}

            {reviewMode === "running" ? (
                <RunningPanel 
                    events={progress?.events ?? []} 
                    runId={runId}
                    runtimeStatus={progress?.runtimeStatus ?? null}
                    runtimeBuildStages={progress?.runtimeBuildStages ?? []}
                />
            ) : null}

            <div 
                className="
                    sticky bottom-0 z-10 mt-auto flex items-center justify-between gap-4
                    bg-egm-bg pt-8 pb-4
                "
            >
                <button 
                    className={backButtonClass} 
                    disabled={operationActive}
                    type="button" 
                    onClick={onBack}
                >
                    <ChevronLeft aria-hidden="true" size={22} strokeWidth={2.0} />
                    Back
                </button>

                <div className="flex gap-3">
                    {operationActive ? (
                        <button
                            className={dangerButtonClass}
                            type="button"
                            onClick={onCancelRun}
                        >
                            Cancel Run
                        </button>
                    ) : (
                        <>
                            <button
                                className={secondaryButtonClass}
                                type="button"
                                onClick={onDryRun}
                            >
                                Dry Run
                            </button>
                            <button
                                className={primaryButtonClass}
                                type="button"
                                onClick={onRun}
                            >
                                Run Model
                            </button>
                        </>
                    )}
                </div>
            </div>
        </>
    )
}

function SummaryPanel({
    selectedModel,
    inputLabel,
    outputRoot,
    dominantHand,
} : {
    selectedModel: ModelInfo;
    inputLabel: string;
    outputRoot: string;
    dominantHand: DominantHand | null;
}) {
    return (
        <>
            <div
                className="
                    mt-8 rounded-2xl border border-egm-card-border bg-white px-6
                    py-7 text-base text-egm-body-copy
                "
            >
                <h2 className="text-xl font-normal leading-none text-black">Summary</h2>

                <dl className="mt-6 grid gap-y-4 sm:grid-cols-[180px_minmax(0,1fr)]">
                    <dt className="border-b border-egm-list-border pb-2">Model:</dt>
                    <dd 
                        className="
                            m-0 font-semibold text-egm-strong-copy text-right border-b 
                            border-egm-list-border pb-2
                        "
                    >
                        {selectedModel.name}
                    </dd>

                    {dominantHand !== null ? (
                        <>
                            <dt className="border-b border-egm-list-border pb-2">
                                Dominant hand:
                            </dt>
                            <dd
                                className="
                                    m-0 font-semibold text-egm-strong-copy text-right
                                    border-b border-egm-list-border pb-2
                                "
                            >
                                {dominantHandLabel(dominantHand)}
                            </dd>
                        </>
                    ) : null}

                    <dt className="border-b border-egm-list-border pb-2">Input:</dt>
                    <dd 
                        className="
                            m-0 font-semibold text-egm-strong-copy text-right border-b 
                            border-egm-list-border pb-2
                        "
                    >
                        {inputLabel}
                    </dd>

                    <dt className="border-b border-egm-list-border pb-2">Output folder:</dt>
                    <dd 
                        className="
                            m-0 break-words font-semibold text-egm-strong-copy text-right
                            border-b border-egm-list-border pb-2
                        "   
                    >
                        {outputRoot}
                    </dd>

                    <dt>Processing mode:</dt>
                    <dd className="m-0 font-semibold text-egm-strong-copy text-right">
                        Local
                    </dd>
                </dl>
            </div>
        </>
    )
}

function ReadyPanel() {
    return (
        <div
            className="
                mt-6 flex min-h-24 items-center rounded-2xl border border-egm-card-border
                bg-white px-6 py-6
            "
        >
            <h2 className="text-2xl font-normal leading-none">Ready to start.</h2>
        </div>
    );
}

function DryRunCompletePanel({ runId } : { runId: string; }) {
    const lines = [
        "Checking selected input...",
        "Checking output folder...",
        "Checking local runtime...",
        "Dry run completed successfully.",
    ];

    const dryRunRowLayout = "grid grid-cols-[26px_1fr] items-center gap-x-4";

    return (
        <div
            className="
                mt-6 rounded-2xl border border-egm-card-border bg-white px-6 py-7
            "
        >
            <div className={dryRunRowLayout}>
                <CircleCheck 
                    aria-hidden="true"
                    className="text-egm-green"
                    size={26}
                    strokeWidth={2.0}
                />

                <h2 className="text-2xl font-normal leading-none">
                    Dry run completed successfully.
                </h2>
            </div>

            <ul className="mt-6 space-y-1 text-base leading-6 text-egm-body-copy">
                {lines.map((line, index) => {
                    const isFinalLine = index === lines.length - 1;
                    
                    return (
                        <li key={line} className={dryRunRowLayout}>
                            <span>
                                {isFinalLine ? (
                                    <CircleCheck
                                        aria-hidden="true"
                                        className="text-egm-green"
                                        size={20}
                                        strokeWidth={2.2}
                                    />
                                ) : null}
                            </span>

                            <span>{line}</span>
                        </li>
                    )
                })}
            </ul>
        </div>
    )
}

function RunningPanel({
    events, 
    runId,
    runtimeStatus,
    runtimeBuildStages,
} : {
    events: ProgressEvent[];
    runId: string;
    runtimeStatus: RuntimeStatus | null;
    runtimeBuildStages: RuntimeBuildStage[];
}) {
    const percent = getPercentage(events, runtimeBuildStages);
    
    return (
        <>
            <div
                className="
                    mt-6 grid grid-cols-[28px_minmax(0,1fr)_28px] gap-x-4 rounded-2xl border 
                    border-egm-card-border bg-white px-6 py-7
                "
            >
                <span 
                    aria-hidden="true"
                    className="
                        col-start-1 row-start-1 h-7 w-7 rounded-full border-[3px] 
                        border-egm-green-soft border-t-egm-green animate-egm-spin
                    "
                />

                <div className="col-start-2 min-w-0">
                    <h2 className="text-2xl font-normal leading-none">Running model...</h2>

                    <p className="mt-4 text-base leading-6 text-egm-secondary-copy">
                        Run ID: {runId}
                    </p>

                    {runtimeStatus ? (
                        <p
                            className="mt-6 text-sm font-medium leading-5 text-egm-danger"
                            role="status"
                        >
                            Building Docker image for {runtimeStatus.modelName}
                            {typeof runtimeStatus.currentStep === "number" &&
                            typeof runtimeStatus.totalSteps === "number" &&
                            runtimeStatus.totalSteps > 0
                                ? ` [${runtimeStatus.currentStep} / ${runtimeStatus.totalSteps}]`
                                : ""}
                        </p>
                    ) : null}

                    <ul 
                        aria-label="Run progress log"
                        className="
                            mt-6 space-y-1 rounded-xl bg-egm-tree-bg px-4 py-3
                            text-base leading-6 text-egm-body-copy
                        "
                        role="log"
                    >
                        {events.map((event, index) => (
                            <li key={`${event.stage}-${index}`}>{event.displayText}</li>
                        ))}
                    </ul>

                    <p className="mt-6 text-sm leading-6 text-egm-body-copy">
                        Overall progress estimate
                    </p>

                    <div 
                        className="
                            mt-1 h-2.5 overflow-hidden rounded-full bg-egm-progress-track
                        "
                    >
                        <div 
                            className="h-full bg-egm-green" 
                            data-testid="progress-bar-fill"
                            style={{ width: `${percent}%` }} 
                        />
                    </div>
                </div>
            </div>

            <p className="mt-3 text-base text-egm-body-copy">
                    This may take several minutes. Please keep this window open.
            </p>
        </>
    )
}

function ResultsScreen({
    selectedModel,
    dominantHand,
    files,
    runId,
    resultSummary,
    progress,
    isBusy,
    canViewOutputPreview,
    onOpenOutputFolder,
    onStartNewRun,
    onViewOutputPreview,
    inputNames,
} : {
    selectedModel: ModelInfo;
    dominantHand: DominantHand | null;
    files: File[];
    runId: string;
    resultSummary: RunSummary | null;
    progress: ProgressResponse | null;
    isBusy: boolean;
    canViewOutputPreview: boolean;
    onOpenOutputFolder: () => void;
    onStartNewRun: () => void;
    onViewOutputPreview: () => void;
    inputNames: string[];
}) {
    const failed = progress?.status === "failed";
    
    return (
        <>
            <PageHeading 
                title={failed ? "Needs attention" : "Run completed"}
                subtitle={
                    failed
                        ? "EgoModelKit could not complete the run."
                        : "Your results were saved successfully."
                }
            />

            <ResultsScreenSummaryPanel 
                selectedModel={selectedModel}
                files={files}
                runId={runId}
                resultSummary={resultSummary}
                progress={progress}
                inputNames={inputNames}
                dominantHand={dominantHand}
            />

            {!failed && progress?.resultVisualization ? (
                <ResultVisualizationPanel
                    visualization={progress.resultVisualization}
                />
            ) : null}

            <div 
                className="
                    sticky bottom-0 z-10 mt-auto flex flex-wrap justify-center gap-4 
                    bg-egm-bg pt-8 pb-4
                "
            >
                <button 
                    className={primaryButtonClass}
                    disabled={isBusy || runId.length === 0}
                    onClick={onOpenOutputFolder}
                    type="button"
                >
                    <Folder aria-hidden="true" />
                    Open Output Folder
                </button>

                <button 
                    className={secondaryButtonClass} 
                    onClick={onStartNewRun} 
                    type="button"
                >
                    Start New Run
                </button>

                <button 
                    className={secondaryButtonClass} 
                    disabled={!canViewOutputPreview}
                    onClick={onViewOutputPreview}
                    type="button"
                >
                    View Output Preview
                </button>
            </div>
        </>
    )
}

function ResultVisualizationPanel({
    visualization,
}: {
    visualization: ResultVisualization;
}) {
    return visualization.kind === "hand-interaction" ? (
        <>
            <HandInteractionMetricsTable visualization={visualization} />
            <HandInteractionTimeline visualization={visualization} />
        </>
    ) : (
        <>
            <AdlActivityTimeline visualization={visualization} />
            <AdlSummaryTable visualization={visualization} />
        </>
    );
}

function HandInteractionMetricsTable({
    visualization,
}: {
    visualization: HandInteractionVisualization;
}) {
    const rows = [
        {
            label: "Percent interaction time",
            values: visualization.metrics.percentInteractionTime,
            format: (value: number) => `${formatNumber(value, 1)}%`,
        },
        {
            label: "Interaction duration",
            values: visualization.metrics.interactionDurationSeconds,
            format: (value: number) => `${formatNumber(value, 1)} s`,
        },
        {
            label: "Number of interaction segments",
            values: visualization.metrics.interactionSegmentCount,
            format: (value: number) => formatNumber(value, 0),
        },
    ];

    return (
        <section
            aria-labelledby="hand-metrics-heading"
            className="mt-8 rounded-2xl border border-egm-list-border bg-white px-6 py-7"
        >
            <h2 id="hand-metrics-heading" className="text-2xl font-semibold">
                Clinical hand-use metrics
            </h2>
            <p className="mt-2 text-base leading-6 text-egm-body-copy">
                Session-level summary of detected hand-object interactions during the
                activity.
            </p>

            <div className="mt-6 overflow-x-auto">
                <table className="w-full min-w-[680px] border-collapse text-left">
                    <thead className="bg-egm-tree-bg text-sm uppercase tracking-wide">
                        <tr>
                            <th className="px-4 py-3 text-egm-body-copy">Metric</th>
                            <th className="px-4 py-3 text-center text-egm-green">
                                Dominant hand
                            </th>
                            <th className="px-4 py-3 text-center text-blue-700">
                                Non-dominant hand
                            </th>
                            <th className="px-4 py-3 text-center text-egm-body-copy">
                                Bilateral / total
                            </th>
                        </tr>
                    </thead>
                    <tbody>
                        {rows.map((row) => (
                            <tr className="border-t border-egm-list-border" key={row.label}>
                                <th className="px-4 py-4 font-medium">{row.label}</th>
                                <td className="px-4 py-4 text-center font-semibold text-egm-green">
                                    {row.format(row.values.dominant)}
                                </td>
                                <td className="px-4 py-4 text-center font-semibold text-blue-700">
                                    {row.format(row.values.nonDominant)}
                                </td>
                                <td className="px-4 py-4 text-center">
                                    {row.format(row.values.bilateralTotal)}
                                </td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>

            <p className="mt-6 text-sm leading-6 text-egm-body-copy">
                These metrics summarize how often and how long each hand interacted
                with objects during the analyzed session.
            </p>
        </section>
    );
}

function HandInteractionTimeline({
    visualization,
}: {
    visualization: HandInteractionVisualization;
}) {
    const rows = [
        {
            label: "Dominant hand",
            role: "dominant" as const,
            color: "#00b97b",
        },
        {
            label: "Non-dominant hand",
            role: "non_dominant" as const,
            color: "#4d94ed",
        },
    ];

    return (
        <section
            aria-labelledby="hand-timeline-heading"
            className="mt-8 rounded-2xl border border-egm-list-border bg-white px-6 py-7"
        >
            <h2 id="hand-timeline-heading" className="text-2xl font-semibold">
                Interaction timeline
            </h2>
            <p className="mt-2 text-base leading-6 text-egm-body-copy">
                Visual overview of when each hand interacted with objects across the
                analyzed session.
            </p>

            <TimelineRows
                durationSeconds={visualization.durationSeconds}
                rows={rows.map((row) => ({
                    label: row.label,
                    segments: visualization.segments
                        .filter((segment) => segment.handRole === row.role)
                        .map((segment) => ({
                            startSeconds: segment.startSeconds,
                            endSeconds: segment.endSeconds,
                            color: row.color,
                            ariaLabel: `${row.label} interaction from ${formatTimelineTime(
                                segment.startSeconds,
                            )} to ${formatTimelineTime(segment.endSeconds)}`,
                        })),
                }))}
            />

            <div className="mt-5 flex flex-wrap gap-x-8 gap-y-3 text-sm text-egm-body-copy">
                {rows.map((row) => (
                    <div className="flex items-center gap-2" key={row.label}>
                        <span
                            aria-hidden="true"
                            className="h-4 w-8 rounded-md"
                            style={{ backgroundColor: row.color }}
                        />
                        {row.label} interaction
                    </div>
                ))}
            </div>
        </section>
    );
}

const ADL_TIMELINE_COLORS = [
    "#00b8a9",
    "#2f7df6",
    "#7c4dff",
    "#e67e22",
    "#d94f70",
    "#5b8c3a",
    "#8d6e63",
    "#00838f",
];

function AdlActivityTimeline({ visualization }: { visualization: AdlVisualization }) {
    const colorByActivity = new Map(
        visualization.activities.map((activity, index) => [
            activity.activity,
            ADL_TIMELINE_COLORS[index % ADL_TIMELINE_COLORS.length],
        ]),
    );

    return (
        <section
            aria-labelledby="adl-timeline-heading"
            className="mt-8 rounded-2xl border border-egm-list-border bg-white px-6 py-7"
        >
            <h2 id="adl-timeline-heading" className="text-2xl font-semibold">
                Activity Timeline
            </h2>
            <p className="mt-2 text-base leading-6 text-egm-body-copy">
                Predicted activities across the analyzed session.
            </p>

            <TimelineRows
                durationSeconds={visualization.durationSeconds}
                rows={[
                    {
                        label: "Activity",
                        segments: visualization.segments.map((segment) => ({
                            startSeconds: segment.startSeconds,
                            endSeconds: segment.endSeconds,
                            color: colorByActivity.get(segment.activity) ?? "#777a80",
                            ariaLabel: `${segment.activity} from ${formatTimelineTime(
                                segment.startSeconds,
                            )} to ${formatTimelineTime(segment.endSeconds)}`,
                        })),
                    },
                ]}
            />

            <div className="mt-5 flex flex-wrap gap-x-7 gap-y-3 text-sm text-egm-body-copy">
                {visualization.activities.map((activity) => (
                    <div className="flex items-center gap-2" key={activity.activity}>
                        <span
                            aria-hidden="true"
                            className="h-4 w-4 rounded-full"
                            style={{
                                backgroundColor: colorByActivity.get(activity.activity),
                            }}
                        />
                        {activity.activity}
                    </div>
                ))}
            </div>
        </section>
    );
}

function AdlSummaryTable({ visualization }: { visualization: AdlVisualization }) {
    const colorByActivity = new Map(
        visualization.activities.map((activity, index) => [
            activity.activity,
            ADL_TIMELINE_COLORS[index % ADL_TIMELINE_COLORS.length],
        ]),
    );

    return (
        <section
            aria-labelledby="adl-summary-heading"
            className="mt-8 rounded-2xl border border-egm-list-border bg-white px-6 py-7"
        >
            <h2 id="adl-summary-heading" className="text-2xl font-semibold">
                Activity Summary
            </h2>
            <p className="mt-2 text-base leading-6 text-egm-body-copy">
                Session-level summary of the activities predicted across all analyzed
                segments.
            </p>

            <div className="mt-6 overflow-x-auto">
                <table className="w-full min-w-[640px] border-collapse text-left">
                    <thead className="bg-egm-tree-bg text-sm uppercase tracking-wide text-egm-body-copy">
                        <tr>
                            <th className="px-4 py-3">Activity</th>
                            <th className="px-4 py-3 text-right">Total duration</th>
                            <th className="px-4 py-3 text-right">Session %</th>
                            <th className="px-4 py-3 text-right">Segments</th>
                        </tr>
                    </thead>
                    <tbody>
                        {visualization.activities.map((activity) => (
                            <tr
                                className="border-t border-egm-list-border"
                                key={activity.activity}
                            >
                                <th className="px-4 py-4 font-medium">
                                    <span className="inline-flex items-center gap-3">
                                        <span
                                            aria-hidden="true"
                                            className="h-4 w-4 rounded-full"
                                            style={{
                                                backgroundColor: colorByActivity.get(
                                                    activity.activity,
                                                ),
                                            }}
                                        />
                                        {activity.activity}
                                    </span>
                                </th>
                                <td className="px-4 py-4 text-right">
                                    {formatDuration(activity.durationSeconds)}
                                </td>
                                <td className="px-4 py-4 text-right">
                                    {formatNumber(activity.sessionPercent, 1)}%
                                </td>
                                <td className="px-4 py-4 text-right">
                                    {activity.segmentCount}
                                </td>
                            </tr>
                        ))}
                    </tbody>
                    <tfoot>
                        <tr className="border-t-2 border-egm-list-border font-semibold">
                            <th className="px-4 py-4">Total analyzed session</th>
                            <td className="px-4 py-4 text-right">
                                {formatDuration(visualization.analyzedDurationSeconds)}
                            </td>
                            <td className="px-4 py-4 text-right">100.0%</td>
                            <td className="px-4 py-4 text-right">
                                {visualization.totalSegmentCount}
                            </td>
                        </tr>
                    </tfoot>
                </table>
            </div>
        </section>
    );
}

type TimelineRow = {
    label: string;
    segments: Array<{
        startSeconds: number;
        endSeconds: number;
        color: string;
        ariaLabel: string;
    }>;
};

function TimelineRows({
    durationSeconds,
    rows,
}: {
    durationSeconds: number;
    rows: TimelineRow[];
}) {
    const safeDuration = Math.max(durationSeconds, 1);
    const width = timelineWidthPixels(safeDuration);

    return (
        <div className="mt-6 flex min-w-0 gap-4">
            <div className="w-36 shrink-0 pt-8 sm:w-40">
                {rows.map((row, index) => (
                    <div
                        className={[
                            "flex h-11 items-center justify-end text-right text-sm",
                            "font-semibold text-egm-body-copy",
                            index > 0 ? "mt-3" : "",
                        ].join(" ")}
                        key={row.label}
                    >
                        {row.label}
                    </div>
                ))}
            </div>

            <div className="min-w-0 flex-1 overflow-x-auto pb-2">
                <div style={{ width: `${width}px` }}>
                    <TimelineAxis durationSeconds={safeDuration} />

                    {rows.map((row, index) => (
                        <div
                            className={[
                                "relative h-11 overflow-hidden rounded-lg bg-egm-tree-bg",
                                index > 0 ? "mt-3" : "",
                            ].join(" ")}
                            key={row.label}
                        >
                            {row.segments.map((segment, segmentIndex) => {
                                const start = Math.max(0, segment.startSeconds);
                                const end = Math.min(
                                    safeDuration,
                                    Math.max(start, segment.endSeconds),
                                );

                                return (
                                    <span
                                        aria-label={segment.ariaLabel}
                                        className="absolute inset-y-0 min-w-[2px]"
                                        key={`${segment.startSeconds}-${segmentIndex}`}
                                        role="img"
                                        style={{
                                            backgroundColor: segment.color,
                                            left: `${100 * start / safeDuration}%`,
                                            width: `${100 * (end - start) / safeDuration}%`,
                                        }}
                                        title={segment.ariaLabel}
                                    />
                                );
                            })}
                        </div>
                    ))}
                </div>
            </div>
        </div>
    );
}

function TimelineAxis({ durationSeconds }: { durationSeconds: number }) {
    const ticks = Array.from(
        { length: 6 },
        (_, index) => durationSeconds * index / 5,
    );

    return (
        <div className="flex h-8 items-start justify-between text-xs text-egm-secondary-copy">
            {ticks.map((tick, index) => (
                <span key={index}>{formatTimelineTime(tick)}</span>
            ))}
        </div>
    );
}

function timelineWidthPixels(durationSeconds: number): number {
    return Math.min(3600, Math.max(720, Math.ceil(durationSeconds * 2)));
}

function formatTimelineTime(seconds: number): string {
    const rounded = Math.max(0, Math.round(seconds));
    const hours = Math.floor(rounded / 3600);
    const minutes = Math.floor((rounded % 3600) / 60);
    const remainingSeconds = rounded % 60;

    if (hours > 0) {
        return `${hours}:${String(minutes).padStart(2, "0")}:${String(
            remainingSeconds,
        ).padStart(2, "0")}`;
    }

    if (rounded >= 60) {
        return `${minutes}:${String(remainingSeconds).padStart(2, "0")}`;
    }

    return `${rounded}s`;
}

function formatDuration(seconds: number): string {
    const rounded = Math.max(0, Math.round(seconds));
    const hours = Math.floor(rounded / 3600);
    const minutes = Math.floor((rounded % 3600) / 60);
    const remainingSeconds = rounded % 60;
    const parts: string[] = [];

    if (hours > 0) {
        parts.push(`${hours} hr`);
    }

    if (minutes > 0) {
        parts.push(`${minutes} min`);
    }

    if (remainingSeconds > 0 || parts.length === 0) {
        parts.push(`${remainingSeconds} sec`);
    }

    return parts.join(" ");
}

function formatNumber(value: number, fractionDigits: number): string {
    return value.toLocaleString(undefined, {
        minimumFractionDigits: fractionDigits,
        maximumFractionDigits: fractionDigits,
    });
}

function ResultsScreenSummaryPanel({
    selectedModel,
    files,
    runId,
    resultSummary,
    progress,
    inputNames,
    dominantHand,
} : {
    selectedModel: ModelInfo;
    files: File[];
    runId: string;
    resultSummary: RunSummary | null;
    progress: ProgressResponse | null;
    inputNames: string[];
    dominantHand : DominantHand | null;
}) {
    const failed = progress?.status === "failed";
    const statusLabel = failed ? "Failed" : "Completed";

    const outputFolder = 
        progress?.outputFolder ?? resultSummary?.outputFolder ?? "Not available";

    const ResultIcon = failed ? Info : CircleCheck;

    return (
        <div
            className={[
                "mt-8 rounded-2xl border px-6 py-7 text-base",
                failed
                    ? "border-egm-danger-border bg-egm-danger-soft text-egm-danger"
                    : "border-egm-card-border bg-egm-success-soft text-egm-body-copy"
            ].join(" ")}
        >
            <div className="flex items-start">
                <ResultIcon 
                    aria-hidden="true"
                    className={failed ? "mr-4 text-egm-danger" : "mr-4 text-egm-green"}
                    size={34}
                    strokeWidth={1.8}
                />

                <div className="min-w-0 flex-1">
                    <h2 className="text-2xl font-normal leading-none text-black">
                        {failed ? "Run could not be completed" : "Completed successfully"}
                    </h2>

                    {failed && progress?.errorMessage ? (
                        <p className="mt-4 leading-6">{progress.errorMessage}</p>
                    ) : null}

                    <dl 
                        className="
                            mt-6 grid sm:gap-y-3 sm:grid-cols-[180px_minmax(0,1fr)]
                        "
                    >
                        <dt className="border-b pb-2 border-egm-list-border">Model:</dt>
                        <dd 
                            className="
                                m-0 pb-2 text-right font-semibold text-egm-strong-copy 
                                border-b border-egm-list-border
                            "
                        >
                            {selectedModel.name}
                        </dd>

                        <dt className="border-b pb-2 border-egm-list-border">Input:</dt>
                        <dd 
                            className="
                                m-0 pb-2 text-right font-semibold text-egm-strong-copy 
                                border-b border-egm-list-border
                            "
                        >
                            {inputLabelFromNames(inputNames)}
                        </dd>

                        {dominantHand !== null ? (
                            <>
                                <dt className="border-b border-egm-list-border pb-2">
                                    Dominant hand:
                                </dt>
                                <dd
                                    className="
                                        m-0 font-semibold text-egm-strong-copy text-right
                                        border-b border-egm-list-border pb-2
                                    "
                                >
                                    {dominantHandLabel(dominantHand)}
                                </dd>
                            </>
                        ) : null}

                        <dt className="border-b pb-2 border-egm-list-border">
                            Output folder:
                        </dt>
                        <dd 
                            className="
                                m-0 pb-2 text-right break-words font-semibold 
                                text-egm-strong-copy border-b border-egm-list-border
                            "
                        >
                            {outputFolder}
                        </dd>

                        <dt className="border-b pb-2 border-egm-list-border">Run ID:</dt>
                        <dd 
                            className="
                                m-0 pb-2 text-right font-semibold text-egm-strong-copy 
                                border-b border-egm-list-border
                            "
                        >
                            {runId}
                        </dd>

                        <dt className="border-b pb-2 border-egm-list-border">
                            Running mode:
                        </dt>
                        <dd 
                            className="
                                m-0 pb-2 text-right font-semibold text-egm-strong-copy 
                                border-b border-egm-list-border
                            "
                        >
                            Local
                        </dd>

                        <dt>Status:</dt>
                        <dd className={[
                            "m-0 text-right font-semibold",
                            failed ? "text-egm-danger" : "text-egm-green",
                        ].join(" ")}>
                            {statusLabel}
                        </dd>
                    </dl>
                </div>
            </div>
        </div>   
    )
}

async function requestProgress(runId: string): Promise<ProgressResponse> {
    const response = await fetch(`/api/runs/${encodeURIComponent(runId)}/progress`);

    if (!response.ok) {
        throw new Error(`Request failed with status ${response.status}.`);
    }

    return (await response.json()) as ProgressResponse;
}

function getPercentage(
    events: ProgressEvent[],
    runtimeBuildStages: RuntimeBuildStage[],
): number {
    const fractions = [
        ...runtimeBuildStages.map(stageFraction),
        ...events
            .filter(isVisibleProgressStage)
            .map(eventFraction),
    ];

    if (fractions.length === 0) {
        return 0;
    }

    const total = fractions.reduce((sum, fraction) => sum + fraction, 0);

    return Math.round(100 * total / fractions.length);
}

function isVisibleProgressStage(event: ProgressEvent): boolean {
    return event.stage !== "current_video";
}

function eventFraction(event: ProgressEvent): number {
    if (
        typeof event.current === "number" &&
        typeof event.total === "number" &&
        event.total > 0
    ) {
        return boundedFraction(event.current, event.total);
    }

    if (event.displayText.toLowerCase().includes("waiting")) {
        return 0;
    }

    return 1;
}

function stageFraction(stage: RuntimeBuildStage): number {
    return boundedFraction(stage.current, stage.total);
}

function boundedFraction(
    current: number | null,
    total: number | null,
): number {
    if (
        typeof current !== "number" ||
        typeof total !== "number" ||
        total <= 0
    ) {
        return 0;
    }

    return Math.max(0, Math.min(1, current / total));
}

function OutputPreviewScreen({
    outputPreview,
    isBusy,
    onBack,
    onOpenOutputFolder,
} : {
    outputPreview: OutputPreview;
    isBusy: boolean;
    onBack: () => void;
    onOpenOutputFolder: () => void;
}) {
    const [contentsOpen, setContentsOpen] = useState<boolean>(false);

    return (
        <>
            <PageHeading 
                title="Output folder preview"
                subtitle="Review what EgoModelKit saved for this run."
            />

            <div
                className="
                    mt-8 rounded-xl border border-egm-blue-border bg-egm-blue-soft
                    px-5 py-4 text-base text-egm-body-copy
                "
            >
                Logs and technical files are kept separately for reproducibility
                and troubleshooting.
            </div>

            <section
                className="
                    mt-8 rounded-2xl border border-egm-card-border bg-white px-6
                    py-7 text-base text-egm-body-copy
                "
            >
                <h2 className="text-xl font-normal leading-none text-black">
                    Output folder structure
                </h2>

                <div
                    aria-label="Output folder structure"
                    className="
                        mt-6 max-h-[420px] overflow-auto rounded-2xl bg-egm-tree-bg
                        px-6 py-5 font-mono text-sm leading-6
                    "
                >
                    {outputPreview.folderTree.split("\n").map((line, index) => (
                        <OutputTreeLine
                            key={`${line}-${index}`}
                            line={line}
                        />
                    ))}
                </div>

                <div
                    className="
                        mt-6 overflow-hidden rounded-xl border border-egm-card-border
                        bg-white
                    "
                >
                    <button
                        aria-expanded={contentsOpen}
                        className="
                            flex min-h-14 w-full items-center justify-between px-5
                            text-left text-base text-egm-body-copy hover:bg-egm-hover
                            focus-visible:outline-3 focus-visible:outline-offset-3
                            focus-visible:outline-egm-green
                        "
                        type="button"
                        onClick={() => setContentsOpen((open) => !open)}
                    >
                        <span className={contentsOpen ? "font-semibold text-black" : ""}>
                            What the output folder contains
                        </span>

                        {contentsOpen ? (
                            <ChevronUp aria-hidden="true" size={22} strokeWidth={2.0} />
                        ) : (
                            <ChevronDown aria-hidden="true" size={22} strokeWidth={2.0} />
                        )}
                    </button>

                    {contentsOpen ? (
                        <dl
                            className="
                                border-t border-egm-card-border px-5 py-4 text-base
                                leading-6
                            "
                        >
                            {outputPreview.files.map((file) => (
                                <div className="mb-4 last:mb-0" key={file.name}>
                                    <dt className="font-semibold text-egm-green">
                                        {file.name}
                                    </dt>
                                    <dd className="m-0 text-egm-body-copy">
                                        {file.description}
                                    </dd>
                                </div>
                            ))}
                        </dl>
                    ) : null}
                </div>

                <p
                    className="
                        mt-6 rounded-xl bg-egm-tree-bg px-5 py-4 text-base
                        text-egm-body-copy
                    "
                >
                    {outputPreview.note}
                </p>
            </section>

            <div
                className="
                    sticky bottom-0 z-10 mt-auto flex flex-wrap justify-start gap-4
                    bg-egm-bg pt-8 pb-4
                "
            >
                <button className={primaryButtonClass} type="button" onClick={onBack}>
                    Back to Results
                </button>

                <button
                    className={secondaryButtonClass}
                    disabled={isBusy}
                    type="button"
                    onClick={onOpenOutputFolder}
                >
                    <Folder aria-hidden="true" />
                    Open Output Folder
                </button>
            </div>
        </>
    )
}

function OutputTreeLine({ line } : { line: string }) {
    const trimmedLine = line.trimStart();
    const depth = line.length - trimmedLine.length;
    const isFolder = trimmedLine.endsWith("/");

    return (
        <div 
            className="flex items-center gap-2 py-0.5 text-sm"
            style={{ paddingLeft: `${depth * 12}px` }}    
        >
            {isFolder ? (
                <>
                    <ChevronRight aria-hidden="true" className="h-3 w-3 text-egm-green" />
                    <span className="font-medium text-egm-green">{trimmedLine}</span>
                </>
            ) : (
                <>
                    <FileText aria-hidden="true" className="h-3 w-3 text-black" />
                    <span className="text-black">{trimmedLine}</span>            
                </>
            )}
        </div>
    );
}
