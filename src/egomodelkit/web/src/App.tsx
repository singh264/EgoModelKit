import { 
    Check, 
    ChevronDown,
    ChevronLeft,
    ChevronRight, 
    ChevronUp,
    CircleCheck,
    FileText,
    Folder,
    Info, 
    Shield, 
    Upload 
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

type GuiRunStatus = "ready" | "running" | "completed" | "failed";

const privacyMessage =
    "Your selected files are processed locally by default. " +
    "No telemetry or cloud upload is used in this MVP.";

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

type ProgressResponse = {
    runId: string;
    status: GuiRunStatus;
    errorMessage: string | null;
    outputFolder: string;
    events: ProgressEvent[];
    outputPreview: OutputPreview;
};

const HAND_OBJECT_MODEL_ID = "hand-object-contact";
const ADL_MODEL_ID = "adl-recognition";

const STEPS: Array<{ id: StepperStep; label: string }> = [
    { id: "select-model", label: "Select model" },
    { id: "choose-input", label: "Choose input" },
    { id: "choose-output", label: "Choose output" },
    { id: "review", label: "Review and run" },
    { id: "results", label: "Results" },
];

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

const backButtonClass =
    "inline-flex min-h-12 items-center justify-center gap-2 rounded-lg " +
    "border border-transparent bg-transparent pl-0 pr-2 py-3 text-base " +
    "font-medium text-egm-back hover:text-black focus-visible:outline-3 " +
    "focus-visible:outline-offset-3 focus-visible:outline-egm-green";

export function App() {
    const [step, setStep] = useState<Step>("welcome");
    const [models, setModels] = useState<ModelInfo[]>([]);
    const [modelsLoading, setModelsLoading] = useState<boolean>(true);
    const [modelsError, setModelsError] = useState<string>("");
    const [modelId, setModelId] = useState<string>("");
    const [files, setFiles] = useState<File[]>([]);
    const [ignoredInputNames, setIgnoredInputNames] = useState<string[]>([]);
    const fileInputRef = useRef<HTMLInputElement | null>(null);
    const [outputRoot, setOutputRoot] = useState<string>("");
    const [privacyOpen, setPrivacyOpen] = useState<boolean>(false);
    const [errorMessage, setErrorMessage] = useState<string>("");
    const [isBusy, setIsBusy] = useState<boolean>(false);
    const [reviewMode, setReviewMode] = useState<ReviewMode>("ready");
    const [runId, setRunId] = useState<string>("");
    const [progress, setProgress] = useState<ProgressResponse | null>(null);
    const [resultSummary, setResultSummary] = useState<RunSummary | null>(null);
    const [outputPreview, setOutputPreview] = useState<OutputPreview | null>(null);

    const selectedModel = models.find((model) => model.id === modelId) ?? null;

    const stepperCurrentStep: StepperStep =
        step === "output-preview"
            ? "results"
            : step === "welcome"
                ? "select-model"
                : step;

    function startNewRun() {
        setModelId("");
        setFiles([]);
        setIgnoredInputNames([]);
        setOutputRoot("");
        setPrivacyOpen(false);
        setErrorMessage("");
        setIsBusy(false);
        clearReviewState();
        setStep("select-model");
    }

    function goHome() {
        setModelId("");
        setFiles([]);
        setIgnoredInputNames([]);
        setOutputRoot("");
        setPrivacyOpen(false);
        setErrorMessage("");
        setIsBusy(false);
        clearReviewState();
        setStep("welcome");
    }

    function selectModel(nextModelId: string) {
        if (nextModelId !== modelId) {
            setModelId(nextModelId);
            setFiles([]);
            setIgnoredInputNames([]);
            setOutputRoot("");
            setPrivacyOpen(false);
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
        setIgnoredInputNames(ignoredNames);
        setOutputRoot("");
        setPrivacyOpen(false);
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

            const selectedOutputRoot =
                backendSelection?.outputRoot ??
                window.prompt(
                    "Enter the output folder path:",
                    "/Users/Research/Desktop/EgoModelKit Results",
                );
            
            if (!selectedOutputRoot?.trim()) {
                return;
            }

            setOutputRoot(selectedOutputRoot.trim());
            clearReviewState();
        } catch {
            setErrorMessage("Unable to choose output folder.")
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
        try {
            setIsBusy(true);
            setErrorMessage("");

            const body = await requestOpenOutputFolder({ runId });

            if (body === null) {
                setErrorMessage(
                    "Opening output folders is not available in this environment."
                );
            }
        } catch {
            setErrorMessage("Unable to open output folder.")
        } finally {
            setIsBusy(false);
        }
    }

    async function runDryRun() {
        try {
            setIsBusy(true);
            setErrorMessage("");

            const body = await postMultipart<DryRunResponse>("/api/dry-run", {
                modelId,
                outputRoot,
                files,
            });

            setRunId(body.runId);
            setResultSummary(body.summary);
            setOutputPreview(body.outputPreview);
            setProgress(null);
            setReviewMode("dry-run-complete");
        } catch {
            setErrorMessage("Unable to complete dry run.");
            setReviewMode("ready");
        } finally {
            setIsBusy(false);
        }
    }

    async function startRun() {
        try {
            setIsBusy(true);
            setErrorMessage("");

            const body = await postMultipart<StartRunResponse>("/api/runs", {
                modelId,
                outputRoot,
                files,
            });

            setRunId(body.runId);
            setResultSummary(body.summary);
            setOutputPreview(body.outputPreview);

            setProgress({
                runId: body.runId,
                status: "running",
                errorMessage: null,
                outputFolder: body.summary.outputFolder,
                events: startingProgressEvents(modelId),
                outputPreview: body.outputPreview,
            });

            setReviewMode("running");
        } catch {
            setErrorMessage("Unable to start model run.");
            setReviewMode("ready");
        } finally {
            setIsBusy(false);
        }
    }

    useEffect(() => {
        if (reviewMode !== "running" || runId.length === 0) {
            return;
        }

        let isMounted = true;

        async function pollProgress() {
            try {
                const body = await requestProgress(runId);

                if (!isMounted) {
                    return;
                }

                setProgress(body);

                setOutputPreview((currentOutputPreview) => (
                    body.outputPreview ?? currentOutputPreview
                ));

                if (body.status === "completed" || body.status === "failed") {
                    setReviewMode("ready");
                    setStep("results");
                }
            } catch {
                if (isMounted) {
                    setErrorMessage("Unable to refresh run progress.");
                }
            }
        }

        void pollProgress();

        const intervalId = window.setInterval(pollProgress, 1000);

        return () => {
            isMounted = false;
            window.clearInterval(intervalId);
        };
    }, [reviewMode, runId]);

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
                            onClick={goHome}
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
                                    privacyOpen={privacyOpen}
                                    onTogglePrivacy={() => setPrivacyOpen((open) => !open)}
                                    onChooseOutputFolder={chooseOutputFolder}
                                    canContinue={outputRoot.trim().length > 0 && !isBusy}
                                    onBack={() => setStep("choose-input")}
                                    onContinue={() => setStep("review")}
                                />
                            ) : step === "review" && selectedModel !== null ? (
                                <ReviewScreen
                                    selectedModel={selectedModel}
                                    files={files}
                                    outputRoot={outputRoot}
                                    reviewMode={reviewMode}
                                    progress={progress}
                                    runId={runId}
                                    isBusy={isBusy}
                                    onBack={() => setStep("choose-output")}
                                    onDryRun={runDryRun}
                                    onRun={startRun}
                                />
                            ) : step === "results" && selectedModel !== null ? (
                                <ResultsScreen
                                    selectedModel={selectedModel}
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
    return (
        <main className="mx-auto w-full max-w-[672px] px-6 pt-24 pb-24 text-center">
            <div
                className="
                    mx-auto flex h-16 w-16 items-center justify-center rounded-full
                    bg-egm-green-soft text-egm-green
                "
            >
                <Shield aria-hidden="true" size={32} strokeWidth={2.2} />
            </div>

            <h1 className="mt-6 text-[30px] font-semibold leading-[1.15] tracking-[-0.03em]">
                EgoModelKit
            </h1>

            <p className="mt-3.5 text-lg font-normal leading-[1.45]">
                Run egocentric video models through a simple local interface.
            </p>

            <div
                className="
                    mx-auto mt-8 flex min-h-[84px] w-full max-w-[544px] items-center
                    gap-4 rounded-xl border border-egm-blue-border bg-egm-blue-soft
                    px-6 py-5 text-left
                "
            >
                <Info
                    aria-hidden="true"
                    className="shrink-0 text-egm-blue-icon"
                    size={20}
                    strokeWidth={2.2}
                />
                <p className="text-sm leading-6">{privacyMessage}</p>
            </div>

            <button
                className="
                    mt-8 min-h-12 min-w-[164px] rounded-lg border border-egm-green
                    bg-egm-green px-8 py-3 text-base font-semibold text-white
                    hover:bg-egm-green-dark focus-visible:outline-3
                    focus-visible:outline-offset-3 focus-visible:outline-egm-green
                "
                type="button"
                onClick={onStart}
            >
                Start New Run
            </button>

            <button
                className="
                    mx-auto mt-[18px] block border-0 bg-transparent text-sm
                    font-medium text-black hover:underline focus-visible:outline-3
                    focus-visible:outline-offset-3 focus-visible:outline-egm-green
                "
                type="button"
            >
                View Previous Output Folder
            </button>

            <p className="mt-7 text-xs font-normal leading-[1.45]">
                Designed for research use. Please confirm approved data handling
                procedures before using clinical data.
            </p>
        </main>
    );
}

function SelectModelScreen({
    models,
    modelsLoading,
    modelsError,
    selectedModelId,
    onSelectModel,
    canContinue,
    onBack,
    onContinue,
}: {
    models: ModelInfo[];
    modelsLoading: boolean;
    modelsError: string;
    selectedModelId: string;
    onSelectModel: (modelId: string) => void;
    canContinue: boolean;
    onBack: () => void;
    onContinue: () => void;
}) {
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

            <FooterActions
                onBack={onBack}
                onContinue={onContinue}
                continueLabel="Continue"
                continueDisabled={!canContinue || modelsLoading || modelsError.length > 0}
            />
        </>
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

function inputLabelFromFiles(files: File[]): string {
    return files.map((file) => file.name).join(", ");
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
} : {
    runId: string;
}) : Promise<OpenOutputFolderResponse | null> {
    const response = await fetch("/api/open-output-folder", {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify({ runId }),
    });

    if ([404, 405].includes(response.status)) {
        return null;
    }

    if (!response.ok) {
        throw new Error(`Request failed with status ${response.status}.`);
    }

    return (await response.json()) as OpenOutputFolderResponse;
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

function startingProgressEvents(modelId: string): ProgressEvent[] {
    if (modelId === HAND_OBJECT_MODEL_ID) {
        return [
            progressLine("setup", "Preparing image input..."),
            progressLine("validate", "Checking selected image..."),
            progressLine(
                "runtime",
                "Running hand-object contact model on the image: 1 / 1 image processed",
            ),
            progressLine("finalize", "Saving detection outputs..."),
        ];
    }

    return [
        progressLine("setup", "Preparing video input..."),
        progressLine("extract", "Extracting frames..."),
        progressLine("runtime", "Running hand-object contact model..."),
        progressLine("runtime", "Running Detic object detection..."),
        progressLine("finalize", "Saving activity recognition outputs..."),
    ];
}

function progressLine(stage: string, displayText: string): ProgressEvent {
    return {
        stage,
        message: displayText,
        current: null,
        total: null,
        unit: null,
        displayText,
    }
}

async function postMultipart<T>(
    url: string,
    {
        modelId,
        outputRoot,
        files,
    }: {
        modelId: string;
        outputRoot: string;
        files: File[];
    },
): Promise<T> {
    const formData = new FormData();

    formData.append("modelId", modelId);
    formData.append("outputRoot", outputRoot);

    for (const file of files) {
        formData.append("files", file, file.name);
    }

    const response = await fetch(url, {
        method: "POST",
        body: formData,
    });

    if (!response.ok) {
        throw new Error(`Request failed with status ${response.status}.`);
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
    privacyOpen,
    onTogglePrivacy,
    onChooseOutputFolder,
    canContinue,
    onBack,
    onContinue,
} : {
    outputRoot: string;
    isBusy: boolean;
    privacyOpen: boolean;
    onTogglePrivacy: () => void;
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

            <div
                className="
                    mt-5 overflow-hidden rounded-xl border border-egm-card-border bg-white
                "
            >
                <button
                    aria-expanded={privacyOpen}
                    className="
                        flex min-h-14 w-full items-center justify-between px-5
                        text-left text-base text-egm-body-copy hover:bg-egm-hover
                        focus-visible:outline-3 focus-visible:outline-offset-3
                        focus-visible:outline-egm-green
                    "
                    type="button"
                    onClick={onTogglePrivacy}
                >
                    <span className={privacyOpen ? "font-semibold text-black" : ""}>
                        Privacy-safe outputs
                    </span>

                    {privacyOpen ? (
                        <ChevronUp
                            aria-hidden="true"
                            size={22}
                            strokeWidth={2.0}
                        />
                    ) : (
                        <ChevronDown
                            aria-hidden="true"
                            size={22}
                            strokeWidth={2.0}
                        />
                    )}
                </button>

                {privacyOpen ? (
                    <div
                        className="
                            border-t border-egm-card-border px-7 py-4 text-sm
                            leading-6 text-egm-body-copy
                        "
                    >
                        <ul className="m-0 list-disc pl-5">
                            <li>Run IDs are neutral names.</li>
                            <li>Logs avoid unnecessary personal details.</li>
                            <li>Temporary files can be cleaned up after processing.</li>
                        </ul>
                    </div>
                ) : null}
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
    files,
    outputRoot,
    reviewMode,
    progress,
    runId,
    isBusy,
    onBack,
    onDryRun,
    onRun,
} : {
    selectedModel: ModelInfo;
    files: File[];
    outputRoot: string;
    reviewMode: ReviewMode;
    progress: ProgressResponse | null;
    runId: string;
    isBusy: boolean;
    onBack: () => void;
    onDryRun: () => void;
    onRun: () => void;
}) {
    const running = reviewMode === "running";

    return (
        <>
            <PageHeading 
                title="Review and run"
                subtitle="Confirm the model, input, and output location before starting."
            />

            <SummaryPanel
                selectedModel={selectedModel}
                inputLabel={inputLabelFromFiles(files)}
                outputRoot={outputRoot}
            />

            <div
                className="
                    mt-5 rounded-xl border border-egm-blue-border bg-egm-blue-soft
                    px-5 py-4 text-base text-egm-body-copy
                "
            >
                Dry run checks the selected input, output folder, and required local
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
                    disabled={running}
                    type="button" 
                    onClick={onBack}
                >
                    <ChevronLeft aria-hidden="true" size={22} strokeWidth={2.0} />
                    Back
                </button>

                <div className="flex gap-3">
                    <button
                        className={secondaryButtonClass}
                        disabled={running || isBusy}
                        type="button"
                        onClick={onDryRun}
                    >
                        Dry Run
                    </button>
                    <button 
                        className={primaryButtonClass}
                        disabled={running || isBusy}
                        type = "button"
                        onClick={onRun}
                    >
                        Run Model
                    </button>
                </div>
            </div>
        </>
    )
}

function SummaryPanel({
    selectedModel,
    inputLabel,
    outputRoot,
} : {
    selectedModel: ModelInfo;
    inputLabel: string;
    outputRoot: string;
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
} : {
    events: ProgressEvent[];
    runId: string;
}) {
    const percent = progressPercentage(events);
    
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
                    
                    <ul 
                        aria-label="Run progress log"
                        className="
                            mt-6 max-h-40 space-y-1 overflow-y-auto rounded-xl bg-egm-tree-bg
                            px-4 py-3 text-base leading-6  text-egm-body-copy
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
    files,
    runId,
    resultSummary,
    progress,
    isBusy,
    canViewOutputPreview,
    onOpenOutputFolder,
    onStartNewRun,
    onViewOutputPreview,
} : {
    selectedModel: ModelInfo;
    files: File[];
    runId: string;
    resultSummary: RunSummary | null;
    progress: ProgressResponse | null;
    isBusy: boolean;
    canViewOutputPreview: boolean;
    onOpenOutputFolder: () => void;
    onStartNewRun: () => void;
    onViewOutputPreview: () => void;
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
            />

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

function ResultsScreenSummaryPanel({
    selectedModel,
    files,
    runId,
    resultSummary,
    progress,
} : {
    selectedModel: ModelInfo;
    files: File[];
    runId: string;
    resultSummary: RunSummary | null;
    progress: ProgressResponse | null;
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
                            {inputLabelFromFiles(files)}
                        </dd>

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

function progressPercentage(events: ProgressEvent[]): number {
    const eventWithTotal = [...events]
        .reverse()
        .find(
            (event) =>
                typeof event.current === "number" &&
                typeof event.total === "number" &&
                event.total > 0,
        )
    
    const current = eventWithTotal?.current;
    const total = eventWithTotal?.total;

    if (typeof current !== "number" || typeof total !== "number" || total <= 0) {
        return events.length > 0 ? 8 : 0;
    }

    return Math.max(
        0,
        Math.min(100,Math.round((current / total) * 100)),
    );
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
