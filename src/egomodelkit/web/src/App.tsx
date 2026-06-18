import { 
    Check, 
    ChevronDown,
    ChevronLeft, 
    ChevronUp,
    Folder,
    Info, 
    Shield, 
    Upload 
} from "lucide-react";
import {
    type ChangeEvent,
    type DragEvent,
    type RefObject,
    useRef, 
    useState 
} from "react";

type Step = 
    | "welcome" 
    | "select-model" 
    | "choose-input" 
    | "choose-output"
    | "review";

const privacyMessage =
    "Your selected files are processed locally by default. " +
    "No telemetry or cloud upload is used in this MVP.";

type ModelInfo = {
    id: string;
    name: string;
    description: string;
    acceptedInputLabel: string;
    outputLabel: string;
};

type SelectOutputFolderResponse = {
    outputRoot: string;
};

const HAND_OBJECT_MODEL_ID = "hand-object-contact";
const ADL_MODEL_ID = "adl-recognition";

const DEFAULT_MODELS: ModelInfo[] = [
    {
        id: HAND_OBJECT_MODEL_ID,
        name: "Hand-object contact",
        description: "Detects hands, object, and hand-object contact in images.",
        acceptedInputLabel: "Input: image folder or image file",
        outputLabel: "Output: detection visualizations and structured results",
    },
    {
        id: ADL_MODEL_ID,
        name: "Activity recognition (ADL)",
        description: 
            "Processes egocentric video clips for activity of daily living (ADL) recognition.",
        acceptedInputLabel: "Input: MP4 video or video folder",
        outputLabel: "Output: predictions and processed frame-level files"
    }
]

const STEPS: Array<{ id: Exclude<Step, "welcome">; label: string }> = [
    { id: "select-model", label: "Select model" },
    { id: "choose-input", label: "Choose input" },
    { id: "choose-output", label: "Choose output" },
    { id: "review", label: "Review and run" },
]

const buttonBaseClass =
    "inline-flex min-h-12 min-w-[132px] items-center justify-center gap-2 " +
    "rounded-lg px-6 py-3 text-base font-semibold transition-colors " +
    "focus-visible:outline-3 focus-visible:outline-offset-3 " +
    "focus-visible:outline-egm-green disabled:cursor-not-allowed";

const primaryButtonClass =
    `${buttonBaseClass} border border-egm-green bg-egm-green text-white text-lg ` +
    "hover:bg-egm-green-dark disabled:border-egm-disabled " +
    "disabled:bg-egm-disabled disabled:text-white"

const backButtonClass =
    "inline-flex min-h-12 items-center justify-center gap-2 rounded-lg " +
    "border border-transparent bg-transparent pl-0 pr-2 py-3 text-base " +
    "font-medium text-egm-back hover:text-black focus-visible:outline-3 " +
    "focus-visible:outline-offset-3 focus-visible:outline-egm-green";

export function App() {
    const [step, setStep] = useState<Step>("welcome");
    const [modelId, setModelId] = useState<string>("");
    const [files, setFiles] = useState<File[]>([]);
    const fileInputRef = useRef<HTMLInputElement | null>(null);
    const [outputRoot, setOutputRoot] = useState<string>("");
    const [privacyOpen, setPrivacyOpen] = useState<boolean>(false);
    const [errorMessage, setErrorMessage] = useState<string>("");
    const [isBusy, setIsBusy] = useState<boolean>(false);

    const selectedModel =
        DEFAULT_MODELS.find((model) => model.id === modelId) ?? DEFAULT_MODELS[0];

    function startNewRun() {
        setModelId("");
        setFiles([]);
        setOutputRoot("");
        setPrivacyOpen(false);
        setErrorMessage("");
        setIsBusy(false);
        setStep("select-model");
    }

    function goHome() {
        setModelId("");
        setFiles([]);
        setOutputRoot("");
        setPrivacyOpen(false);
        setErrorMessage("");
        setIsBusy(false);
        setStep("welcome");
    }

    function selectModel(nextModelId: string) {
        if (nextModelId !== modelId) {
            setModelId(nextModelId);
            setFiles([]);
            setOutputRoot("");
            setPrivacyOpen(false);
            setErrorMessage("");
        }
    }

    function handleFilesChange(event: ChangeEvent<HTMLInputElement>) {
        const selectedFiles = event.currentTarget.files
            ? Array.from(event.currentTarget.files)
            : [];
        
        setFiles(selectedFiles);
        setOutputRoot("");
        setPrivacyOpen(false);
        setErrorMessage("");
    }

    function handleDrop(event: DragEvent<HTMLDivElement>) {
        event.preventDefault();

        const droppedFiles = event.dataTransfer.files
            ? Array.from(event.dataTransfer.files)
            : [];
        
        if (droppedFiles.length === 0) {
            return;
        }

        setFiles(droppedFiles);
        setOutputRoot("");
        setPrivacyOpen(false);
        setErrorMessage("");
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
        } catch {
            setErrorMessage("Unable to choose output folder.")
        } finally {
            setIsBusy(false);
        }
    }

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
                        <Stepper currentStep={step} />

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
                                    models={DEFAULT_MODELS}
                                    selectedModelId={modelId}
                                    onSelectModel={selectModel}
                                    canContinue={modelId.length > 0}
                                    onBack={() => setStep("welcome")}
                                    onContinue={() => setStep("choose-input")}
                                />
                            ) : step === "choose-input" ? (
                                <ChooseInputScreen 
                                    selectedModel={selectedModel}
                                    files={files}
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
                            ) : (
                                <ReviewPlaceholder
                                    selectedModel={selectedModel}
                                    files={files}
                                    outputRoot={outputRoot}
                                    onBack={() => setStep("choose-output")}
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
    selectedModelId,
    onSelectModel,
    canContinue,
    onBack,
    onContinue,
}: {
    models: ModelInfo[];
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
                                    ? "border-2 border-black bg-egm-green-tint"
                                    : "border border-egm-card-border hover:bg-egm-hover bg-white"
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
                                    <span className="h-4 w-4 rounded-full bg-white" />
                                </span>
                            </span>
                            <span className="ml-5 flex flex-col">
                                <span className="text-2xl font-medium leading-none">
                                    {model.name}
                                </span>

                                <span className="mt-4 text-base leading-6 text-egm-body-copy">
                                    {model.description}
                                </span>

                                <span className="mt-6 text-base leading-6 text-egm-secondary-copy">
                                    {model.acceptedInputLabel}
                                    <br />
                                    {model.outputLabel}
                                </span>
                            </span>
                        </button>
                    )
                })}
            </div>

            <FooterActions
                onBack={onBack}
                onContinue={onContinue}
                continueLabel="Continue"
                continueDisabled={!canContinue}
            />
        </>
    );
}

function Stepper({ currentStep }: { currentStep: Exclude<Step, "welcome"> }) {
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
    if (files.length === 1) {
        return files[0].name;
    }

    return `${files.length} files`;
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

function ChooseInputScreen({
    selectedModel,
    files,
    fileInputRef,
    onFilesChange,
    onDrop,
    canContinue,
    onBack,
    onContinue,
} : {
    selectedModel: ModelInfo;
    files: File[];
    fileInputRef: RefObject<HTMLInputElement | null>;
    onFilesChange: (event: ChangeEvent<HTMLInputElement>) => void;
    onDrop: (event: DragEvent<HTMLDivElement>) => void;
    canContinue: boolean;
    onBack: () => void;
    onContinue: () => void;
}) {
    const subtitle =
        selectedModel.id === HAND_OBJECT_MODEL_ID
            ? "Select an image or folder of images"
            : "Select a video or folder of videos";

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
                    Choose input files
                </button>

                <p className="mt-5 text-sm leading-6 text-egm-secondary-copy">
                    Supported files depend on the selected model.
                </p>

                <input
                    ref={fileInputRef}
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
                <>
                    <div
                        className="
                            mt-6 rounded-2xl border border-egm-card-border bg-white
                            px-6 py-4 text-base text-egm-body-copy
                        "
                    >
                        Selected: {inputLabelFromFiles(files)}
                    </div>

                    <div
                        className="
                            mt-4 rounded-2xl bg-egm-success-soft px-6 py-4
                            text-base text-egm-body-copy
                        "
                    >
                        Input selected.
                    </div>
                </>
            )}

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

function ReviewPlaceholder({
    selectedModel,
    files,
    outputRoot,
    onBack,
} : {
    selectedModel: ModelInfo;
    files: File[];
    outputRoot: string;
    onBack: () => void;
}) {
    return (
        <>
            <PageHeading 
                title="Review and run"
                subtitle="Confirm the model, input, and output location before starting."
            />

            <div
                className="
                    mt-8 rounded-2xl border border-egm-card-border bg-white px-6
                    py-7 text-base text-egm-body-copy
                "
            >
                <dl className="grid gap-4 sm:grid-cols-[180px_minmax(0,1fr)]">
                    <dt>Model:</dt>
                    <dd className="m-0 font-semibold text-egm-strong-copy text-right">
                        {selectedModel.name}
                    </dd>

                    <dt>Input:</dt>
                    <dd className="m-0 font-semibold text-egm-strong-copy text-right">
                        {inputLabelFromFiles(files)}
                    </dd>

                    <dt>Output folder:</dt>
                    <dd 
                        className="
                            m-0 break-words font-semibold text-egm-strong-copy text-right
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

            <div
                className="
                    mt-5 rounded-xl border border-egm-blue-border bg-egm-blue-soft
                    px-5 py-4 text-base text-egm-body-copy
                "
            >
                Run review will be added in the next commit.
            </div>

            <div className="sticky bottom-0 z-10 mt-auto bg-egm-bg pt-8 pb-4">
                <button className={backButtonClass} type="button" onClick={onBack}>
                    <ChevronLeft aria-hidden="true" size={22} strokeWidth={2.0} />
                    Back
                </button>
            </div>
        </>
    )
}
