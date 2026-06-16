import { Check, ChevronLeft, Info, Shield, Upload } from "lucide-react";
import {
    type ChangeEvent,
    type DragEvent,
    type RefObject,
    useRef, 
    useState 
} from "react";

type Step = "welcome" | "select-model" | "choose-input" | "choose-output";

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

    const selectedModel =
        DEFAULT_MODELS.find((model) => model.id === modelId) ?? DEFAULT_MODELS[0];

    function startNewRun() {
        setModelId("");
        setFiles([]);
        setStep("select-model");
    }

    function goHome() {
        setModelId("");
        setFiles([]);
        setStep("welcome");
    }

    function selectModel(nextModelId: string) {
        if (nextModelId !== modelId) {
            setModelId(nextModelId);
            setFiles([]);
        }
    }

    function handleFilesChange(event: ChangeEvent<HTMLInputElement>) {
        const selectedFiles = event.currentTarget.files
            ? Array.from(event.currentTarget.files)
            : [];
        
        setFiles(selectedFiles);
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
                            {step === "select-model" ? (
                                <SelectModelScreen
                                    models={DEFAULT_MODELS}
                                    selectedModelId={modelId}
                                    onSelectModel={selectModel}
                                    canContinue={modelId.length > 0}
                                    onBack={() => setStep("welcome")}
                                    onContinue={() => setStep("choose-input")}
                                />
                            ) : (
                                step === "choose-input" ? (
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
                                ) : (
                                    <ChooseOutputPlaceholder 
                                        onBack={() => setStep("choose-input")}
                                    />
                                )
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

function ChooseOutputPlaceholder({
    onBack,
} : {
    onBack: () => void;
}) {
    return (
        <>
            <PageHeading
                title="Choose output folder"
                subtitle="Select a folder where EgoModelKit should save the results."
            />

            <div
                className="
                    mt-8 rounded-2xl border border-egm-card-border bg-white
                    px-6 py-8 text-egm-body-copy
                "
            >
                Output folder selection will be added in the next commit.
            </div>

            <div className="sticky bottom-0 z-10 mt-auto bg-egm-bg pt-8 pb-4">
                <button className={backButtonClass} type="button" onClick={onBack}>
                    <ChevronLeft aria-hidden="true" size={22} strokeWidth={2.4} />
                    Back
                </button>
            </div>
        </>
    )
}
