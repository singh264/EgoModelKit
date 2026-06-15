import { Info, Shield } from "lucide-react";
import { useState } from "react";

type Step = "welcome" | "select-model";

const privacyMessage =
    "Your selected files are processed locally by default. " +
    "No telemetry or cloud upload is used in this MVP.";

export function App() {
    const [step, setStep] = useState<Step>("welcome");

    function startNewRun() {
        setStep("select-model");
    }

    return (
        <div className="min-h-screen bg-egm-bg text-black">
            <div className="flex min-h-screen flex-col">
                <header
                    className="
                        flex h-[68px] items-center border-b border-egm-header-border
                        bg-white px-6 text-[26px] font-normal leading-none tracking-[0.01em]
                    "
                >
                    EgoModelKit
                </header>

                {step === "welcome" ? (
                    <WelcomeScreen onStart={startNewRun} />
                ) : (
                    <SelectModelScreen />
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

function SelectModelScreen() {
    return (
        <main className="mx-auto w-full max-w-[672px] px-6 pt-24">
            <h1 className="text-[30px] font-semibold leading-[1.15] tracking-[-0.03em]">
                Select a model
            </h1>

            <p className="mt-3.5 text-lg font-normal leading-[1.45] text-egm-subtitle">
                Choose the workflow you want to run.
            </p>
        </main>
    );
}
