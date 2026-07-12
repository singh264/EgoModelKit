import { act, createEvent, fireEvent, render, screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { App } from "./App";

function dropInputFiles(target: HTMLElement, files: File[]) {
    const dropEvent = createEvent.drop(target);

    Object.defineProperty(dropEvent, "dataTransfer", {
        value: {
            files,
        },
    });

    fireEvent(target, dropEvent);
}

const APP_STATE_STORAGE_KEY = "egomodelkit.gui.state.v1";

function createTestLocalStorage() {
    const storedValues = new Map<string, string>();

    return {
        getItem: vi.fn((key: string) => storedValues.get(key) ?? null),
        setItem: vi.fn((key: string, value: string) => {
            storedValues.set(key, value);
        }),
        removeItem: vi.fn((key: string) => {
            storedValues.delete(key);
        }),
        clear: vi.fn(() => {
            storedValues.clear();
        }),
        key: vi.fn((index: number) => Array.from(storedValues.keys())[index] ?? null),
        get length() {
            return storedValues.size;
        },
    };
}

beforeEach(() => {
    vi.stubGlobal("localStorage", createTestLocalStorage());
});

afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
});

function dropWithoutFileList(target: HTMLElement) {
    const dropEvent = createEvent.drop(target);

    Object.defineProperty(dropEvent, "dataTransfer", {
        value: {},
    });

    fireEvent(target, dropEvent);
}

function modelsResponse() {
    return {
        models: [
            {
                id: "hand-object-contact",
                name: "Hand-object contact",
                description: "Detects hands, objects, and hand-object contact in images.",
                acceptedInputLabel: "an image or multiple images",
                supportedInputExtensions: [".jpg", ".jpeg", ".png", ".bmp", ".webp"],
                outputLabel: "detection visualizations and structured results",
            },
            {
                id: "adl-recognition",
                name: "Activity recognition (ADL)",
                description:
                    "Processes egocentric video clips for " +
                    "activity of daily living (ADL) recognition.",
                acceptedInputLabel: "a video or multiple videos",
                supportedInputExtensions: [".mp4"],
                outputLabel: "predictions and processed frame-level files",
            },
        ],
    };
}

function okJson(body: unknown) {
    return {
        ok: true,
        status: 200,
        json: async () => body,
    }
}

function stubFetchWithModels(
    handler: (input: RequestInfo | URL, init?: RequestInit) => Promise<unknown> =
        async (input) => {
            throw new Error(`Unexpected fetch call: ${String(input)}`);
        },
) {
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
        const url = String(input);

        if (url === "api/models" || url === "/api/models") {
            return okJson(modelsResponse());
        }

        return handler(input, init);
    });

    vi.stubGlobal("fetch", fetchMock);

    return fetchMock;
}

beforeEach(() => {
    stubFetchWithModels();
});

async function navigateToOutputStep(
    user: ReturnType<typeof userEvent.setup>,
    {
        modelName = /Hand-object contact/,
        file = new File(["fake image"], "frame.jpg", { type: "image/jpeg" }),
    }: {
        modelName?: RegExp;
        file?: File;
    } = {},
) {
    const existingFetch = globalThis.fetch;

    if (existingFetch && vi.isMockFunction(existingFetch)) {
        stubFetchWithModels((input, init) => existingFetch(input, init));
    } else {
        stubFetchWithModels();
    }

    render(<App />);

    await user.click(await screen.findByRole("button", { name: "Start New Run" }));
    await user.click(await screen.findByRole("button", { name: modelName }));
    await user.click(screen.getByRole("button", { name: "Continue" }));

    await user.upload(screen.getByLabelText("Choose input files"), file);
    await user.click(screen.getByRole("button", { name: "Continue" }));
}

function outputPreview(runId: string) {
    return {
        runId,
        scenario: "single_image",
        folderTree:
            "EgoModelKit Results/\n" +
            `  ${runId}/\n` +
            "    README.txt\n" +
            "    run_summary.json\n" +
            "    visual_outputs/\n" +
            "      hand_object_contact/\n" +
            "        frame_det.png\n" +
            "    logs/\n" +
            "      progress.jsonl\n" +
            "      runtime.log",
        note: "Frame-level metrics are not generated for a single image.",
        files: [
            {
                name: "README.txt",
                description: "Explanation of the output folder contents.",
            },
            {
                name: "run_summary.json",
                description: "Summary of the run and completion status.",
            },
            {
                name: "progress.jsonl",
                description: "Progress events written during the run.",
            },
        ],
    }
}

function dryRunResponse(runId = "dry-run-1") {
    return {
        runId,
        status: "ready",
        scenario: "single_iamge",
        summary: {
            modelId: "hand-object-contact",
            model: "Hand-object contact",
            input: "frame.jpg",
            outputFolder: "/tmp/egomodelkit-results",
            status: "ready",
        },
        outputPreview: outputPreview(runId),
    }
}

function startRunResponse(runId = "run-1") {
    return {
        runId,
        status: "running",
        scenario: "single_image",
        summary: {
            modelId: "hand-object-contact",
            model: "Hand-object contact",
            input: "frame.jpg",
            outputFolder: "/tmp/egomodelkit-results",
            status: "running",
        },
        outputPreview: outputPreview(runId),
    }
}

function progressEvent({
    stage,
    displayText,
    current = null,
    total = null,
} : {
    stage: string;
    displayText: string;
    current?: number | null;
    total?: number | null;
}) {
    return {
        stage,
        message: displayText,
        current,
        total,
        unit: null,
        displayText,
    }
}

function progressResponse({
    runId = "run-1",
    status = "running",
    errorMessage = null,
    outputFolder = "/tmp/egomodelkit-results/run-1",
    events = [
        progressEvent({
            stage: "setup",
            displayText: "Preparing image input...",
            current: 1,
            total: 4,
        }),
        progressEvent({
            stage: "runtime",
            displayText: "Running hand-object contact model...",
            current: 2,
            total: 4,
        }),
    ],
    runtimeStatus = null,
    runtimeBuildStages = [],
} : {
    runId?: string;
    status?: string;
    errorMessage?: string | null;
    outputFolder?: string;
    events?: object[];
    runtimeStatus?: object | null;
    runtimeBuildStages?: object[] | null;
}) {
    return {
        runId,
        status,
        errorMessage,
        outputFolder,
        events,
        runtimeStatus,
        runtimeBuildStages,
        outputPreview: outputPreview(runId),
    }
}

async function navigateToReviewStep(user: ReturnType<typeof userEvent.setup>) {
    vi.stubGlobal(
        "fetch",
        vi.fn().mockResolvedValue({
            ok: true,
            status: 200,
            json: async () => ({ outputRoot: "/tmp/egomodelkit-results" }),
        }),
    );

    await navigateToOutputStep(user);
    await user.click(screen.getByRole("button", { name: "Choose Output Folder" }));
    await user.click(screen.getByRole("button", { name: "Continue" }));
}

async function startRunWithProgressResponses(
    user: ReturnType<typeof userEvent.setup>,
    progressResponses: unknown[],
) {
    const fetchMock = vi
        .fn()
        .mockResolvedValueOnce({
            ok: true,
            status: 200,
            json: async () => ({ outputRoot: "/tmp/egomodelkit-results" }),
        })
        .mockResolvedValueOnce({
            ok: true,
            status: 200,
            json: async () => startRunResponse(),
        });

    for (const response of progressResponses) {
        fetchMock.mockResolvedValueOnce({
            ok: true,
            status: 200,
            json: async () => response,
        });
    }

    const wrappedFetchMock = stubFetchWithModels(fetchMock);

    await navigateToOutputStep(user);

    await user.click(screen.getByRole("button", { name: "Choose Output Folder" }));
    await user.click(screen.getByRole("button", { name: "Continue" }));
    await user.click(screen.getByRole("button", { name: "Run Model" }));    
    
    return wrappedFetchMock;
}

async function renderCompletedResultsWithFetchMock(
    user: ReturnType<typeof userEvent.setup>,
    openOutputFolderResponse: {
        ok: boolean;
        status: number;
        json: () => Promise<unknown>;
    },
) {
    const fetchMock = vi.fn(
        async (input: RequestInfo | URL, _init?: RequestInit) => {
            const url = String(input);

            if (url === "/api/select-output-folder") {
                return {
                    ok: true,
                    status: 200,
                    json: async () => ({ outputRoot: "/tmp/egomodelkit-results" }),
                }
            }

            if (url === "/api/runs") {
                return {
                    ok: true,
                    status: 200,
                    json: async () => startRunResponse(),
                }
            }

            if (url === "/api/runs/run-1/progress") {
                return {
                    ok: true,
                    status: 200,
                    json: async () => progressResponse({ status: "completed" }),
                };
            }

            if (url === "/api/open-output-folder") {
                return openOutputFolderResponse;
            }

            throw new Error(`Unexpected fetch call: ${url}`);
    });

    const wrappedFetchMock = stubFetchWithModels(fetchMock);

    await navigateToOutputStep(user);
    await user.click(screen.getByRole("button", { name: "Choose Output Folder" }));

    await waitFor(() => {
        expect(screen.getByRole("button", { name: "Continue" })).toBeEnabled();
    });

    await user.click(screen.getByRole("button", { name: "Continue" }));
    await user.click(screen.getByRole("button", { name: "Run Model" }));

    expect(
        await screen.findByRole("heading", { name: "Run completed" })
    ).toBeInTheDocument();

    return wrappedFetchMock;
}

async function renderCompletedResultsWithCustomProgress(
    user: ReturnType<typeof userEvent.setup>,
    completedProgressResponse: unknown,
) {
    const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
        const url = String(input);

        if (url === "/api/select-output-folder") {
            return {
                ok: true,
                status: 200,
                json: async () => ({ outputRoot: "/tmp/egomodelkit-results" }),
            }
        }

        if (url === "/api/runs") {
            return {
                ok: true,
                status: 200,
                json: async () => startRunResponse(),
            }
        }

        if (url === "/api/runs/run-1/progress") {
            return {
                ok: true,
                status: 200,
                json: async () => completedProgressResponse,
            };
        }

        throw new Error(`Unexpected fetch call: ${url}`);
    });

    const wrappedFetchMock = stubFetchWithModels(fetchMock);

    await navigateToOutputStep(user);
    await user.click(screen.getByRole("button", { name: "Choose Output Folder" }));

    await waitFor(() => {
        expect(screen.getByRole("button", { name: "Continue" })).toBeEnabled();
    });

    await user.click(screen.getByRole("button", { name: "Continue" }));
    await user.click(screen.getByRole("button", { name: "Run Model" }));

    expect(
        await screen.findByRole("heading", { name: "Run completed" })
    ).toBeInTheDocument();

    return wrappedFetchMock;
}

describe("App", () => {
    it("renders the welcome screen", () => {
        render(<App />);

        expect(
            screen.getByRole("heading", { name: "EgoModelKit" }),
        ).toBeInTheDocument();

        expect(
            screen.getByText(
                "Run egocentric video models through a simple local interface.",
            ),
        ).toBeInTheDocument();

        expect(
            screen.getByText(
                "Your selected files are processed locally by default. " +
                "No telemetry or cloud upload is used in this MVP."
            ),
        ).toBeInTheDocument();

        expect(
            screen.getByRole("button", { name: "Start New Run" }),
        ).toBeEnabled();

        expect(
            screen.getByText(
                "Designed for research use. " +
                "Please confirm approved data handling procedures before using clinical data."
            ),
        ).toBeInTheDocument()
    });

    it("opens the model-selection screen with the first wizard step active", async () => {
        const user = userEvent.setup();

        render(<App />);

        await user.click(screen.getByRole("button", { name: "Start New Run" }));

        expect(
            screen.getByRole("heading", { name: "Select a model" }),
        ).toBeInTheDocument();

        expect(
            screen.getByText("Choose the workflow you want to run."),
        ).toBeInTheDocument();

        expect(screen.getByText("Select model")).toBeInTheDocument();
        expect(screen.getByText("Choose input")).toBeInTheDocument();

        expect(screen.getByRole("button", { name: "Continue" })).toBeDisabled();
    });

    it("renders both supported model choices", async () => {
        const user = userEvent.setup();

        render(<App />);

        await user.click(screen.getByRole("button", { name: "Start New Run" }));

        expect(
            screen.getByRole("button", { name: /Hand-object contact/}),
        ).toHaveAttribute("aria-pressed", "false");

        expect(
            screen.getByRole("button", { name: /Activity recognition \(ADL\)/ })
        ).toHaveAttribute("aria-pressed", "false");

        expect(
            screen.getByText("Detects hands, objects, and hand-object contact in images."),
        ).toBeInTheDocument();

        expect(
            screen.getByText(
                "Processes egocentric video clips for " + 
                "activity of daily living (ADL) recognition.",
            ),
        ).toBeInTheDocument();

        expect(
            screen.getByRole("group", { name: "Available models" }),
        ).toBeInTheDocument();

        expect(
            screen.getByText(/Output: detection visualizations and structured results/),
        ).toBeInTheDocument();

        expect(
            screen.getByText(/Output: predictions and processed frame-level files/),
        ).toBeInTheDocument();
    });

   it("does not duplicate input and output label prefixes from backend models", async () => {
        const user = userEvent.setup();

        vi.stubGlobal(
            "fetch",
            vi.fn(async (input: RequestInfo | URL) => {
                const url = String(input);

                if (url === "/api/models") {
                    return okJson({
                        models: [
                            {
                                id: "prefixed-model",
                                name: "Prefixed model",
                                description: "Uses labels already formatted by the backend.",
                                acceptedInputLabel: "Input: prepared images",
                                supportedInputExtensions: [".jpg"],
                                outputLabel: "Output: prepared results",
                            },
                        ],
                    });
                }

                throw new Error(`Unexpected fetch call: ${url}`);
            }),
        );

        render(<App />);

        await user.click(screen.getByRole("button", { name: "Start New Run" }));

        expect(
            await screen.findByRole("button", { name: /Prefixed model/ }),
        ).toBeInTheDocument();

        expect(screen.getByText(/Input: prepared images/)).toBeInTheDocument();
        expect(screen.getByText(/Output: prepared results/)).toBeInTheDocument();

        expect(screen.queryByText(/Input: Input:/)).not.toBeInTheDocument();
        expect(screen.queryByText(/Output: Output:/)).not.toBeInTheDocument();
    });

    it("selects a model and enables continue", async () => {
        const user = userEvent.setup();

        render(<App />);

        await user.click(screen.getByRole("button", { name: "Start New Run" }));

        const handObjectModel = screen.getByRole("button", {
            name: /Hand-object contact/,
        });

        await user.click(handObjectModel);

        expect(handObjectModel).toHaveAttribute("aria-pressed", "true");
        expect(screen.getByRole("button", { name: "Continue" })).toBeEnabled();
    });

    it("allows switching the selected model", async() => {
        const user = userEvent.setup();

        render(<App />);

        await user.click(screen.getByRole("button", { name: "Start New Run" }));

        const handObjectModel = screen.getByRole("button", {
            name: /Hand-object contact/,
        });

        const adlModel = screen.getByRole("button", {
            name: /Activity recognition \(ADL\)/,
        });

        await user.click(handObjectModel);
        await user.click(adlModel);

        expect(handObjectModel).toHaveAttribute("aria-pressed", "false");
        expect(adlModel).toHaveAttribute("aria-pressed", "true");
    });

    it("loads model choices from the backend model endpoint", async () => {
        const user = userEvent.setup();

        const fetchMock = stubFetchWithModels(async () => {
            throw new Error("No other calls expected."); 
        });

        render(<App />);

        await user.click(screen.getByRole("button", { name: "Start New Run" }));

        expect(
            screen.getByRole("button", { name: /Hand-object contact/}),
        ).toBeInTheDocument();

        expect(
            screen.getByRole("button", { name: /Activity recognition \(ADL\)/ })
        ).toBeInTheDocument();
    });

    it("shows a loading state while backend models are loading", async () => {
        const user = userEvent.setup();

        vi.stubGlobal(
            "fetch",
            vi.fn(
                () =>
                    new Promise(() => {
                        // keep pending
                    }),
            ),
        );

        render(<App />);

        await user.click(screen.getByRole("button", { name: "Start New Run" }));

        expect(screen.getByRole("status")).toHaveTextContent(
            "Loading available models...",
        );

        expect(screen.getByRole("button", { name: "Continue" })).toBeDisabled();
    });

   it("shows an empty model list message when the backend returns no models", async () => {
        const user = userEvent.setup();

        vi.stubGlobal(
            "fetch",
            vi.fn(async (input: RequestInfo | URL) => {
                const url = String(input);

                if (url === "/api/models") {
                    return okJson({ models: [] });
                }

                throw new Error(`Unexpected fetch call: ${url}`);
            }),
        );

        render(<App />);

        await user.click(screen.getByRole("button", { name: "Start New Run" }));

        expect(
            await screen.findByText("No models are available from the local backend."),
        ).toBeInTheDocument();

        expect(screen.getByRole("button", { name: "Continue" })).toBeDisabled();
    });

    it("shows a graceful error when backend models cannot be loaded", async () => {
        const user = userEvent.setup();

        vi.stubGlobal(
            "fetch",
            vi.fn(async () => ({
                ok: false,
                status: 500,
                json: async() => ({}),
            })),
        );

        render(<App />);

        await user.click(screen.getByRole("button", { name: "Start New Run" }));

        expect(await screen.findByRole("alert")).toHaveTextContent(
            "Unable to load available models.",
        );

        expect(screen.getByRole("button", { name: "Continue" })).toBeDisabled();
    });

    it("ignores backend model responses after unmounting", async () => {
        let resolveModels!: (response: ReturnType<typeof okJson>) => void;

        const pendingModels = new Promise<ReturnType<typeof okJson>>((resolve) => {
            resolveModels = resolve;
        });

        const fetchMock = vi.fn(() => pendingModels);

        vi.stubGlobal("fetch", fetchMock);

        const { unmount } = render(<App />);

        unmount();

        await act(async () => {
            resolveModels(okJson(modelsResponse()));
            await pendingModels;
            await Promise.resolve();
        });

        expect(fetchMock).toHaveBeenCalledWith("/api/models");
    });

   it("ignores backend model errors after unmounting", async () => {
        let rejectModels!: (reason?: unknown) => void;

        const pendingModels = new Promise<ReturnType<typeof okJson>>((_, reject) => {
            rejectModels = reject;
        });

        const fetchMock = vi.fn(() => pendingModels);

        vi.stubGlobal("fetch", fetchMock);

        const { unmount } = render(<App />);

        unmount();

        await act(async () => {
            rejectModels(new Error("Model request failed."));
            await pendingModels.catch(() => undefined);
            await Promise.resolve();
        });

        expect(fetchMock).toHaveBeenCalledWith("/api/models");
    });

    it("continues from hand-object model selection to an image input screen", async () => {
        const user = userEvent.setup();

        render(<App />);

        await user.click(screen.getByRole("button", { name: "Start New Run" }));
        await user.click(screen.getByRole("button", { name: /Hand-object contact/ }));
        await user.click(screen.getByRole("button", { name: "Continue" }));

        expect(
            screen.getByRole("heading", { name: "Choose input" }),
        ).toBeInTheDocument();

        expect(
            screen.getByText("Select an image or multiple images")
        ).toBeInTheDocument();
        
        expect(
            screen.getByText("Drop input or choose from your computer"),
        ).toBeInTheDocument();

        expect(screen.getByText("No input selected yet.")).toBeInTheDocument();
        expect(screen.getByRole("button", { name: "Continue" })).toBeDisabled();
    });

    it("continues from adl model selection to a video input screen", async () => {
        const user = userEvent.setup()

        render(<App />);

        await user.click(screen.getByRole("button", { name: "Start New Run" }));
        await user.click(screen.getByRole("button", { name: /Activity recognition \(ADL\)/ }));
        await user.click(screen.getByRole("button", { name: "Continue" }));

        expect(
            screen.getByRole("heading", { name: "Choose input" }),
        ).toBeInTheDocument();

        expect(
            screen.getByText("Select a video or multiple videos"),
        ).toBeInTheDocument();

        expect(screen.getByRole("button", {name: "Continue" })).toBeDisabled();
    });

    it(
        "uses the fallback model screen actions when the selected model is unavailable",
        async () => {
            const user = userEvent.setup();

            const unavailableSelectedModelList = modelsResponse().models.slice(0, 1);

            unavailableSelectedModelList.find = vi.fn(() => undefined);

            vi.stubGlobal(
                "fetch",
                vi.fn(async (input: RequestInfo | URL) => {
                    const url = String(input);

                    if (url === "/api/models") {
                        return okJson({
                            models: unavailableSelectedModelList,
                        });
                    }

                    throw new Error(`Unexpected fetch call: ${url}`);
                }),
            )

            render(<App />);

            await user.click(screen.getByRole("button", { name: "Start New Run" }));

            await user.click(
                await screen.findByRole("button", { name: /Hand-object contact/ })
            );

            await user.click(screen.getByRole("button", { name: "Continue" }));

            expect(
                screen.getByRole("heading", { name: "Select a model" })
            ).toBeInTheDocument();

            await user.click(screen.getByRole("button", { name: "Continue" }));

            expect(
                screen.getByRole("heading", { name: "Select a model" })
            ).toBeInTheDocument();

            await user.click(screen.getByRole("button", { name: "Back" }));

            expect(
                screen.getByRole("heading", { name: "EgoModelKit" }),
            ).toBeInTheDocument();
        }
    );

    it("selects one input file and enables continue", async () => {
        const user = userEvent.setup();
        
        render(<App />);

        await user.click(screen.getByRole("button", { name: "Start New Run" }));
        await user.click(screen.getByRole("button", { name: /Hand-object contact/}));
        await user.click(screen.getByRole("button", { name: "Continue" }));

        await user.upload(
            screen.getByLabelText("Choose input files"),
            new File(["fake image"], "frame.jpg", { type: "image/jpeg" }),
        );

        expect(screen.getByText("Selected: 1 file")).toBeInTheDocument();
        expect(screen.getByText("frame.jpg")).toBeInTheDocument();
        expect(screen.getByRole("button", { name: "Continue" })).toBeEnabled();   
    });

    it("summarizes multiple selected input files", async () => {
        const user = userEvent.setup();

        render(<App />);

        await user.click(screen.getByRole("button", { name: "Start New Run" }));
        await user.click(screen.getByRole("button", { name: /Hand-object contact/}));
        await user.click(screen.getByRole("button", { name: "Continue" }));

        await user.upload(screen.getByLabelText("Choose input files"), [
            new File(["fake image"], "frame-1.jpg", { type: "image/jpeg" }),
            new File(["fake image"], "frame-2.jpg", { type: "image/jpeg" }),
        ]);

        expect(screen.getByText("Selected: 2 files")).toBeInTheDocument();
        expect(screen.getByText("frame-1.jpg")).toBeInTheDocument();
        expect(screen.getByText("frame-2.jpg")).toBeInTheDocument();
        expect(screen.getByRole("button", { name: "Continue" })).toBeEnabled();
    });

    it("filters unsupported files from hand-object input selection", async () => {
        const user = userEvent.setup({ applyAccept: false });

        render(<App />);

        await user.click(screen.getByRole("button", { name: "Start New Run" }));
        await user.click(screen.getByRole("button", { name: /Hand-object contact/ }));
        await user.click(screen.getByRole("button", { name: "Continue" }));

        await user.upload(screen.getByLabelText("Choose input files"), [
            new File(["fake image"], "frame.jpg", { type: "image/jpeg" }),
            new File(["fake video"], "clip.mp4", { type: "video/mp4" }),
            new File(["notes"], "notes.txt", { type: "text/plain" }),
        ]);

        expect(screen.getByText("Selected: 1 file")).toBeInTheDocument();
        expect(screen.getByText("frame.jpg")).toBeInTheDocument();
        expect(screen.getByText("Ignored: 2 files")).toBeInTheDocument();
        expect(screen.getByText("clip.mp4")).toBeInTheDocument();
        expect(screen.getByText("notes.txt")).toBeInTheDocument();
        expect(screen.getByRole("button", { name: "Continue" })).toBeEnabled();
    });

    it("filters unsupported files from ADL input selection", async () => {
        const user = userEvent.setup({ applyAccept: false });

        render(<App />);

        await user.click(screen.getByRole("button", { name: "Start New Run" }));
        await user.click(screen.getByRole("button", { name: /Activity recognition \(ADL\)/ }));
        await user.click(screen.getByRole("button", { name: "Continue" }));

        await user.upload(screen.getByLabelText("Choose input files"), [
            new File(["fake video"], "clip.mp4", { type: "video/mp4" }),
            new File(["fake image"], "frame.jpg", { type: "image/jpeg" }),
            new File(["notes"], "notes.txt", { type: "text/plain" }),
        ]);

        expect(screen.getByText("Selected: 1 file")).toBeInTheDocument();
        expect(screen.getByText("clip.mp4")).toBeInTheDocument();
        expect(screen.getByText("Ignored: 2 files")).toBeInTheDocument();
        expect(screen.getByText("frame.jpg")).toBeInTheDocument();
        expect(screen.getByText("notes.txt")).toBeInTheDocument();
        expect(screen.getByRole("button", { name: "Continue" })).toBeEnabled();
    });

    it(
        "falls back to no supported extensions when filtering input without model extensions", 
        async () => {
            const user = userEvent.setup();

            const supportedInputExtensions = vi
                .fn()
                .mockReturnValueOnce([".jpg"])
                .mockReturnValueOnce([".jpg"])
                .mockReturnValueOnce(undefined)
                .mockReturnValue([".jpg"]);

            vi.stubGlobal(
                "fetch",
                vi.fn(async (input: RequestInfo | URL) => {
                    const url = String(input);

                    if (url === "/api/models") {
                        return okJson({
                            models: [
                                {
                                    id: "missing-filter-extensions",
                                    name: "Missing filter extensions",
                                    description:
                                        "Uses available picker extensions but " +
                                        "omits them during filtering.",
                                    acceptedInputLabel: "an image or multiple images",
                                    get supportedInputExtensions() {
                                        return supportedInputExtensions();
                                    },
                                    outputLabel: "filtered test results",
                                },
                            ],
                        });
                    }

                    throw new Error(`Unexpected fetch call: ${url}`);
                }),
            );

            render(<App />);

            await user.click(screen.getByRole("button", { name: "Start New Run" }));
            
            await user.click(
                await screen.findByRole("button", { name: /Missing filter extensions/ }),
            );
            
            await user.click(screen.getByRole("button", { name: "Continue" }));

            expect(screen.getByText("Supported files: .jpg")).toBeInTheDocument();

            await user.upload(
                screen.getByLabelText("Choose input files"),
                new File(["fake image"], "frame.jpg", { type: "image/jpeg" }),
            );

            expect(screen.getByText("No input selected yet.")).toBeInTheDocument();
            expect(screen.getByText("Ignored: 1 file")).toBeInTheDocument();
            expect(screen.getByText("frame.jpg")).toBeInTheDocument();
            expect(screen.getByRole("button", { name: "Continue" })).toBeDisabled();
        }
    );

    it("rejects input selection with no supported files", async () => {
        const user = userEvent.setup({ applyAccept: false });

        render(<App />);

        await user.click(screen.getByRole("button", { name: "Start New Run" }));
        await user.click(screen.getByRole("button", { name: /Activity recognition \(ADL\)/ }));
        await user.click(screen.getByRole("button", { name: "Continue" }));

        await user.upload(screen.getByLabelText("Choose input files"), [
            new File(["fake image"], "frame.jpg", { type: "image/jpeg" }),
            new File(["notes"], "notes.txt", { type: "text/plain" }),
        ]);

        expect(screen.getByText("No input selected yet.")).toBeInTheDocument();
        expect(screen.getByText("Ignored: 2 files")).toBeInTheDocument();
        expect(screen.getByText("frame.jpg")).toBeInTheDocument();
        expect(screen.getByText("notes.txt")).toBeInTheDocument();
        expect(screen.getByRole("button", { name: "Continue" })).toBeDisabled();
    });

    it("sends only supported selected files to the backend", async () => {
        const user = userEvent.setup({ applyAccept: false });

        const fetchMock = vi
            .fn()
            .mockResolvedValueOnce({
                ok: true,
                status: 200,
                json: async () => ({ outputRoot: "/tmp/egomodelkit-results" }),
            })
            .mockResolvedValueOnce({
                ok: true,
                status: 200,
                json: async () => dryRunResponse(),
            });

        const wrappedFetchMock = stubFetchWithModels(fetchMock);

        render(<App />);

        await user.click(screen.getByRole("button", { name: "Start New Run" }));
        await user.click(screen.getByRole("button", { name: /Hand-object contact/ }));
        await user.click(screen.getByRole("button", { name: "Continue" }));

        await user.upload(screen.getByLabelText("Choose input files"), [
            new File(["fake image"], "frame.jpg", { type: "image/jpeg" }),
            new File(["fake video"], "clip.mp4", { type: "video/mp4" }),
        ]);

        await user.click(screen.getByRole("button", { name: "Continue" }));
        await user.click(screen.getByRole("button", { name: "Choose Output Folder" }));

        await waitFor(() => {
            expect(screen.getByRole("button", { name: "Continue" })).toBeEnabled();
        });

        await user.click(screen.getByRole("button", { name: "Continue" }));
        await user.click(screen.getByRole("button", { name: "Dry Run" }));

        const dryRunCall = wrappedFetchMock.mock.calls.find(
            ([input]) => String(input) === "/api/dry-run",
        );

        expect(dryRunCall).toBeDefined();

        const dryRunRequest = dryRunCall?.[1] as RequestInit;
        const formData = dryRunRequest.body as FormData;

        expect(formData.getAll("files").map((file) => (file as File).name)).toEqual([
            "frame.jpg",
        ]);
    });

    it("ignores empty input drops", async () => {
        const user = userEvent.setup();

        render(<App />);

        await user.click(screen.getByRole("button", { name: "Start New Run" }));
        await user.click(screen.getByRole("button", { name: /Hand-object contact/}));
        await user.click(screen.getByRole("button", { name: "Continue" }));

        fireEvent.drop(screen.getByTestId("input-drop-zone"), {
            dataTransfer: {
                files: [],
            },
        });

        expect(screen.getByText("No input selected yet.")).toBeInTheDocument();
        expect(screen.getByRole("button", { name: "Continue" })).toBeDisabled();
    });

    it("accepts input files by drag and drop", async () => {
        const user = userEvent.setup();

        render(<App />);

        await user.click(screen.getByRole("button", { name: "Start New Run" }));
        await user.click(screen.getByRole("button", { name: /Hand-object contact/}));
        await user.click(screen.getByRole("button", { name: "Continue" }));
        
        const dropZone = screen.getByTestId("input-drop-zone");
        
        const droppedFile = new File(["fake image"], "dropped-frame.png", {
            type: "image/png",
        });

        fireEvent.dragOver(dropZone)
        dropInputFiles(dropZone, [droppedFile]);

        expect(screen.getByText("Selected: 1 file")).toBeInTheDocument();
        expect(screen.getByText("dropped-frame.png")).toBeInTheDocument();
        expect(screen.getByRole("button", { name: "Continue" })).toBeEnabled();
    });

    it("opens the native file picker from the visible choose-input button", async () => {
        const user = userEvent.setup();

        render(<App />);

        await user.click(screen.getByRole("button", { name: "Start New Run" }));
        await user.click(screen.getByRole("button", { name: /Hand-object contact/ }));
        await user.click(screen.getByRole("button", { name: "Continue" }));

        const input = screen.getByLabelText("Choose input files");
        const inputClickSpy = vi.spyOn(input, "click");

        await user.click(screen.getByRole("button", { name: "Choose input" }));

        expect(inputClickSpy).toHaveBeenCalledOnce();
        expect(input).toHaveAttribute("accept", ".jpg,.jpeg,.png,.bmp,.webp");
    });

    it("returns from choose input to the model selection screen", async () => {
        const user = userEvent.setup();

        render(<App />);

        expect(
            screen.getByRole("heading", { name: "EgoModelKit" }),
        ).toBeInTheDocument();

        expect(
            screen.getByRole("button", { name: "Start New Run" }),
        ).toBeEnabled();

        await user.click(screen.getByRole("button", { name: "Start New Run" }));

        expect(
            screen.getByRole("button", { name: /Hand-object contact/}),
        ).toHaveAttribute("aria-pressed", "false");

        expect(
            screen.getByRole("button", { name: /Activity recognition \(ADL\)/ })
        ).toHaveAttribute("aria-pressed", "false");


        await user.click(screen.getByRole("button", { name: /Hand-object contact/ }));
        await user.click(screen.getByRole("button", { name: "Continue" }));

        expect(
            screen.getByText("Select an image or multiple images"),
        ).toBeInTheDocument();
        
        expect(
            screen.getByText("No input selected yet."),
        ).toBeInTheDocument();

        await user.click(screen.getByRole("button", { name: "Back" }));

        expect(
            screen.getByRole("button", { name: /Hand-object contact/}),
        ).toHaveAttribute("aria-pressed", "true");
    });

    it("returns from model selection to the welcome screen", async () => {
        const user = userEvent.setup();

        render(<App />);

        await user.click(screen.getByRole("button", { name: "Start New Run" }));
        await user.click(screen.getByRole("button", {name: "Back"}));

        expect(
            screen.getByRole("heading", { name: "EgoModelKit" }),
        ).toBeInTheDocument();

        expect(
            screen.getByRole("button", { name: "Start New Run" }),
        ).toBeEnabled();
    });

    it("continues from selected input to the output-folder screen", async () => {
        const user = userEvent.setup();

        await navigateToOutputStep(user);

        expect(
            screen.getByRole("heading", { name: "Choose output folder"}),
        ).toBeInTheDocument();

        expect(
            screen.getByText("Select a folder where EgoModelKit should save the results."),
        ).toBeInTheDocument();

        expect(screen.getByText("No output folder selected")).toBeInTheDocument();
        expect(screen.getByRole("button", { name: "Choose Output Folder"} )).toBeEnabled();
        expect(screen.getByRole("button", { name: "Continue"} )).toBeDisabled();

        expect(
            screen.getByText(
                "A new run folder will be created inside the selected output folder."
            ),
        ).toBeInTheDocument();
    });

    it("returns from output folder screen to the selected input screen", async () => {
        const user = userEvent.setup();

        await navigateToOutputStep(user);

        expect(
            screen.getByRole("heading", { name: "Choose output folder"}),
        ).toBeInTheDocument();
        
        await user.click(screen.getByRole("button", { name: "Back" }));

        expect(
            screen.getByRole("heading", { name: "Choose input" }),
        ).toBeInTheDocument();

        expect(screen.getByText("Selected: 1 file")).toBeInTheDocument();
        expect(screen.getByText("frame.jpg")).toBeInTheDocument();
    });

    it("keeps the selected model when selecting the same model again", async () => {
        const user = userEvent.setup();

        render(<App />);

        await user.click(screen.getByRole("button", { name: "Start New Run" }));

        const handObjectModel = screen.getByRole("button", { name: /Hand-object contact/  });

        await user.click(handObjectModel);
        await user.click(handObjectModel);

        expect(handObjectModel).toHaveAttribute("aria-pressed", "true");
        expect(screen.getByRole("button", { name: "Continue" })).toBeEnabled();
    });

    it("clears selected input when the file picker reports no files", async () => {
        const user = userEvent.setup();

        render(<App />);

        await user.click(screen.getByRole("button", { name: "Start New Run" }));
        await user.click(screen.getByRole("button", { name: /Hand-object contact/ }));
        await user.click(screen.getByRole("button", { name: "Continue" }));

        await user.upload(
            screen.getByLabelText("Choose input files"),
            new File(["fake image"], "frame.jpg", { type: "image/jpeg" }),
        );

        expect(screen.getByText("Selected: 1 file")).toBeInTheDocument();
        expect(screen.getByText("frame.jpg")).toBeInTheDocument();

        fireEvent.change(screen.getByLabelText("Choose input files"), {
            target: {
                files: null,
            },
        });

        expect(screen.getByText("No input selected yet.")).toBeInTheDocument();
        expect(screen.getByRole("button", { name: "Continue" })).toBeDisabled();
    });

    it("ignores dropped input when the drop event has no file list", async () => {
        const user = userEvent.setup();

        render(<App />);

        await user.click(screen.getByRole("button", { name: "Start New Run" }));
        await user.click(screen.getByRole("button", { name: /Hand-object contact/ }));
        await user.click(screen.getByRole("button", { name: "Continue" }));
    
        dropWithoutFileList(screen.getByTestId("input-drop-zone"));

        expect(screen.getByText("No input selected yet.")).toBeInTheDocument();
        expect(screen.getByRole("button", { name: "Continue" })).toBeDisabled();
    });

    it("returns home when the EgoModelKit header title is clicked", async () => {
        const user = userEvent.setup();

        render(<App />);

        await user.click(screen.getByRole("button", { name: "Start New Run" }));
        await user.click(screen.getByRole("button", { name: /Hand-object contact/ }));
        await user.click(screen.getByRole("button", { name: "Continue" }));
    
        expect(
            screen.getByRole("heading", { name: "Choose input" }),
        ).toBeInTheDocument();

        await user.click(screen.getByRole("button", { name: "EgoModelKit" }));

        expect(
            screen.getByRole("heading", { name: "EgoModelKit" }),
        ).toBeInTheDocument();

        expect(
            screen.getByRole("button", { name: "Start New Run" }),
        ).toBeInTheDocument();
    });

    it("selects an output folder from the backend picker response", async () => {
        const user = userEvent.setup();

        vi.stubGlobal(
            "fetch",
            vi.fn().mockResolvedValue({
                ok: true,
                status: 200,
                json: async () => ({ outputRoot: "/tmp/egomodelkit-results" }),
            }),
        );

        await navigateToOutputStep(user);

        await user.click(screen.getByRole("button", { name: "Choose Output Folder" }));

        expect(fetch).toHaveBeenCalledWith("/api/select-output-folder", {
            method: "POST",
        });

        expect(screen.getByText("/tmp/egomodelkit-results")).toBeInTheDocument();
        expect(screen.getByRole("button", { name: "Continue" })).toBeEnabled();
    });

    it(
        "falls back to manually entered output path when the backend picker is unavailable",
        async () => {
            const user = userEvent.setup();

            vi.stubGlobal(
                "fetch",
                vi.fn().mockResolvedValue({
                    ok: false,
                    status: 405,
                    json: async () => ({}),
                }),
            );        
            
            vi.spyOn(window, "prompt").mockReturnValue("/manual/results");

            await navigateToOutputStep(user);

            await user.click(screen.getByRole("button", { name: "Choose Output Folder" }));

            expect(window.prompt).toHaveBeenCalledWith(
                "Enter the output folder path:",
                "/Users/Research/Desktop/EgoModelKit Results"
            );

            expect(screen.getByText("/manual/results")).toBeInTheDocument();
            expect(screen.getByRole("button", { name: "Continue" })).toBeEnabled();
        }
    );

    it("keeps output selection empty when the fallback prompt is cancelled", async () => {
        const user = userEvent.setup();

        vi.stubGlobal(
            "fetch",
            vi.fn().mockResolvedValue({
                ok: false,
                status: 404,
                json: async () => ({}),
            }),
        );

        vi.spyOn(window, "prompt").mockReturnValue(null);

        await navigateToOutputStep(user);

        await user.click(screen.getByRole("button", { name: "Choose Output Folder"}));

        expect(screen.getByText("No output folder selected")).toBeInTheDocument();
        expect(screen.getByRole("button", { name: "Continue" })).toBeDisabled();
    });

    it("shows an error if output folder selection fails", async () => {
        const user = userEvent.setup();

        vi.stubGlobal(
            "fetch",
            vi.fn().mockResolvedValue({
                ok: false,
                status: 500,
                json: async () => ({}),
            }),
        );

        await navigateToOutputStep(user);

        await user.click(screen.getByRole("button", { name: "Choose Output Folder" }));

        expect(screen.getByRole("alert")).toHaveTextContent(
            "Unable to choose output folder.",
        );

        expect(screen.getByText("No output folder selected")).toBeInTheDocument();
    });   

    it("continues from selected output folder to the review screen", async () => {
        const user = userEvent.setup();

        await navigateToReviewStep(user);

        expect(screen.getByRole("heading", { name: "Review and run" }),).toBeInTheDocument();

        expect(
            screen.getByText("Confirm the model, input, and output location before starting."),
        ).toBeInTheDocument();

        expect(screen.getByText("Summary")).toBeInTheDocument();
        expect(screen.getByText("Hand-object contact")).toBeInTheDocument();
        expect(screen.getByText("frame.jpg")).toBeInTheDocument();
        expect(screen.getByText("/tmp/egomodelkit-results")).toBeInTheDocument();
        expect(screen.getByText("Local")).toBeInTheDocument();
        expect(screen.getByText("Ready to start.")).toBeInTheDocument();
        
        expect(screen.getByRole("button", { name: "Dry Run"})).toBeEnabled();
        expect(screen.getByRole("button", { name: "Run Model"})).toBeEnabled();
    });

    it("runs a dry run and shows the dry-run success panel", async () => {
        const user = userEvent.setup();
        
        const fetchMock = vi
            .fn()
            .mockResolvedValueOnce({
                ok: true,
                status: 200,
                json: async () => ({ outputRoot: "/tmp/egomodelkit-results" }),
            })
            .mockResolvedValueOnce({
                ok: true,
                status: 200,
                json: async () => dryRunResponse(),
            });

        const wrappedFetchMock = stubFetchWithModels(fetchMock);
        
        await navigateToOutputStep(user);
        await user.click(screen.getByRole("button", { name: "Choose Output Folder" }));
        await user.click(screen.getByRole("button", { name: "Continue" }));
        await user.click(screen.getByRole("button", { name: "Dry Run" }));

        expect(wrappedFetchMock).toHaveBeenLastCalledWith(
            "/api/dry-run",
            expect.objectContaining({
                method: "POST",
                body: expect.any(FormData)
            }),
        );

        const dryRunRequest = wrappedFetchMock.mock.calls[1][1] as RequestInit;
        const formData = dryRunRequest.body as FormData;

        expect(formData.get("modelId")).toBe("hand-object-contact");
        expect(formData.get("outputRoot")).toBe("/tmp/egomodelkit-results");
        expect((formData.getAll("files")[0] as File).name).toBe("frame.jpg");

        expect(screen.getByRole(
            "heading", { name: "Dry run completed successfully." })
        ).toBeInTheDocument();

        expect(screen.getByText("Checking selected input...")).toBeInTheDocument();
        expect(screen.getByText("Checking output folder...")).toBeInTheDocument();
        expect(screen.getByText("Checking local runtime...")).toBeInTheDocument();
    });

    it("shows an error when dry run fails", async () => {
        const user = userEvent.setup();
        
        const fetchMock = vi
            .fn()
            .mockResolvedValueOnce({
                ok: true,
                status: 200,
                json: async () => ({ outputRoot: "/tmp/egomodelkit-results" }),
            })
            .mockResolvedValueOnce({
                ok: false,
                status: 500,
                json: async () => ({}),
            });

        stubFetchWithModels(fetchMock);

        await navigateToOutputStep(user);
        await user.click(screen.getByRole("button", { name: "Choose Output Folder" }));
        await user.click(screen.getByRole("button", { name: "Continue" }));
        await user.click(screen.getByRole("button", { name: "Dry Run" }));
    
        expect(screen.getByRole("alert")).toHaveTextContent("Unable to complete dry run.");
        expect(screen.getByText("Ready to start.")).toBeInTheDocument();
    });

    it("shows dry-run runtime-check details from the backend", async () => {
        const user = userEvent.setup();

        const runtimeError = [
            "EgoModelKit model runs require a Linux host with an NVIDIA GPU;",
            "detected Darwin.",
        ].join(" ");
        
        const fetchMock = vi
            .fn()
            .mockResolvedValueOnce({
                ok: true,
                status: 200,
                json: async () => ({ outputRoot: "/tmp/egomodelkit-results" }),
            })
            .mockResolvedValueOnce({
                ok: false,
                status: 400,
                json: async () => ({ detail: runtimeError }),
            });

        stubFetchWithModels(fetchMock);

        await navigateToOutputStep(user);
        await user.click(screen.getByRole("button", { name: "Choose Output Folder" }));
        await user.click(screen.getByRole("button", { name: "Continue" }));
        await user.click(screen.getByRole("button", { name: "Dry Run" }));
    
        expect(screen.getByRole("alert")).toHaveTextContent(runtimeError);
        expect(screen.getByText("Ready to start.")).toBeInTheDocument();
    });

    it("shows the fallback dry-run error when the error body is unreadable", async () => {
        const user = userEvent.setup();
        
        const fetchMock = vi
            .fn()
            .mockResolvedValueOnce({
                ok: true,
                status: 200,
                json: async () => ({ outputRoot: "/tmp/egomodelkit-results" }),
            })
            .mockResolvedValueOnce({
                ok: false,
                status: 500,
                json: async () => {
                    throw new Error("Invalid error body.");
                },
            });

        stubFetchWithModels(fetchMock);

        await navigateToOutputStep(user);
        await user.click(screen.getByRole("button", { name: "Choose Output Folder" }));
        await user.click(screen.getByRole("button", { name: "Continue" }));
        await user.click(screen.getByRole("button", { name: "Dry Run" }));

        expect(screen.getByRole("alert")).toHaveTextContent("Unable to complete dry run.");
        expect(screen.getByText("Ready to start.")).toBeInTheDocument();
    });

    it("shows a clear dry-run error when the output folder does not exist", async () => {
        const user = userEvent.setup();

        const fetchMock = vi
            .fn()
            .mockResolvedValueOnce({
                ok: false,
                status: 405,
                json: async () => ({}),
            })
            .mockResolvedValueOnce({
                ok: false,
                status: 400,
                json: async () => ({
                    detail: 
                        "Output folder does not exist. " + 
                        "Choose an existing folder before continuing.",
                }),
            });

        stubFetchWithModels(fetchMock);

        vi.spyOn(window, "prompt").mockReturnValue("/manual/missing-results");

        await navigateToOutputStep(user);
        
        await user.click(screen.getByRole("button", { name: "Choose Output Folder" }));
        await user.click(screen.getByRole("button", { name: "Continue" }));
        await user.click(screen.getByRole("button", { name: "Dry Run" }));

        expect(screen.getByRole("alert")).toHaveTextContent(
            "Output folder does not exist. Choose an existing folder before continuing.",
        );

        expect(screen.getByText("Ready to start.")).toBeInTheDocument();
    });

    it("shows only Cancel Run while dry run is in progress", async () => {
        const user = userEvent.setup();

        const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
            const url = String(input);

            if (url === "/api/select-output-folder") {
                return okJson({ outputRoot: "/tmp/egomodelkit-results" });
            }

            if (url === "/api/dry-run") {
                return new Promise(() => {
                    // Keep dry run pending.
                });
            }

            if (url === "/api/cancel-run") {
                return okJson({ cancelled: true, runId: null, operationId: "operation-1" });
            }

            throw new Error(`Unexpected fetch call: ${url}`);
        });

        stubFetchWithModels(fetchMock);

        await navigateToOutputStep(user);

        await user.click(screen.getByRole("button", { name: "Choose Output Folder" }));
        await user.click(screen.getByRole("button", { name: "Continue" }));
        await user.click(screen.getByRole("button", { name: "Dry Run" }));

        expect(screen.getByRole("button", { name: "Cancel Run" })).toBeInTheDocument();
        expect(screen.queryByRole("button", { name: "Dry Run" })).not.toBeInTheDocument();
        expect(screen.queryByRole("button", { name: "Run Model" })).not.toBeInTheDocument();
    });

    it("restores Dry Run and Run Model buttons after dry run completes", async () => {
        const user = userEvent.setup();

        const fetchMock = vi
            .fn()
            .mockResolvedValueOnce(okJson({ outputRoot: "/tmp/egomodelkit-results" }))
            .mockResolvedValueOnce(okJson(dryRunResponse()));

        stubFetchWithModels(fetchMock);

        await navigateToOutputStep(user);

        await user.click(screen.getByRole("button", { name: "Choose Output Folder" }));
        await user.click(screen.getByRole("button", { name: "Continue" }));
        await user.click(screen.getByRole("button", { name: "Dry Run" }));

        expect(
            await screen.findByRole("heading", {
                name: "Dry run completed successfully.",
            }),
        ).toBeInTheDocument();

        expect(screen.getByRole("button", { name: "Dry Run" })).toBeInTheDocument();
        expect(screen.getByRole("button", { name: "Run Model" })).toBeInTheDocument();
        expect(screen.queryByRole("button", { name: "Cancel Run" })).not.toBeInTheDocument();
    });

    it(
        "keeps the running screen when header navigation cancellation is declined",
        async () => {
            const user = userEvent.setup();
            const confirmMock = vi.fn(() => false);

            vi.stubGlobal("confirm", confirmMock);

            const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
                const url = String(input);

                if (url === "/api/select-output-folder") {
                    return okJson({ outputRoot: "/tmp/egomodelkit-results" });
                }

                if (url === "/api/runs") {
                    return okJson(startRunResponse("run-stay-running"));
                }

                if (url === "/api/runs/run-stay-running/progress") {
                    return okJson(
                        progressResponse({
                            runId: "run-stay-running",
                            status: "running",
                        }),
                    );
                }

                throw new Error(`Unexpected fetch call: ${url}`);
            });

            stubFetchWithModels(fetchMock);

            await navigateToOutputStep(user);
            
            await user.click(screen.getByRole("button", { name: "Choose Output Folder" }));
            await user.click(screen.getByRole("button", { name: "Continue" }));
            await user.click(screen.getByRole("button", { name: "Run Model" }));

            expect(await screen.findByText("Running model...")).toBeInTheDocument();

            await user.click(screen.getByRole("button", { name: "EgoModelKit" }));

            expect(confirmMock).toHaveBeenCalledOnce();
            expect(screen.getByText("Running model...")).toBeInTheDocument();

            expect(
                fetchMock.mock.calls.some(([input]) => String(input) === "/api/cancel-run"),
            ).toBe(false);
        }
    );

    it("shows an alert when cancelling the backend operation fails", async () => {
        const user = userEvent.setup();

        const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
            const url = String(input);

            if (url === "/api/select-output-folder") {
                return okJson({ outputRoot: "/tmp/egomodelkit-results" });
            }

            if (url === "/api/runs") {
                return okJson(startRunResponse("run-cancel-fails"));
            }

            if (url === "/api/runs/run-cancel-fails/progress") {
                return okJson(
                    progressResponse({
                        runId: "run-cancel-fails",
                        status: "running",
                    }),
                );
            }

            if (url === "/api/cancel-run") {
                return {
                    ok: false,
                    status: 500,
                    json: async () => ({}),
                };
            }

            throw new Error(`Unexpected fetch call: ${url}`);
        });

        stubFetchWithModels(fetchMock);

        await navigateToOutputStep(user);

        await user.click(screen.getByRole("button", { name: "Choose Output Folder" }));
        await user.click(screen.getByRole("button", { name: "Continue" }));
        await user.click(screen.getByRole("button", { name: "Run Model" }));

        expect(await screen.findByText("Running model...")).toBeInTheDocument();

        await user.click(screen.getByRole("button", { name: "Cancel Run" }));

        expect(screen.getByRole("alert")).toHaveTextContent(
            "Unable to cancel the backend operation.",
        );

        expect(screen.getByText("Ready to start.")).toBeInTheDocument();
    });

    it("returns home when polling reports that a run was cancelled", async () => {
        const user = userEvent.setup();

        const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
            const url = String(input);

            if (url === "/api/select-output-folder") {
                return okJson({ outputRoot: "/tmp/egomodelkit-results" });
            }

            if (url === "/api/runs") {
                return okJson(startRunResponse("run-backend-cancelled"));
            }

            if (url === "/api/runs/run-backend-cancelled/progress") {
                return okJson(
                    progressResponse({
                        runId: "run-backend-cancelled",
                        status: "cancelled",
                        errorMessage: "Run was cancelled.",
                    }),
                );
            }

            throw new Error(`Unexpected fetch call: ${url}`);
        });

        stubFetchWithModels(fetchMock);

        await navigateToOutputStep(user);

        await user.click(screen.getByRole("button", { name: "Choose Output Folder" }));
        await user.click(screen.getByRole("button", { name: "Continue" }));
        await user.click(screen.getByRole("button", { name: "Run Model" }));

        expect(
            await screen.findByRole("heading", { name: "EgoModelKit" }),
        ).toBeInTheDocument();

        expect(screen.getByRole("button", { name: "Start New Run" })).toBeInTheDocument();
    });

    it("does not call the backend when opening output folder without a run id", async () => {
        window.localStorage.setItem(
            APP_STATE_STORAGE_KEY,
            JSON.stringify({
                step: "output-preview",
                modelId: "",
                inputNames: [],
                ignoredInputNames: [],
                outputRoot: "/tmp/egomodelkit-results",
                reviewMode: "ready",
                runId: "",
                activeOperationId: "",
                progress: null,
                resultSummary: null,
                outputPreview: outputPreview("preview-only"),
            }),
        );

        const fetchMock = stubFetchWithModels();
        const user = userEvent.setup();

        render(<App />);

        expect(
            await screen.findByRole("heading", { name: "Output folder preview" }),
        ).toBeInTheDocument();

        await user.click(screen.getByRole("button", { name: "Open Output Folder" }));

        expect(
            fetchMock.mock.calls.some(
                ([input]) => String(input) === "/api/open-output-folder",
            ),
        ).toBe(false);
    });

    it("shows not available for restored results without input names", async () => {
        window.localStorage.setItem(
            APP_STATE_STORAGE_KEY,
            JSON.stringify({
                step: "results",
                modelId: "hand-object-contact",
                inputNames: [],
                ignoredInputNames: [],
                outputRoot: "/tmp/egomodelkit-results",
                reviewMode: "ready",
                runId: "run-empty-input",
                activeOperationId: "",
                progress: progressResponse({
                    runId: "run-empty-input",
                    status: "completed",
                    outputFolder: "/tmp/egomodelkit-results/run-empty-input",
                }),
                resultSummary: null,
                outputPreview: outputPreview("run-empty-input"),
            }),
        );

        render(<App />);

        expect(await screen.findByText("Completed successfully")).toBeInTheDocument();
        expect(screen.getByText("Not available")).toBeInTheDocument();
    });

    it("summarizes restored multiple input names on the results screen", async () => {
        window.localStorage.setItem(
            APP_STATE_STORAGE_KEY,
            JSON.stringify({
                step: "results",
                modelId: "hand-object-contact",
                inputNames: ["frame-1.jpg", "frame-2.jpg"],
                ignoredInputNames: [],
                outputRoot: "/tmp/egomodelkit-results",
                reviewMode: "ready",
                runId: "run-restored-results",
                activeOperationId: "",
                progress: progressResponse({
                    runId: "run-restored-results",
                    status: "completed",
                    outputFolder: "/tmp/egomodelkit-results/run-restored-results",
                }),
                resultSummary: null,
                outputPreview: outputPreview("run-restored-results"),
            }),
        );

        render(<App />);

        expect(await screen.findByText("2 files")).toBeInTheDocument();
        expect(screen.getByText("Completed successfully")).toBeInTheDocument();
    });

    it("uses the fallback client operation id when randomUUID is unavailable", async () => {
        const user = userEvent.setup();

        vi.stubGlobal("crypto", {});
        vi.spyOn(Date, "now").mockReturnValue(12345);
        vi.spyOn(Math, "random").mockReturnValue(0.5);

        const fetchMock = vi
            .fn()
            .mockResolvedValueOnce(okJson({ outputRoot: "/tmp/egomodelkit-results" }))
            .mockImplementationOnce(async (_input: RequestInfo | URL, init?: RequestInit) => {
                const formData = init?.body as FormData;

                expect(formData.get("operationId")).toBe("operation-12345-8");

                return okJson(dryRunResponse("dry-run-fallback-operation"));
            });

        stubFetchWithModels(fetchMock);

        await navigateToOutputStep(user);

        await user.click(screen.getByRole("button", { name: "Choose Output Folder" }));
        await user.click(screen.getByRole("button", { name: "Continue" }));
        await user.click(screen.getByRole("button", { name: "Dry Run" }));

        expect(
            await screen.findByRole("heading", {
                name: "Dry run completed successfully.",
            }),
        ).toBeInTheDocument();
    });

    it("ignores malformed persisted state", async () => {
        window.localStorage.setItem(APP_STATE_STORAGE_KEY, "not-json");

        render(<App />);

        expect(
            await screen.findByRole("heading", { name: "EgoModelKit" }),
        ).toBeInTheDocument();
    });

    it("normalizes invalid persisted state fields", async () => {
        window.localStorage.setItem(
            APP_STATE_STORAGE_KEY,
            JSON.stringify({
                step: "results",
                modelId: "hand-object-contact",
                inputNames: "frame.jpg",
                ignoredInputNames: "ignored.txt",
                outputRoot: 123,
                reviewMode: "not-a-review-mode",
                runId: 456,
                activeOperationId: 789,
                progress: { status: "completed" },
                resultSummary: { status: "completed" },
                outputPreview: { folderTree: "missing run id" },
            }),
        );

        render(<App />);

        expect(await screen.findByText("Completed successfully")).toBeInTheDocument();
        expect(screen.getAllByText("Not available")).toHaveLength(2);
    });

    it("ignores persisted state without a valid step", async () => {
        window.localStorage.setItem(
            APP_STATE_STORAGE_KEY,
            JSON.stringify({ step: "not-a-step" }),
        );

        render(<App />);

        expect(
            await screen.findByRole("heading", { name: "EgoModelKit" }),
        ).toBeInTheDocument();
    });

    it("ignores aborted dry-run requests after cancellation", async () => {
        const user = userEvent.setup();

        const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
            const url = String(input);

            if (url === "/api/select-output-folder") {
                return okJson({ outputRoot: "/tmp/egomodelkit-results" });
            }

            if (url === "/api/dry-run") {
                return new Promise((_resolve, reject) => {
                    init?.signal?.addEventListener("abort", () => {
                        reject(new DOMException("Aborted", "AbortError"));
                    });
                });
            }

            if (url === "/api/cancel-run") {
                return okJson({ cancelled: true, runId: null, operationId: "operation-dry-run" });
            }

            throw new Error(`Unexpected fetch call: ${url}`);
        });

        stubFetchWithModels(fetchMock);

        await navigateToOutputStep(user);

        await user.click(screen.getByRole("button", { name: "Choose Output Folder" }));
        await user.click(screen.getByRole("button", { name: "Continue" }));
        await user.click(screen.getByRole("button", { name: "Dry Run" }));

        expect(screen.getByRole("button", { name: "Cancel Run" })).toBeInTheDocument();

        await user.click(screen.getByRole("button", { name: "Cancel Run" }));

        expect(screen.getByText("Ready to start.")).toBeInTheDocument();
        expect(screen.queryByRole("alert")).not.toBeInTheDocument();
    });

    it("ignores aborted start-run requests after cancellation", async () => {
        const user = userEvent.setup();

        const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
            const url = String(input);

            if (url === "/api/select-output-folder") {
                return okJson({ outputRoot: "/tmp/egomodelkit-results" });
            }

            if (url === "/api/runs") {
                return new Promise((_resolve, reject) => {
                    init?.signal?.addEventListener("abort", () => {
                        reject(new DOMException("Aborted", "AbortError"));
                    });
                });
            }

            if (url === "/api/cancel-run") {
                return okJson({ cancelled: true, runId: null, operationId: "operation-run" });
            }

            throw new Error(`Unexpected fetch call: ${url}`);
        });

        stubFetchWithModels(fetchMock);

        await navigateToOutputStep(user);

        await user.click(screen.getByRole("button", { name: "Choose Output Folder" }));
        await user.click(screen.getByRole("button", { name: "Continue" }));
        await user.click(screen.getByRole("button", { name: "Run Model" }));

        expect(screen.getByRole("button", { name: "Cancel Run" })).toBeInTheDocument();

        await user.click(screen.getByRole("button", { name: "Cancel Run" }));

        expect(screen.getByText("Ready to start.")).toBeInTheDocument();
        expect(screen.queryByRole("alert")).not.toBeInTheDocument();
    });

    it("starts a hand-object model run and shows the running panel", async () => {
        const user = userEvent.setup();
        
        const fetchMock = vi
            .fn()
            .mockResolvedValueOnce({
                ok: true,
                status: 200,
                json: async () => ({ outputRoot: "/tmp/egomodelkit-results" }),
            })
            .mockResolvedValueOnce({
                ok: true,
                status: 200,
                json: async () => startRunResponse(),
            });

        const wrappedFetchMock = stubFetchWithModels(fetchMock);

        await navigateToOutputStep(user);
        await user.click(screen.getByRole("button", { name: "Choose Output Folder" }));
        await user.click(screen.getByRole("button", { name: "Continue" }));
        await user.click(screen.getByRole("button", { name: "Run Model" }));
    
        expect(wrappedFetchMock).toHaveBeenCalledWith(
            "/api/runs",
            expect.objectContaining({
                method: "POST",
                body: expect.any(FormData)
            }),
        );

        expect(screen.getByText("Running model...")).toBeInTheDocument();
        expect(screen.getByText("Run ID: run-1")).toBeInTheDocument();
        expect(screen.getByText("Overall progress estimate")).toBeInTheDocument();

        expect(
            screen.getByText("This may take several minutes. Please keep this window open.")
        ).toBeInTheDocument();


        expect(screen.getByRole("button", { name: "Back" })).toBeDisabled();
        expect(screen.getByRole("button", { name: "Cancel Run" })).toBeInTheDocument();
        expect(screen.queryByRole("button", { name: "Dry Run" })).not.toBeInTheDocument();
        expect(screen.queryByRole("button", { name: "Run Model" })).not.toBeInTheDocument();
    });

    it("starts an ADL model run without frontend-generated progress rows", async () => {
        const user = userEvent.setup();

        const fetchMock = vi
            .fn()
            .mockResolvedValueOnce({
                ok: true,
                status: 200,
                json: async () => ({ outputRoot: "/tmp/egomodelkit-results" }),
            })
            .mockResolvedValueOnce({
                ok: true,
                status: 200,
                json: async () => startRunResponse("adl-run-1"),
            });

        stubFetchWithModels(fetchMock);
        
        await navigateToOutputStep(user, {
            modelName: /Activity recognition \(ADL\)/,
            file: new File(["fake video"], "clip.mp4", { type: "video/mp4" }),
        });

        await user.click(screen.getByRole("button", { name: "Choose Output Folder" }));
        await user.click(screen.getByRole("button", { name: "Continue" }));
        await user.click(screen.getByRole("button", { name: "Run Model" }));

        expect(screen.getByText("Running model...")).toBeInTheDocument();
        expect(screen.getByText("Run ID: adl-run-1")).toBeInTheDocument();
        expect(screen.getByText("Overall progress estimate")).toBeInTheDocument();

        expect(
            screen.queryByText("Preparing video input..."),
        ).not.toBeInTheDocument();

        expect(
            screen.queryByText("Extracting frames..."),
        ).not.toBeInTheDocument();

        expect(
            screen.queryByText("Running Detic object detection..."),
        ).not.toBeInTheDocument();
    });

    it("shows an error when starting the model run fails", async () => {
        const user = userEvent.setup();

        const fetchMock = vi
            .fn()
            .mockResolvedValueOnce({
                ok: true,
                status: 200,
                json: async () => ({ outputRoot: "/tmp/egomodelkit-results" }),
            })
            .mockResolvedValueOnce({
                ok: false,
                status: 500,
                json: async () => ({}),
            });

        const wrappedFetchMock = stubFetchWithModels(fetchMock);
        
        await navigateToOutputStep(user);
        await user.click(screen.getByRole("button", { name: "Choose Output Folder" }));
        await user.click(screen.getByRole("button", { name: "Continue" }));
        await user.click(screen.getByRole("button", { name: "Run Model" }));

        expect(await screen.findByRole("alert")).toHaveTextContent(
            "Unable to start model run",
        );

        expect(screen.getByText("Ready to start.")).toBeInTheDocument();
    });

    it("cancels the backend run and resets state from Cancel Run", async () => {
        const user = userEvent.setup();

        const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
            const url = String(input);

            if (url === "/api/select-output-folder") {
                return okJson({ outputRoot: "/tmp/egomodelkit-results" });
            }

            if (url === "/api/runs") {
                return okJson(startRunResponse("run-cancel"));
            }

            if (url === "/api/runs/run-cancel/progress") {
                return okJson(progressResponse({ runId: "run-cancel", status: "running" }));
            }

            if (url === "/api/cancel-run") {
                return okJson({
                    cancelled: true,
                    runId: "run-cancel",
                    operationId: "operation-run-cancel",
                });
            }

            throw new Error(`Unexpected fetch call: ${url}`);
        });

        stubFetchWithModels(fetchMock);

        await navigateToOutputStep(user);

        await user.click(screen.getByRole("button", { name: "Choose Output Folder" }));
        await user.click(screen.getByRole("button", { name: "Continue" }));
        await user.click(screen.getByRole("button", { name: "Run Model" }));

        expect(await screen.findByText("Running model...")).toBeInTheDocument();

        await user.click(screen.getByRole("button", { name: "Cancel Run" }));

        expect(screen.getByText("Ready to start.")).toBeInTheDocument();
        expect(screen.getByRole("button", { name: "Dry Run" })).toBeInTheDocument();
        expect(screen.getByRole("button", { name: "Run Model" })).toBeInTheDocument();
        expect(screen.queryByRole("button", { name: "Cancel Run" })).not.toBeInTheDocument();

        expect(fetchMock).toHaveBeenCalledWith(
            "/api/cancel-run",
            expect.objectContaining({ method: "POST" }),
        );

        expect(screen.getByRole("heading", { name: "Review and run" })).toBeInTheDocument();
    });

    it(
        "warns and cancels backend operation when heading is clicked during a run", 
        async () => {
            const user = userEvent.setup();
            const confirmMock = vi.fn(() => true);

            vi.stubGlobal("confirm", confirmMock);

            const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
                const url = String(input);

                if (url === "/api/select-output-folder") {
                    return okJson({ outputRoot: "/tmp/egomodelkit-results" });
                }

                if (url === "/api/runs") {
                    return okJson(startRunResponse("run-heading-cancel"));
                }

                if (url === "/api/runs/run-heading-cancel/progress") {
                    return okJson(
                        progressResponse({
                            runId: "run-heading-cancel",
                            status: "running",
                        }),
                    );
                }

                if (url === "/api/cancel-run") {
                    return okJson({
                        cancelled: true,
                        runId: "run-heading-cancel",
                        operationId: "operation-heading-cancel",
                    });
                }

                throw new Error(`Unexpected fetch call: ${url}`);
            }
        );

        stubFetchWithModels(fetchMock);

        await navigateToOutputStep(user);

        await user.click(screen.getByRole("button", { name: "Choose Output Folder" }));
        await user.click(screen.getByRole("button", { name: "Continue" }));
        await user.click(screen.getByRole("button", { name: "Run Model" }));

        expect(await screen.findByText("Running model...")).toBeInTheDocument();

        expect(screen.getByRole("button", { name: "Cancel Run" })).toHaveClass(
            "bg-egm-danger",
        );

        await user.click(screen.getByRole("button", { name: "EgoModelKit" }));

        expect(confirmMock).toHaveBeenCalledWith(
            "A model operation is currently in progress. Leaving this page will cancel " +
            "the backend operation and progress will be lost. Continue?",
        );

        expect(fetchMock).toHaveBeenCalledWith(
            "/api/cancel-run",
            expect.objectContaining({ method: "POST" }),
        );

        expect(
            await screen.findByRole("heading", { name: "EgoModelKit" }),
        ).toBeInTheDocument();
    });

    it("polls run progress and shows backend progress events", async () => {
        const user = userEvent.setup();

        await startRunWithProgressResponses(user, [
            progressResponse({
                status: "running",
                events: [
                    progressEvent({
                        stage: "setup",
                        displayText: "Backend: preparing image input",
                        current: 1,
                        total: 4,
                    }),
                    progressEvent({
                        stage: "runtime",
                        displayText: "Backend: running model",
                        current: 2,
                        total: 4,
                    }),
                ],
            }),
        ]);

        expect(
            await screen.findByText("Backend: preparing image input")
        ).toBeInTheDocument();

        expect(screen.getByText("Backend: running model")).toBeInTheDocument();

        expect(
            screen.getByRole("log", { name: "Run progress log" }),
        ).not.toHaveClass("max-h-40", "overflow-y-auto");
        
        expect(screen.getByTestId("progress-bar-fill")).toHaveStyle({ width: "38%" });
    });

    it("returns from review screen to the selected output folder screen", async () => {
        const user = userEvent.setup();

        await navigateToReviewStep(user);
        await user.click(screen.getByRole("button", { name: "Back" }));

        expect(
            screen.getByRole("heading", { name: "Choose output folder" }),
        ).toBeInTheDocument();

        expect(screen.getByText("/tmp/egomodelkit-results")).toBeInTheDocument();
    });

    it(
        "shows a small progress estimate when progress events do not include totals",
        async () => {
            const user = userEvent.setup();

            await startRunWithProgressResponses(user, [
                progressResponse({
                    status: "running",
                    events: [
                        progressEvent({
                            stage: "running",
                            displayText: "Backend: running without numeric progress",
                        }),
                    ],
                }),
            ]);

            expect(
                await screen.findByText("Backend: running without numeric progress"),
            ).toBeInTheDocument();

            expect(screen.getByTestId("progress-bar-fill")).toHaveStyle({ width: "100%" });
        }
    );

    it("shows zero progress when no progress events are available", async () => {
        const user = userEvent.setup();

        await startRunWithProgressResponses(user, [
            progressResponse({
                status: "running",
                events: [],
            }),
        ]);

        expect(await screen.findByText("Running model...")).toBeInTheDocument();

        expect(screen.getByTestId("progress-bar-fill")).toHaveStyle({ width: "0%" });    
    });

    it("shows zero progress when backend progress omits events", async () => {
        const user = userEvent.setup();

        await startRunWithProgressResponses(user, [
            {
                runId: "run-1",
                status: "running",
                errorMessage: null,
                outputFolder: "/tmp/egomodelkit-results/run-1",
                outputPreview: outputPreview("run-1"),
            },
        ]);

        expect(await screen.findByText("Running model...")).toBeInTheDocument();
        expect(screen.getByText("Run ID: run-1")).toBeInTheDocument();
        expect(screen.getByTestId("progress-bar-fill")).toHaveStyle({ width: "0%" });
    });

    it("caps progress at 100 percent when backend progress exceeds the total", async () => {
        const user = userEvent.setup();

        await startRunWithProgressResponses(user, [
            progressResponse({
                status: "running",
                events: [
                    progressEvent({
                        stage: "runtime",
                        displayText: "Backend: progress exceeded total",
                        current: 6,
                        total: 4,
                    }),
                ],
            }),
        ]);

        expect(
            await screen.findByText("Backend: progress exceeded total"),
        ).toBeInTheDocument();

        expect(screen.getByTestId("progress-bar-fill")).toHaveStyle({ width: "100%" });
    });

    it("keeps progress at zero when the backend progress is negative", async () => {
        const user = userEvent.setup();

        await startRunWithProgressResponses(user, [
            progressResponse({
                status: "running",
                events: [
                    progressEvent({
                        stage: "runtime",
                        displayText: "Backend: negative progress",
                        current: -1,
                        total: 4,
                    }),
                ],
            }),
        ]);

        expect(
            await screen.findByText("Backend: negative progress"),
        ).toBeInTheDocument();

        expect(screen.getByTestId("progress-bar-fill")).toHaveStyle({ width: "0%" });
    });

    it("moves to the completed results screen when progress completes", async () => {
        const user = userEvent.setup();

        const fetchMock = await startRunWithProgressResponses(user, [
            progressResponse({
                status: "completed",
                events: [
                    progressEvent({
                        stage: "finalize",
                        displayText: "Saving detection outputs...",
                        current: 4,
                        total: 4,
                    }),
                ],
            }),
        ]);

        expect(
            await screen.findByRole("heading", { name: "Run completed" }),
        ).toBeInTheDocument();
    
        expect(
            screen.getByText("Your results were saved successfully.")
        ).toBeInTheDocument();

        expect(screen.getByText("Hand-object contact")).toBeInTheDocument();
        expect(screen.getByText("frame.jpg")).toBeInTheDocument();
        expect(screen.getByText("/tmp/egomodelkit-results/run-1")).toBeInTheDocument();
        expect(screen.getByText("Completed")).toBeInTheDocument();

        expect(fetchMock).toHaveBeenCalledWith(
            "/api/runs/run-1/progress",
            undefined
        );
    });

    it("shows a graceful failed results screen when progress reports failure", async () => {
        const user = userEvent.setup();

        await startRunWithProgressResponses(user, [
            progressResponse({
                status: "failed",
                errorMessage:  "Docker or an NVIDIA GPU was not available on this machine.",
                events: [
                    progressEvent({
                        stage: "finalize",
                        displayText: "Saving detection outputs...",
                        current: 4,
                        total: 4,
                    }),
                ],
            }),
        ]);

        expect(
            await screen.findByRole("heading", { name: "Needs attention" }),
        ).toBeInTheDocument();

        expect(
            screen.getByText("EgoModelKit could not complete the run."),
        ).toBeInTheDocument();

        expect(
            screen.getByText("Docker or an NVIDIA GPU was not available on this machine."),
        ).toBeInTheDocument();

        expect(screen.getByText("Failed")).toBeInTheDocument();
    });

    it("keeps the output-preview action disabled when progress reports failure", async () => {
        const user = userEvent.setup();

        await startRunWithProgressResponses(user, [
            progressResponse({ status: "failed" }),
        ]);

        expect(
            await screen.findByRole("heading", { name: "Needs attention" }),
        ).toBeInTheDocument();

        expect(
            screen.getByRole("button", { name: "View Output Preview" }),
        ).toBeDisabled();
    });

    it(
        "keeps running screen visible and shows an alert when progress polling fails",
        async () => {
            const user = userEvent.setup();

            const fetchMock = vi
                .fn()
                .mockResolvedValueOnce({
                    ok: true,
                    status: 200,
                    json: async () => ({ outputRoot: "/tmp/egomodelkit-results" }),
                })
                .mockResolvedValueOnce({
                    ok: true,
                    status: 200,
                    json: async () => startRunResponse(),
                })
                .mockResolvedValueOnce({
                    ok: false,
                    status: 500,
                    json: async () => ({}),
                });
            
            stubFetchWithModels(fetchMock);

            await navigateToOutputStep(user);
            await user.click(screen.getByRole("button", { name: "Choose Output Folder" }));

            expect(
                await screen.findByText("/tmp/egomodelkit-results"),
            ).toBeInTheDocument();

            await user.click(screen.getByRole("button", { name: "Continue" }));
            await user.click(screen.getByRole("button", { name: "Run Model" }));

            expect(await screen.findByRole("alert")).toHaveTextContent(
                "Unable to refresh run progress.",
            );

            expect(screen.getByText("Running model...")).toBeInTheDocument();
        }
    );

    it("encodes run IDs before polling progress", async () => {
        const user = userEvent.setup();

        const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
            const url = String(input);

            if (url === "/api/select-output-folder") {
                return {
                    ok: true,
                    status: 200,
                    json: async () => ({ outputRoot: "/tmp/egomodelkit-results" }),
                };
            }

            if (url === "/api/runs") {
                return {
                    ok: true,
                    status: 200,
                    json: async () => startRunResponse("run 1"),
                };
            }

            if (url === "/api/runs/run%201/progress") {
                return {
                    ok: true,
                    status: 200,
                    json: async () => 
                        progressResponse({
                            runId: "run 1",
                            status: "completed",
                            outputFolder: "/tmp/egomodelkit-results/run 1"
                        }),
                };
            }

            throw new Error(`Unexpected fetch call: ${url}`);
        });

        const wrappedFetchMock = stubFetchWithModels(fetchMock);

        await navigateToOutputStep(user);
        await user.click(screen.getByRole("button", { name: "Choose Output Folder" }));

        await waitFor(() => {
            expect(screen.getByRole("button", { name: "Continue" })).toBeEnabled();
        });

        await user.click(screen.getByRole("button", { name: "Continue" }));
        await user.click(screen.getByRole("button", { name: "Run Model" }));

        expect(
            await screen.findByRole("heading", { name: "Run completed" })
        ).toBeInTheDocument();

        expect(wrappedFetchMock).toHaveBeenCalledWith(
            "/api/runs/run%201/progress",
            undefined,
        );
    });

    it(
        "ignores a stale progress polling error after leaving the running screen", 
        async () => {
            const user = userEvent.setup();

            vi.stubGlobal("confirm", vi.fn(() => true));    

            let rejectProgress!: (reason?: unknown) => void;

            const pendingProgressResponse = new Promise((_resolve, reject) => {
                rejectProgress = reject;
            });

            const fetchMock = vi.fn(
                (input: RequestInfo | URL, _init?: RequestInit) => {
                    const url = String(input);

                    if (url === "/api/select-output-folder") {
                        return Promise.resolve({
                            ok: true,
                            status: 200,
                            json: async () => ({ outputRoot: "/tmp/egomodelkit-results" }),
                        });
                    }

                    if (url === "/api/runs") {
                        return Promise.resolve({
                            ok: true,
                            status: 200,
                            json: async () => startRunResponse(),
                        });
                    }

                    if (url === "/api/runs/run-1/progress") {
                        return pendingProgressResponse;
                    }

                    if (url === "/api/cancel-run") {
                        return Promise.resolve(
                            okJson({
                                cancelled: true,
                                runId: "run-1",
                                operationId: "operation-stale",
                            }),
                        );
                    }

                    return Promise.reject(new Error(`Unexpected fetch call: ${url}`));
                }
            );

            stubFetchWithModels(fetchMock);

            await navigateToOutputStep(user);
            await user.click(screen.getByRole("button", { name: "Choose Output Folder" }));

            expect(
                await screen.findByText("/tmp/egomodelkit-results"),
            ).toBeInTheDocument();

            await user.click(screen.getByRole("button", { name: "Continue" }));
            await user.click(screen.getByRole("button", { name: "Run Model" }));

            expect(await screen.findByText("Running model...")).toBeInTheDocument();

            await user.click(screen.getByRole("button", { name: "EgoModelKit" }));

            expect(
                await screen.findByRole("heading", { name: "EgoModelKit" }),
            ).toBeInTheDocument();

            await act(async () => {
                rejectProgress(new Error("stale progress failure"));

                await Promise.resolve();
                await Promise.resolve();
            });

            expect(screen.queryByRole("alert")).not.toBeInTheDocument();
            expect(screen.getByRole("heading", { name: "EgoModelKit" }),).toBeInTheDocument();
        }
    );
    
    it("ignores a stale progress response after leaving the running screen", async () => {
        const user = userEvent.setup();

        vi.stubGlobal("confirm", vi.fn(() => true));

        let resolveProgress!: (value: unknown) => void;

        const pendingProgressResponse = new Promise((resolve) => {
            resolveProgress = resolve;
        });

        const fetchMock = vi.fn(
            (input: RequestInfo | URL, _init?: RequestInit) => {
                const url = String(input);

                if (url === "/api/select-output-folder") {
                    return Promise.resolve({
                        ok: true,
                        status: 200,
                        json: async () => ({ outputRoot: "/tmp/egomodelkit-results" }),
                    });
                }
                
                if (url === "/api/runs") {
                    return Promise.resolve({
                        ok: true,
                        status: 200,
                        json: async () => startRunResponse(),
                    });
                }

                if (url === "/api/runs/run-1/progress") {
                    return pendingProgressResponse;
                }

                if (url === "/api/cancel-run") {
                    return Promise.resolve(
                        okJson({
                            cancelled: true,
                            runId: "run-1",
                            operationId: "operation-stale",
                        }),
                    );
                }

                return Promise.reject(new Error(`Unexpected fetch call: ${url}`));
            }
        );
        
        stubFetchWithModels(fetchMock);

        await navigateToOutputStep(user);
        await user.click(screen.getByRole("button", { name: "Choose Output Folder" }));

        expect(await screen.findByText("/tmp/egomodelkit-results")).toBeInTheDocument();

        await user.click(screen.getByRole("button", { name:  "Continue" }));
        await user.click(screen.getByRole("button", { name: "Run Model" }));

        expect(await screen.findByText("Running model...")).toBeInTheDocument();

        await user.click(screen.getByRole("button", { name: "EgoModelKit" }));

        expect(
            await screen.findByRole("heading", { name: "EgoModelKit" }),
        ).toBeInTheDocument();

        await act(async () => {
            resolveProgress({
                ok: true,
                status: 200,
                json: async () => progressResponse({ status: "completed" }),
            });

            await Promise.resolve();
            await Promise.resolve();
        });

        expect(
            screen.queryByRole("heading", { name: "Run completed" }),
        ).not.toBeInTheDocument();
    });

    it("starts a new run from the results screen", async () => {
        const user = userEvent.setup();

        await startRunWithProgressResponses(user, [
            progressResponse({ status: "completed" }),
        ]);

        expect(
            await screen.findByRole("heading", { name: "Run completed" }),
        ).toBeInTheDocument();

        await user.click(screen.getByRole("button", { name: "Start New Run" }));

        expect(screen.getByRole("heading", { name: "Select a model" })).toBeInTheDocument();
        expect(screen.getByRole("button", { name: "Continue" })).toBeDisabled();
    });

    it("opens the completed run output folder", async () => {
        const user = userEvent.setup();

        const fetchMock = await renderCompletedResultsWithFetchMock(user, {
            ok: true,
            status: 200,
            json: async () => ({
                opened: true,
                runId: "run-1",
                outputFolder: "/tmp/egomodelkit-results/run-1",
            }),
        });

        await user.click(screen.getByRole("button", { name: "Open Output Folder" }));

        await waitFor(() => {
            expect(fetchMock).toHaveBeenCalledWith(
                "/api/open-output-folder",
                expect.objectContaining({
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json",
                    },
                    body: JSON.stringify({ runId: "run-1" }),
                }),
            );
        });
    });

    it("shows an alert when opening the output folder is unavailable", async () => {
        const user = userEvent.setup();
        
        await renderCompletedResultsWithFetchMock(user, {
            ok: false,
            status: 405,
            json: async () => ({}),
        });

        await user.click(screen.getByRole("button", { name: "Open Output Folder" }));

        expect(await screen.findByRole("alert")).toHaveTextContent(
            "Opening output folders is not available in this environment."
        );
    });

    it("shows an alert when opening the output folder fails", async () => {
        const user = userEvent.setup();
        
        await renderCompletedResultsWithFetchMock(user, {
            ok: false,
            status: 500,
            json: async () => ({}),
        });
        
        await user.click(screen.getByRole("button", { name: "Open Output Folder" }));

        expect(await screen.findByRole("alert")).toHaveTextContent(
            "Unable to open output folder."
        );
    });

    it("opens the output-preview screen from completed results", async () => {
        const user = userEvent.setup();

        const fetchMock = await startRunWithProgressResponses(user, [
            progressResponse({ status: "completed" }),
        ]);

        expect(
            await screen.findByRole("heading", { name: "Run completed" }),
        ).toBeInTheDocument();

        const previewButton = screen.getByRole("button", {
            name: "View Output Preview",
        });

        expect(previewButton).toBeEnabled();

        await user.click(previewButton);

        expect(
            screen.getByRole("heading", { name: "Output folder preview" }),
        ).toBeInTheDocument();

        expect(
            screen.getByText("Review what EgoModelKit saved for this run."),
        ).toBeInTheDocument();

        expect(screen.getByText("visual_outputs/")).toBeInTheDocument();
        expect(screen.getByText("run_summary.json")).toBeInTheDocument();

        expect(
            screen.getByText("Frame-level metrics are not generated for a single image."),
        ).toBeInTheDocument();

        const contentsButton = screen.getByRole("button", {
            name: "What the output folder contains",
        });

        expect(contentsButton).toHaveAttribute("aria-expanded", "false");

        expect(
            screen.queryByText("Explanation of the output folder contents."),
        ).not.toBeInTheDocument();

        await user.click(contentsButton);

        expect(contentsButton).toHaveAttribute("aria-expanded", "true");

        expect(screen.getAllByText("README.txt")).toHaveLength(2);

        expect(
            screen.getByText("Explanation of the output folder contents."),
        ).toBeInTheDocument();

        expect(
            screen.getByText("Progress events written during the run."),
        ).toBeInTheDocument();

        await user.click(screen.getByRole("button", { name: "Back to Results" }));

        expect(screen.getByRole("heading", { name: "Run completed" })).toBeInTheDocument();

        expect(fetchMock).not.toHaveBeenCalledWith(
            "/api/output-preview",
            expect.anything(),
        );
    });

    it(
        "opens the output-preview screen from the start-run preview when progress omits it",
        async () => {
            const user = userEvent.setup();

            await renderCompletedResultsWithCustomProgress(user, {
                runId: "run-1",
                status: "completed",
                errorMessage: null,
                outputFolder: "/tmp/egomodelkit-results/run-1",
                events: [
                    progressEvent({
                        stage: "finalize",
                        displayText: "Saving detection outputs...",
                        current: 4,
                        total: 4,
                    }),
                ],
            });

            const previewButton = screen.getByRole("button", {
                name: "View Output Preview",
            });

            expect(previewButton).toBeEnabled();

            await user.click(previewButton);

            expect(
                screen.getByRole("heading", { name: "Output folder preview" }),
            ).toBeInTheDocument();

            expect(screen.getByText("visual_outputs/")).toHaveClass("text-egm-green");
            expect(screen.getByText("run_summary.json")).toHaveClass("text-black");
        },
    );

    it(
        "skips backend cancellation when restored running state has no operation or run id", 
        async () => {
            window.localStorage.setItem(
                APP_STATE_STORAGE_KEY,
                JSON.stringify({
                    step: "review",
                    modelId: "hand-object-contact",
                    inputNames: ["frame.jpg"],
                    ignoredInputNames: [],
                    outputRoot: "/tmp/egomodelkit-results",
                    reviewMode: "running",
                    runId: "",
                    activeOperationId: "",
                    progress: null,
                    resultSummary: null,
                    outputPreview: null,
                }),
            );

            const fetchMock = stubFetchWithModels();
            const user = userEvent.setup();

            render(<App />);

            expect(await screen.findByText("Running model...")).toBeInTheDocument();

            await user.click(screen.getByRole("button", { name: "Cancel Run" }));

            expect(screen.getByText("Ready to start.")).toBeInTheDocument();

            expect(
                fetchMock.mock.calls.some(([input]) => String(input) === "/api/cancel-run"),
            ).toBe(false);
        }
    );

    it(
        "uses the start-run summary output folder when progress has no output folder",
        async () => {
            const user = userEvent.setup();

            await renderCompletedResultsWithCustomProgress(user, {
                ...progressResponse({ status: "completed" }),
                outputFolder: null,
            });

            expect(screen.getByText("/tmp/egomodelkit-results")).toBeInTheDocument();
        }
    );

    it(
        "shows not available when no output folder is available in run or progress responses",
        async () => {
            const user = userEvent.setup();

            const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
                const url = String(input);

                if (url === "/api/select-output-folder") {
                    return {
                        ok: true,
                        status: 200,
                        json: async () => ({ outputRoot: "/tmp/egomodelkit-results" }),
                    };
                }

                if (url === "/api/runs") {
                    const response = startRunResponse();

                    return {
                        ok: true,
                        status: 200,
                        json: async () => ({
                            ...response,
                            summary: {
                                ...response.summary,
                                outputFolder: undefined,
                            },
                        }),
                    };
                }

                if (url === "/api/runs/run-1/progress") {
                    return {
                        ok: true,
                        status: 200,
                        json: async () => ({
                            ...progressResponse({ status: "completed" }),
                            outputFolder: undefined,
                        }),
                    };
                }

                throw new Error(`Unexpected fetch call: ${url}`);
            });

            stubFetchWithModels(fetchMock);

            await navigateToOutputStep(user);
            await user.click(screen.getByRole("button", { name: "Choose Output Folder" }));

            await waitFor(() => {
                expect(screen.getByRole("button", { name: "Continue" })).toBeEnabled();
            });

            await user.click(screen.getByRole("button", { name: "Continue" }));
            await user.click(screen.getByRole("button", { name: "Run Model" }));

            expect(
                await screen.findByRole("heading", { name: "Run completed" }),
            ).toBeInTheDocument();

            expect(screen.getByText("Not available")).toBeInTheDocument();
        }
    );

    it("counts Docker image builds as independent progress stages", async () => {
        const user = userEvent.setup();

        await startRunWithProgressResponses(user, [
            progressResponse({
                status: "running",
                runtimeStatus: {
                    modelName: "Detic",
                    currentStep: 2,
                    totalSteps: 4,
                },
                runtimeBuildStages: [
                    {
                        stageId: "docker:Detic",
                        modelName: "Detic",
                        current: 2,
                        total: 4,
                    },
                ],
                events: [
                    progressEvent({
                        stage: "run_detic",
                        displayText: "Running object detection model: waiting",
                    }),
                ],
            }),
        ]);

        expect(
            await screen.findByRole("status"),
        ).toHaveTextContent("Building Docker image for Detic [2 / 4]");

        expect(
            screen.getByRole("log", { name: "Run progress log" }),
        ).not.toHaveTextContent("Building Docker image");

        expect(screen.getByTestId("progress-bar-fill")).toHaveStyle({
            width: "25%",
        });
    });

    it("treats Docker stages with invalid totals as zero progress", async () => {
        const user = userEvent.setup();

        await startRunWithProgressResponses(user, [
            progressResponse({
                status: "running",
                runtimeBuildStages: [
                    {
                        stageId: "docker:missing-current",
                        modelName: "Missing current",
                        current: null,
                        total: 4,
                    },
                    {
                        stageId: "docker:missing-total",
                        modelName: "Missing total",
                        current: 2,
                        total: null,
                    },
                    {
                        stageId: "docker:zero-total",
                        modelName: "Zero total",
                        current: 2,
                        total: 0,
                    },
                ],
                events: [],
            }),
        ]);

        expect(await screen.findByText("Running model...")).toBeInTheDocument();

        expect(screen.getByTestId("progress-bar-fill")).toHaveStyle({
            width: "0%",
        });
    });

    it(
        "shows Docker build status without step counts when runtime status omits totals", 
        async () => {
            const user = userEvent.setup();

            await startRunWithProgressResponses(user, [
                progressResponse({
                    status: "running",
                    runtimeStatus: {
                        modelName: "EgoVizML",
                        currentStep: null,
                        totalSteps: null,
                    },
                    runtimeBuildStages: [],
                    events: [],
                }),
            ]);

            const status = await screen.findByRole("status");

            expect(status).toHaveTextContent("Building Docker image for EgoVizML");
            expect(status).not.toHaveTextContent("[");
            expect(status).not.toHaveTextContent("]");
        }
    );

    it("restores a running review screen after refresh", async () => {
        const user = userEvent.setup();

        const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
            const url = String(input);

            if (url === "/api/select-output-folder") {
                return okJson({ outputRoot: "/tmp/egomodelkit-results" });
            }

            if (url === "/api/runs") {
                return okJson(startRunResponse("run-refresh"));
            }

            if (url === "/api/runs/run-refresh/progress") {
                return okJson(
                    progressResponse({
                        runId: "run-refresh",
                        status: "running",
                        events: [
                            progressEvent({
                                stage: "setup",
                                displayText: "Preparing image input...",
                            }),
                        ],
                    }),
                );
            }

            throw new Error(`Unexpected fetch call: ${url}`);
        });

        stubFetchWithModels(fetchMock);

        const { unmount } = render(<App />);

        await user.click(screen.getByRole("button", { name: "Start New Run" }));
        await user.click(await screen.findByRole("button", { name: /Hand-object contact/ }));
        await user.click(screen.getByRole("button", { name: "Continue" }));
        
        await user.upload(
            screen.getByLabelText("Choose input files"),
            new File(["fake image"], "frame.jpg", { type: "image/jpeg" }),
        );
        
        await user.click(screen.getByRole("button", { name: "Continue" }));
        await user.click(screen.getByRole("button", { name: "Choose Output Folder" }));
        await user.click(screen.getByRole("button", { name: "Continue" }));
        await user.click(screen.getByRole("button", { name: "Run Model" }));

        expect(await screen.findByText("Running model...")).toBeInTheDocument();

        await waitFor(() => {
            expect(
                JSON.parse(
                    window.localStorage.getItem(APP_STATE_STORAGE_KEY) ?? "{}",
                ),
            ).toEqual(
                expect.objectContaining({
                    step: "review",
                    reviewMode: "running",
                    runId: "run-refresh",
                    inputNames: ["frame.jpg"],
                }),
            );
        });

        unmount();

        render(<App />);

        expect(await screen.findByText("Running model...")).toBeInTheDocument();
        expect(screen.getByText("Run ID: run-refresh")).toBeInTheDocument();
        expect(screen.getByText("frame.jpg")).toBeInTheDocument();

        await waitFor(() => {
            expect(
                fetchMock.mock.calls.some(
                    ([input]) => String(input) === "/api/runs/run-refresh/progress",
                ),
            ).toBe(true);
        });
    });
});

it("shows ADL dominant-hand settings only when ADL is selected", async () => {
    const user = userEvent.setup();

    render(<App />);

    await user.click(screen.getByRole("button", { name: "Start New Run" }));

    expect(screen.queryByText("ADL settings")).not.toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /Hand-object contact/ }));

    expect(screen.queryByText("ADL settings")).not.toBeInTheDocument();

    await user.click(screen.getByRole("button", {
        name: /Activity recognition \(ADL\)/,
    }));

    expect(screen.getByText("ADL settings")).toBeInTheDocument();
    expect(screen.getByLabelText("Right")).toBeChecked();
    expect(screen.getByLabelText("Left")).not.toBeChecked();

    await user.click(screen.getByLabelText("Left"));

    expect(screen.getByLabelText("Left")).toBeChecked();
});

it("sends selected ADL dominant hand with dry-run requests", async () => {
    const user = userEvent.setup();

    const fetchMock = vi
        .fn()
        .mockResolvedValueOnce(okJson({ outputRoot: "/tmp/egomodelkit-results" }))
        .mockResolvedValueOnce(okJson(dryRunResponse()));

    const wrappedFetchMock = stubFetchWithModels(fetchMock);

    await navigateToOutputStep(user, {
        modelName: /Activity recognition \(ADL\)/,
        file: new File(["fake video"], "clip.mp4", { type: "video/mp4" }),
    });

    await user.click(screen.getByRole("button", { name: "Back" }));
    await user.click(screen.getByRole("button", { name: "Back" }));
    await user.click(screen.getByLabelText("Left"));
    await user.click(screen.getByRole("button", { name: "Continue" }));
    await user.click(screen.getByRole("button", { name: "Continue" }));

    await user.click(screen.getByRole("button", { name: "Choose Output Folder" }));
    await user.click(screen.getByRole("button", { name: "Continue" }));

    expect(screen.getByText("Dominant hand:")).toBeInTheDocument();
    expect(screen.getByText("Left")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Dry Run" }));

    const dryRunCall = wrappedFetchMock.mock.calls.find(
        ([input]) => String(input) === "/api/dry-run",
    );

    const formData = dryRunCall?.[1]?.body as FormData;

    expect(formData.get("modelId")).toBe("adl-recognition");
    expect(formData.get("dominantHand")).toBe("left");
});

it("shows the ADL multi-file session note on input and review screens", async () => {
    const user = userEvent.setup();

    render(<App />);

    await user.click(screen.getByRole("button", { name: "Start New Run" }));

    await user.click(await screen.findByRole("button", {
        name: /Activity recognition \(ADL\)/,
    }));

    await user.click(screen.getByRole("button", { name: "Continue" }));

    expect(
        screen.getByText(/multiple selected ADL videos are grouped as one session/i),
    ).toBeInTheDocument();

    await user.upload(screen.getByLabelText("Choose input files"), [
        new File(["fake video 1"], "clip1.mp4", { type: "video/mp4" }),
        new File(["fake video 2"], "clip2.mp4", { type: "video/mp4" }),
    ]);

    await user.click(screen.getByRole("button", { name: "Continue" }));

    const fetchMock = vi
        .fn()
        .mockResolvedValueOnce(okJson({ outputRoot: "/tmp/egomodelkit-results" }));

    stubFetchWithModels(fetchMock);

    await user.click(screen.getByRole("button", { name: "Choose Output Folder" }));
    await user.click(screen.getByRole("button", { name: "Continue" }));
});

it("keeps Run Model preflight failures on Review and restores run buttons", async () => {
    const user = userEvent.setup();

    const fetchMock = vi
        .fn()
        .mockResolvedValueOnce(okJson({ outputRoot: "/tmp/egomodelkit-results" }))
        .mockResolvedValueOnce({
            ok: false,
            status: 400,
            json: async () => ({
                detail:
                    "ADL recognition requires a Linux host with an NVIDIA GPU.",
            }),
        });

    const wrappedFetchMock = stubFetchWithModels(fetchMock);

    await navigateToOutputStep(user, {
        modelName: /Activity recognition \(ADL\)/,
        file: new File(["fake video"], "clip.mp4", { type: "video/mp4" }),
    });

    await user.click(screen.getByRole("button", { name: "Choose Output Folder" }));
    await user.click(screen.getByRole("button", { name: "Continue" }));
    await user.click(screen.getByRole("button", { name: "Run Model" }));

    expect(await screen.findByRole("alert")).toHaveTextContent(
        "ADL recognition requires a Linux host with an NVIDIA GPU.",
    );

    expect(screen.getByRole("heading", { name: "Review and run" })).toBeInTheDocument();
    expect(screen.queryByRole("heading", { name: "Needs attention" })).not.toBeInTheDocument();
    expect(screen.getByText("Ready to start.")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Dry Run" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Run Model" })).toBeInTheDocument();

    expect(
        wrappedFetchMock.mock.calls.some(([input]) =>
            String(input).includes("/api/runs/run-1/progress"),
        ),
    ).toBe(false);
});

it("does not start progress polling after Run Model preflight failure", async () => {
    const user = userEvent.setup();

    const fetchMock = vi
        .fn()
        .mockResolvedValueOnce(okJson({ outputRoot: "/tmp/egomodelkit-results" }))
        .mockResolvedValueOnce({
            ok: false,
            status: 400,
            json: async () => ({
                detail: "EgoModelKit model runs require a Linux host with an NVIDIA GPU.",
            }),
        });

    const wrappedFetchMock = stubFetchWithModels(fetchMock);

    await navigateToOutputStep(user);

    await user.click(screen.getByRole("button", { name: "Choose Output Folder" }));
    await user.click(screen.getByRole("button", { name: "Continue" }));
    await user.click(screen.getByRole("button", { name: "Run Model" }));

    await screen.findByRole("alert");

    expect(
        wrappedFetchMock.mock.calls.some(([input]) =>
            String(input).includes("/api/runs/") &&
            String(input).includes("/progress"),
        ),
    ).toBe(false);
});

it("shows the selected ADL dominant hand on completed results", async () => {
    const user = userEvent.setup();

    const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
        const url = String(input);

        if (url === "/api/select-output-folder") {
            return okJson({
                outputRoot: "/tmp/egomodelkit-results",
            });
        }

        if (url === "/api/runs") {
            return okJson({
                ...startRunResponse("adl-run-1"),
                summary: {
                    modelId: "adl-recognition",
                    model: "Activity recognition (ADL)",
                    input: "clip.mp4",
                    outputFolder: "/tmp/egomodelkit-results/adl-run-1",
                    status: "running",
                },
            });
        }

        if (url === "/api/runs/adl-run-1/progress") {
            return okJson(
                progressResponse({
                    runId: "adl-run-1",
                    status: "completed",
                    outputFolder: "/tmp/egomodelkit-results/adl-run-1",
                }),
            );
        }

        throw new Error(`Unexpected fetch call: ${url}`);
    });

    stubFetchWithModels(fetchMock);

    await navigateToOutputStep(user, {
        modelName: /Activity recognition \(ADL\)/,
        file: new File(
            ["fake video"],
            "clip.mp4",
            {
                type: "video/mp4",
            },
        ),
    });

    await user.click(screen.getByRole("button", {name: "Choose Output Folder"}));
    await user.click(screen.getByRole("button", {name: "Continue"}));
    await user.click(screen.getByRole("button", {name: "Run Model"}));

    expect(await screen.findByRole("heading", {name: "Run completed"})).toBeInTheDocument();
    expect(screen.getByText("Dominant hand:")).toBeInTheDocument();
    expect(screen.getByText("Right")).toBeInTheDocument();
});
