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
                acceptedInputLabel: "an image or folder of images",
                outputLabel: "detection visualizations and structured results",
            },
            {
                id: "adl-recognition",
                name: "Activity recognition (ADL)",
                description:
                    "Processes egocentric video clips for " +
                    "activity of daily living (ADL) recognition.",
                acceptedInputLabel: "a video or folder of videos",
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

    await user.click(screen.getByRole("button", { name: "Start New Run" }));
    await user.click(await screen.findByRole("button", { name: modelName }));
    await user.click(screen.getByRole("button", { name: "Continue" }));

    await user.upload(screen.getByLabelText("Choose input files"), file);
    await user.click(screen.getByRole("button", { name: "Continue" }));
}

function outputPreview(runId: string) {
    return {
        runId,
        scenario: "single_image",
        folderTree: "results/",
        note: "Preview only.",
        files: [],
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
} : {
    runId?: string;
    status?: string;
    errorMessage?: string | null;
    outputFolder?: string;
    events?: object[];
}) {
    return {
        runId,
        status,
        errorMessage,
        outputFolder,
        events,
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
            screen.getByRole("button", { name: "View Previous Output Folder" }),
        ).toBeInTheDocument()

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
            screen.getByText("Select an image or folder of images"),
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
            screen.getByText("Select a video or folder of videos"),
        ).toBeInTheDocument();
        
        expect(
            screen.getByText("Select a video or folder of videos"),
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

        expect(screen.getByText("Selected: frame.jpg")).toBeInTheDocument();
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
        expect(screen.getByRole("button", { name: "Continue" })).toBeEnabled();
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

        expect(screen.getByText("Selected: dropped-frame.png")).toBeInTheDocument();
        expect(screen.getByRole("button", { name: "Continue" })).toBeEnabled();
    });

    it("opens the native file picker from the visible choose-input button", async () => {
        const user = userEvent.setup();
        const inputClickSpy = vi.spyOn(HTMLInputElement.prototype, "click");

        render(<App />);

        await user.click(screen.getByRole("button", { name: "Start New Run" }));
        await user.click(screen.getByRole("button", { name: /Hand-object contact/ }));
        await user.click(screen.getByRole("button", { name: "Continue" }));

        await user.click(screen.getByRole("button", { name: "Choose input" }));

        expect(inputClickSpy).toHaveBeenCalledOnce();

        inputClickSpy.mockRestore();
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
            screen.getByText("Select an image or folder of images"),
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

        expect(
            screen.getByRole("button", { name: "Privacy-safe outputs" }),
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

        expect(screen.getByText("Selected: frame.jpg")).toBeInTheDocument();
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

        expect(screen.getByText("Selected: frame.jpg")).toBeInTheDocument();

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

    it("toggles the privacy-safe outputs details", async () => {
        const user = userEvent.setup();

        await navigateToOutputStep(user);

        const privacyButton = screen.getByRole("button", { name: "Privacy-safe outputs" });

        expect(privacyButton).toHaveAttribute("aria-expanded", "false");

        await user.click(privacyButton);

        expect(privacyButton).toHaveAttribute("aria-expanded", "true");
        expect(screen.getByText("Run IDs are neutral names.")).toBeInTheDocument();
        
        expect(
            screen.getByText("Logs avoid unnecessary personal details."),
        ).toBeInTheDocument();

        expect(
            screen.getByText("Temporary files can be cleaned up after processing."),
        ).toBeInTheDocument();

        await user.click(privacyButton);

        expect(privacyButton).toHaveAttribute("aria-expanded", "false");
        expect(screen.queryByText("Run IDs are neutral names.")).not.toBeInTheDocument();
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

        const wrappedFetchMock = stubFetchWithModels(fetchMock);

        await navigateToOutputStep(user);
        await user.click(screen.getByRole("button", { name: "Choose Output Folder" }));
        await user.click(screen.getByRole("button", { name: "Continue" }));
        await user.click(screen.getByRole("button", { name: "Dry Run" }));
    
        expect(screen.getByRole("alert")).toHaveTextContent("Unable to complete dry run.");
        expect(screen.getByText("Ready to start.")).toBeInTheDocument();
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
        expect(screen.getByText("Preparing image input...")).toBeInTheDocument();
        expect(screen.getByText("Saving detection outputs...")).toBeInTheDocument();
        expect(screen.getByText("Overall progress estimate")).toBeInTheDocument();

        expect(
            screen.getByText("This may take several minutes. Please keep this window open.")
        ).toBeInTheDocument();


        expect(screen.getByRole("button", { name: "Back" })).toBeDisabled();
        expect(screen.getByRole("button", { name: "Dry Run" })).toBeDisabled();
        expect(screen.getByRole("button", { name: "Run Model" })).toBeDisabled();
    });

    it("starts an ADL model run with video-oriented progress messages", async () => {
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

        expect(screen.getByText("Run ID: adl-run-1")).toBeInTheDocument();
        expect(screen.getByText("Preparing video input...")).toBeInTheDocument();
        expect(screen.getByText("Extracting frames...")).toBeInTheDocument();
        expect(screen.getByText("Running Detic object detection...")).toBeInTheDocument();
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
        ).toHaveClass("max-h-40", "overflow-y-auto");
        
        expect(screen.getByTestId("progress-bar-fill")).toHaveStyle({ width: "50%" });
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

            expect(screen.getByTestId("progress-bar-fill")).toHaveStyle({ width: "8%" });
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

            expect(screen.getByRole("heading", { name: "EgoModelKit" })).toBeInTheDocument();

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

        expect(screen.getByRole("heading", { name: "EgoModelKit" })).toBeInTheDocument();

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

    it(
        "keeps the output-preview action disabled until the output preview page is added",
        async () => {
            const user = userEvent.setup();

            await startRunWithProgressResponses(user, [
                progressResponse({ status: "completed" }),
            ]);

            expect(
                await screen.findByRole("heading", { name: "Run completed" }),
            ).toBeInTheDocument();
    
            expect(
                screen.getByRole("button", { name: "View Output Preview" }),
            ).toBeDisabled();
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
});
