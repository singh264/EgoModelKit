import { createEvent, fireEvent, render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

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
    render(<App />);

    await user.click(screen.getByRole("button", { name: "Start New Run" }));
    await user.click(screen.getByRole("button", { name: modelName }));
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
            screen.getByText("Detects hands, object, and hand-object contact in images."),
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

    it("continues from hand-object model selection to an image input screen", async () => {
        const user = userEvent.setup()

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
        expect(screen.getByText("Input selected.")).toBeInTheDocument();
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

        await user.click(screen.getByRole("button", { name: "Choose input files" }));

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
        expect(screen.getByText("Input selected.")).toBeInTheDocument();
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

        vi.stubGlobal("fetch", fetchMock);
        
        await navigateToOutputStep(user);
        await user.click(screen.getByRole("button", { name: "Choose Output Folder" }));
        await user.click(screen.getByRole("button", { name: "Continue" }));
        await user.click(screen.getByRole("button", { name: "Dry Run" }));

        expect(fetchMock).toHaveBeenLastCalledWith(
            "/api/dry-run",
            expect.objectContaining({
                method: "POST",
                body: expect.any(FormData)
            }),
        );

        const dryRunRequest = fetchMock.mock.calls[1][1] as RequestInit;
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

        vi.stubGlobal("fetch", fetchMock);

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

        vi.stubGlobal("fetch", fetchMock);

        await navigateToOutputStep(user);
        await user.click(screen.getByRole("button", { name: "Choose Output Folder" }));
        await user.click(screen.getByRole("button", { name: "Continue" }));
        await user.click(screen.getByRole("button", { name: "Run Model" }));
    
        expect(fetchMock).toHaveBeenLastCalledWith(
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

        vi.stubGlobal("fetch", fetchMock);
        
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

        vi.stubGlobal("fetch", fetchMock);
        
        await navigateToOutputStep(user);
        await user.click(screen.getByRole("button", { name: "Choose Output Folder" }));
        await user.click(screen.getByRole("button", { name: "Continue" }));
        await user.click(screen.getByRole("button", { name: "Run Model" }));

        expect(await screen.getByRole("alert")).toHaveTextContent(
            "Unable to start model run",
        );
        
        expect(screen.getByText("Ready to start.")).toBeInTheDocument();
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
});
