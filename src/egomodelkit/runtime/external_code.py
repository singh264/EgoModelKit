""" Pinned external model-code and model-asset provenance for hidden runtimes. """

from dataclasses import dataclass
from typing import Final

PROJECT_GITHUB_OWNER: Final[str] = "singh264"
PROJECT_GITHUB_BASE_URL: Final[str] = f"https://github.com/{PROJECT_GITHUB_OWNER}"
DOCKER_LABEL_PREFIX: Final[str] = "org.egomodelkit.provenance"

@dataclass(frozen = True, slots = True)
class ExternalModelCodePin:
    """ Auditable source-code pin for one external model repository. """

    model_id: str
    fork_repository_url: str
    upstream_repository_url: str
    commit_sha: str
    
    def as_manifest_entry(self) -> dict[str, str]:
        """ Return a future run-manifest-friendly respresentation. """
        
        return {
            "model_id": self.model_id,
            "fork_repository_url":  self.fork_repository_url,
            "upstream_repository_url": self.upstream_repository_url,
            "commit_sha": self.commit_sha,
        }

@dataclass(frozen = True, slots = True)
class ExternalModelAssetPin:
    """ Auditable source pin for one external model asset. """
    
    asset_id: str
    source_url: str
    original_source_url: str
    filename: str
    download_tool: str
    
    def as_manifest_entry(self) -> dict[str, str]:
        """ Return a future run-manifest-freindly represntation. """
        
        return {
            "asset_id": self.asset_id,
            "source_url": self.source_url,
            "original_source_url": self.original_source_url,
            "filename": self.filename,
            "download_tool": self.download_tool,
        }

HAND_OBJECT_DETECTOR_PIN: Final[ExternalModelCodePin] = ExternalModelCodePin(
    model_id = "hand-object-detector",
    fork_repository_url = f"{PROJECT_GITHUB_BASE_URL}/hand_object_detector",
    upstream_repository_url = "https://github.com/ddshan/hand_object_detector",
    commit_sha = "70146cabaeffb41ecc02e6edb605fc021dbdb555",
)

EGOVIZML_PIN: Final[ExternalModelCodePin] = ExternalModelCodePin(
    model_id = "egovizml",
    fork_repository_url = f"{PROJECT_GITHUB_BASE_URL}/EgoVizML",
    upstream_repository_url = "https://github.com/adeshkadambi/EgoVizML",
    commit_sha = "b3c24d065179289cd0d99091de06ba6fe54083c8",
)

DETIC_PIN: Final[ExternalModelCodePin] = ExternalModelCodePin(
    model_id = "detic",
    fork_repository_url = f"{PROJECT_GITHUB_BASE_URL}/Detic",
    upstream_repository_url = "https://github.com/facebookresearch/Detic",
    commit_sha = "436cda2a2347df60a7c66daca0e8c59f93dc5e79",
)

DETECTRON2_PIN: Final[ExternalModelCodePin] = ExternalModelCodePin(
    model_id = "detectron2",
    fork_repository_url = f"{PROJECT_GITHUB_BASE_URL}/detectron2",
    upstream_repository_url = "https://github.com/facebookresearch/detectron2",
    commit_sha = "e0ec4e189d438848521aee7926f9900e114229f5",
)

HAND_OBJECT_DETECTOR_WEIGHTS_PIN: Final[ExternalModelAssetPin] = ExternalModelAssetPin(
    asset_id = "faster_rcnn_1_8_132028",
    source_url = (
        "https://drive.google.com/file/d/1b_BkGgmYAe8VNbsFeljrSP7V1Jd8E82v/view?usp=sharing"
    ),
    original_source_url = (
        "https://drive.google.com/file/d/1H2tWsZkS7tDF8q1-jdjx6V9XrK25EDbE/view"
    ),
    filename = "faster_rcnn_1_8_132028.pth",
    download_tool = "gdown",
)

DETIC_WEIGHTS_PIN: Final[ExternalModelAssetPin] = ExternalModelAssetPin(
    asset_id = "detic-lcocoi21k-clip-swinb-896b32-4x-ft4x-max-size",
    source_url = (
        "https://drive.google.com/file/d/1WFgtv1_o30BzNICmSJmA-a-92ZIHcMS4/view?usp=sharing"
    ),
    original_source_url = (
        "https://dl.fbaipublicfiles.com/detic/"
        "Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth"
    ),
    filename = "Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth",
    download_tool = "gdown",
)

EXTERNAL_MODEL_CODE_PINS: Final[tuple[ExternalModelCodePin, ...]] = (
    HAND_OBJECT_DETECTOR_PIN,
    EGOVIZML_PIN,
    DETIC_PIN,
    DETECTRON2_PIN,
)

EXTERNAL_MODEL_ASSET_PINS: Final[tuple[ExternalModelAssetPin, ...]] = (
    HAND_OBJECT_DETECTOR_WEIGHTS_PIN,
    DETIC_WEIGHTS_PIN,
)

def docker_code_label_arguments(*pins: ExternalModelCodePin) -> list[str]:
    """ Return Docker build labels that preserve the external code provenance. """
    arguments: list[str] = []
    
    for pin in pins:
        label_prefix = f"{DOCKER_LABEL_PREFIX}.code.{pin.model_id}"
        
        arguments.extend(
            [
                "--label",
                f"{label_prefix}.fork-repository-url={pin.fork_repository_url}",
                "--label",
                f"{label_prefix}.upstream-repository-url={pin.upstream_repository_url}",
                "--label",
                f"{label_prefix}.commit-sha={pin.commit_sha}",
            ]
        )
        
    return arguments

def docker_asset_label_arguments(*pins: ExternalModelAssetPin) -> list[str]:
    """ Return Docker build labels that preserve external asset provenance. """
    arguments: list[str] = []
    
    for pin in pins:
        label_prefix = f"{DOCKER_LABEL_PREFIX}.asset.{pin.asset_id}"
        
        arguments.extend(
            [
                "--label",
                f"{label_prefix}.source-url={pin.source_url}",
                "--label",
                f"{label_prefix}.original-source-url={pin.original_source_url}",
                "--label",
                f"{label_prefix}.filename={pin.filename}",
                "--label",
                f"{label_prefix}.download-tool={pin.download_tool}",
            ]
        )
        
    return arguments
