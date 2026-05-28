import re

from egomodelkit.runtime.external_code import (
    DETIC_WEIGHTS_PIN,
    DOCKER_LABEL_PREFIX,
    EXTERNAL_MODEL_ASSET_PINS,
    EXTERNAL_MODEL_CODE_PINS,
    HAND_OBJECT_DETECTOR_WEIGHTS_PIN,
    PROJECT_GITHUB_OWNER,
    docker_asset_label_arguments,
    docker_code_label_arguments,
)

FULL_SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")

def test_external_model_code_pins_are_project_controlled_forks() -> None:
    assert {pin.model_id for pin in EXTERNAL_MODEL_CODE_PINS} == {
        "hand-object-detector",
        "egovizml",
        "detic",
        "detectron2",
    }
    
    for pin in EXTERNAL_MODEL_CODE_PINS:
        assert pin.fork_repository_url.startswith(
            f"https://github.com/{PROJECT_GITHUB_OWNER}/"
        )
        
        assert pin.fork_repository_url != pin.upstream_repository_url
        assert FULL_SHA_PATTERN.fullmatch(pin.commit_sha)
    
def test_external_model_code_pins_are_manifest_ready() -> None:
    entries = [
        pin.as_manifest_entry()
        for pin in EXTERNAL_MODEL_CODE_PINS
    ]
    
    assert entries
    assert all("model_id" in entry for entry in entries)
    assert all("fork_repository_url" in entry for entry in entries)
    assert all("upstream_repository_url" in entry for entry in entries)
    assert all("commit_sha" in entry for entry in entries)

def test_external_model_asset_pins_are_manifest_ready() -> None:
    entries = [
        pin.as_manifest_entry()
        for pin in EXTERNAL_MODEL_ASSET_PINS
    ]

    assert entries
    assert all("asset_id" in entry for entry in entries)
    assert all("source_url" in entry for entry in entries)
    assert all("original_source_url" in entry for entry in entries)
    assert all("filename" in entry for entry in entries)
    assert all("download_tool" in entry for entry in entries)

def test_hand_object_detector_weights_pin_uses_google_drive_copy_with_gdown() -> None:
    assert HAND_OBJECT_DETECTOR_WEIGHTS_PIN in EXTERNAL_MODEL_ASSET_PINS
    
    assert HAND_OBJECT_DETECTOR_WEIGHTS_PIN.source_url == (
        "https://drive.google.com/file/d/1b_BkGgmYAe8VNbsFeljrSP7V1Jd8E82v/view?usp=sharing"
    )
    
    assert HAND_OBJECT_DETECTOR_WEIGHTS_PIN.original_source_url == (
        "https://drive.google.com/file/d/1H2tWsZkS7tDF8q1-jdjx6V9XrK25EDbE/view"
    )
    
    assert HAND_OBJECT_DETECTOR_WEIGHTS_PIN.download_tool == "gdown"
    
    assert HAND_OBJECT_DETECTOR_WEIGHTS_PIN.filename == (
        "faster_rcnn_1_8_132028.pth"
    )

def test_detic_weights_pin_uses_google_drive_copy_with_gdown() -> None:
    assert DETIC_WEIGHTS_PIN in EXTERNAL_MODEL_ASSET_PINS
    
    assert DETIC_WEIGHTS_PIN.source_url == (
        "https://drive.google.com/file/d/1WFgtv1_o30BzNICmSJmA-a-92ZIHcMS4/view?usp=sharing"
    )
    
    assert DETIC_WEIGHTS_PIN.original_source_url == (
        "https://dl.fbaipublicfiles.com/detic/"
        "Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth"
    )
    
    assert DETIC_WEIGHTS_PIN.download_tool == "gdown"
    
    assert DETIC_WEIGHTS_PIN.filename == (
        "Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth"
    )
    
def test_code_docker_label_arguments_inside_external_code_provenance() -> None:
    labels = docker_code_label_arguments(*EXTERNAL_MODEL_CODE_PINS)

    for pin in EXTERNAL_MODEL_CODE_PINS:        
        label_prefix = f"{DOCKER_LABEL_PREFIX}.code.{pin.model_id}"
        
        assert f"{label_prefix}.fork-repository-url={pin.fork_repository_url}" in labels
        assert f"{label_prefix}.upstream-repository-url={pin.upstream_repository_url}" in labels
        assert f"{label_prefix}.commit-sha={pin.commit_sha}" in labels       

def test_asset_docker_label_arguments_include_external_asset_provenance() -> None:
    labels = docker_asset_label_arguments(*EXTERNAL_MODEL_ASSET_PINS)

    for pin in EXTERNAL_MODEL_ASSET_PINS:
        label_prefix = f"{DOCKER_LABEL_PREFIX}.asset.{pin.asset_id}"
        
        assert f"{label_prefix}.source-url={pin.source_url}" in labels
        assert f"{label_prefix}.original-source-url={pin.original_source_url}" in labels
        assert f"{label_prefix}.filename={pin.filename}" in labels
        assert f"{label_prefix}.download-tool={pin.download_tool}" in labels

def test_code_docker_label_arguments_returns_empty_list_without_pins() -> None:
    assert docker_code_label_arguments() == []

def test_asset_docker_label_arguments_returns_empty_list_without_pins() -> None:
    assert docker_asset_label_arguments() == []
