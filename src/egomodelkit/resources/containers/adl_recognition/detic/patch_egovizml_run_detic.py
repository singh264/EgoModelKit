""" Apply compatibility fixes to EgoVizML's scripts/run_detic.py. 

The fixes are:

1. current Detic can return 2 values instead of 3 from demo.run_on_image
2. metadata class lookup should tolerate missing class names
3. thread worker exceptions must be surfaced with future.result()
"""

import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: patch_egovizml_run_detic.py /path/to/run_detic.py")
    
    path = Path(sys.argv[1])
    text = path.read_text()
    
    text = _patch_run_on_image_return(text)
    text = _patch_class_name_extraction(text)
    text = _patch_future_result(text)
    
    path.write_text(text)

def _patch_run_on_image_return(text: str) -> str:
    old = "    predictions, visualized_output, metadata = demo.run_on_image(img)\n"

    new = """    result = demo.run_on_image(img)
    
    if isinstance(result, tuple) and len(result) == 3:
        predictions, visualized_output, metadata = result
    elif isinstance(result, tuple) and len(result) == 2:
        predictions, visualized_output = result
        metadata = getattr(demo, "metadata", None)
        
        if metadata is None:
            raise RuntimeError("Detic returned 2 values and demo.metadata was not found.")
    else:
        raise RuntimeError(f"Unexpected Detic run_on_image return value: {type(result)}")
"""

    if old in text:
        return text.replace(old, new)
    
    if "result = demo.run_on_image(img)" in text:
        return text

    raise RuntimeError("Could not find EgoVizML run_on_image call to patch.")

def _patch_class_name_extraction(text: str) -> str:
    old = "    class_names = [metadata.thing_classes[i] for i in classes]\n"
    
    new = """    thing_classes = getattr(metadata, "thing_classes", [])
    class_names = [
        thing_classes[int(index)]
        if int(index) < len(thing_classes) else str(int(index))
        for index in classes
    ]
"""

    if old in text:
        return text.replace(old, new)
    
    if "thing_classes = getattr(metadata" in text:
        return text
    
    raise RuntimeError("Could not find EgoVizML class-name extraction to patch.")

def _patch_future_result(text: str) -> str:
    old = """        for future in tqdm.tqdm(
            concurrent.futures.as_completed(futures), total=len(futures)
        ):
            pass
"""

    new = """        for future in tqdm.tqdm(
            concurrent.futures.as_completed(futures), total=len(futures)
        ):
            future.result()
"""

    if old in text:
        return text.replace(old, new)
    
    if "future.result()" in text:
        return text
    
    raise RuntimeError("Could not find EgoVizML future loop to patch.")

if __name__ == "__main__":
    main()
