"""Execute only the offline notebook and retain a timed local verification record."""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import os
import sys
import time
import traceback

import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter

ROOT=Path(__file__).resolve().parent


def main():
    started=time.monotonic()
    notebook=nbformat.read(ROOT/"START_HERE.ipynb",as_version=4)
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1",
                      MPLCONFIGDIR=str(ROOT/"outputs/matplotlib-cache"), OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    os.environ["PATH"]=str(Path(sys.executable).parent)+os.pathsep+os.environ["PATH"]
    versions={p:importlib.metadata.version(p) for p in
              ("numpy","jsonschema","pandas","matplotlib","nbformat","nbclient","nbconvert","ipykernel","jupyterlab")}
    record={"python":sys.version.split()[0], "libraries":versions, "status":"STARTED",
            "scope":"Local clean-kernel offline analysis replay; no new research generation"}
    try:
        NotebookClient(notebook,timeout=180,kernel_name="python3",resources={"metadata":{"path":str(ROOT)}}).execute()
        nbformat.validate(notebook)
        assert not any(o.output_type=="error" for c in notebook.cells if c.cell_type=="code" for o in c.outputs)
        nbformat.write(notebook,ROOT/"START_HERE.ipynb")
        html,_=HTMLExporter().from_notebook_node(notebook)
        (ROOT/"PREVIEW.html").write_text(html)
        record.update(status="PASS", code_cells=sum(c.cell_type=="code" for c in notebook.cells),
                      inline_figures=sum("image/png" in o.get("data",{}) for c in notebook.cells if c.cell_type=="code" for o in c.outputs),
                      notebook_sha256=hashlib.sha256((ROOT/"START_HERE.ipynb").read_bytes()).hexdigest())
    except Exception:
        record.update(status="FAIL", diagnostic=traceback.format_exc())
        raise
    finally:
        record["elapsed_seconds"]=round(time.monotonic()-started,2)
        destination=ROOT/"verification/notebook_execution.json"
        if destination.exists():
            destination=destination.with_name(f"notebook_execution_{time.time_ns()}.json")
        with destination.open("x") as handle:json.dump(record,handle,indent=2)
        print(json.dumps(record,indent=2))


if __name__=="__main__":main()
