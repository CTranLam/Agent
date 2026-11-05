"""REST API + trang web. Chạy cục bộ: uvicorn diabetes_mas.api:app --host 127.0.0.1"""
import re
from dataclasses import asdict
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from .agents.orchestrator import Orchestrator
from .kb.rag import RAG
from .kb.store import load_kb
from .llm import LLM
from .scenarios import SCENARIOS
from .schemas import PatientProfile

MMOL_TO_MGDL = 18.016
INDEX = Path(__file__).parent / "web" / "index.html"
CONDITION_LABELS = {"egfr_lt_20": "eGFR < 20", "egfr_lt_30": "eGFR < 30", "egfr_lt_45": "eGFR < 45",
                    "egfr_lt_60": "eGFR < 60", "heart_failure": "Suy tim", "pregnancy": "Mang thai",
                    "mtc_men2": "Ung thư giáp thể tủy / MEN2", "pancreatitis_history": "Tiền sử viêm tụy",
                    "age_ge_75": "Từ 75 tuổi", "type1_dka": "Type 1 (nguy cơ toan ceton)"}


class Strict(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False, extra="forbid")


class ProfileIn(Strict):
    name: str = Field("", max_length=100)
    age: float = Field(50, ge=1, le=120)
    sex: Literal["M", "F"] = "M"
    weight_kg: float = Field(70, ge=20, le=300)
    height_cm: float = Field(165, ge=100, le=250)
    hba1c: float = Field(5.5, ge=3, le=20)
    fasting_glucose: float | None = Field(None, ge=20, le=600)
    duration_years: float = Field(0, ge=0, le=100)
    hypertension: bool = False
    heart_disease: bool = False
    heart_failure: bool = False
    pregnant: bool = False
    mtc_men2_history: bool = False
    pancreatitis_history: bool = False
    smoking: str = Field("never", max_length=20)
    egfr: float = Field(90, ge=0, le=200)
    c_peptide: float | None = Field(None, ge=0, le=50)
    autoantibody: bool | None = None
    on_insulin: bool = False
    medications: list[str] = Field(default_factory=list, max_length=30)
    isf: float | None = Field(None, ge=5, le=200)
    icr: float | None = Field(None, ge=1, le=100)
    diabetes_type_known: Literal["T1", "T2"] | None = None
    allergies: list[str] = Field(default_factory=list, max_length=30)
    vegetarian: bool = False


class AnalyzeIn(Strict):
    profile: ProfileIn
    readings: list[float] | None = Field(None, min_length=1, max_length=288)
    series_text: str | None = Field(None, max_length=5000)
    unit: Literal["mg/dL", "mmol/L"] = "mg/dL"
    planned_carbs: float | None = Field(None, ge=0, le=300)
    minutes_since_insulin: float | None = Field(None, ge=0, le=1440)
    last_insulin_units: float | None = Field(None, ge=0, le=100)
    reading_age_min: float | None = Field(None, ge=0, le=100000)
    meal_type: Literal["breakfast", "lunch", "dinner"] | None = None


class AskIn(Strict):
    question: str = Field(max_length=1000)


def parse_series(text: str, unit: str) -> list[float]:
    """Chuỗi người dùng gõ. Với mmol/L, dấu phẩy giữa hai chữ số là dấu thập phân (2,9 = 2.9)."""
    if unit == "mmol/L":
        text = re.sub(r"(\d),(\d)", r"\1.\2", text)
    parts = [x for x in re.split(r"[\s,;]+", text) if x]
    try:
        values = [float(x) for x in parts]
    except ValueError:
        raise ValueError("Chuỗi đường huyết chỉ được chứa số.")
    if not values or len(values) > 288:
        raise ValueError("Chuỗi đường huyết cần từ 1 đến 288 số đo.")
    return values


def graph_data(kb) -> dict:
    nodes = [{"id": d["name"], "label": d["name"], "type": "drug"} for d in kb.drugs]
    edges, conditions = [], set()
    for d in kb.drugs:
        for kind, key in (("contraindicated", "contraindications"), ("caution", "cautions")):
            for c in d[key]:
                conditions.add(c)
                edges.append({"from": d["name"], "to": c, "kind": kind, "note": d["notes"].get(c, "")})
    nodes += [{"id": c, "label": CONDITION_LABELS.get(c, c), "type": "condition"} for c in sorted(conditions)]
    edges += [{"from": i["a"], "to": i["b"], "kind": "interacts", "note": i["message"]} for i in kb.drug_interactions]
    edges += [{"from": s["drug"], "to": s["alternative"], "kind": "substitute",
               "note": f"khi {CONDITION_LABELS.get(s['condition'], s['condition'])}: {s['why']}"} for s in kb.substitutes]
    return {"nodes": nodes, "edges": edges,
            "counts": {"recommendations": len(kb.recommendations), "foods": len(kb.foods), "drugs": len(kb.drugs)},
            "meta": kb.meta}


def create_app(orchestrator: Orchestrator | None = None, rag: RAG | None = None) -> FastAPI:
    if orchestrator is None:
        from .dl.predictor import Predictor
        from .policy.rl_insulin import QPolicy
        kb = load_kb()
        rag = rag or RAG(kb)
        orchestrator = Orchestrator(Predictor(), kb, rag, QPolicy().load("artifacts/q_table.npy"), LLM())
    rag = rag or orchestrator.rag
    app = FastAPI(title="Hệ đa agent hỗ trợ tiểu đường")

    @app.exception_handler(RequestValidationError)
    def invalid(request, exc):
        # không trả lại giá trị đầu vào: NaN làm hỏng JSON, và đây là dữ liệu sức khoẻ
        detail = [{"loc": list(e["loc"]), "msg": e["msg"]} for e in exc.errors()]
        return JSONResponse({"detail": detail}, status_code=422)

    @app.post("/analyze")
    def analyze(req: AnalyzeIn):
        profile = PatientProfile(**req.profile.model_dump())
        mmol = req.unit == "mmol/L"
        try:
            raw = parse_series(req.series_text, req.unit) if req.series_text is not None else req.readings
            if raw is None:
                raise ValueError("Cần ít nhất 1 số đo đường huyết.")
            readings = [x * MMOL_TO_MGDL for x in raw] if mmol else raw
            report = orchestrator.analyze(profile, readings, req.planned_carbs, req.minutes_since_insulin,
                                          req.last_insulin_units, req.reading_age_min, req.meal_type, unit_confirmed=mmol)
        except ValueError as e:
            raise HTTPException(422, str(e))
        return report.to_dict()

    @app.post("/ask")
    def ask(req: AskIn):
        return rag.answer(req.question, llm=orchestrator.llm)

    @app.get("/scenarios")
    def scenarios():
        return {k: {**s, "profile": asdict(s["profile"])} for k, s in SCENARIOS.items()}

    @app.get("/graph")
    def graph():
        return graph_data(orchestrator.kb)

    @app.get("/health")
    def health():
        llm = orchestrator.llm
        return {"kb": "bộ nhớ", "dl": "torch" if type(orchestrator.predictor).__name__ == "Predictor" else "giả",
                "llm": bool(llm and llm.enabled)}

    @app.get("/", response_class=HTMLResponse)
    def index():
        return INDEX.read_text(encoding="utf-8")

    return app


def __getattr__(name: str):
    # dựng app thật khi uvicorn cần, để import module (trong test) không phải nạp model
    if name == "app":
        return create_app()
    raise AttributeError(name)
