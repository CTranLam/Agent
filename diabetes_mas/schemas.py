"""Kiểu dữ liệu chung giữa các lớp: hồ sơ, ngữ cảnh, kết quả agent, báo cáo."""
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class PatientProfile:
    name: str = ""
    age: float = 50
    sex: str = "M"
    weight_kg: float = 70.0
    height_cm: float = 165.0
    hba1c: float = 5.5
    fasting_glucose: float | None = None  # None = chưa xét nghiệm, không được coi là bình thường
    duration_years: float = 0
    hypertension: bool = False
    heart_disease: bool = False
    heart_failure: bool = False
    pregnant: bool = False
    mtc_men2_history: bool = False
    pancreatitis_history: bool = False
    smoking: str = "never"
    egfr: float = 90.0
    c_peptide: float | None = None
    autoantibody: bool | None = None
    on_insulin: bool = False
    medications: list[str] = field(default_factory=list)
    isf: float | None = None  # bác sĩ cá thể hoá; không có thì không tính liều, không dùng giá trị mặc định
    icr: float | None = None
    diabetes_type_known: str | None = None
    allergies: list[str] = field(default_factory=list)
    vegetarian: bool = False

    @property
    def bmi(self) -> float:
        return self.weight_kg / (self.height_cm / 100) ** 2


@dataclass
class Evidence:
    source: str
    title: str
    text: str
    score: float = 0.0


@dataclass
class Alert:
    level: str  # "critical" | "warning" | "info"
    message: str
    agent: str = ""


@dataclass
class AgentResult:
    agent: str
    summary: str
    details: dict[str, Any] = field(default_factory=dict)
    actions: list[str] = field(default_factory=list)
    alerts: list[Alert] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)


@dataclass
class AnalysisContext:
    profile: PatientProfile
    readings: list[float]
    classification: dict
    forecast: dict
    planned_carbs: float | None = None
    minutes_since_insulin: float | None = None
    last_insulin_units: float | None = None
    reading_age_min: float | None = None
    meal_type: str | None = None
    shared: dict = field(default_factory=dict)

    @property
    def glucose(self) -> float:
        return self.readings[-1]

    @property
    def n_readings(self) -> int:
        return len(self.readings)

    @property
    def forecast_min(self) -> float:
        return min(self.forecast["values_conservative"])

    @property
    def diabetes_type(self) -> str:
        return self.classification["code"]


LEVEL_ORDER = {"critical": 0, "warning": 1, "info": 2}


@dataclass
class Report:
    plan: str
    steps: list[str]
    classification: dict
    forecast: dict
    results: dict[str, AgentResult]
    alerts: list[Alert]
    audit_findings: list[str]
    synthesis: str | None
    synthesis_note: str | None
    sources: list[str]
    disclaimer: str
    readings: list[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_text(self) -> str:
        lines = [f"KẾ HOẠCH: {self.plan} ({' -> '.join(self.steps)})",
                 f"PHÂN LOẠI: {self.classification.get('class_name', '')}"]
        lines += [f"[{a.level.upper()}] {a.message}" for a in self.alerts]
        for key, r in self.results.items():
            lines.append(f"\n== {r.agent}: {r.summary}")
            lines += [f"  - {x}" for x in r.actions]
        if self.audit_findings:
            lines += ["\nAUDITOR ĐÃ CAN THIỆP:"] + [f"  - {x}" for x in self.audit_findings]
        if self.synthesis:
            lines += ["\nTỔNG HỢP:", self.synthesis]
        lines += ["\nNGUỒN:"] + [f"  - {s}" for s in self.sources] + ["", self.disclaimer]
        return "\n".join(lines)
